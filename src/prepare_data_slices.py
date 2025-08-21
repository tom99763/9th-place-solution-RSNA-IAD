import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ast
from pathlib import Path
import os
import pydicom
import cv2
import multiprocessing
from tqdm import tqdm
import sys
from typing import Tuple

# Add the parent directory to the path to find configs module
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from configs.data_config import *

# Globals used by multiprocessing workers
label_df_global = None  # original full label df (for fallback / reference)
labels_by_uid_global = None  # dict: uid -> list of label tuples (row_index, instance_number, location,x,y)
target_dir_global = None

# Raw HU clipping range (aligned with MIP preprocessing) so we can window later at training time
RAW_MIN_HU = -1200.0
RAW_MAX_HU = 4000.0

def initializer(label_df, labels_by_uid, target_dir):
    """Initializer for worker processes.

    Parameters
    ----------
    label_df : pd.DataFrame
        Original label dataframe (used only for reference / index updates)
    labels_by_uid : dict
        Mapping UID -> list[(row_index, instance_number, location, x, y)] for O(1) lookup.
    target_dir : Path
        Base target directory.
    """
    global label_df_global, labels_by_uid_global, target_dir_global
    label_df_global = label_df
    labels_by_uid_global = labels_by_uid
    target_dir_global = target_dir

def apply_ct_window(image: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    lower = window_level - (window_width / 2)
    upper = window_level + (window_width / 2)
    image = np.clip(image, lower, upper)
    image = ((image - lower) / (window_width + 1e-7)) * 255.0

    return image


def preprocess_dcm_slice(image, dcm, output_size=(IMG_SIZE, IMG_SIZE), **kwargs):
    """Convert a raw DICOM slice to float32 HU clipped to a fixed range and resized.

    We intentionally DO NOT apply any windowing or uint8 normalization here so that
    training code can perform dynamic / multi-window augmentations later (same idea
    as in MIP preprocessing). Returned array is float32 in raw HU space clipped to
    [RAW_MIN_HU, RAW_MAX_HU].
    """
    image = image.astype(np.float32)

    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        # Convert stored values to Hounsfield Units
        image = image * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

    # Clip to predefined HU range
    image = np.clip(image, RAW_MIN_HU, RAW_MAX_HU)

    # Resize to model target size (keeping float32)
    if image.shape != output_size:
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    return image

    

def process_dicom_series_to_slices(uid: str):
    """Process one DICOM series, subsample slices, and save ONE NPZ per series.

    Returns
    -------
    slices_metadata : list[dict]
        Per-slice rows for the slice dataframe (still one row per kept slice).
    label_z_updates : list[tuple]
        List of (label_row_index, new_z) used later to update label_df inplace.
    """
    global target_dir_global, labels_by_uid_global

    series_path = Path(f"{data_path}/series/{uid}")
    all_filepaths = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(series_path)
        for f in files if f.endswith('.dcm')
    ])

    if not all_filepaths:
        return [], []

    temp_slices = []  # (z_pos, processed_img, instance_number, modality)
    instance_numbers_encountered = []
    modality_global = None

    # Read ALL slices (first pass) – unavoidable but we keep logic tight
    for fp in all_filepaths:
        try:
            ds = pydicom.dcmread(fp, force=True)
            pix = ds.pixel_array.astype(np.float32)
            if pix.ndim == 3 and pix.shape[-1] == 3:
                # Convert RGB to grayscale
                pix_list = [cv2.cvtColor(pix, cv2.COLOR_BGR2GRAY).astype(np.float32)]
            elif pix.ndim == 3:
                # Multiframe (frames, H, W) -> list of frames
                pix_list = [p.astype(np.float32) for p in pix]
            else:
                pix_list = [pix]

            modality_global = modality_global or str(getattr(ds, "Modality", "")).upper()
            inst_num = int(getattr(ds, "InstanceNumber", 0))
            z_val = getattr(ds, "ImagePositionPatient", [0])[-1] if hasattr(ds, "ImagePositionPatient") else inst_num

            for frame_img in pix_list:
                processed = preprocess_dcm_slice(frame_img, ds)
                temp_slices.append((z_val, processed, inst_num, modality_global))
                instance_numbers_encountered.append(inst_num)
        except Exception as e:
            print(f"[WARN] {uid}: failed reading {fp}: {e}")
            continue

    if not temp_slices:
        return [], []

    # Sort by z (robust ordering)
    temp_slices.sort(key=lambda x: x[0])
    total_slices = len(temp_slices)

    # Build mapping original InstanceNumber -> ALL sorted indices (handle duplicates) – usually one
    instance_to_sorted_idxs = {}
    for sorted_idx, (_z, _img, inst_num, _m) in enumerate(temp_slices):
        instance_to_sorted_idxs.setdefault(inst_num, []).append(sorted_idx)

    # Base subsampling by FACTOR
    base_selected = set(range(0, total_slices, FACTOR))

    # Required indices from labels (robust mapping)
    label_tuples = labels_by_uid_global.get(uid, [])  # list[(row_index, instance_number, location,x,y)]
    required_sorted_idxs = set()
    for (_row_idx, inst_num, _loc, _x, _y) in label_tuples:
        if inst_num in instance_to_sorted_idxs:
            required_sorted_idxs.update(instance_to_sorted_idxs[inst_num])

    final_sorted_idxs = sorted(base_selected.union(required_sorted_idxs))

    # Build mapping sorted_idx -> new_kept_index
    kept_index_mapping = {sorted_idx: new_i for new_i, sorted_idx in enumerate(final_sorted_idxs)}

    # Compute new z for each label row (first occurrence of its instance number among kept slices)
    label_z_updates = []
    for (row_idx, inst_num, _loc, _x, _y) in label_tuples:
        new_z = None
        for s_idx in instance_to_sorted_idxs.get(inst_num, []):
            if s_idx in kept_index_mapping:
                new_z = kept_index_mapping[s_idx]
                break
        if new_z is not None:
            label_z_updates.append((row_idx, new_z))
        else:
            # Edge: label slice dropped by subsampling (should not happen since we forced required)
            pass

    # Collect kept slices into array (N, H, W) float16 to save space
    kept_slices = []
    kept_instance_numbers = []
    kept_z_positions = []
    for sorted_idx in final_sorted_idxs:
        z_val, img_arr, inst_num, modality = temp_slices[sorted_idx]
        kept_slices.append(img_arr)
        kept_instance_numbers.append(inst_num)
        kept_z_positions.append(z_val)

    series_dir = target_dir_global / "series_slices"
    series_dir.mkdir(exist_ok=True, parents=True)
    series_filename = f"{uid}.npz"
    series_path = series_dir / series_filename
    try:
        np.savez_compressed(
            series_path,
            slices=np.stack(kept_slices, axis=0).astype(np.float16),  # (K, H, W)
            instance_numbers=np.array(kept_instance_numbers, dtype=np.int32),
            z_positions=np.array(kept_z_positions, dtype=np.float32),
            raw_hu_range=np.array([RAW_MIN_HU, RAW_MAX_HU], dtype=np.float32),
            modality=np.array(modality_global or "")
        )
    except Exception as e:
        print(f"[ERR] saving series {uid}: {e}")
        return [], []

    # Per-slice metadata rows for dataframe
    slices_metadata = []
    # Build quick label lookup: inst_num -> list of (location,x,y)
    label_lookup = {}
    for (_row_idx, inst_num, loc, x, y) in label_tuples:
        label_lookup.setdefault(inst_num, []).append((loc, x, y))

    for new_idx, sorted_idx in enumerate(final_sorted_idxs):
        z_val, _img, inst_num, modality = temp_slices[sorted_idx]
        label_list = label_lookup.get(inst_num, [])
        has_aneurysm = len(label_list) > 0
        locations = [l for (l, _x, _y) in label_list]
        x_coords = [x for (_l, x, _y) in label_list]
        y_coords = [y for (_l, _x, y) in label_list]
        slices_metadata.append({
            'series_npz': series_filename,
            'series_uid': uid,
            'slice_idx_in_series': new_idx,
            'original_sorted_idx': sorted_idx,
            'z_position': z_val,
            'instance_number': inst_num,
            'has_aneurysm': has_aneurysm,
            'num_aneurysms': len(label_list) if has_aneurysm else 0,
            'aneurysm_locations': locations if has_aneurysm else [],
            'aneurysm_x_coords': x_coords if has_aneurysm else [],
            'aneurysm_y_coords': y_coords if has_aneurysm else []
        })

    return slices_metadata, label_z_updates

def process_and_save_slices(uid: str):
    """Worker wrapper for a single UID."""
    try:
        slices_metadata, label_updates = process_dicom_series_to_slices(uid)
        return {"uid": uid, "slices_metadata": slices_metadata, "label_updates": label_updates}
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        return {"uid": uid, "slices_metadata": [], "label_updates": []}

if __name__ == "__main__":
    root_path = Path(data_path)
    target_dir = root_path / "processed"
    os.makedirs(target_dir, exist_ok=True)
    
    # New directory for per-series NPZ files (created lazily in worker too)
    series_slices_dir = target_dir / "series_slices"
    os.makedirs(series_slices_dir, exist_ok=True)

    train_df = pd.read_csv(root_path / "train.csv")
    label_df = pd.read_csv(root_path / "train_localizers.csv")
    mf_dicom_uids = pd.read_csv(root_path / "multiframe_dicoms.csv")

    ignore_uids = [
        "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
        "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
        "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242"
    ] + list(mf_dicom_uids["SeriesInstanceUID"])

    train_df = train_df[~train_df["SeriesInstanceUID"].isin(ignore_uids)].reset_index(drop=True)
    train_df["fold_id"] = 0

    # Create folds
    sgkf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    for i, (_, test_index) in enumerate(sgkf.split(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"])):
        train_df.loc[test_index, "fold_id"] = i

    # Process label coordinates
    label_df["x"] = label_df["coordinates"].map(lambda x: ast.literal_eval(x)['x'])
    label_df["y"] = label_df["coordinates"].map(lambda x: ast.literal_eval(x)['y'])
    label_df["z"] = -1

    # Get z-coordinates (instance numbers) from DICOM files (store original InstanceNumber as z+1)
    for idx, row in label_df.iterrows():
        uid, sop = row["SeriesInstanceUID"], row["SOPInstanceUID"]
        dcm_path = root_path / "series" / uid / f"{sop}.dcm"
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            inst_num = int(getattr(ds, 'InstanceNumber', 0))
            label_df.at[idx, 'z'] = inst_num - 1  # maintain legacy semantics
        except Exception as e:
            print(f"Failed to read DICOM for label {uid}/{sop}: {e}")

    label_df.drop(columns=["coordinates"], inplace=True)

    # Precompute label lookup mapping: uid -> list[(row_index, instance_number, location, x, y)]
    labels_by_uid = {}
    for idx, row in label_df.iterrows():
        uid = row['SeriesInstanceUID']
        # original instance number = z + 1
        inst_num = int(row['z']) + 1 if row['z'] >= 0 else -1
        labels_by_uid.setdefault(uid, []).append((idx, inst_num, row['location'], row['x'], row['y']))

    uids_to_process = train_df["SeriesInstanceUID"].unique()
    print(f"Starting series-wise processing (per-series NPZ) for {len(uids_to_process)} UIDs...")

    processes = min( max(1, multiprocessing.cpu_count() - 1), 6)  # cap to 16 to avoid oversubscription
    with multiprocessing.Pool(
        processes=processes,
        initializer=initializer,
        initargs=(label_df, labels_by_uid, target_dir)
    ) as pool:
        results = list(tqdm(pool.imap_unordered(process_and_save_slices, uids_to_process), total=len(uids_to_process)))

    print("\nSlice processing complete.")

    # Flatten results and create slice dataframe
    all_slice_metadata = []
    label_updates_all = []
    for result in results:
        if result:
            all_slice_metadata.extend(result["slices_metadata"])
            label_updates_all.extend(result.get("label_updates", []))

    slice_df = pd.DataFrame(all_slice_metadata)
    
    # Add fold information based on series UID
    uid_to_fold = dict(zip(train_df["SeriesInstanceUID"], train_df["fold_id"]))
    slice_df["fold_id"] = slice_df["series_uid"].map(uid_to_fold)
    
    # Add series-level aneurysm information
    uid_to_aneurysm_present = dict(zip(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"]))
    slice_df["series_has_aneurysm"] = slice_df["series_uid"].map(uid_to_aneurysm_present)

    # Save the slice-level dataframe
    slice_df.to_csv(target_dir / "slice_df.csv", index=False)

    # Update label_df z with new subsampled indices
    for (row_idx, new_z) in label_updates_all:
        label_df.at[row_idx, 'z'] = new_z

    
    # Also save the original dataframes for compatibility
    label_df.to_csv(target_dir / "label_df_slices.csv", index=False)
    train_df.to_csv(target_dir / "train_df_slices.csv", index=False)
    
    print(f"Created slice dataframe with {len(slice_df)} individual slices (stored in per-series NPZ)")
    print(f"Slices with aneurysms: {slice_df['has_aneurysm'].sum()}")
    print(f"Slices without aneurysms: {(~slice_df['has_aneurysm']).sum()}")
    print(f"Files saved to: {target_dir}")