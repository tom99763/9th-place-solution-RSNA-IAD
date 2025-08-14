import os
import sys
from pathlib import Path
from typing import List, Tuple

import ast
import cv2
from matplotlib import image
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import multiprocessing
import glob

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from configs.data_config import * 


def apply_ct_window(image: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    lower = window_level - (window_width / 2)
    upper = window_level + (window_width / 2)
    image = np.clip(image, lower, upper)
    image = ((image - lower) / (window_width + 1e-7)) * 255.0
    image = np.clip(image, 0, 255)  
    return image




def load_series_slices(series_dir: str) -> Tuple[List[np.ndarray], pydicom.dataset.FileDataset]:
    """Load all slices from a series directory and return list of pixel arrays and sample metadata"""
    dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    
    slices = []
    sample_dcm = None
    
    for dicom_file in sorted(dicom_files):
        try:
            dcm = pydicom.dcmread(dicom_file, force=True)
            px = dcm.pixel_array
            
            # Convert to HU if CT and rescale parameters are available
            if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                px = px.astype(np.float64)
                px = px * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
            
            # Handle multi-dimensional arrays
            if px.ndim == 3:
                if px.shape[-1] == 3:  # RGB
                    px = cv2.cvtColor(px.astype(np.float64), cv2.COLOR_BGR2GRAY).astype(np.float64)
                else:  # Multi-frame, take all slices
                    for slice_idx in range(px.shape[0]):
                        slices.append(px[slice_idx].astype(np.float64))
            else:
                slices.append(px.astype(np.float64))
            
            if sample_dcm is None:
                sample_dcm = dcm
                
        except Exception as e:
            print(f"Error loading {dicom_file}: {e}")
            continue
    
    return slices, sample_dcm



def compute_axial_mip(slices: List[np.ndarray]) -> np.ndarray:
    """Compute Maximum Intensity Projection from list of slices"""
    if not slices:
        return None
    
    # Ensure all slices have the same shape
    target_shape = slices[0].shape
    processed_slices = []
    
    for slice_data in slices[2:len(slices)-2]:
        if slice_data.shape != target_shape:
            # Resize to match target shape
            slice_resized = cv2.resize(slice_data.astype(np.float32), 
                                     (target_shape[1], target_shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            processed_slices.append(slice_resized.astype(np.float64))
        else:
            processed_slices.append(slice_data)
    
    # Compute MIP
    stack = np.stack(processed_slices, axis=0)
    mip = np.max(stack, axis=0)
    
    return mip


def process_series_to_mip(uid: str, root_path: Path, mip_dir: Path) -> dict:
    """
    For a given SeriesInstanceUID, read slices, preprocess, compute axial MIP, and save.
    Returns a record with metadata for building a dataframe.
    """
    try:
        series_dir = root_path / "series" / uid
        slices, _ = load_series_slices(series_dir)
        #print(slices)
        #slices = [sl for _, sl, _ in slices]

        if len(slices) == 0:
            return {"uid": uid, "mip_filename": None}

        mip = compute_axial_mip(slices)
        if mip is None:
            return {"uid": uid, "mip_filename": None}
            # Apply percentile 0.5-99.5 windowing
        low_val = np.percentile(mip, 0.5)
        high_val = np.percentile(mip, 99.5)
        
        window_center = (low_val + high_val) / 2
        window_width = high_val - low_val

        mip = apply_ct_window(mip, window_center, window_width)
        image_uint8 = np.clip(mip, 0, 255).astype(np.uint8)
        resized = cv2.resize(image_uint8, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        mip_filename = f"{uid}_mip.npz"
        mip_path = mip_dir / mip_filename
        np.savez_compressed(mip_path, mip=resized)

        return {"uid": uid, "mip_filename": mip_filename}
    except Exception as e:
        print(f"Error processing UID {uid}: {e}")
        return {"uid": uid, "mip_filename": None}


def _worker_init(global_root: Path, global_mip_dir: Path):
    global _ROOT_PATH, _MIP_DIR
    _ROOT_PATH = global_root
    _MIP_DIR = global_mip_dir


def _worker(uid: str) -> dict:
    return process_series_to_mip(uid, _ROOT_PATH, _MIP_DIR)


if __name__ == "__main__":
    root_path = Path(data_path)
    processed_dir = root_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    mip_dir = processed_dir / "mip_images"
    mip_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    train_df = pd.read_csv(root_path / "train.csv")
    label_df = pd.read_csv(root_path / "train_localizers.csv")
    mf_dicom_uids = pd.read_csv(root_path / "multiframe_dicoms.csv")

    ignore_uids = [
        "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
        "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
        "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242",
    ] + list(mf_dicom_uids["SeriesInstanceUID"])

    train_df = train_df[~train_df["SeriesInstanceUID"].isin(ignore_uids)].reset_index(drop=True)
    train_df["fold_id"] = 0

    sgkf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    for i, (_, test_index) in enumerate(
        sgkf.split(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"])
    ):
        train_df.loc[test_index, "fold_id"] = i

    uids = train_df["SeriesInstanceUID"].unique().tolist()
    print(f"Starting MIP computation for {len(uids)} UIDs...")

    with multiprocessing.Pool(
        processes=CORES,
        initializer=_worker_init,
        initargs=(root_path, mip_dir),
    ) as pool:
        results = list(tqdm(pool.imap_unordered(_worker, uids), total=len(uids)))

    # Build dataframe of MIPs
    mip_records = [r for r in results if r and r.get("mip_filename")]
    mip_df = pd.DataFrame(mip_records)
    mip_df.rename(columns={"uid": "series_uid"}, inplace=True)
    # Map fold and target
    uid_to_fold = dict(zip(train_df["SeriesInstanceUID"], train_df["fold_id"]))
    uid_to_target = dict(zip(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"]))
    mip_df["fold_id"] = mip_df["series_uid"].map(uid_to_fold)
    mip_df["series_has_aneurysm"] = mip_df["series_uid"].map(uid_to_target)
    mip_df.to_csv(processed_dir / "mip_df.csv", index=False)
    print(f"Created {len(mip_df)} MIP images. Saved to: {mip_dir}")
    print(f"Metadata saved to: {processed_dir / 'mip_df.csv'}")

