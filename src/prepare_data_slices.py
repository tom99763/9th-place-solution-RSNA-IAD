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
label_df_global = None
target_dir_global = None

def initializer(label_df, target_dir):
    global label_df_global
    global target_dir_global
    label_df_global = label_df
    target_dir_global = target_dir

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float, preserve_contrast=True) -> np.ndarray:
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    img_clipped = np.clip(img, img_min, img_max)
    img_normalized = (img_clipped - img_min) / (img_max - img_min + 1e-7)
    img_processed = (img_normalized * 255).astype(np.uint8)
    if preserve_contrast:
        img_processed = cv2.equalizeHist(img_processed)
    return img_processed

def get_windowing_params(modality: str) -> Tuple[float, float]:
    return windows.get(modality, (40, 80))

def process_slice(img, ds):
    modality = getattr(ds, 'Modality', 'CT')
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    #window_center, window_width = get_windowing_params(modality)
    #img = apply_dicom_windowing(img, window_center, window_width)
    img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-7)
    img = (img_normalized * 255).astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def preprocess_dcm_slice(image, dcm, output_size=(IMG_SIZE, IMG_SIZE), window_level=150, window_width=350):
    """
    Reads and preprocesses a single DICOM slice from a CTA or MRA scan.

    For CTA scans, it applies a specific vascular window to highlight arteries.
    For other modalities like MRA, it performs standard min-max normalization.
    The final image is resized and returned as an 8-bit grayscale numpy array.

    Args:
        dcm_path (str): The full path to the .dcm file.
        output_size (tuple): The target dimensions for the output image (width, height).
        window_level (int): The window level (center) for CTA windowing in HU.
        window_width (int): The window width for CTA windowing in HU.

    Returns:
        numpy.ndarray: The preprocessed 8-bit grayscale image, or None if an error occurs.
    """
    try:

        # Get the pixel data from the DICOM file
        image = image.astype(np.float64)

        # 2. Check if the modality is 'CT' to decide on the processing method
        # The DICOM tag (0008,0060) specifies the modality
        is_ct_scan = 'CT' in dcm.get('Modality', '').upper()

        if is_ct_scan:
            # For CT scans, convert pixel data to Hounsfield Units (HU)
            # using the Rescale Slope and Intercept values from DICOM metadata
            if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
                image = image * dcm.RescaleSlope + dcm.RescaleIntercept

            # Apply the vascular windowing
            lower_bound = window_level - (window_width / 2)
            upper_bound = window_level + (window_width / 2)
            
            # Clip the image to the window range
            image = np.clip(image, lower_bound, upper_bound)
            
            # Normalize the windowed image to a 0-255 scale
            image = ((image - lower_bound) / window_width) * 255.0

        else:
            # 3. For non-CT scans (like MRA), perform standard min-max normalization
            if np.max(image) != np.min(image):
                image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0

        # Convert the final processed image to an 8-bit unsigned integer format
        image = image.astype(np.uint8)

        # 4. Resize the image to the desired output size
        processed_image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)

        return processed_image

    except Exception as e:
        print(f"Error processing the file {dcm_path}: {e}")
        return None
def process_dicom_series_to_slices(uid: str):
    """
    Process DICOM series and save individual slices instead of volumes.
    Returns a tuple of (slices_metadata, mapped_idxs) for creating the slice dataframe.
    """
    global label_df_global, target_dir_global
    
    series_path = Path(f"{data_path}/series/{uid}")
    all_filepaths = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(series_path)
        for file in files if file.endswith('.dcm')
    ])

    if not all_filepaths:
        print(f"No DCM files found for {uid}")
        return [], []

    slices_metadata = []
    instance_numbers = []

    # First pass: collect all slices and instance numbers
    temp_slices = []
    for filepath in all_filepaths:
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)

            if img.ndim == 3 and img.shape[-1] == 3:
                imgs = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)]
            elif img.ndim == 3:
                imgs = list(img)
            else:
                imgs = [img]

            for img_idx, img in enumerate(imgs):
                if hasattr(ds, "InstanceNumber"):
                    instance_numbers.append(ds.InstanceNumber)
                z_val = getattr(ds, "ImagePositionPatient", [0])[-1] if hasattr(ds, "ImagePositionPatient") else int(getattr(ds, "InstanceNumber", 0))
                processed_img = preprocess_dcm_slice(img, ds)
                temp_slices.append((z_val, processed_img, ds.InstanceNumber if hasattr(ds, "InstanceNumber") else 0))
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    if not temp_slices:
        return [], []

    # Sort slices by z-position
    temp_slices = sorted(temp_slices, key=lambda x: x[0])
    
    # Determine which slices to keep (subsampling + required labeled slices)
    instance_numbers = sorted(instance_numbers) if instance_numbers else [0]
    start_instance_number = instance_numbers[0] - 1 if instance_numbers else 0
    
    total_slices = len(temp_slices)
    selected_idxs = list(range(0, total_slices, FACTOR))
    
    # Get required indices for this UID from labels
    uid_label_df = label_df_global[label_df_global["SeriesInstanceUID"] == uid]
    required_idxs = []
    if not uid_label_df.empty:
        required_idxs = [int(idx) - start_instance_number for idx in uid_label_df["z"]]
        # Ensure required indices are within bounds
        required_idxs = [idx for idx in required_idxs if 0 <= idx < total_slices]

    # Combine selected and required indices
    if required_idxs:
        final_idxs = sorted(set(selected_idxs).union(required_idxs))
        # Calculate mapped indices for the required slices
        mapped_idxs = [final_idxs.index(idx) for idx in required_idxs]
    else:
        final_idxs = sorted(selected_idxs)
        mapped_idxs = []

    # Save individual slices and collect metadata
    for new_idx, original_idx in enumerate(final_idxs):
        if original_idx < len(temp_slices):
            z_val, processed_img, instance_num = temp_slices[original_idx]
            
            # Create unique slice filename
            slice_filename = f"{uid}_{new_idx:03d}.npz"
            slice_path = target_dir_global / "individual_slices" / slice_filename
            
            # Save individual slice
            np.savez_compressed(slice_path, slice=processed_img)
            
            # Check if this slice has labels
            slice_labels = uid_label_df[uid_label_df["z"] == (instance_num - 1)]
            has_aneurysm = len(slice_labels) > 0
            
            # Collect metadata
            slice_metadata = {
                'slice_filename': slice_filename,
                'series_uid': uid,
                'slice_idx_in_series': new_idx,
                'original_slice_idx': original_idx,
                'z_position': z_val,
                'instance_number': instance_num,
                'has_aneurysm': has_aneurysm,
                'num_aneurysms': len(slice_labels) if has_aneurysm else 0
            }
            
            # Add location information if labels exist
            if has_aneurysm:
                locations = slice_labels['location'].tolist()
                x_coords = slice_labels['x'].tolist()
                y_coords = slice_labels['y'].tolist()
                slice_metadata.update({
                    'aneurysm_locations': locations,
                    'aneurysm_x_coords': x_coords,
                    'aneurysm_y_coords': y_coords
                })
            
            slices_metadata.append(slice_metadata)

    return slices_metadata, mapped_idxs

def process_and_save_slices(uid: str):
    """Process a single UID and save its slices individually."""
    try:
        slices_metadata, mapped_idxs = process_dicom_series_to_slices(uid)
        return {"uid": uid, "slices_metadata": slices_metadata, "mapped_idx": mapped_idxs}
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        return {"uid": uid, "slices_metadata": [], "mapped_idx": []}

if __name__ == "__main__":
    root_path = Path(data_path)
    target_dir = root_path / "processed"
    os.makedirs(target_dir, exist_ok=True)
    
    # Create directory for individual slices
    individual_slices_dir = target_dir / "individual_slices"
    os.makedirs(individual_slices_dir, exist_ok=True)

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

    # Get z-coordinates from DICOM files
    for idx, row in label_df.iterrows():
        uid, sop = row["SeriesInstanceUID"], row["SOPInstanceUID"]
        dcm_path = root_path / "series" / uid / f"{sop}.dcm"
        try:
            ds = pydicom.dcmread(dcm_path)
            label_df.at[idx, 'z'] = int(ds.InstanceNumber) - 1
        except Exception as e:
            print(f"Failed to read DICOM for label {uid}/{sop}: {e}")

    label_df.drop(columns=["coordinates"], inplace=True)

    uids_to_process = train_df["SeriesInstanceUID"].unique()
    print(f"Starting slice-wise processing for {len(uids_to_process)} UIDs...")

    # Process UIDs and collect slice metadata
    with multiprocessing.Pool(
        processes=4,
        initializer=initializer,
        initargs=(label_df, target_dir)
    ) as pool:
        results = list(tqdm(pool.imap_unordered(process_and_save_slices, uids_to_process), total=len(uids_to_process)))

    print("\nSlice processing complete.")

    # Flatten results and create slice dataframe
    all_slice_metadata = []
    for result in results:
        if result:  # Skip empty results
            all_slice_metadata.extend(result["slices_metadata"])

    slice_df = pd.DataFrame(all_slice_metadata)
    
    # Add fold information based on series UID
    uid_to_fold = dict(zip(train_df["SeriesInstanceUID"], train_df["fold_id"]))
    slice_df["fold_id"] = slice_df["series_uid"].map(uid_to_fold)
    
    # Add series-level aneurysm information
    uid_to_aneurysm_present = dict(zip(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"]))
    slice_df["series_has_aneurysm"] = slice_df["series_uid"].map(uid_to_aneurysm_present)

    # Save the slice-level dataframe
    slice_df.to_csv(target_dir / "slice_df.csv", index=False)

    # Create a mapping dictionary for easier updates
    uid_to_mapped_z = {}
    for r in results:
        uid, mapped = r["uid"], r["mapped_idx"]
        if mapped:
            uid_to_mapped_z[uid] = mapped
    
    # Update the z values in label_df
    for uid, mapped_indices in uid_to_mapped_z.items():
        uid_rows = label_df[label_df["SeriesInstanceUID"] == uid].copy()
        if len(mapped_indices) == len(uid_rows):
            # Update z values for this UID
            for i, (idx, row) in enumerate(uid_rows.iterrows()):
                label_df.at[idx, 'z'] = mapped_indices[i]
        else:
            print(f"Warning: Mismatch in number of labels for UID {uid}. Expected {len(uid_rows)}, got {len(mapped_indices)}")

    
    # Also save the original dataframes for compatibility
    label_df.to_csv(target_dir / "label_df_slices.csv", index=False)
    train_df.to_csv(target_dir / "train_df_slices.csv", index=False)
    
    print(f"Created slice dataframe with {len(slice_df)} individual slices")
    print(f"Slices with aneurysms: {slice_df['has_aneurysm'].sum()}")
    print(f"Slices without aneurysms: {(~slice_df['has_aneurysm']).sum()}")
    print(f"Files saved to: {target_dir}")