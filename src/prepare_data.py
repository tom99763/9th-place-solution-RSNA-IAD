#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ast
import pydicom

from pathlib import Path
import os
import pydicom
import cv2
import multiprocessing
from tqdm import tqdm


def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float, preserve_contrast=True) -> np.ndarray:
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    img_clipped = np.clip(img, img_min, img_max)
    img_normalized = (img_clipped - img_min) / (img_max - img_min + 1e-7)
    img_processed = (img_normalized * 255).astype(np.uint8)
    if preserve_contrast:
        img_processed = cv2.equalizeHist(img_processed)
    return img_processed

def get_windowing_params(modality: str) -> tuple[float, float]:
    """Get appropriate windowing for different modalities"""
    windows = {
        'CT': (40, 80),
        'CTA': (50, 350),
        'MR': (600, 1200),
        'MRA': (600, 1200),
        'MRI': (40, 80),
    }
    return windows.get(modality, (40, 80))

def process_slice(img,ds):
    modality = getattr(ds, 'Modality', 'CT')
    
    # Apply rescale if available
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept
        
    window_center, window_width = get_windowing_params(modality)
    img = apply_dicom_windowing(img, window_center, window_width)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def process_dicom_series(uid: str):
    """Process a DICOM series and extract metadata"""
    series_path = Path(f"../data/series/{uid}")
    
    # Find all DICOM files
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    if len(all_filepaths) == 0:
        print(f"No DCM files found for {uid}")
        return np.array([])
        
    # Process DICOM files
    slices = []
    instance_numbers = []
    
    for _, filepath in enumerate(all_filepaths):
        ds = pydicom.dcmread(filepath, force=True)
        
        # print(ds.InstanceNumber)
        img = ds.pixel_array.astype(np.float32)
        if img.ndim == 3:
            if img.shape[-1] == 3:
                imgs = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)]
            else:
                imgs = []
                for i in range(img.shape[0]):
                    imgs.append(img[i, :, :])

        else:
            imgs = [img]
            
        for img in imgs:
            if hasattr(ds, "InstanceNumber"):
                instance_numbers.append(ds.InstanceNumber)
            
            if hasattr(ds, "ImagePositionPatient"):
                slices.append((ds.ImagePositionPatient[-1],process_slice(img,ds)))
            elif hasattr(ds, "InstanceNumber"):
                slices.append((int(ds.InstanceNumber),process_slice(img,ds)))
            else:
                slices.append((0,process_slice(img,ds)))


    # sometimes it's the case that the starting instance number is greater than 1. So we want to get the start_instance_number and then substract it from the z axis.
    instance_numbers = sorted(instance_numbers)
    start_instance_number = instance_numbers[0] - 1


    # we sort all the slices by ImagePositionPatient or InstanceNumber
    slices = sorted(slices, key = lambda x: x[0])
    
    volume = np.array([slice[-1] for slice in slices])

    # We get nth slice of the volume
    selected_idxs = [*range(0,volume.shape[0],FACTOR)]

    uid_label_df = label_df[label_df["SeriesInstanceUID"] == uid]

    # make sure that all the instance_numbers starts from 0.
    required_idxs = [idx - start_instance_number for idx in list(uid_label_df["z"])]
    
    if len(required_idxs) != 0:

        # If the current case is positive, then we make sure that we get all the corresponding slices in the z axis regardless of the factor
        final_idxs = sorted(list(set(selected_idxs).union(required_idxs)))
    else:
        final_idxs = sorted(selected_idxs)

    if len(required_idxs) != 0:

        # Since the factor have changed the original indexes, we want to remap them.
        mapped_idxs = [final_idxs.index(idx) for idx in required_idxs]

        return volume[final_idxs], mapped_idxs
    else:
        return volume[final_idxs], []

def process_and_save(uid):
    """Processes a single DICOM series and saves it to a .npz file."""
    try:
        vol, mapped_idx = process_dicom_series(uid)
        np.savez_compressed(target_dir / f"{uid}.npz", vol=vol) # Use savez_compressed for smaller files
        return {"uid": uid, "mapped_idx": mapped_idx}, None # Return UID on success
    except Exception as e:
        return {"uid": uid, "mapped_idx": []}, e # Return UID and the error if something fails


if __name__ == "__main__":

    IMG_SIZE = 512
    FACTOR = 3
    SEED = 42
    N_FOLDS = 5
    CORES = 16


    root_path = Path("../data")
    target_dir = root_path / "processed"

    os.mkdir(target_dir, exist_ok=True)

    train_df = pd.read_csv(root_path / "train.csv")
    label_df = pd.read_csv(root_path / "train_localizers.csv")
    mf_dicom_uids = pd.read_csv(root_path / "multiframe_dicoms.csv")


    # We don't want to include multiframe dicoms as we can't get there z axis
    # Discussion: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/591546
    ignore_uids = [
        "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
        "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
        "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242"
    ] + list(mf_dicom_uids["SeriesInstanceUID"])

    train_df = train_df[~train_df["SeriesInstanceUID"].isin(ignore_uids)].reset_index(drop=True)
    train_df["fold_id"] = 0

    sgkf = StratifiedKFold(n_splits=N_FOLDS,random_state=SEED, shuffle=True)

    for i, (train_index, test_index) in enumerate(sgkf.split(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"])):
        train_df.loc[test_index, "fold_id"] =  i


    label_df["x"] = [s['x'] for s in list(label_df["coordinates"].map(ast.literal_eval)) ]
    label_df["y"] = [s['y'] for s in list(label_df["coordinates"].map(ast.literal_eval)) ]
    label_df["z"] = -1

    for idx,rowdf in label_df.iterrows():
        uid,f = rowdf["SeriesInstanceUID"],rowdf["SOPInstanceUID"]

        # we want to get the corresponding z axis for the given volume, usually InstanceNumber starts with 1 so we substract 1 from it.
        label_df.loc[idx,'z'] = int(pydicom.dcmread(root_path / "series" / f"{uid}/{f}.dcm").InstanceNumber) - 1

    del label_df["coordinates"]
        

    # Get the list of unique UIDs to process
    uids_to_process = train_df["SeriesInstanceUID"].unique()
    total_uids = len(uids_to_process)

    print(f"Starting processing for {total_uids} UIDs...")

    with multiprocessing.Pool(processes=CORES) as pool:
    
        results = list(tqdm(pool.imap_unordered(process_and_save, uids_to_process), total=total_uids))

    print("\nProcessing complete.")

    for r in results:
        data = r[0]
        if len(data["mapped_idx"]) != 0:
            label_df.loc[label_df["SeriesInstanceUID"] == data["uid"],'z'] = data["mapped_idx"]

    label_df.to_csv(target_dir / "label_df.csv", index=False)
    train_df.to_csv(target_dir / "train_df.csv", index=False)
