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

sys.path.append('../')
from configs.data_config import *

# Globals used by multiprocessing workers
label_df_global = None
target_dir_global = None

def initializer(label_df, target_dir):
    global label_df_global
    global target_dir_global
    label_df_global = label_df
    target_dir_global = target_dir

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def get_windowing_params(modality: str) -> Tuple[float, float]:
    return windows.get(modality, (40, 80))

def process_slice(img, ds):
    modality = getattr(ds, 'Modality', 'CT')
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    window_center, window_width = get_windowing_params(modality)
    img = apply_dicom_windowing(img, window_center, window_width)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def process_dicom_series(uid: str):
    global label_df_global
    series_path = Path(f"{data_path}/series/{uid}")
    all_filepaths = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(series_path)
        for file in files if file.endswith('.dcm')
    ])

    if not all_filepaths:
        print(f"No DCM files found for {uid}")
        return np.array([]), []

    slices = []
    instance_numbers = []

    for filepath in all_filepaths:
        ds = pydicom.dcmread(filepath, force=True)
        img = ds.pixel_array.astype(np.float32)

        if img.ndim == 3 and img.shape[-1] == 3:
            imgs = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)]
        elif img.ndim == 3:
            imgs = list(img)
        else:
            imgs = [img]

        for img in imgs:
            if hasattr(ds, "InstanceNumber"):
                instance_numbers.append(ds.InstanceNumber)
            z_val = getattr(ds, "ImagePositionPatient", [0])[-1] if hasattr(ds, "ImagePositionPatient") else int(getattr(ds, "InstanceNumber", 0))
            slices.append((z_val, process_slice(img, ds)))

    instance_numbers = sorted(instance_numbers)
    start_instance_number = instance_numbers[0] - 1 if instance_numbers else 0
    slices = sorted(slices, key=lambda x: x[0])
    volume = np.array([s[1] for s in slices])

    selected_idxs = list(range(0, volume.shape[0], FACTOR))
    uid_label_df = label_df_global[label_df_global["SeriesInstanceUID"] == uid]
    required_idxs = [int(idx) - start_instance_number for idx in uid_label_df["z"]]

    if required_idxs:
        final_idxs = sorted(set(selected_idxs).union(required_idxs))
        mapped_idxs = [final_idxs.index(idx) for idx in required_idxs]
    else:
        final_idxs = sorted(selected_idxs)
        mapped_idxs = []

    return volume[final_idxs], mapped_idxs

def process_and_save(uid: str):
    try:
        vol, mapped_idx = process_dicom_series(uid)
        np.savez_compressed(target_dir_global / f"slices/{uid}.npz", vol=vol)
        return {"uid": uid, "mapped_idx": mapped_idx}
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        return {"uid": uid, "mapped_idx": []}

if __name__ == "__main__":
    root_path = Path(data_path)
    target_dir = root_path / "processed"
    os.makedirs(target_dir, exist_ok=True)

    train_df = pd.read_csv(root_path / "train.csv")
    label_df = pd.read_csv(root_path / "train_localizers.csv")
    mf_dicom_uids = pd.read_csv(root_path / "multiframe_dicoms.csv")

    if not os.path.exists(f'{target_dir}/slices'):
        os.makedirs(f'{target_dir}/slices')

    ignore_uids = [
        "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
        "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
        "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242"
    ] + list(mf_dicom_uids["SeriesInstanceUID"])

    train_df = train_df[~train_df["SeriesInstanceUID"].isin(ignore_uids)].reset_index(drop=True)
    train_df["fold_id"] = 0

    sgkf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    for i, (_, test_index) in enumerate(sgkf.split(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"])):
        train_df.loc[test_index, "fold_id"] = i

    label_df["x"] = label_df["coordinates"].map(lambda x: ast.literal_eval(x)['x'])
    label_df["y"] = label_df["coordinates"].map(lambda x: ast.literal_eval(x)['y'])
    label_df["z"] = -1

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
    print(f"Starting processing for {len(uids_to_process)} UIDs...")

    with multiprocessing.Pool(
        processes=CORES,
        initializer=initializer,
        initargs=(label_df, target_dir)
    ) as pool:
        results = list(tqdm(pool.imap_unordered(process_and_save, uids_to_process), total=len(uids_to_process)))

    print("\nProcessing complete.")

    for r in results:
        uid, mapped = r["uid"], r["mapped_idx"]
        if mapped:
            label_df.loc[label_df["SeriesInstanceUID"] == uid, "z"] = mapped

    label_df.to_csv(target_dir / "label_df.csv", index=False)
    train_df.to_csv(target_dir / "train_df.csv", index=False)
