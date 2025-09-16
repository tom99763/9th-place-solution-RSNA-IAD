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
from preprocess import *

data_path = '../data'

def process_vol_and_save(series_path: str):
    try:
        vol = load_dicom_to_nii_series(series_path)
        print(vol.shape)
        np.savez_compressed(target_dir / f"{uid}.npz", vol=vol)
    except Exception as e:
        print(f"Error processing {uid}: {e}")

if __name__ == "__main__":
    root_path = Path(data_path)
    target_dir = root_path / "seg_vols"
    os.makedirs(target_dir, exist_ok=True)

    train_df = pd.read_csv(root_path / "train.csv")
    mf_dicom_uids = pd.read_csv(root_path / "multiframe_dicoms.csv")

    ignore_uids = [
        "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
        "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
        "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242"
    ] + list(mf_dicom_uids["SeriesInstanceUID"])

    #train_df = train_df[~train_df.SeriesInstanceUID.isin(ignore_uids)]
    #valid_uids = train_df.SeriesInstanceUID.values
    seg_uids = [uid.split('.nii')[0]
                for uid in os.listdir(root_path/'segmentations') if 'cowseg' not in uid]
    seg_uids = [uid for uid in seg_uids if uid not in ignore_uids]
    print('total uids:', len(seg_uids))
    for uid in tqdm(seg_uids):
        series_path = root_path/f'series/{uid}'
        process_vol_and_save(series_path)
