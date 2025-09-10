import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import os
import pydicom
import cv2
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig

import cupy as cp
from cupyx.scipy.ndimage import zoom


def preprocess_dcm_slice(slice, dcm):

    # Get the pixel data from the DICOM file
    slice = slice.astype(np.float32)

    # 2. Check if the modality is 'CT' to decide on the processing method
    # The DICOM tag (0008,0060) specifies the modality
    is_ct_scan = 'CT' in dcm.get('Modality', '').upper()

    if is_ct_scan:

        # For CT scans, convert pixel data to Hounsfield Units (HU)
        # using the Rescale Slope and Intercept values from DICOM metadata
        if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
            slice = slice * dcm.RescaleSlope + dcm.RescaleIntercept

    return slice

def normalize(image):
    """
    Normalizes the image intensity to the range [0, 1].
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image

def process_dicom(filepath):
    ds = pydicom.dcmread(filepath, force=True)
    img = ds.pixel_array.astype(np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        imgs = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)]
    elif img.ndim == 3:
        imgs = list(img)
    else:
        imgs = [img]

    results = []
    for img in imgs:
        z_val = getattr(ds, "ImagePositionPatient")[-1] \
            if hasattr(ds, "ImagePositionPatient") else int(getattr(ds, "InstanceNumber", 0))
        results.append((z_val, preprocess_dcm_slice(img, ds)))

    return results

def parallel_process(all_filepaths, max_workers=8):
    slices = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_dicom, fp): fp for fp in all_filepaths}
        for future in as_completed(futures):
            slices.extend(future.result())
    return slices


def process_dicom_series(uid: str, data_path, cfg):

    target_depth  = cfg.preprocess.volume.depth
    target_height = cfg.preprocess.volume.height
    target_width  = cfg.preprocess.volume.width

    series_path = data_path / f"series/{uid}"
    all_filepaths = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(series_path)
        for file in files if file.endswith('.dcm')
    ])

    slices = parallel_process(all_filepaths, max_workers=cfg.preprocess.cores)
    slices = sorted(slices, key=lambda x: x[0])
    vol = np.array([s[1] for s in slices])

    depth,height,width = vol.shape

    vol = cp.asarray(vol)
    vol = zoom(vol, (target_depth/depth, target_height/height, target_width/width), order=3)
    vol = cp.asnumpy(vol)

    vol = (normalize(vol) * 255).astype(np.uint8)
    np.savez( data_path / f"processed/series/{uid}.npz"
            , vol=vol
            , original_shape=np.array([depth, height, width])
    )

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def preprocess_data(cfg: DictConfig) -> None:

    root_path = Path(cfg.preprocess.data_path)

    target_dir = root_path / "processed"
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(f'{target_dir}/series'):
        os.makedirs(f'{target_dir}/series')

    train_df = pd.read_csv(root_path / "train.csv")

    skf = StratifiedKFold( n_splits=cfg.preprocess.n_folds
                         , random_state=cfg.seed
                         , shuffle=True)

    train_df["fold_id"] = 0
    for i, (_, test_index) in enumerate(skf.split(train_df["SeriesInstanceUID"], train_df["Aneurysm Present"])):
        train_df.loc[test_index, "fold_id"] = i

    uids = train_df["SeriesInstanceUID"].unique().tolist()

    for uid in tqdm(uids):
        process_dicom_series(uid, root_path, cfg)

    train_df.to_csv(target_dir / "train_df.csv", index=False)

if __name__ == "__main__":
    preprocess_data()
