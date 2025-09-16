"""Prepare 3D volumes (D=32, H=W=384) from raw DICOM series.

This script:
 1. Reads train.csv + train_localizers.csv (labels) and multiframe exceptions.
 2. Builds stratified folds (same logic as other prep scripts).
 3. Loads each series' DICOM slices, converts to HU using RescaleSlope/Intercept.
 4. Sorts slices by z-position (ImagePositionPatient[2]) falling back to InstanceNumber.
 5. Stacks into 3D volume (float32 HU), clips to a fixed HU range, then rescales to [0,1].
 6. Resamples to target depth (32) using linear interpolation (scipy.ndimage.zoom).
 7. Resizes in-plane to (384, 384) with cv2.INTER_LINEAR.
 8. Saves volume as NPZ with shape (D, H, W) float32.
 9. Writes volume_df.csv with per-series metadata (uid, filename, fold, target).

Notes on voxel size / spacing:
 - You DO have voxel size info in the limited DICOM tags: PixelSpacing (row, col) and either
   SpacingBetweenSlices or SliceThickness and/or ImagePositionPatient difference. For rigorous
   physical isotropic resampling you'd rescale each axis proportionally to spacing; here we
   *only* normalize slice count and spatial dims, ignoring anisotropy. This is a pragmatic
   baseline; anisotropic spacing can be incorporated later by computing zoom factors using
   (target_spacing / current_spacing) per axis before cropping/resizing.
 - ImageOrientationPatient is ignored assuming consistent axial orientation across the dataset.
   If not, a future improvement should re-orient into RAS / LPS canonical space.

Future improvements (optional):
 - Multi-window channels (store CxDxHxW) by applying different HU windows.
 - Per-axis spacing aware isotropic resampling.
 - 3D augmentations (using monai / torchio) instead of 2D ones.
"""
from __future__ import annotations
import os
import ast
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import StratifiedKFold

try:
    from scipy.ndimage import zoom as nd_zoom
except ImportError:
    raise SystemExit("scipy is required. Install via: pip install scipy")
    
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from configs.data_config import * 

TARGET_DEPTH = 32
TARGET_SIZE = 384
HU_MIN = -1200.0
HU_MAX = 4000.0
STORE_NORMALIZED = False  # Set True to revert to [0,1] scaling

# Globals for worker processes
_label_df = None


def _load_series_dicom_paths(series_uid: str, root: Path) -> List[Path]:
    series_dir = root / 'series' / series_uid
    paths = []
    for r, _, files in os.walk(series_dir):
        for f in files:
            if f.endswith('.dcm'):
                paths.append(Path(r) / f)
    return paths


def _read_dicom(path: Path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return ds, arr


def _extract_slice_position(ds) -> float:
    # Prefer ImagePositionPatient z, fallback to InstanceNumber
    if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) == 3:
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, 'InstanceNumber'):
        try:
            return float(ds.InstanceNumber)
        except Exception:
            pass
    return 0.0


def _resample_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    if volume.shape[0] == target_depth:
        return volume
    depth_zoom = target_depth / volume.shape[0]
    # zoom along depth only, order=1 linear
    return nd_zoom(volume, (depth_zoom, 1.0, 1.0), order=1)


def _resize_inplane(volume: np.ndarray, target_hw: int) -> np.ndarray:
    d, h, w = volume.shape
    if h == target_hw and w == target_hw:
        return volume
    resized = np.empty((d, target_hw, target_hw), dtype=volume.dtype)
    for i in range(d):
        resized[i] = cv2.resize(volume[i], (target_hw, target_hw), interpolation=cv2.INTER_LINEAR)
    return resized


def _clip_or_normalize(volume: np.ndarray) -> np.ndarray:
    """Either clip-only (raw HU retained in range) or clip+normalize to [0,1]."""
    vol = np.clip(volume, HU_MIN, HU_MAX).astype(np.float32)
    if STORE_NORMALIZED:
        vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol


def _process_single_series(uid: str, root: Path, out_dir: Path) -> Dict[str, Any]:
    try:
        dcm_paths = _load_series_dicom_paths(uid, root)
        if not dcm_paths:
            return {"series_uid": uid, "volume_filename": None, "num_slices_raw": 0}
        slices: List[Tuple[float, np.ndarray]] = []
        for p in dcm_paths:
            try:
                ds, arr = _read_dicom(p)
                # If multi-frame (arr.ndim==3) stack frames individually
                if arr.ndim == 3 and arr.shape[-1] != 3:
                    for fi in range(arr.shape[0]):
                        slices.append((_extract_slice_position(ds) + fi * 0.001, arr[fi].astype(np.float32)))
                else:
                    if arr.ndim == 3 and arr.shape[-1] == 3:
                        # Convert RGB to grayscale
                        arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
                    slices.append((_extract_slice_position(ds), arr.astype(np.float32)))
            except Exception:
                continue
        if not slices:
            return {"series_uid": uid, "volume_filename": None, "num_slices_raw": 0}
        # Sort by z
        slices.sort(key=lambda x: x[0])
        vol = np.stack([s[1] for s in slices], axis=0)  # (D, H, W)
        num_raw = vol.shape[0]
        # Clip HU range (optionally normalize based on STORE_NORMALIZED)
        vol = _clip_or_normalize(vol)
        # Depth resample
        vol = _resample_depth(vol, TARGET_DEPTH)
        # In-plane resize
        vol = _resize_inplane(vol, TARGET_SIZE)
        # Save
        vol_filename = f"{uid}_d{TARGET_DEPTH}_sz{TARGET_SIZE}.npz"
        # Save meta: [HU_MIN, HU_MAX, normalized_flag]
        meta = np.array([HU_MIN, HU_MAX, 1.0 if STORE_NORMALIZED else 0.0], dtype=np.float32)
        np.savez_compressed(out_dir / vol_filename, vol=vol, meta=meta)
        return {"series_uid": uid, "volume_filename": vol_filename, "num_slices_raw": num_raw}
    except Exception as e:
        return {"series_uid": uid, "volume_filename": None, "error": str(e), "num_slices_raw": 0}


def _worker_init(root: Path, out_dir: Path):
    global _ROOT, _OUT
    _ROOT = root
    _OUT = out_dir


def _worker(uid: str):
    return _process_single_series(uid, _ROOT, _OUT)


def main():
    root = Path(data_path)
    processed = root / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    vol_dir = processed / 'volumes_3d'
    vol_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(root / 'train.csv')
    label_df = pd.read_csv(root / 'train_localizers.csv')
    mf_dicom_uids = pd.read_csv(root / 'multiframe_dicoms.csv') if (root / 'multiframe_dicoms.csv').exists() else pd.DataFrame(columns=['SeriesInstanceUID'])

    ignore_uids = set([
        '1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068',
        '1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444',
        '1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242',
    ]) | set(mf_dicom_uids['SeriesInstanceUID'].tolist())

    train_df = train_df[~train_df['SeriesInstanceUID'].isin(ignore_uids)].reset_index(drop=True)
    train_df['fold_id'] = 0

    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    for fold, (_, val_idx) in enumerate(skf.split(train_df['SeriesInstanceUID'], train_df['Aneurysm Present'])):
        train_df.loc[val_idx, 'fold_id'] = fold

    uids = train_df['SeriesInstanceUID'].unique().tolist()
    print(f"Preparing 3D volumes for {len(uids)} series -> target shape ({TARGET_DEPTH}, {TARGET_SIZE}, {TARGET_SIZE})")

    cores = min( max(1, mp.cpu_count() - 1), 8)
    with mp.Pool(processes=cores, initializer=_worker_init, initargs=(root, vol_dir)) as pool:
        from tqdm import tqdm
        results = list(tqdm(pool.imap_unordered(_worker, uids), total=len(uids)))

    vol_records = [r for r in results if r.get('volume_filename')]
    volume_df = pd.DataFrame(vol_records)
    uid_to_fold = dict(zip(train_df['SeriesInstanceUID'], train_df['fold_id']))
    uid_to_target = dict(zip(train_df['SeriesInstanceUID'], train_df['Aneurysm Present']))
    volume_df['fold_id'] = volume_df['series_uid'].map(uid_to_fold)
    volume_df['series_has_aneurysm'] = volume_df['series_uid'].map(uid_to_target)

    volume_df.to_csv(processed / 'volume_df.csv', index=False)
    train_df.to_csv(processed / 'train_df_volume.csv', index=False)
    print(f"Saved {len(volume_df)} volumes to {vol_dir}")
    print(f"Metadata saved to {processed / 'volume_df.csv'}")

    # Optional: quick stats
    if not volume_df.empty:
        print(volume_df[['num_slices_raw']].describe())

if __name__ == '__main__':
    main()
