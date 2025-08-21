"""Prepare multi-fraction axial MIPs (v3) in cumulative OR interval mode.

Modes
-----
1. cumulative (default):
     Channel i is the maximum projection from slice 0 up to fraction f_i of the
     stack. Fractions are treated as cumulative endpoints.
     Example fractions: 0.125,0.25,...,1.0 produce:
             ch0 = max(0   : 12%)
             ch1 = max(0   : 25%)
             ...
             ch7 = max(0   :100%)

2. interval (non-overlapping):
     Channel i is the maximum projection over the *interval* (f_{i-1}, f_i].
     Implicit start fraction 0.0 is added. Using the same fraction list above:
             ch0 = max(0%  : 12%]
             ch1 = max(12% : 25%]
             ...
             ch7 = max(87% :100%]

Default fractions list (interpreted as endpoints):
    0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0  -> 8 channels

Outputs:
    processed/mip_fraction_images/<SeriesInstanceUID>_mip_fracs.npz containing:
         - mip_frac_uint8: (H,W,C) uint8 0-255
         - meta: [raw_min_hu, raw_max_hu]
         - fractions: fraction endpoints used (float32)
         - mode: 'cumulative' or 'interval'
         - interval_starts (only in interval mode): starting fractions per channel

Also writes processed/mip_df_v3.csv with columns:
    series_uid, mip_filename, fold_id, series_has_aneurysm

Usage examples:
    cumulative:
        python src/prepare_data_mip_slice_v3.py --fractions 0.125,0.25,0.5,1.0 \
                --mode cumulative --img-size 512 --cores 8
    interval:
        python src/prepare_data_mip_slice_v3.py --fractions 0.125,0.25,0.5,1.0 \
                --mode interval --img-size 512 --cores 8

"""
from __future__ import annotations

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict
import glob

import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Project config import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, N_FOLDS, SEED, CORES  # type: ignore

RAW_MIN_HU = -1200.0
RAW_MAX_HU = 4000.0

# ----------------------------- Helpers ---------------------------------

def read_series_slices(series_dir: Path) -> List[np.ndarray]:
    dcm_files = sorted(glob.glob(str(series_dir / "*.dcm")))
    slices: List[np.ndarray] = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            px = ds.pixel_array
            # Handle multi-frame (stack frames) but ignore RGB
            if px.ndim == 3:
                if px.shape[-1] == 3:
                    # convert to grayscale then treat as single slice
                    px = cv2.cvtColor(px.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    px = px.astype(np.float32)
                    slope = float(getattr(ds, 'RescaleSlope', 1.0))
                    inter = float(getattr(ds, 'RescaleIntercept', 0.0))
                    px = px * slope + inter
                    px = np.clip(px, RAW_MIN_HU, RAW_MAX_HU)
                    slices.append(px)
                else:  # assume (frames, H, W)
                    slope = float(getattr(ds, 'RescaleSlope', 1.0))
                    inter = float(getattr(ds, 'RescaleIntercept', 0.0))
                    for i in range(px.shape[0]):
                        frame = px[i].astype(np.float32)
                        frame = frame * slope + inter
                        frame = np.clip(frame, RAW_MIN_HU, RAW_MAX_HU)
                        slices.append(frame)
                continue
            img = px.astype(np.float32)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            inter = float(getattr(ds, 'RescaleIntercept', 0.0))
            img = img * slope + inter
            img = np.clip(img, RAW_MIN_HU, RAW_MAX_HU)
            slices.append(img)
        except Exception:
            continue
    return slices


def fraction_mips(
    slices: List[np.ndarray],
    fractions: List[float],
    out_size: int,
    mode: str = 'cumulative'
) -> Tuple[np.ndarray, List[float]]:
    """Compute multi-channel MIPs.

    Parameters
    ----------
    slices : list of 2D float arrays (already in HU range)
    fractions : sorted list of fraction endpoints (0<f<=1]
    out_size : output spatial size (square)
    mode : 'cumulative' or 'interval'

    Returns
    -------
    arr : (H,W,C) uint8
    interval_starts : list of starting fractions per channel (for interval mode; else empty)
    """
    if not slices:
        return np.zeros((out_size, out_size, len(fractions)), dtype=np.uint8), []
    fractions = sorted(fractions)
    h0, w0 = slices[0].shape
    proc = []
    for s in slices:
        if s.shape != (h0, w0):
            s = cv2.resize(s, (w0, h0), interpolation=cv2.INTER_LINEAR)
        proc.append(s)
    stack = np.stack(proc, axis=0)  # (S,H,W)
    total = stack.shape[0]
    mips = []
    interval_starts: List[float] = []
    if mode == 'cumulative':
        for f in fractions:
            n = max(1, int(round(f * total)))
            sub = stack[:n]
            mip = sub.max(axis=0)
            norm = (mip - RAW_MIN_HU) / (RAW_MAX_HU - RAW_MIN_HU)
            norm = np.clip(norm, 0.0, 1.0)
            mip_u8 = (norm * 255.0).astype(np.uint8)
            if mip_u8.shape[0] != out_size or mip_u8.shape[1] != out_size:
                mip_u8 = cv2.resize(mip_u8, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            mips.append(mip_u8)
        arr = np.stack(mips, axis=-1)
        return arr, interval_starts
    elif mode == 'interval':
        prev_f = 0.0
        for f in fractions:
            start_idx = int(round(prev_f * total))
            end_idx = int(round(f * total))
            if end_idx <= start_idx:
                end_idx = min(total, start_idx + 1)
            sub = stack[start_idx:end_idx]
            mip = sub.max(axis=0)
            norm = (mip - RAW_MIN_HU) / (RAW_MAX_HU - RAW_MIN_HU)
            norm = np.clip(norm, 0.0, 1.0)
            mip_u8 = (norm * 255.0).astype(np.uint8)
            if mip_u8.shape[0] != out_size or mip_u8.shape[1] != out_size:
                mip_u8 = cv2.resize(mip_u8, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            interval_starts.append(prev_f)
            mips.append(mip_u8)
            prev_f = f
        arr = np.stack(mips, axis=-1)
        return arr, interval_starts
    else:  # defensive
        raise ValueError(f"Unknown mode '{mode}' (expected 'cumulative' or 'interval')")

# ----------------------------- Worker ---------------------------------

def _worker_init(global_root: Path, out_dir: Path, fractions: List[float], img_size: int, mode: str):
    global _ROOT_PATH, _OUT_DIR, _FRACTIONS, _IMG_SIZE, _MODE
    _ROOT_PATH = global_root
    _OUT_DIR = out_dir
    _FRACTIONS = fractions
    _IMG_SIZE = img_size
    _MODE = mode


def _worker(uid: str) -> Dict[str, str | None]:
    try:
        series_dir = _ROOT_PATH / 'series' / uid
        slices = read_series_slices(series_dir)
        frac_mips, interval_starts = fraction_mips(slices, _FRACTIONS, _IMG_SIZE, mode=_MODE)
        fname = f"{uid}_mip_fracs.npz"
        fpath = _OUT_DIR / fname
        save_kwargs = dict(
            mip_frac_uint8=frac_mips,
            meta=np.array([RAW_MIN_HU, RAW_MAX_HU], dtype=np.float32),
            fractions=np.array(_FRACTIONS, dtype=np.float32),
            mode=np.array([_MODE]),
        )
        if interval_starts:
            save_kwargs['interval_starts'] = np.array(interval_starts, dtype=np.float32)
        np.savez_compressed(fpath, **save_kwargs)
        return {"uid": uid, "mip_filename": fname}
    except Exception as e:
        print(f"[ERR] {uid}: {e}")
        return {"uid": uid, "mip_filename": None}

# ----------------------------- Main ---------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare multi-fraction cumulative MIPs (v3)")
    ap.add_argument('--fractions', type=str, default='0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0',
                    help='Comma-separated cumulative fractions (0-1]')
    ap.add_argument('--img-size', type=int, default=512, help='Output square size for each channel')
    ap.add_argument('--mode', choices=['cumulative','interval'], default='interval',
                    help='cumulative = each channel max over 0->f. interval = non-overlapping (prev_f,f]')
    ap.add_argument('--cores', type=int, default=CORES, help='Parallel workers')
    ap.add_argument('--overwrite', action='store_true', help='Recompute even if file exists')
    return ap.parse_args()


def main():
    args = parse_args()
    fractions = [float(x) for x in args.fractions.split(',') if x.strip()]
    root_path = Path(data_path)
    processed_dir = root_path / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_dir = processed_dir / 'mip_fraction_images'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input dataframes
    train_df = pd.read_csv(root_path / 'train.csv')
    mf_uids = pd.read_csv(root_path / 'multiframe_dicoms.csv') if (root_path / 'multiframe_dicoms.csv').exists() else pd.DataFrame(columns=['SeriesInstanceUID'])

    ignore_uids = set(mf_uids['SeriesInstanceUID'].tolist())

    # Filter
    if 'SeriesInstanceUID' not in train_df.columns:
        raise RuntimeError('train.csv missing SeriesInstanceUID column')

    if 'Aneurysm Present' not in train_df.columns:
        raise RuntimeError('train.csv missing "Aneurysm Present" column')

    work_df = train_df[~train_df['SeriesInstanceUID'].isin(ignore_uids)].reset_index(drop=True)

    # Build folds if needed
    if 'fold_id' not in work_df.columns:
        work_df['fold_id'] = 0
        sgkf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
        for i, (_, test_index) in enumerate(
            sgkf.split(work_df['SeriesInstanceUID'], work_df['Aneurysm Present'])
        ):
            work_df.loc[test_index, 'fold_id'] = i

    uids = work_df['SeriesInstanceUID'].unique().tolist()

    print(f"Preparing fraction MIPs for {len(uids)} series | fractions={fractions} | mode={args.mode} | img_size={args.img_size}")

    # Parallel processing
    records = []
    with mp.Pool(processes=args.cores, initializer=_worker_init, initargs=(root_path, out_dir, fractions, args.img_size, args.mode)) as pool:
        for rec in tqdm(pool.imap_unordered(_worker, uids), total=len(uids)):
            if rec and rec.get('mip_filename'):
                records.append(rec)

    frac_df = pd.DataFrame(records)
    frac_df.rename(columns={'uid': 'series_uid'}, inplace=True)
    uid_to_fold = dict(zip(work_df['SeriesInstanceUID'], work_df['fold_id']))
    uid_to_target = dict(zip(work_df['SeriesInstanceUID'], work_df['Aneurysm Present']))
    frac_df['fold_id'] = frac_df['series_uid'].map(uid_to_fold)
    frac_df['series_has_aneurysm'] = frac_df['series_uid'].map(uid_to_target)

    out_csv = processed_dir / 'mip_df_v3.csv'
    frac_df.to_csv(out_csv, index=False)
    print(f"Created {len(frac_df)} fraction MIP files -> {out_dir}")
    print(f"Metadata saved to: {out_csv}")

if __name__ == '__main__':
    main()
