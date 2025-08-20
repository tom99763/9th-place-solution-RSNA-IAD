"""Prepare a YOLO detection dataset (single class: aneurysm) from raw DICOM slices.

This script:
  * Reads label coordinates from train_localizers.csv (x,y point labels)
  * Loads corresponding DICOM slices, converts to HU, clips, and min-max normalizes per-slice
  * Creates square bounding boxes centered at the point (configurable size)
  * Splits images into train/val via existing fold assignments in train.csv (or random if missing)
  * Optionally samples negative slices (empty label files) for context
  * Outputs YOLO-format dataset under data/yolo_dataset/
    - images/train, images/val
    - labels/train, labels/val
    - configs/yolo_aneurysm.yaml (dataset descriptor – created separately)

YOLO txt label format:
  class x_center y_center width height (all normalized 0-1)

Assumptions / Notes:
  * train_localizers.csv has columns: SeriesInstanceUID, SOPInstanceUID, coordinates (dict-like string with 'x','y'), location
  * Coordinates are pixel indices matching the stored DICOM pixel array orientation (no flips applied here)
  * For multi-frame DICOMs we skip frames to keep logic simple (can extend later)
  * Bounding box size is clamped to image boundaries

Usage:
  python src/prepare_yolo_dataset.py --val-fold 0 --box-size 48 --negatives-per-positive 1

After running, you can train with:
  python src/train_yolo_aneurysm.py --data configs/yolo_aneurysm.yaml --model yolov8n.pt --epochs 50
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import random
import sys
import json
import math
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import pydicom
import cv2

# Allow importing project configs
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, N_FOLDS, SEED  # type: ignore

# Raw HU clipping range (same as other preprocessing) before per-slice min-max
RAW_MIN_HU = -1200.0
RAW_MAX_HU = 4000.0

rng = random.Random(SEED)


def read_dicom_hu(path: Path) -> np.ndarray:
    """Read a single-slice DICOM and return float32 array in HU clipped to standard range.
    Multi-frame DICOMs are skipped by raising ValueError.
    """
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    if pix.ndim == 3:  # multi-frame or RGB
        raise ValueError("Multi-frame / RGB DICOM not supported in this simple script")
    img = pix.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    img = img * slope + intercept
    img = np.clip(img, RAW_MIN_HU, RAW_MAX_HU)
    return img


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def build_box(x: float, y: float, box_size: int, w: int, h: int) -> Tuple[float, float, float, float]:
    """Return normalized (xc, yc, bw, bh) for YOLO given center point and desired box size in pixels."""
    half = box_size / 2.0
    x0 = max(0.0, x - half)
    y0 = max(0.0, y - half)
    x1 = min(w - 1.0, x + half)
    y1 = min(h - 1.0, y + half)
    bw = x1 - x0
    bh = y1 - y0
    xc = x0 + bw / 2.0
    yc = y0 + bh / 2.0
    # Normalize
    return xc / w, yc / h, bw / w, bh / h


def ensure_dirs(base: Path):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare YOLO aneurysm dataset from DICOM slices")
    ap.add_argument('--val-fold', type=int, default=0, help='Fold ID to use for validation (from train.csv)')
    ap.add_argument('--box-size', type=int, default=48, help='Square box size in pixels around aneurysm point')
    ap.add_argument('--negatives-per-positive', type=float, default=0.0, help='Ratio of negative (no label) slices to add per positive example (can be fractional)')
    ap.add_argument('--max-negatives', type=int, default=5000, help='Hard cap on total negative slices to avoid explosion')
    ap.add_argument('--limit', type=int, default=0, help='Optional limit on number of positives (for quick tests)')
    ap.add_argument('--image-ext', type=str, default='png', choices=['png','jpg','jpeg'], help='Image file extension')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def load_folds(root: Path) -> Dict[str, int]:
    train_csv = root / 'train.csv'
    df = pd.read_csv(train_csv)
    if 'fold_id' not in df.columns:
        # deterministic fold assignment if missing
        uids = df['SeriesInstanceUID'].unique().tolist()
        folds = {}
        for i, uid in enumerate(sorted(uids)):
            folds[uid] = i % N_FOLDS
        return folds
    return dict(zip(df['SeriesInstanceUID'], df['fold_id']))


def load_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / 'train_localizers.csv')
    # Parse coordinate dict
    if 'x' not in label_df.columns or 'y' not in label_df.columns:
        label_df['x'] = label_df['coordinates'].map(lambda s: ast.literal_eval(s)['x'])
        label_df['y'] = label_df['coordinates'].map(lambda s: ast.literal_eval(s)['y'])
    return label_df


def collect_series_sops(label_df: pd.DataFrame) -> Dict[str, Set[str]]:
    d: Dict[str, Set[str]] = {}
    for _, r in label_df.iterrows():
        d.setdefault(r.SeriesInstanceUID, set()).add(str(r.SOPInstanceUID))
    return d


def pick_negative_slices(series_dir: Path, positive_sops: Set[str], target_n: int) -> List[Path]:
    # gather candidate dcm paths
    cands = []
    for p in series_dir.glob('*.dcm'):
        if p.stem not in positive_sops:
            cands.append(p)
    if not cands:
        return []
    rng.shuffle(cands)
    return cands[:target_n]


def main():
    args = parse_args()
    root = Path(data_path)
    out_base = root / 'yolo_dataset'
    ensure_dirs(out_base)

    folds = load_folds(root)
    label_df = load_labels(root)

    # Build per-series groups
    label_groups = label_df.groupby('SeriesInstanceUID')

    total_pos_written = 0
    total_neg_written = 0
    neg_budget = args.max_negatives

    for series_uid, grp in label_groups:
        fold_id = folds.get(series_uid, 0)
        split = 'val' if fold_id == args.val_fold else 'train'
        series_dir = root / 'series' / series_uid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] Series directory missing: {series_dir}")
            continue

        # Positive slices first
        for _, row in grp.iterrows():
            sop = str(row.SOPInstanceUID)
            dcm_path = series_dir / f"{sop}.dcm"
            if not dcm_path.exists():
                if args.verbose:
                    print(f"[MISS] DICOM missing: {dcm_path}")
                continue
            try:
                img_hu = read_dicom_hu(dcm_path)
            except Exception as e:
                if args.verbose:
                    print(f"[SKIP] {dcm_path.name}: {e}")
                continue
            h, w = img_hu.shape
            img_uint8 = min_max_normalize(img_hu)
            xc, yc, bw, bh = build_box(float(row.x), float(row.y), args.box_size, w, h)
            stem = f"{series_uid}_{sop}"
            img_path = out_base / 'images' / split / f"{stem}.{args.image_ext}"
            label_path = out_base / 'labels' / split / f"{stem}.txt"
            if img_path.exists() and label_path.exists() and not args.overwrite:
                continue
            cv2.imwrite(str(img_path), img_uint8)
            with open(label_path, 'w') as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            total_pos_written += 1
            if args.limit and total_pos_written >= args.limit:
                break
        if args.limit and total_pos_written >= args.limit:
            break

        # Negatives (sample from same series if requested)
        if args.negatives_per_positive > 0 and neg_budget > 0:
            positives_in_series = len(grp)
            desired_neg = args.negatives_per_positive * positives_in_series
            # accumulate fractional across series
            take_n = int(desired_neg)
            if rng.random() < (desired_neg - take_n):
                take_n += 1
            take_n = min(take_n, neg_budget)
            if take_n > 0:
                pos_sops = {str(s) for s in grp.SOPInstanceUID.values}
                cand_paths = pick_negative_slices(series_dir, pos_sops, take_n)
                for neg_path in cand_paths:
                    try:
                        img_hu = read_dicom_hu(neg_path)
                    except Exception:
                        continue
                    img_uint8 = min_max_normalize(img_hu)
                    stem = f"{series_uid}_{neg_path.stem}"
                    img_path = out_base / 'images' / split / f"{stem}.{args.image_ext}"
                    label_path = out_base / 'labels' / split / f"{stem}.txt"
                    if img_path.exists() and label_path.exists() and not args.overwrite:
                        continue
                    cv2.imwrite(str(img_path), img_uint8)
                    # Empty label file indicates no objects
                    open(label_path, 'w').close()
                    total_neg_written += 1
                    neg_budget -= 1
                    if neg_budget <= 0:
                        break

    print(f"Positives written: {total_pos_written}")
    print(f"Negatives written: {total_neg_written}")
    print(f"Dataset root: {out_base}")
    # Summaries
    for split in ['train','val']:
        n_imgs = len(list((out_base / 'images' / split).glob('*')))
        n_lbls = len(list((out_base / 'labels' / split).glob('*.txt')))
        print(f"{split}: images={n_imgs} labels={n_lbls}")

if __name__ == '__main__':
    main()
#sersasj@DESKTOP-8U9D0KJ:~/RSNA-IAD-Codebase$ python3 src/prepare_yolo_dataset.py --val-fold 0 --box-size 24 --negatives-per-positive 0