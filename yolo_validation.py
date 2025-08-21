"""Validate YOLO aneurysm detector at series level.

Per user request:
  * Use same preprocessing as in `src/prepare_yolo_dataset.py` (DICOM -> HU -> per-slice min-max -> uint8)
  * Series-level aneurysm present probability = max detection confidence across ALL slices in the series
  * Location probabilities (13 anatomical sites) are set to constant 0.5 (baseline) since YOLO model is single-class

Outputs:
  * Prints classification AUC (aneurysm present)
  * Prints (degenerate) location macro AUC (expected ~0.5 due to constant probs)
  * Optional CSV with per-series probabilities

Usage:
  python3 yolo_validation.py --weights runs/detect/train/weights/best.pt --val-fold 0 
  (adjust weights path to your YOLO trained weights)

Additional modes:
    --rgb-slices : replicate RGB slice construction used in dataset prep (channels = [prev, current, next])
python3 yolo_validation.py --weights /home/sersasj/RSNA-IAD-Codebase/runs/yolo_aneurysm/exp-yolon-new-data5/weights/best.pt --val-fold 0 --rgb-slices 
Notes:
  * Requires ultralytics package (YOLOv8/11). Assumes single-class model trained on aneurysm dataset.
  * For speed you can set --max-slices to limit number of slices per series during quick experiments.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import math
from typing import List, Dict

import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.metrics import roc_auc_score

# Project root & config imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))  # to allow importing configs.data_config
from configs.data_config import data_path  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit("ultralytics not installed. Install with `pip install ultralytics`.") from e


LABELS_TO_IDX = {
    'Anterior Communicating Artery': 0,
    'Basilar Tip': 1,
    'Left Anterior Cerebral Artery': 2,
    'Left Infraclinoid Internal Carotid Artery': 3,
    'Left Middle Cerebral Artery': 4,
    'Left Posterior Communicating Artery': 5,
    'Left Supraclinoid Internal Carotid Artery': 6,
    'Other Posterior Circulation': 7,
    'Right Anterior Cerebral Artery': 8,
    'Right Infraclinoid Internal Carotid Artery': 9,
    'Right Middle Cerebral Artery': 10,
    'Right Posterior Communicating Artery': 11,
    'Right Supraclinoid Internal Carotid Artery': 12,
}
LOCATION_LABELS = sorted(list(LABELS_TO_IDX.keys()))
N_LOC = len(LOCATION_LABELS)


def parse_args():
    ap = argparse.ArgumentParser(description="Series-level validation for YOLO aneurysm detector")
    ap.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt)')
    ap.add_argument('--val-fold', type=int, default=0, help='Fold id to evaluate (matches train.csv fold_id)')
    ap.add_argument('--series-limit', type=int, default=0, help='Optional limit on number of validation series (debug)')
    ap.add_argument('--max-slices', type=int, default=0, help='Optional cap on number of slices per series (debug)')
    ap.add_argument('--save-csv', type=str, default='', help='Optional path to save per-series predictions CSV')
    ap.add_argument('--batch-size', type=int, default=16, help='Batch size for slice inference (higher = faster, more VRAM)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--rgb-slices', action='store_true', help='Use 3-channel [prev,current,next] slice stacking (matches dataset prep RGB mode)')
    return ap.parse_args()


def read_dicom_hu(path: Path) -> np.ndarray:
    """Copy of logic from prepare_yolo_dataset (without global clipping) and per-slice min-max."""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    if pix.ndim == 3:  # skip multi-frame for this simple loop
        raise ValueError('Multi-frame DICOM not supported in this script')
    img = pix.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    img = img * slope + intercept
    return img


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def collect_series_slices(series_dir: Path) -> List[Path]:
    return sorted(series_dir.glob('*.dcm'))


def slice_confidence(model: YOLO, img_uint8: np.ndarray) -> float:
    """Run YOLO on single slice (uint8 grayscale) and return max detection confidence for class 0."""
    # Ensure 3-channel for model
    if img_uint8.ndim == 2:
        img3 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img3 = img_uint8
    results = model.predict(img3, verbose=False)
    if not results:
        return 0.0
    r = results[0]
    if r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
        return 0.0
    # Single-class so take max confidence
    return float(r.boxes.conf.max().item())


def main():
    args = parse_args()
    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_df = pd.read_csv(data_root / 'train_df.csv') if (data_root / 'train_df.csv').exists() else pd.read_csv(data_root / 'train.csv')
    if 'Aneurysm Present' not in train_df.columns:
        raise SystemExit("train_df.csv requires 'Aneurysm Present' column for classification label")

    val_series = train_df[train_df['fold_id'] == args.val_fold]['SeriesInstanceUID'].unique().tolist()
    if args.series_limit:
        val_series = val_series[:args.series_limit]

    print(f"Validation fold {args.val_fold}: {len(val_series)} series")
    model = YOLO(args.weights)

    series_probs: Dict[str, float] = {}
    cls_labels: List[int] = []
    loc_labels: List[np.ndarray] = []
    const_loc_probs = np.full(N_LOC, 0.5, dtype=np.float32)
    series_pred_loc_probs: List[np.ndarray] = []

    for sid in val_series:
        series_dir = series_root / sid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] {sid} (no directory)")
            continue
        dicoms = collect_series_slices(series_dir)
        if not dicoms:
            if args.verbose:
                print(f"[EMPTY] {sid}")
            continue
        if args.max_slices and len(dicoms) > args.max_slices:
            dicoms = dicoms[:args.max_slices]

        max_conf = 0.0
        batch: list[np.ndarray] = []
        def flush_batch(batch_imgs: list[np.ndarray]):
            nonlocal max_conf
            if not batch_imgs:
                return
            # Ultralytics can take list/np.array. Provide list for variable shapes (should be consistent though).
            results = model.predict(batch_imgs, verbose=False, conf=0.005)
            for r in results:
                if not r or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                    continue
                batch_max = float(r.boxes.conf.max().item())
                if batch_max > max_conf:
                    max_conf = batch_max

        if args.rgb_slices:
            # Pre-read & normalize all usable slices first (skip failures silently unless verbose)
            slice_imgs: list[np.ndarray] = []  # each uint8 single channel
            for dcm_path in dicoms:
                try:
                    img_hu = read_dicom_hu(dcm_path)
                except Exception as e:
                    if args.verbose:
                        print(f"[SKIP] {dcm_path.name}: {e}")
                    continue
                slice_imgs.append(min_max_normalize(img_hu))
            if not slice_imgs:
                if args.verbose:
                    print(f"[NO_VALID_SLICES] {sid}")
            else:
                # Harmonize shapes (choose most common HxW)
                shape_counts: Dict[tuple[int,int], int] = {}
                for arr in slice_imgs:
                    shape_counts[arr.shape] = shape_counts.get(arr.shape, 0) + 1
                target_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
                if len(shape_counts) > 1 and args.verbose:
                    print(f"[RESIZE] {sid}: {len(shape_counts)} shapes -> {target_shape}")
                if len(shape_counts) > 1:
                    th, tw = target_shape
                    for i, arr in enumerate(slice_imgs):
                        if arr.shape != target_shape:
                            slice_imgs[i] = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LINEAR)
                for i in range(1,len(slice_imgs)-1):
                    prev_img = slice_imgs[i-1] if i > 0 else slice_imgs[i]
                    cur_img = slice_imgs[i]
                    next_img = slice_imgs[i+1] if i+1 < len(slice_imgs) else slice_imgs[i]
                    try:
                        rgb = np.stack([prev_img, cur_img, next_img], axis=-1)  # H,W,3
                    except ValueError:
                        # Shape mismatch despite harmonization; skip
                        if args.verbose:
                            print(f"[STACK_FAIL] {sid} slice index {i} shape mismatch; skipping triplet")
                        continue
                    batch.append(rgb)
                    if len(batch) >= args.batch_size:
                        flush_batch(batch)
                        batch.clear()
                flush_batch(batch)
        else:
            for dcm_path in dicoms:
                try:
                    img_hu = read_dicom_hu(dcm_path)
                except Exception as e:
                    if args.verbose:
                        print(f"[SKIP] {dcm_path.name}: {e}")
                    continue
                img_uint8 = min_max_normalize(img_hu)
                if img_uint8.ndim == 2:
                    img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                batch.append(img_uint8)
                if len(batch) >= args.batch_size:
                    flush_batch(batch)
                    batch.clear()
            # flush remaining
            flush_batch(batch)
        series_probs[sid] = max_conf
        print(f"Series {sid} max_conf={max_conf:.4f}")
        if args.verbose:
            print(f"Series {sid} max_conf={max_conf:.4f}")

        row = train_df[train_df['SeriesInstanceUID'] == sid].iloc[0]
        cls_labels.append(int(row['Aneurysm Present']))
        # Build location label vector (13) if present; else zeros
        loc_vec = np.zeros(N_LOC, dtype=np.float32)
        missing_loc_cols = False
        for idx, name in enumerate(LOCATION_LABELS):
            if name in row:
                try:
                    loc_vec[idx] = float(row[name])
                except Exception:
                    loc_vec[idx] = 0.0
            else:
                missing_loc_cols = True
        if missing_loc_cols and args.verbose:
            print("Warning: Some location columns missing in train_df; filled with 0")
        loc_labels.append(loc_vec)
        series_pred_loc_probs.append(const_loc_probs.copy())

    if not series_probs:
        print("No series processed.")
        return

    # Metrics
    y_true = np.array(cls_labels)
    y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
    cls_auc = roc_auc_score(y_true, y_scores)

    loc_labels_arr = np.stack(loc_labels)
    loc_pred_arr = np.stack(series_pred_loc_probs)
    # guard against all-zero or single-class columns (roc_auc_score will error); handle try/except per column then macro average
    per_loc_aucs = []
    for i in range(N_LOC):
        try:
            auc_i = roc_auc_score(loc_labels_arr[:, i], loc_pred_arr[:, i])
        except ValueError:
            auc_i = float('nan')
        per_loc_aucs.append(auc_i)
    loc_macro_auc = np.nanmean(per_loc_aucs)

    print(f"Classification AUC (aneurysm present): {cls_auc:.4f}")
    print(f"Location macro AUC (constant 0.5 baseline): {loc_macro_auc:.4f}")
    print(f"Combined (mean) CV metric: {(cls_auc + (loc_macro_auc if not math.isnan(loc_macro_auc) else 0))/2:.4f}")

    if args.save_csv:
        out_rows = []
        for sid, prob in series_probs.items():
            out_rows.append({
                'SeriesInstanceUID': sid,
                'aneurysm_prob': prob,
                **{f'loc_prob_{i}': 0.5 for i in range(N_LOC)}
            })
        pd.DataFrame(out_rows).to_csv(args.save_csv, index=False)
        print(f"Saved per-series predictions to {args.save_csv}")


if __name__ == '__main__':
    main()
