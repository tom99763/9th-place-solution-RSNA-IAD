"""Validate a 13-class YOLO aneurysm detector at series level.

Preprocessing:
    - DICOM -> HU -> per-slice min-max -> uint8 (BGR for YOLO inference)

Series scores:
    - Aneurysm Present = max detection confidence across ALL 13 classes over processed slices/MIP windows
    - Location probs   = per-class max detection confidence over processed slices/MIP windows

Outputs:
    - Classification AUC (aneurysm present)
    - Location macro AUC (macro over 13 classes)
    - Optional CSV with per-series predictions

Notes:
    - Requires ultralytics (YOLOv8/11) and 13-class weights.
    - Use --mip-window > 0 for sliding-window MIPs; 0 for per-slice.
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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

# Project root & config imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from configs.data_config import data_path, LABELS_TO_IDX  # type: ignore

LOCATION_LABELS = sorted(list(LABELS_TO_IDX.keys()))
N_LOC = len(LOCATION_LABELS)


def parse_args():
    ap = argparse.ArgumentParser(description="Series-level validation for YOLO (13-class)")
    #ap.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt) with 13 classes')
    ap.add_argument('--val-fold', type=int, default=0, help='Fold id to evaluate (matches train.csv fold_id)')
    ap.add_argument('--series-limit', type=int, default=0, help='Optional limit on number of validation series (debug)')
    ap.add_argument('--max-slices', type=int, default=0, help='Optional cap on number of slices/windows per series (debug)')
    ap.add_argument('--save-csv', type=str, default='', help='Optional path to save per-series predictions CSV')
    ap.add_argument('--batch-size', type=int, default=16, help='Batch size for inference (higher = faster, more VRAM)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice (default=1)')
    # Sliding-window MIP mode
    ap.add_argument('--mip-window', type=int, default=0, help='Half-window (in slices) for MIP; 0 = per-slice mode')
    ap.add_argument('--mip-img-size', type=int, default=0, help='Optional resize of MIP/slice to this square size before inference (0 keeps original)')
    ap.add_argument('--mip-no-overlap', action='store_true', help='Use non-overlapping MIP windows (stride = 2*w+1 instead of slice_step)')
    return ap.parse_args()


def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:
        # RGB or multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def collect_series_slices(series_dir: Path) -> List[Path]:
    return sorted(series_dir.glob('*.dcm'))


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
    print(f"Processing every {args.slice_step} slice(s); MIP half-window={args.mip_window}")
    model = YOLO("/home/sersasj/RSNA-IAD-Codebase/runs/yolo_aneurysm_locations/baseline_slice_24bbox2/weights/best.pt")

    series_probs: Dict[str, float] = {}
    cls_labels: List[int] = []
    loc_labels: List[np.ndarray] = []
    series_pred_loc_probs: List[np.ndarray] = []

    # Optional summaries
    series_pred_counts: Dict[str, int] = {}

    for sid in tqdm(val_series, desc="Validating series", unit="series"):
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

        # In per-slice mode, apply stepping BEFORE max_slices
        if args.mip_window <= 0:
            dicoms = dicoms[::args.slice_step]
            if args.max_slices and len(dicoms) > args.max_slices:
                dicoms = dicoms[:args.max_slices]

        max_conf_all = 0.0
        per_class_max = np.zeros(N_LOC, dtype=np.float32)
        total_dets = 0
        batch: list[np.ndarray] = []

        def flush_batch(batch_imgs: list[np.ndarray]):
            nonlocal max_conf_all, total_dets, per_class_max
            if not batch_imgs:
                return
            results = model.predict(batch_imgs, verbose=False, conf=0.01)
            for r in results:
                if not r or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                    continue
                try:
                    confs = r.boxes.conf
                    clses = r.boxes.cls
                    n = len(confs)
                    total_dets += int(n)
                    for i in range(n):
                        c = float(confs[i].item())
                        k = int(clses[i].item())
                        if c > max_conf_all:
                            max_conf_all = c
                        if 0 <= k < N_LOC and c > per_class_max[k]:
                            per_class_max[k] = c
                except Exception:
                    # Fallback: just use max conf
                    try:
                        batch_max = float(r.boxes.conf.max().item())
                        if batch_max > max_conf_all:
                            max_conf_all = batch_max
                        total_dets += int(len(r.boxes))
                    except Exception:
                        pass

        if args.mip_window > 0:
            # Build HU slices and slide MIP
            slices_hu: list[np.ndarray] = []
            shapes_count: Dict[tuple[int, int], int] = {}
            for dcm_path in dicoms:  # tqdm(dicoms, desc=f"{sid} slices", leave=False, unit="slice"):
                try:
                    frames = read_dicom_frames_hu(dcm_path)
                except Exception as e:
                    if args.verbose:
                        print(f"[SKIP] {dcm_path.name}: {e}")
                    continue
                for f in frames:
                    f = f.astype(np.float32)
                    slices_hu.append(f)
                    shapes_count[f.shape] = shapes_count.get(f.shape, 0) + 1
            if not slices_hu:
                series_probs[sid] = 0.0
                series_pred_loc_probs.append(np.zeros(N_LOC, dtype=np.float32))
                series_pred_counts[sid] = 0
                continue
            target_shape = max(shapes_count.items(), key=lambda kv: kv[1])[0]
            th, tw = target_shape
            n = len(slices_hu)
            if args.mip_no_overlap:
                w = max(0, int(args.mip_window))
                window_size = 2 * w + 1
                if n <= 0:
                    centers: list[int] = []
                elif n < window_size:
                    centers = [n // 2]
                else:
                    centers = list(range(w, n, window_size))
                    if centers and centers[-1] + w < n - 1:
                        centers.append(min(n - 1 - w, n - 1))
            else:
                stride = max(1, args.slice_step)
                centers = list(range(0, n, stride))
            if args.max_slices and len(centers) > args.max_slices:
                centers = centers[:args.max_slices]

            resized_cache: Dict[int, np.ndarray] = {}
            w = args.mip_window
            for c in centers:
                lo = max(0, c - w)
                hi = min(n - 1, c + w)
                mip_hu = None
                for i in range(lo, hi + 1):
                    arr = resized_cache.get(i)
                    if arr is None:
                        a = slices_hu[i]
                        if a.shape != target_shape:
                            a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_LINEAR)
                        resized_cache[i] = a
                        arr = a
                    mip_hu = arr if mip_hu is None else np.maximum(mip_hu, arr)
                mip_u8 = min_max_normalize(mip_hu)
                if args.mip_img_size > 0 and (mip_u8.shape[0] != args.mip_img_size or mip_u8.shape[1] != args.mip_img_size):
                    mip_u8 = cv2.resize(mip_u8, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
                mip_rgb = cv2.cvtColor(mip_u8, cv2.COLOR_GRAY2BGR)
                batch.append(mip_rgb)
                if len(batch) >= args.batch_size:
                    flush_batch(batch)
                    batch.clear()
            flush_batch(batch)
        else:
            # Per-slice inference
            for dcm_path in dicoms:  # tqdm(dicoms, desc=f"{sid} slices", leave=False, unit="slice"):
                try:
                    frames = read_dicom_frames_hu(dcm_path)
                except Exception as e:
                    if args.verbose:
                        print(f"[SKIP] {dcm_path.name}: {e}")
                    continue
                for f in frames:
                    img_uint8 = min_max_normalize(f)
                    if args.mip_img_size > 0 and (img_uint8.shape[0] != args.mip_img_size or img_uint8.shape[1] != args.mip_img_size):
                        img_uint8 = cv2.resize(img_uint8, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
                    if img_uint8.ndim == 2:
                        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    batch.append(img_uint8)
                    if len(batch) >= args.batch_size:
                        flush_batch(batch)
                        batch.clear()
            flush_batch(batch)

        series_probs[sid] = max_conf_all
        series_pred_loc_probs.append(per_class_max.copy())
        series_pred_counts[sid] = total_dets
        print(f"Series {sid} max_conf_any={max_conf_all:.4f} dets={total_dets} (processed {len(dicoms)} slices)")

        # Labels and metadata for this series
        row = train_df[train_df['SeriesInstanceUID'] == sid].iloc[0]
        label = int(row['Aneurysm Present'])
        cls_labels.append(label)
        # Build 13-dim location label vector from columns (if present)
        loc_vec = np.zeros(N_LOC, dtype=np.float32)
        for idx, name in enumerate(LOCATION_LABELS):
            if name in row:
                try:
                    loc_vec[idx] = float(row[name])
                except Exception:
                    loc_vec[idx] = 0.0
        loc_labels.append(loc_vec)

    if not series_probs:
        print("No series processed.")
        return

    # Metrics
    y_true = np.array(cls_labels)
    y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
    cls_auc = roc_auc_score(y_true, y_scores)

    loc_labels_arr = np.stack(loc_labels)
    loc_pred_arr = np.stack(series_pred_loc_probs)
    per_loc_aucs = []
    for i in range(N_LOC):
        try:
            auc_i = roc_auc_score(loc_labels_arr[:, i], loc_pred_arr[:, i])
        except ValueError:
            auc_i = float('nan')
        per_loc_aucs.append(auc_i)
    loc_macro_auc = np.nanmean(per_loc_aucs)

    print(f"Classification AUC (aneurysm present): {cls_auc:.4f}")
    print(f"Location macro AUC: {loc_macro_auc:.4f}")
    print(f"Combined (mean) metric: {(cls_auc + (loc_macro_auc if not math.isnan(loc_macro_auc) else 0))/2:.4f}")

    if args.save_csv:
        out_rows = []
        for idx, sid in enumerate(series_probs.keys()):
            row = {
                'SeriesInstanceUID': sid,
                'aneurysm_prob': series_probs[sid],
            }
            probs = series_pred_loc_probs[idx]
            for i, name in enumerate(LOCATION_LABELS):
                row[f'loc_prob_{i}'] = float(probs[i])
            out_rows.append(row)
        pd.DataFrame(out_rows).to_csv(args.save_csv, index=False)
        print(f"Saved per-series predictions to {args.save_csv}")


if __name__ == '__main__':
    main()


#Baseline yolo11n 48bbox slice-based
# yolo11n.pt 48 bbox fold1
#Classification AUC (aneurysm present): 0.6960
#Location macro AUC: 0.7659
#Combined (mean) metric: 0.7309