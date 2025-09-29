"""Validate a 13-class YOLO aneurysm detector at series level.

Preprocessing:
    - DICOM -> HU -> per-slice min-max -> uint8 (BGR for YOLO inference)
    - BGR mode: stacks 3 consecutive slices [i-1, i, i+1] as 3-channel images

Series scores:
    - Aneurysm Present = max detection confidence across ALL 13 classes over processed slices/MIP windows
    - Location probs   = per-class max detection confidence over processed slices/MIP windows

Outputs:
    - Classification AUC (aneurysm present)
    - Location macro AUC (macro over 13 classes)
    - Optional CSV with per-series predictions

Notes:
    - Requires ultralytics (YOLOv8/11) and 13-class weights.
    - Use --mip-window > 0 for sliding-window MIPs; 0 for per-slice; --bgr-mode for 3-slice stacking.
"""
import argparse
from pathlib import Path
import sys
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import time
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
    ap.add_argument('--weights', type=str, required=False, default='', help='Path to YOLO weights (.pt) with 13 classes')
    ap.add_argument('--val-fold', type=int, default=1, help='Fold id to evaluate (matches  fold_id)')
    ap.add_argument('--series-limit', type=int, default=0, help='Optional limit on number of validation series (debug)')
    ap.add_argument('--max-slices', type=int, default=0, help='Optional cap on number of slices/windows per series (debug)')
    ap.add_argument('--save-csv', type=str, default='', help='Optional path to save per-series predictions CSV (deprecated, prefer --out-dir)')
    ap.add_argument('--out-dir', type=str, default='', help='If set, writes metrics JSON/CSV into this directory')
    ap.add_argument('--batch-size', type=int, default=16, help='Batch size for inference (higher = faster, more VRAM)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice (default=1)')
    # Sliding-window MIP mode
    ap.add_argument('--mip-window', type=int, default=0, help='Half-window (in slices) for MIP; 0 = per-slice mode')
    ap.add_argument('--img-size', type=int, default=0, help='Optional resize of MIP/slice to this square size before inference (0 keeps original)')
    ap.add_argument('--mip-no-overlap', action='store_true', help='Use non-overlapping MIP windows (stride = 2*w+1 instead of slice_step)')
    # CV
    ap.add_argument('--cv', action='store_true', help='Run validation across all folds found in train_df.csv/')
    ap.add_argument('--folds', type=str, default='', help='Comma-separated fold ids to run (overrides --val-fold when provided)')
    # Weights & Biases
    ap.add_argument('--wandb', default=True, action='store_true', help='Log metrics and outputs to Weights & Biases')
    ap.add_argument('--wandb-project', type=str, default='yolo_aneurysm_locations', help='W&B project name (no slashes)')
    ap.add_argument('--wandb-entity', type=str, default='', help='W&B entity (team/user)')
    ap.add_argument('--wandb-run-name', type=str, default='', help='W&B run name (defaults to val_fold{fold})')
    ap.add_argument('--wandb-group', type=str, default='', help='Optional W&B group')
    ap.add_argument('--wandb-tags', type=str, default='', help='Comma-separated W&B tags')
    ap.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline'], help='W&B mode (online/offline)')
    ap.add_argument('--bgr-mode', action='store_true', help='Use BGR mode (3-channel images from stacked slices)')
    ap.add_argument('--bgr-non-overlapping', action='store_true', help='Use non-overlapping 3-slice windows in BGR mode (e.g., [1,2,3],[4,5,6],[7,8,9])')
    ap.add_argument('--wandb-resume-id', type=str, default='', help='Resume W&B run id (optional)')
    ap.add_argument('--single-cls', action='store_true', help='Single class mode: only classify aneurysm present/absent (no location prediction)')
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


def _run_validation_for_fold(args: argparse.Namespace, weights_path: str, fold_id: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Runs validation for a single fold and returns (metrics_dict, per_series_df)."""
    # Load split
    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_df = pd.read_csv(data_root / 'train.csv')
    if 'Aneurysm Present' not in train_df.columns:
        raise SystemExit("train_df.csv requires 'Aneurysm Present' column for classification label")

    val_series = train_df[train_df['fold_id'] == fold_id]['SeriesInstanceUID'].unique().tolist()
    if args.series_limit:
        val_series = val_series[:args.series_limit]

    print(f"Validation fold {fold_id}: {len(val_series)} series")
    print(f"Processing every {args.slice_step} slice(s); MIP half-window={args.mip_window}")
    model = YOLO(weights_path)

    series_probs: Dict[str, float] = {}
    cls_labels: List[int] = []
    loc_labels: List[np.ndarray] = []
    series_pred_loc_probs: List[np.ndarray] = []
    series_pred_counts: Dict[str, int] = {}
    
    # For partial AUC tracking
    processed_count = 0
    partial_auc_interval = 10

    def compute_partial_metrics(series_ids, probs_dict, cls_labels_list, loc_labels_list, loc_probs_list):
        """Compute partial AUC metrics with current data"""
        if len(cls_labels_list) < 2:
            return float('nan'), float('nan'), float('nan')
        
        # Classification AUC
        try:
            y_true = np.array(cls_labels_list)
            y_scores = np.array([probs_dict[sid] for sid in series_ids])
            if len(np.unique(y_true)) < 2:  # Need both classes
                cls_auc = float('nan')
            else:
                cls_auc = roc_auc_score(y_true, y_scores)
        except Exception:
            cls_auc = float('nan')
        
        # Location macro AUC (skip in single-cls mode)
        if args.single_cls or len(loc_labels_list) == 0:
            loc_macro_auc = float('nan')
            competition_score = cls_auc
        else:
            try:
                loc_labels_arr = np.stack(loc_labels_list)
                loc_pred_arr = np.stack(loc_probs_list)
                per_loc_aucs = []
                for i in range(N_LOC):
                    try:
                        auc_i = roc_auc_score(loc_labels_arr[:, i], loc_pred_arr[:, i])
                    except ValueError:
                        auc_i = float('nan')
                    per_loc_aucs.append(auc_i)
                loc_macro_auc = np.nanmean(per_loc_aucs)
                
                # Compute competition score
                valid_loc_aucs = [auc for auc in per_loc_aucs if not math.isnan(auc)]
                if valid_loc_aucs:
                    aneurysm_weight = 13
                    location_weight = 1
                    total_weights = aneurysm_weight + location_weight * N_LOC
                    weighted_sum = aneurysm_weight * cls_auc + location_weight * sum(valid_loc_aucs)
                    competition_score = weighted_sum / total_weights
                else:
                    competition_score = cls_auc
            except Exception:
                loc_macro_auc = float('nan')
                competition_score = cls_auc
                
        return cls_auc, loc_macro_auc, competition_score

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
                        # In single-cls mode, all detections are treated as aneurysm class
                        if not args.single_cls and 0 <= k < N_LOC and c > per_class_max[k]:
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
            for dcm_path in dicoms:
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
                if args.img_size > 0 and (mip_u8.shape[0] != args.img_size or mip_u8.shape[1] != args.img_size):
                    mip_u8 = cv2.resize(mip_u8, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
                mip_rgb = cv2.cvtColor(mip_u8, cv2.COLOR_GRAY2BGR) if mip_u8.ndim == 2 else mip_u8
                batch.append(mip_rgb)
                if len(batch) >= args.batch_size:
                    flush_batch(batch)
                    batch.clear()
            flush_batch(batch)
        elif getattr(args, 'bgr_mode', False):
            # BGR mode: stack 3 consecutive slices as channels
            slices_hu: list[np.ndarray] = []
            for dcm_path in dicoms:
                try:
                    frames = read_dicom_frames_hu(dcm_path)
                except Exception as e:
                    if args.verbose:
                        print(f"[SKIP] {dcm_path.name}: {e}")
                    continue
                for f in frames:
                    f = f.astype(np.float32)
                    slices_hu.append(f)

            if len(slices_hu) < 3:
                # Not enough slices for BGR mode, skip this series
                series_probs[sid] = 0.0
                series_pred_loc_probs.append(np.zeros(N_LOC, dtype=np.float32))
                series_pred_counts[sid] = 0
                continue

            if getattr(args, 'bgr_non_overlapping', False):
                # Non-overlapping 3-slice windows: [0,1,2], [3,4,5], [6,7,8], etc.
                window_starts = list(range(0, len(slices_hu) - 2, 3))  # Start every 3 slices, ensure we have at least 3 slices
                if args.max_slices and len(window_starts) > args.max_slices:
                    window_starts = window_starts[:args.max_slices]
                
                for start_idx in window_starts:
                    # Get exactly 3 consecutive slices: [start_idx, start_idx+1, start_idx+2]
                    slice_indices_3 = [start_idx, start_idx + 1, start_idx + 2]
                    
                    # Load and normalize the 3 slices
                    processed_slices = []
                    target_shape = slices_hu[0].shape
                    for idx in slice_indices_3:
                        slice_hu = slices_hu[idx]
                        # Resize if needed to match target shape
                        if slice_hu.shape != target_shape:
                            slice_hu = cv2.resize(slice_hu, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                        slice_u8 = min_max_normalize(slice_hu)
                        processed_slices.append(slice_u8)

                    # Stack as 3-channel image
                    bgr_img = np.stack(processed_slices, axis=-1)  # Shape: (H, W, 3)

                    # Resize if needed
                    if args.img_size > 0 and (bgr_img.shape[0] != args.img_size or bgr_img.shape[1] != args.img_size):
                        bgr_img = cv2.resize(bgr_img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

                    batch.append(bgr_img)
                    if len(batch) >= args.batch_size:
                        flush_batch(batch)
                        batch.clear()
            else:
                # Original overlapping BGR mode: [i-1,i,i+1] for each i
                step = max(1, args.slice_step)
                slice_indices = list(range(0, len(slices_hu), step))
                if args.max_slices and len(slice_indices) > args.max_slices:
                    slice_indices = slice_indices[:args.max_slices]

                for i in slice_indices:
                    # Get 3 consecutive slices: [i-1, i, i+1], handling boundaries
                    slice_indices_3 = []
                    for offset in [-1, 0, 1]:
                        idx = i + offset
                        if 0 <= idx < len(slices_hu):
                            slice_indices_3.append(idx)
                        else:
                            # For boundary cases, duplicate the center slice
                            slice_indices_3.append(i)

                    # Ensure we have exactly 3 slices
                    while len(slice_indices_3) < 3:
                        slice_indices_3.append(i)

                    # Load and normalize the 3 slices
                    processed_slices = []
                    target_shape = slices_hu[0].shape
                    for idx in slice_indices_3[:3]:
                        slice_hu = slices_hu[idx]
                        # Resize if needed to match target shape
                        if slice_hu.shape != target_shape:
                            slice_hu = cv2.resize(slice_hu, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                        slice_u8 = min_max_normalize(slice_hu)
                        processed_slices.append(slice_u8)

                    # Stack as 3-channel image
                    bgr_img = np.stack(processed_slices, axis=-1)  # Shape: (H, W, 3)

                    # Resize if needed
                    if args.img_size > 0 and (bgr_img.shape[0] != args.img_size or bgr_img.shape[1] != args.img_size):
                        bgr_img = cv2.resize(bgr_img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

                    batch.append(bgr_img)
                    if len(batch) >= args.batch_size:
                        flush_batch(batch)
                        batch.clear()
            flush_batch(batch)
        else:
            # Per-slice inference
            for dcm_path in dicoms:
                try:
                    frames = read_dicom_frames_hu(dcm_path)
                except Exception as e:
                    if args.verbose:
                        print(f"[SKIP] {dcm_path.name}: {e}")
                    continue
                for f in frames:
                    img_uint8 = min_max_normalize(f)
                    if args.img_size > 0 and (img_uint8.shape[0] != args.img_size or img_uint8.shape[1] != args.img_size):
                        img_uint8 = cv2.resize(img_uint8, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
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
        if args.verbose:
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
        
        # Track partial AUC every 10 series
        processed_count += 1
        if processed_count % partial_auc_interval == 0:
            current_series = list(series_probs.keys())
            partial_cls_auc, partial_loc_auc, partial_comp_score = compute_partial_metrics(
                current_series, series_probs, cls_labels, loc_labels, series_pred_loc_probs
            )
            
            if not math.isnan(partial_cls_auc):
                if args.single_cls:
                    print(f"[Partial after {processed_count} series] Cls AUC: {partial_cls_auc:.4f}")
                else:
                    if not math.isnan(partial_loc_auc):
                        print(f"[Partial after {processed_count} series] Cls AUC: {partial_cls_auc:.4f}, Loc AUC: {partial_loc_auc:.4f}, Comp Score: {partial_comp_score:.4f}")
                    else:
                        print(f"[Partial after {processed_count} series] Cls AUC: {partial_cls_auc:.4f}, Loc AUC: insufficient data")
            else:
                print(f"[Partial after {processed_count} series]: insufficient data")

    if not series_probs:
        print("No series processed.")
        # Return empty metrics and df
        return {
            'cls_auc': float('nan'),
            'cls_ap': float('nan'),
            'loc_macro_auc': float('nan'),
            'combined_mean': float('nan'),
            'mean_num_detections': 0.0,
            'per_loc_aucs': {name: float('nan') for name in LOCATION_LABELS},
            'fold_id': fold_id,
        }, pd.DataFrame()

    # Metrics
    y_true = np.array(cls_labels)
    y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
    cls_auc = roc_auc_score(y_true, y_scores)
    try:
        cls_ap = average_precision_score(y_true, y_scores)
    except Exception:
        cls_ap = float('nan')

    # In single-cls mode, skip location metrics
    if args.single_cls:
        per_loc_aucs = []
        loc_macro_auc = float('nan')
        competition_score = cls_auc  # Only aneurysm classification matters
        combined_mean = cls_auc
    else:
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

        # Compute competition-style weighted metric
        # Competition weights: 13 for aneurysm present, 1 for each location
        aneurysm_weight = 13
        location_weight = 1
        total_weights = aneurysm_weight + location_weight * N_LOC  # 13 + 13 = 26

        valid_loc_aucs = [auc for auc in per_loc_aucs if not math.isnan(auc)]
        if valid_loc_aucs:
            weighted_sum = aneurysm_weight * cls_auc + location_weight * sum(valid_loc_aucs)
            competition_score = weighted_sum / total_weights
        else:
            competition_score = cls_auc  # fallback if no location AUCs

        # For backward compatibility, also compute the simple average
        combined_mean = (cls_auc + (loc_macro_auc if not math.isnan(loc_macro_auc) else 0)) / 2

    # Final partial AUC report if not already printed at this exact count
    if processed_count % partial_auc_interval != 0:
        current_series = list(series_probs.keys())
        partial_cls_auc, partial_loc_auc, partial_comp_score = compute_partial_metrics(
            current_series, series_probs, cls_labels, loc_labels, series_pred_loc_probs
        )
        
        if not math.isnan(partial_cls_auc):
            if args.single_cls:
                print(f"[Final partial after {processed_count} series] Cls AUC: {partial_cls_auc:.4f}")
            else:
                if not math.isnan(partial_loc_auc):
                    print(f"[Final partial after {processed_count} series] Cls AUC: {partial_cls_auc:.4f}, Loc AUC: {partial_loc_auc:.4f}, Comp Score: {partial_comp_score:.4f}")
                else:
                    print(f"[Final partial after {processed_count} series] Cls AUC: {partial_cls_auc:.4f}, Loc AUC: insufficient data")

    print(f"Classification AUC (aneurysm present): {cls_auc:.4f}")
    print(f"Classification AP (PR AUC): {cls_ap:.4f}")
    if args.single_cls:
        print("Single-cls mode: Location metrics skipped")
        print(f"Final score (classification only): {cls_auc:.4f}")
    else:
        print(f"Location macro AUC: {loc_macro_auc:.4f}")
        print(f"Competition score (weighted): {competition_score:.4f}")
        print(f"Combined (mean) metric: {combined_mean:.4f}")  # Should be same as competition_score

    # Build per-series dataframe
    out_rows = []
    keys = list(series_probs.keys())
    for idx, sid in enumerate(keys):
        row_df = train_df[train_df['SeriesInstanceUID'] == sid]
        label = int(row_df['Aneurysm Present'].iloc[0]) if not row_df.empty else 0
        row = {
            'SeriesInstanceUID': sid,
            'aneurysm_prob': float(series_probs[sid]),
            'label_aneurysm': label,
            'num_detections': int(series_pred_counts.get(sid, 0)),
        }
        # Only add location data if not in single-cls mode
        if not args.single_cls:
            probs = series_pred_loc_probs[idx]
            for i, name in enumerate(LOCATION_LABELS):
                row[f'loc_prob_{i}'] = float(probs[i])
                # add label if present
                if name in row_df.columns:
                    try:
                        row[f'loc_label_{i}'] = float(row_df[name].iloc[0])
                    except Exception:
                        row[f'loc_label_{i}'] = 0.0
        out_rows.append(row)
    per_series_df = pd.DataFrame(out_rows)

    metrics = {
        'fold_id': fold_id,
        'cls_auc': float(cls_auc),
        'cls_ap': float(cls_ap) if not (isinstance(cls_ap, float) and math.isnan(cls_ap)) else float('nan'),
        'loc_macro_auc': float(loc_macro_auc) if not math.isnan(loc_macro_auc) else float('nan'),
        'competition_score': float(competition_score) if not math.isnan(competition_score) else float('nan'),
        'combined_mean': float(combined_mean) if not math.isnan(combined_mean) else float('nan'),
        'mean_num_detections': float(np.mean(list(series_pred_counts.values())) if series_pred_counts else 0.0),
        'per_loc_aucs': {} if args.single_cls else {LOCATION_LABELS[i]: (float(per_loc_aucs[i]) if not math.isnan(per_loc_aucs[i]) else float('nan')) for i in range(N_LOC)},
        'single_cls_mode': bool(args.single_cls),
        'num_series': int(len(keys)),
    }
    return metrics, per_series_df


def _maybe_init_wandb(args: argparse.Namespace, fold_id: int, weights_path: str):
    if not getattr(args, 'wandb', False):
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"W&B not available: {e}. Install with 'pip install wandb' or disable --wandb.")
        return None
    if wandb.run is not None:
        return wandb
    run_name = args.wandb_run_name or f"val_fold{fold_id}"
    tags = [t.strip() for t in (args.wandb_tags.split(',') if args.wandb_tags else []) if t.strip()]
    # Basic config for traceability
    config = {
        'weights_path': weights_path,
        'val_fold': fold_id,
        'slice_step': args.slice_step,
        'mip_window': args.mip_window,
        'img_size': args.img_size,
        'mip_no_overlap': args.mip_no_overlap,
        'bgr_mode': args.bgr_mode,
        'bgr_non_overlapping': args.bgr_non_overlapping,
        'batch_size': args.batch_size,
        'series_limit': args.series_limit,
        'max_slices': args.max_slices,
        'single_cls': args.single_cls,
    }
    project = (args.wandb_project or 'rsna_iad').replace('/', '_')
    wandb.init(
        project=project,
        entity=args.wandb_entity or None,
        name=run_name,
        group=args.wandb_group or None,
        tags=tags or None,
        mode=args.wandb_mode or 'online',
        id=args.wandb_resume_id or None,
        resume='allow' if args.wandb_resume_id else None,
        config=config,
    )
    return wandb


def main():
    args = parse_args()
    # Resolve weights path
    weights = args.weights.strip()
    if not weights:
        # Try to infer a latest best.pt under runs dir if present
        default = ROOT / 'runs'
        raise SystemExit('Please provide --weights path to YOLO .pt file')

    # Prepare out dir
    out_dir: Optional[Path] = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Determine folds to run
    folds: List[int]
    if args.cv or args.folds:
        data_root = Path(data_path)
        train_df = pd.read_csv(data_root / 'train.csv')
        if args.folds:
            folds = [int(x) for x in args.folds.split(',') if x.strip() != '']
        else:
            folds = sorted(int(x) for x in pd.unique(train_df['fold_id']))
    else:
        folds = [int(args.val_fold)]

    all_fold_metrics: List[Dict[str, float]] = []
    for f in folds:
        fold_out_dir = out_dir / f'fold_{f}' if out_dir else None
        if fold_out_dir:
            fold_out_dir.mkdir(parents=True, exist_ok=True)
        # Init W&B per-fold (separate runs per fold to mirror training)
        wandb = _maybe_init_wandb(args, f, weights)
        metrics, per_series_df = _run_validation_for_fold(args, weights, f)
        all_fold_metrics.append(metrics)
        # Save fold artifacts
        if fold_out_dir:
            # Per-series CSV
            per_series_csv = fold_out_dir / 'per_series_predictions.csv'
            per_series_df.to_csv(per_series_csv, index=False)
            # Metrics JSON and per-class CSV
            import json
            with open(fold_out_dir / 'metrics.json', 'w') as fjs:
                json.dump(metrics, fjs, indent=2)
            per_loc_items = list(metrics['per_loc_aucs'].items()) if isinstance(metrics.get('per_loc_aucs'), dict) else []
            if per_loc_items:
                pd.DataFrame(per_loc_items, columns=['location', 'auc']).to_csv(fold_out_dir / 'per_location_auc.csv', index=False)

        # W&B logging of metrics and tables
        if wandb is not None:
            try:
                # Scalars
                scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                wandb.log(scalars)
                # Per-location AUCs as a table
                if isinstance(metrics.get('per_loc_aucs'), dict) and metrics['per_loc_aucs']:
                    loc_df = pd.DataFrame(list(metrics['per_loc_aucs'].items()), columns=['location', 'auc'])
                    wandb.log({'per_location_auc_table': wandb.Table(dataframe=loc_df)})
                # Upload per-series predictions as a table and as artifact
                if not per_series_df.empty:
                    wandb.log({'per_series_predictions': wandb.Table(dataframe=per_series_df)})
                # Also store as artifact files if available
                if fold_out_dir:
                    art = wandb.Artifact(name=f"val_fold{f}_artifacts", type='validation')
                    if (fold_out_dir / 'metrics.json').exists():
                        art.add_file(str(fold_out_dir / 'metrics.json'))
                    if (fold_out_dir / 'per_series_predictions.csv').exists():
                        art.add_file(str(fold_out_dir / 'per_series_predictions.csv'))
                    if (fold_out_dir / 'per_location_auc.csv').exists():
                        art.add_file(str(fold_out_dir / 'per_location_auc.csv'))
                    wandb.log_artifact(art)
            finally:
                # End run after each fold
                try:
                    import wandb as _wandb  # type: ignore
                    if _wandb.run is not None:
                        _wandb.finish()
                except Exception:
                    pass

    # CV summary
    if out_dir and len(all_fold_metrics) > 1:
        # Aggregate simple means across folds for scalar metrics
        agg_keys = ['cls_auc', 'cls_ap', 'loc_macro_auc', 'competition_score', 'combined_mean', 'mean_num_detections']
        summary = {'num_folds': len(all_fold_metrics)}
        for k in agg_keys:
            vals = [m[k] for m in all_fold_metrics if isinstance(m.get(k), (int, float)) and not math.isnan(m.get(k))]
            summary[f'mean_{k}'] = float(np.mean(vals)) if vals else float('nan')
        # Per-location AUCS mean
        loc_aucs_df_rows = []
        for m in all_fold_metrics:
            if isinstance(m.get('per_loc_aucs'), dict):
                loc_aucs_df_rows.append(pd.Series(m['per_loc_aucs']))
        if loc_aucs_df_rows:
            loc_aucs_df = pd.DataFrame(loc_aucs_df_rows)
            summary_per_loc = loc_aucs_df.mean(skipna=True).to_dict()
            # Save per-location mean CSV
            pd.DataFrame(list(summary_per_loc.items()), columns=['location', 'mean_auc']).to_csv(out_dir / 'cv_per_location_mean_auc.csv', index=False)
            summary['per_loc_mean_auc'] = {k: (float(v) if not pd.isna(v) else float('nan')) for k, v in summary_per_loc.items()}
    # Save summary JSON
        import json
        with open(out_dir / 'cv_summary.json', 'w') as fjs:
            json.dump(summary, fjs, indent=2)

    # Back-compat: --save-csv if requested and no out_dir
    if args.save_csv and not args.out_dir and all_fold_metrics:
        # Only for the last fold run
        metrics, per_series_df = all_fold_metrics[-1], None
        # We didn't retain per_series_df here; encourage using out_dir.
        print('Tip: prefer --out-dir to save structured outputs per fold.')


if __name__ == '__main__':
    main()

# python3 /home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_validation.py --weights yolo_aneurysm_locations/cv_mobilenet_more_negatives_fold22/weights/best.pt --val-fold 2 --out-dir yolo_aneurysm_locations/cv_mobilenet_more_negatives_fold22/series_validation --batch-size 16 --slice-step 1 