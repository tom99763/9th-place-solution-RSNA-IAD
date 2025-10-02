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
    ap.add_argument('--agg-strategies', type=str, default='max', help='Comma-separated list of aggregation strategies: max,mean,top3,top5,p95,p90,median,max_mean (default: max)')
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


def aggregate_confidences(all_confs: List[float], strategy: str) -> float:
    """Aggregate list of confidences using specified strategy.
    
    Strategies:
    - max: Maximum confidence
    - mean: Mean of all confidences
    - top3: Mean of top 3 confidences
    - top5: Mean of top 5 confidences
    - top10: Mean of top 10 confidences
    - p95: 95th percentile
    - p90: 90th percentile
    - p75: 75th percentile
    - median: Median confidence
    - max_mean: Weighted combination of max and mean (0.7*max + 0.3*mean)
    """
    if not all_confs:
        return 0.0
    
    arr = np.array(all_confs)
    
    if strategy == 'max':
        return float(np.max(arr))
    elif strategy == 'mean':
        return float(np.mean(arr))
    elif strategy.startswith('top'):
        k = int(strategy[3:])  # e.g., 'top3' -> 3
        if len(arr) < k:
            return float(np.mean(arr))
        top_k = np.partition(arr, -k)[-k:]
        return float(np.mean(top_k))
    elif strategy.startswith('p'):
        percentile = int(strategy[1:])  # e.g., 'p95' -> 95
        return float(np.percentile(arr, percentile))
    elif strategy == 'median':
        return float(np.median(arr))
    elif strategy == 'max_mean':
        return float(0.7 * np.max(arr) + 0.3 * np.mean(arr))
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


def _run_validation_for_fold(args: argparse.Namespace, weights_path: str, fold_id: int, strategies: List[str]) -> Tuple[Dict[str, float], pd.DataFrame]:
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
    print(f"Testing aggregation strategies: {', '.join(strategies)}")
    model = YOLO(weights_path)

    # Store raw detection data per series for each strategy
    series_all_confs: Dict[str, List[float]] = {}  # All detection confidences per series
    series_all_class_confs: Dict[str, List[List[float]]] = {}  # Per-class confidences [series][class_idx] = [confs]
    cls_labels: List[int] = []
    loc_labels: List[np.ndarray] = []
    series_pred_counts: Dict[str, int] = {}
    
    # For partial AUC tracking
    processed_count = 0
    partial_auc_interval = 10

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

        # Collect all confidences for aggregation strategies
        all_confs: List[float] = []  # All detection confidences across slices
        per_class_confs: List[List[float]] = [[] for _ in range(N_LOC)]  # Per-class confidences
        total_dets = 0
        batch: list[np.ndarray] = []

        def flush_batch(batch_imgs: list[np.ndarray]):
            nonlocal total_dets, all_confs, per_class_confs
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
                        all_confs.append(c)
                        # In single-cls mode, all detections are treated as aneurysm class
                        if not args.single_cls and 0 <= k < N_LOC:
                            per_class_confs[k].append(c)
                except Exception:
                    # Fallback: just collect confidences
                    try:
                        for c in r.boxes.conf:
                            all_confs.append(float(c.item()))
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

        # Store raw data for later aggregation
        series_all_confs[sid] = all_confs
        series_all_class_confs[sid] = per_class_confs
        series_pred_counts[sid] = total_dets
        if args.verbose:
            max_conf = max(all_confs) if all_confs else 0.0
            print(f"Series {sid} max_conf={max_conf:.4f} dets={total_dets} total_confs={len(all_confs)} (processed {len(dicoms)} slices)")

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

    if not series_all_confs:
        print("No series processed.")
        # Return empty metrics and df for each strategy
        empty_metrics = {}
        for strat in strategies:
            empty_metrics[strat] = {
                'cls_auc': float('nan'),
                'cls_ap': float('nan'),
                'loc_macro_auc': float('nan'),
                'combined_mean': float('nan'),
                'competition_score': float('nan'),
                'mean_num_detections': 0.0,
                'per_loc_aucs': {name: float('nan') for name in LOCATION_LABELS},
                'fold_id': fold_id,
                'strategy': strat,
            }
        return empty_metrics, pd.DataFrame()

    # Get series keys in consistent order
    series_keys = list(series_all_confs.keys())
    y_true = np.array(cls_labels)
    loc_labels_arr = np.stack(loc_labels) if loc_labels else None

    # Compute metrics for EACH aggregation strategy
    print(f"\n{'='*80}")
    print(f"COMPARING AGGREGATION STRATEGIES")
    print(f"{'='*80}\n")
    
    all_strategy_metrics = {}
    all_strategy_dfs = {}
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.upper()} ---")
        
        # Aggregate confidences using this strategy
        series_probs = {}
        series_pred_loc_probs = []
        
        for sid in series_keys:
            # Aggregate overall confidence
            series_probs[sid] = aggregate_confidences(series_all_confs[sid], strategy)
            
            # Aggregate per-class confidences
            if not args.single_cls:
                per_class_agg = np.zeros(N_LOC, dtype=np.float32)
                for class_idx in range(N_LOC):
                    class_confs = series_all_class_confs[sid][class_idx]
                    per_class_agg[class_idx] = aggregate_confidences(class_confs, strategy)
                series_pred_loc_probs.append(per_class_agg)
        
        # Compute metrics for this strategy
        y_scores = np.array([series_probs[sid] for sid in series_keys])
        cls_auc = roc_auc_score(y_true, y_scores)
        try:
            cls_ap = average_precision_score(y_true, y_scores)
        except Exception:
            cls_ap = float('nan')

        # In single-cls mode, skip location metrics
        if args.single_cls:
            per_loc_aucs = []
            loc_macro_auc = float('nan')
            competition_score = cls_auc
            combined_mean = cls_auc
        else:
            loc_pred_arr = np.stack(series_pred_loc_probs)
            per_loc_aucs = []
            for i in range(N_LOC):
                try:
                    auc_i = roc_auc_score(loc_labels_arr[:, i], loc_pred_arr[:, i])
                except ValueError:
                    auc_i = float('nan')
                per_loc_aucs.append(auc_i)
            loc_macro_auc = np.nanmean(per_loc_aucs)

            # Competition-style weighted metric
            aneurysm_weight = 13
            location_weight = 1
            total_weights = aneurysm_weight + location_weight * N_LOC

            valid_loc_aucs = [auc for auc in per_loc_aucs if not math.isnan(auc)]
            if valid_loc_aucs:
                weighted_sum = aneurysm_weight * cls_auc + location_weight * sum(valid_loc_aucs)
                competition_score = weighted_sum / total_weights
            else:
                competition_score = cls_auc

            combined_mean = (cls_auc + (loc_macro_auc if not math.isnan(loc_macro_auc) else 0)) / 2

        # Print metrics for this strategy
        print(f"  Classification AUC: {cls_auc:.4f}")
        print(f"  Classification AP:  {cls_ap:.4f}")
        if not args.single_cls:
            print(f"  Location macro AUC: {loc_macro_auc:.4f}")
            print(f"  Competition score:  {competition_score:.4f}")

        # Store metrics
        metrics = {
            'fold_id': fold_id,
            'strategy': strategy,
            'cls_auc': float(cls_auc),
            'cls_ap': float(cls_ap) if not (isinstance(cls_ap, float) and math.isnan(cls_ap)) else float('nan'),
            'loc_macro_auc': float(loc_macro_auc) if not math.isnan(loc_macro_auc) else float('nan'),
            'competition_score': float(competition_score) if not math.isnan(competition_score) else float('nan'),
            'combined_mean': float(combined_mean) if not math.isnan(combined_mean) else float('nan'),
            'mean_num_detections': float(np.mean(list(series_pred_counts.values())) if series_pred_counts else 0.0),
            'per_loc_aucs': {} if args.single_cls else {LOCATION_LABELS[i]: (float(per_loc_aucs[i]) if not math.isnan(per_loc_aucs[i]) else float('nan')) for i in range(N_LOC)},
            'single_cls_mode': bool(args.single_cls),
            'num_series': int(len(series_keys)),
        }
        all_strategy_metrics[strategy] = metrics
        
        # Build per-series dataframe for this strategy
        out_rows = []
        for idx, sid in enumerate(series_keys):
            row_df = train_df[train_df['SeriesInstanceUID'] == sid]
            label = int(row_df['Aneurysm Present'].iloc[0]) if not row_df.empty else 0
            row = {
                'SeriesInstanceUID': sid,
                'aneurysm_prob': float(series_probs[sid]),
                'label_aneurysm': label,
                'num_detections': int(series_pred_counts.get(sid, 0)),
                'num_confidences': len(series_all_confs[sid]),
            }
            if not args.single_cls:
                probs = series_pred_loc_probs[idx]
                for i, name in enumerate(LOCATION_LABELS):
                    row[f'loc_prob_{i}'] = float(probs[i])
                    if name in row_df.columns:
                        try:
                            row[f'loc_label_{i}'] = float(row_df[name].iloc[0])
                        except Exception:
                            row[f'loc_label_{i}'] = 0.0
            out_rows.append(row)
        all_strategy_dfs[strategy] = pd.DataFrame(out_rows)

    # Print summary comparison
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<15} {'Cls AUC':>10} {'Cls AP':>10} {'Loc AUC':>10} {'Comp Score':>12}")
    print(f"{'-'*80}")
    for strategy in strategies:
        m = all_strategy_metrics[strategy]
        cls_auc_str = f"{m['cls_auc']:.4f}" if not math.isnan(m['cls_auc']) else "N/A"
        cls_ap_str = f"{m['cls_ap']:.4f}" if not math.isnan(m['cls_ap']) else "N/A"
        loc_auc_str = f"{m['loc_macro_auc']:.4f}" if not math.isnan(m['loc_macro_auc']) else "N/A"
        comp_str = f"{m['competition_score']:.4f}" if not math.isnan(m['competition_score']) else "N/A"
        print(f"{strategy:<15} {cls_auc_str:>10} {cls_ap_str:>10} {loc_auc_str:>10} {comp_str:>12}")
    print(f"{'='*80}\n")

    # Find best strategy by competition score
    best_strategy = max(strategies, key=lambda s: all_strategy_metrics[s].get('competition_score', float('-inf')))
    print(f"🏆 Best strategy: {best_strategy.upper()} (Competition Score: {all_strategy_metrics[best_strategy]['competition_score']:.4f})")

    return all_strategy_metrics, all_strategy_dfs


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

    # Parse aggregation strategies
    strategies = [s.strip() for s in args.agg_strategies.split(',') if s.strip()]
    if not strategies:
        strategies = ['max']
    print(f"Will test {len(strategies)} aggregation strategy(ies): {', '.join(strategies)}")

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

    all_fold_metrics: List[Dict[str, Dict[str, float]]] = []  # List of {strategy: metrics} dicts
    for f in folds:
        fold_out_dir = out_dir / f'fold_{f}' if out_dir else None
        if fold_out_dir:
            fold_out_dir.mkdir(parents=True, exist_ok=True)
        # Init W&B per-fold (separate runs per fold to mirror training)
        wandb = _maybe_init_wandb(args, f, weights)
        strategy_metrics, strategy_dfs = _run_validation_for_fold(args, weights, f, strategies)
        all_fold_metrics.append(strategy_metrics)
        # Save fold artifacts for EACH strategy
        if fold_out_dir:
            import json
            
            # Save comparison summary
            comparison_rows = []
            for strategy in strategies:
                m = strategy_metrics[strategy]
                comparison_rows.append({
                    'strategy': strategy,
                    'cls_auc': m.get('cls_auc', float('nan')),
                    'cls_ap': m.get('cls_ap', float('nan')),
                    'loc_macro_auc': m.get('loc_macro_auc', float('nan')),
                    'competition_score': m.get('competition_score', float('nan')),
                    'combined_mean': m.get('combined_mean', float('nan')),
                })
            comparison_df = pd.DataFrame(comparison_rows)
            comparison_df.to_csv(fold_out_dir / 'strategy_comparison.csv', index=False)
            
            # Save detailed results for each strategy
            for strategy in strategies:
                metrics = strategy_metrics[strategy]
                per_series_df = strategy_dfs[strategy]
                
                # Create strategy subdirectory
                strat_dir = fold_out_dir / strategy
                strat_dir.mkdir(parents=True, exist_ok=True)
                
                # Per-series CSV
                per_series_csv = strat_dir / 'per_series_predictions.csv'
                per_series_df.to_csv(per_series_csv, index=False)
                
                # Metrics JSON
                with open(strat_dir / 'metrics.json', 'w') as fjs:
                    json.dump(metrics, fjs, indent=2)
                
                # Per-location AUC CSV
                per_loc_items = list(metrics['per_loc_aucs'].items()) if isinstance(metrics.get('per_loc_aucs'), dict) else []
                if per_loc_items:
                    pd.DataFrame(per_loc_items, columns=['location', 'auc']).to_csv(strat_dir / 'per_location_auc.csv', index=False)

        # W&B logging of metrics and tables
        if wandb is not None:
            try:
                # Log comparison table
                comparison_rows = []
                for strategy in strategies:
                    m = strategy_metrics[strategy]
                    comparison_rows.append({
                        'strategy': strategy,
                        'cls_auc': m.get('cls_auc', float('nan')),
                        'cls_ap': m.get('cls_ap', float('nan')),
                        'loc_macro_auc': m.get('loc_macro_auc', float('nan')),
                        'competition_score': m.get('competition_score', float('nan')),
                    })
                wandb.log({'strategy_comparison': wandb.Table(dataframe=pd.DataFrame(comparison_rows))})
                
                # Log metrics for each strategy with prefix
                for strategy in strategies:
                    metrics = strategy_metrics[strategy]
                    scalars = {f"{strategy}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                    wandb.log(scalars)
                    
                    # Per-series predictions table
                    per_series_df = strategy_dfs[strategy]
                    if not per_series_df.empty:
                        wandb.log({f'{strategy}/per_series_predictions': wandb.Table(dataframe=per_series_df)})
                
                # Upload artifacts if available
                if fold_out_dir:
                    art = wandb.Artifact(name=f"val_fold{f}_artifacts", type='validation')
                    if (fold_out_dir / 'strategy_comparison.csv').exists():
                        art.add_file(str(fold_out_dir / 'strategy_comparison.csv'))
                    for strategy in strategies:
                        strat_dir = fold_out_dir / strategy
                        if strat_dir.exists():
                            for fname in ['metrics.json', 'per_series_predictions.csv', 'per_location_auc.csv']:
                                fpath = strat_dir / fname
                                if fpath.exists():
                                    art.add_file(str(fpath), name=f'{strategy}/{fname}')
                    wandb.log_artifact(art)
            finally:
                # End run after each fold
                try:
                    import wandb as _wandb  # type: ignore
                    if _wandb.run is not None:
                        _wandb.finish()
                except Exception:
                    pass

    # CV summary - aggregate across folds AND strategies
    if out_dir and len(all_fold_metrics) > 1:
        import json
        
        # For each strategy, aggregate across folds
        cv_summary = {'num_folds': len(all_fold_metrics), 'strategies': {}}
        
        for strategy in strategies:
            agg_keys = ['cls_auc', 'cls_ap', 'loc_macro_auc', 'competition_score', 'combined_mean', 'mean_num_detections']
            strat_summary = {}
            
            for k in agg_keys:
                vals = [fold_metrics[strategy][k] for fold_metrics in all_fold_metrics 
                        if strategy in fold_metrics and isinstance(fold_metrics[strategy].get(k), (int, float)) 
                        and not math.isnan(fold_metrics[strategy].get(k))]
                strat_summary[f'mean_{k}'] = float(np.mean(vals)) if vals else float('nan')
                strat_summary[f'std_{k}'] = float(np.std(vals)) if vals else float('nan')
            
            # Per-location AUCs mean
            loc_aucs_df_rows = []
            for fold_metrics in all_fold_metrics:
                if strategy in fold_metrics and isinstance(fold_metrics[strategy].get('per_loc_aucs'), dict):
                    loc_aucs_df_rows.append(pd.Series(fold_metrics[strategy]['per_loc_aucs']))
            if loc_aucs_df_rows:
                loc_aucs_df = pd.DataFrame(loc_aucs_df_rows)
                summary_per_loc = loc_aucs_df.mean(skipna=True).to_dict()
                strat_summary['per_loc_mean_auc'] = {k: (float(v) if not pd.isna(v) else float('nan')) 
                                                     for k, v in summary_per_loc.items()}
            
            cv_summary['strategies'][strategy] = strat_summary
        
        # Save CV summary JSON
        with open(out_dir / 'cv_summary.json', 'w') as fjs:
            json.dump(cv_summary, fjs, indent=2)
        
        # Save strategy comparison across folds
        comparison_rows = []
        for strategy in strategies:
            row = {'strategy': strategy}
            row.update(cv_summary['strategies'][strategy])
            comparison_rows.append(row)
        pd.DataFrame(comparison_rows).to_csv(out_dir / 'cv_strategy_comparison.csv', index=False)
        
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION SUMMARY (n={len(all_fold_metrics)} folds)")
        print(f"{'='*80}")
        print(f"{'Strategy':<15} {'Mean Cls AUC':>15} {'Mean Comp Score':>18}")
        print(f"{'-'*80}")
        for strategy in strategies:
            mean_cls = cv_summary['strategies'][strategy].get('mean_cls_auc', float('nan'))
            mean_comp = cv_summary['strategies'][strategy].get('mean_competition_score', float('nan'))
            cls_str = f"{mean_cls:.4f}" if not math.isnan(mean_cls) else "N/A"
            comp_str = f"{mean_comp:.4f}" if not math.isnan(mean_comp) else "N/A"
            print(f"{strategy:<15} {cls_str:>15} {comp_str:>18}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

# python3 /home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_validation.py --weights yolo_aneurysm_locations/cv_mobilenet_more_negatives_fold22/weights/best.pt --val-fold 2 --out-dir yolo_aneurysm_locations/cv_mobilenet_more_negatives_fold22/series_validation --batch-size 16 --slice-step 1 