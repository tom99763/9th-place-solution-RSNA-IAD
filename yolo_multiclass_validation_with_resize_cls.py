"""Validate a binary YOLO aneurysm classifier at series level.

Preprocessing:
    - DICOM -> HU -> per-slice min-max -> uint8 (rgb for YOLO inference)
    - rgb mode: stacks 3 consecutive slices [i-1, i, i+1] as 3-channel images
    - Volume resize mode: loads entire series as 3D volume and resizes to target shape (default: original_z x 512x512)
    - Uses OpenCV for per-slice resizing when z-axis unchanged (faster), CuPy for 3D resizing when z-axis changes

Series scores:
    - Aneurysm Present = max detection confidence across processed slices/MIP windows

Outputs:
    - Classification AUC (aneurysm present)
    - Optional CSV with per-series predictions

Fold Management:
    - If train.csv has no 'fold_id' column, automatically generates stratified folds using StratifiedKFold
    - Uses N_FOLDS=5 and SEED=42 from configs/data_config.py for consistent fold generation
    - Updates train.csv with the generated fold_id column for future runs

Notes:
    - Requires ultralytics (YOLOv8/11) and binary classification weights.
    - Use --mip-window > 0 for sliding-window MIPs; 0 for per-slice; --rgb-mode for 3-slice stacking; --use-volume-resize for volume resizing.
    - By default preserves original z-axis dimension while resizing x,y to 512x512 using optimized OpenCV per-slice approach.
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
from scipy import ndimage

# Try to import CuPy for GPU acceleration, fallback to CPU if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available - will use for 3D resizing when z-axis changes")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - using CPU/OpenCV processing")
sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

# Project root & config imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from configs.data_config import data_path, N_FOLDS, SEED  # type: ignore
from src.prepare_yolo_dataset_v3_cls import load_and_process_volume, ordered_dcm_paths  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(description="Series-level validation for YOLO (binary classification)")
    ap.add_argument('--weights', type=str, required=False, default='', help='Path to YOLO weights (.pt) for binary classification')
    ap.add_argument('--val-fold', type=int, default=1, help='Fold id to evaluate (matches train.csv fold_id)')
    ap.add_argument('--series-limit', type=int, default=0, help='Optional limit on number of validation series (debug)')
    ap.add_argument('--max-slices', type=int, default=0, help='Optional cap on number of slices/windows per series (debug)')
    ap.add_argument('--save-csv', type=str, default='', help='Optional path to save per-series predictions CSV (deprecated, prefer --out-dir)')
    ap.add_argument('--out-dir', type=str, default='', help='If set, writes metrics JSON/CSV into this directory')
    ap.add_argument('--batch-size', type=int, default=16, help='Batch size for inference (higher = faster, more VRAM)')
    ap.add_argument('--verbose', default=True, action='store_true')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice (default=1)')
    # Sliding-window MIP mode
    ap.add_argument('--mip-window', type=int, default=0, help='Half-window (in slices) for MIP; 0 = per-slice mode')
    ap.add_argument('--mip-img-size', type=int, default=0, help='Optional resize of MIP/slice to this square size before inference (0 keeps original)')
    ap.add_argument('--mip-no-overlap', action='store_true', help='Use non-overlapping MIP windows (stride = 2*w+1 instead of slice_step)')
    # CV
    ap.add_argument('--cv', action='store_true', help='Run validation across all folds found in train_df.csv/train.csv')
    ap.add_argument('--folds', type=str, default='', help='Comma-separated fold ids to run (overrides --val-fold when provided)')
    # Weights & Biases
    ap.add_argument('--wandb', default=True, action='store_true', help='Log metrics and outputs to Weights & Biases')
    ap.add_argument('--wandb-project', type=str, default='yolo_aneurysm_classification', help='W&B project name (no slashes)')
    ap.add_argument('--wandb-entity', type=str, default='', help='W&B entity (team/user)')
    ap.add_argument('--wandb-run-name', type=str, default='', help='W&B run name (defaults to val_fold{fold})')
    ap.add_argument('--wandb-group', type=str, default='', help='Optional W&B group')
    ap.add_argument('--wandb-tags', type=str, default='', help='Comma-separated W&B tags')
    ap.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline'], help='W&B mode (online/offline)')
    ap.add_argument('--rgb-mode', default=True, action='store_true', help='Use rgb mode (3-channel images from stacked slices)')
    ap.add_argument('--rgb-non-overlapping', action='store_true', help='Use non-overlapping 3-slice windows in rgb mode (e.g., [1,2,3],[4,5,6],[7,8,9])')
    ap.add_argument('--wandb-resume-id', type=str, default='', help='Resume W&B run id (optional)')
    # Volume resizing arguments
    ap.add_argument('--use-volume-resize', default=True, action='store_true', help='Enable volume resizing using CuPy/scipy.ndimage.zoom')
    ap.add_argument('--volume-target-depth', type=int, default=32, help='Target depth for volume resizing (None = keep original z-axis)')
    ap.add_argument('--volume-target-height', type=int, default=512, help='Target height for volume resizing')
    ap.add_argument('--volume-target-width', type=int, default=512, help='Target width for volume resizing')
    return ap.parse_args()


def load_folds(root: Path) -> Dict[str, int]:
    """Map SeriesInstanceUID -> fold_id using stratified folds from train.csv.

    Requirements:
      - data/train.csv must exist
      - Columns: 'SeriesInstanceUID', 'Aneurysm Present'
      - Will create N_FOLDS stratified splits on the series-level label
    """
    df_path = root / "train.csv"

    df = pd.read_csv(df_path)

    series_df = df[["SeriesInstanceUID", "Aneurysm Present"]].drop_duplicates().reset_index(drop=True)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_map: Dict[str, int] = {}
    for i, (_, test_idx) in enumerate(
        skf.split(series_df["SeriesInstanceUID"], series_df["Aneurysm Present"]) 
    ):
        for uid in series_df.loc[test_idx, "SeriesInstanceUID"].tolist():
            fold_map[uid] = i
    
    # Update train.csv with the new folds
    df["fold_id"] = df["SeriesInstanceUID"].astype(str).map(fold_map)
    df.to_csv(df_path, index=False)
    print(f"Updated {df_path} with new fold_id column based on N_FOLDS={N_FOLDS}")
    
    return fold_map


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
        # If rgb (H,W,3), take first channel; else assume multi-frame (N,H,W)
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
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


def min_max_normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize entire volume consistently to uint8."""
    mn, mx = float(volume.min()), float(volume.max())
    if mx - mn < 1e-6:
        return np.zeros_like(volume, dtype=np.uint8)
    norm = (volume - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def collect_series_slices(series_dir: Path) -> List[Path]:
    """Collect DICOM files ordered by spatial position using same method as training."""
    paths, _ = ordered_dcm_paths(series_dir)
    return paths


def calculate_partial_auc(cls_labels: List[int], series_probs: Dict[str, float]) -> float:
    """Calculate partial AUC score for classification.
    
    Returns:
        cls_auc
    """
    if len(cls_labels) < 2:
        return float('nan')
    
    try:
        # Classification AUC
        y_true = np.array(cls_labels)
        y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
        cls_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        cls_auc = float('nan')
    
    return cls_auc


def resize_volume_to_target(volume: np.ndarray, target_shape: Tuple[int | None, int, int] = (None, 512, 512)) -> np.ndarray:
    """Resize a 3D volume to target shape using CuPy (if available) or scipy.ndimage.zoom.
    
    Args:
        volume: Input 3D volume of shape (D, H, W)
        target_shape: Target shape (target_D, target_H, target_W). If target_D is None, keep original D dimension
    
    Returns:
        Resized volume with target shape
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D array")
    
    current_shape = volume.shape
    if target_shape[0] is None:
        # Keep original z dimension
        final_target_shape = (current_shape[0], target_shape[1], target_shape[2])
    else:
        final_target_shape = target_shape
    
    zoom_factors = tuple(final_target_shape[i] / current_shape[i] for i in range(3))
    
    # Choose optimal resizing method based on whether z-axis needs resizing
    z_needs_resize = abs(zoom_factors[0] - 1.0) > 1e-6
    
    if z_needs_resize:
        # Use CuPy for 3D resizing when z-axis changes (faster for 3D operations)
        if CUPY_AVAILABLE:
            try:
                # Import CuPy's scipy-compatible ndimage module
                from cupyx.scipy import ndimage as cp_ndimage
                # Move volume to GPU
                volume_gpu = cp.asarray(volume)
                # Use CuPy's zoom function
                resized_volume_gpu = cp_ndimage.zoom(volume_gpu, zoom_factors, order=1)
                # Move back to CPU
                resized_volume = cp.asnumpy(resized_volume_gpu)
                # Clean up GPU memory
                del volume_gpu, resized_volume_gpu
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"[WARNING] CuPy processing failed: {e}, falling back to CPU")
                # Fallback to CPU processing
                resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        else:
            # Use CPU processing for 3D
            resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
    else:
        # Use OpenCV for per-slice 2D resizing when z-axis unchanged (faster per experiments)
        target_h, target_w = final_target_shape[1], final_target_shape[2]
        
        resized_slices = []
        for slice_idx in range(volume.shape[0]):
            slice_2d = volume[slice_idx]  # Shape: (H, W)
            resized_slice = cv2.resize(slice_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(resized_slice)
        
        resized_volume = np.stack(resized_slices, axis=0)
        del resized_slices
    
    return resized_volume


def load_series_as_volume(series_dir: Path) -> np.ndarray | None:
    """Load entire series as a 3D volume (D, H, W).
    
    Args:
        series_dir: Directory containing DICOM files
    
    Returns:
        3D numpy array of shape (D, H, W) or None if loading fails
    """
    paths = collect_series_slices(series_dir)
    if not paths:
        return None
    
    slices = []
    for dcm_path in paths:
        try:
            frames = read_dicom_frames_hu(dcm_path)
            for f in frames:
                slices.append(f)
        except Exception:
            continue
    
    if not slices:
        return None
    
    # Stack slices into volume (D, H, W)
    try:
        volume = np.stack(slices, axis=0)
        return volume
    except Exception:
        return None


def _run_validation_for_fold(args: argparse.Namespace, weights_path: str, fold_id: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Runs validation for a single fold and returns (metrics_dict, per_series_df)."""
    # Load split
    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_df = pd.read_csv(data_root / 'train.csv')
    if 'Aneurysm Present' not in train_df.columns:
        raise SystemExit("train.csv requires 'Aneurysm Present' column for classification label")
    
    # Check if fold_id column exists, if not generate folds using StratifiedKFold
    if 'fold_id' not in train_df.columns:
        print("fold_id column not found in train.csv, generating stratified folds...")
        fold_map = load_folds(data_root)
        # Reload the dataframe with the new fold_id column
        train_df = pd.read_csv(data_root / 'train.csv')

    val_series = train_df[train_df['fold_id'] == fold_id]['SeriesInstanceUID'].unique().tolist()
    if args.series_limit:
        val_series = val_series[:args.series_limit]

    print(f"Validation fold {fold_id}: {len(val_series)} series")
    if args.use_volume_resize:
        depth_str = f"{args.volume_target_depth}" if args.volume_target_depth is not None else "original"
        print(f"Processing with volume resizing to {depth_str}x{args.volume_target_height}x{args.volume_target_width}; every {args.slice_step} slice(s)")
    else:
        print(f"Processing every {args.slice_step} slice(s); MIP half-window={args.mip_window}")
    model = YOLO(weights_path)

    series_probs: Dict[str, float] = {}
    cls_labels: List[int] = []
    series_pred_counts: Dict[str, int] = {}
    
    # For partial AUC calculation every 10 tomograms
    processed_count = 0
    partial_series_probs: Dict[str, float] = {}
    partial_cls_labels: List[int] = []

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
        total_dets = 0
        batch: list[np.ndarray] = []

        def flush_batch(batch_imgs: list[np.ndarray]):
            nonlocal max_conf_all, total_dets
            if not batch_imgs:
                return
            results = model.predict(batch_imgs, verbose=False)
            for r in results:
                if not r or r.probs is None:
                    continue
                try:
                    # For classification, r.probs.data contains class probabilities
                    probs = r.probs.data
                    total_dets += 1  # Each image gets one classification
                    
                    # For binary classification (aneurysm present/absent)
                    if len(probs) == 2:  # [no_aneurysm, aneurysm_present]
                        aneurysm_prob = float(probs[0].item())  # Index 0 = no_aneurysm
                        if aneurysm_prob > max_conf_all:
                            max_conf_all = aneurysm_prob
                    else:
                        # Handle unexpected number of classes
                        max_prob = float(max(probs).item())
                        if max_prob > max_conf_all:
                            max_conf_all = max_prob
                except Exception as e:
                    if args.verbose:
                        print(f"[ERROR] Processing classification result: {e}")
                    pass

        if args.use_volume_resize:
            # Volume resizing mode: use SAME pipeline as training data preparation
            dicoms = collect_series_slices(series_dir)
            if not dicoms:
                series_probs[sid] = 0.0
                series_pred_counts[sid] = 0
                continue
            
            # Use the EXACT same processing pipeline as training - preserve z-axis
            target_shape = (args.volume_target_depth, args.volume_target_height, args.volume_target_width)
            try:
                volume, zoom_factors, origin_to_resized = load_and_process_volume(
                    dicoms, target_shape=target_shape, verbose=args.verbose
                )
            except Exception as e:
                if args.verbose:
                    print(f"[SKIP] {sid}: Volume processing failed: {e}")
                series_probs[sid] = 0.0
                series_pred_counts[sid] = 0
                continue
            
            # volume is already normalized to uint8 by load_and_process_volume
            resized_volume_normalized = volume
            
            # Process slices from resized volume
            for slice_idx in range(0, resized_volume_normalized.shape[0], args.slice_step):
                if args.max_slices and len(batch) >= args.max_slices:
                    break
                    
                slice_img = resized_volume_normalized[slice_idx]
                
                # Handle rgb mode (3-channel) to match preparation script
                if getattr(args, 'rgb_mode', False):
                    # Create 3-channel image: [prev, current, next] as [R, G, B] (matching preparation)
                    prev_idx = max(0, slice_idx - 1)
                    next_idx = min(resized_volume_normalized.shape[0] - 1, slice_idx + 1)
                    r = resized_volume_normalized[prev_idx]
                    g = resized_volume_normalized[slice_idx] 
                    b = resized_volume_normalized[next_idx]
                    slice_img = np.stack([r, g, b], axis=-1)
                else:
                    # Convert to rgb format for single-slice mode
                    if slice_img.ndim == 2:
                        # Use BGR for Ultralytics inference
                        slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
                
                batch.append(slice_img)
                if len(batch) >= args.batch_size:
                    flush_batch(batch)
                    batch.clear()
            flush_batch(batch)
            
            if args.verbose:
                actual_target = (volume.shape[0], args.volume_target_height, args.volume_target_width)
                print(f"  Volume {sid}: Resized to {actual_target}, processed {resized_volume_normalized.shape[0]} slices")
        elif args.mip_window > 0:
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
                series_pred_counts[sid] = 0
                continue
            target_shape = max(shapes_count.items(), key=lambda kv: kv[1])[0]
            th, tw = target_shape
            n = len(slices_hu)
            
            # Stack into volume for consistent normalization
            volume_hu = np.stack([cv2.resize(s, (tw, th), interpolation=cv2.INTER_LINEAR) if s.shape != target_shape else s for s in slices_hu], axis=0)
            volume_normalized = min_max_normalize_volume(volume_hu)
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

            w = args.mip_window
            for c in centers:
                lo = max(0, c - w)
                hi = min(n - 1, c + w)
                mip_u8 = None
                for i in range(lo, hi + 1):
                    slice_u8 = volume_normalized[i]
                    mip_u8 = slice_u8 if mip_u8 is None else np.maximum(mip_u8, slice_u8)
                if args.mip_img_size > 0 and (mip_u8.shape[0] != args.mip_img_size or mip_u8.shape[1] != args.mip_img_size):
                    mip_u8 = cv2.resize(mip_u8, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
                # Convert to BGR for Ultralytics inference
                mip_bgr = cv2.cvtColor(mip_u8, cv2.COLOR_GRAY2BGR) if mip_u8.ndim == 2 else mip_u8
                batch.append(mip_bgr)
                if len(batch) >= args.batch_size:
                    flush_batch(batch)
                    batch.clear()
            flush_batch(batch)
        elif getattr(args, 'rgb_mode', False):
            # rgb mode: stack 3 consecutive slices as channels
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

            if len(slices_hu) < 3:
                # Not enough slices for rgb mode, skip this series
                series_probs[sid] = 0.0
                series_pred_counts[sid] = 0
                continue
                
            # Stack into volume for consistent normalization (matching preparation script)
            target_shape = max(shapes_count.items(), key=lambda kv: kv[1])[0]
            th, tw = target_shape
            volume_hu = np.stack([cv2.resize(s, (tw, th), interpolation=cv2.INTER_LINEAR) if s.shape != target_shape else s for s in slices_hu], axis=0)
            volume_normalized = min_max_normalize_volume(volume_hu)


            # Original overlapping rgb mode: [i-1,i,i+1] for each i
            step = max(1, args.slice_step)
            slice_indices = list(range(0, len(slices_hu), step))
            if args.max_slices and len(slice_indices) > args.max_slices:
                slice_indices = slice_indices[:args.max_slices]

            for i in slice_indices:
                # Get 3 consecutive slices: [i-1, i, i+1], handling boundaries
                # Order as [prev, current, next] to match preparation script rgb order
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

                # Use volume-normalized slices in rgb order: [prev, current, next] (matching preparation)
                processed_slices = []
                for idx in slice_indices_3[:3]:
                    # Use pre-normalized slice from volume (matches preparation approach)
                    slice_u8 = volume_normalized[idx]
                    processed_slices.append(slice_u8)

                # Stack as 3-channel image (rgb order: [prev, current, next] to match preparation)
                RGB_img = np.stack(processed_slices, axis=-1)  # Shape: (H, W, 3)

                # Resize if needed
                if args.mip_img_size > 0 and (RGB_img.shape[0] != args.mip_img_size or RGB_img.shape[1] != args.mip_img_size):
                    RGB_img = cv2.resize(RGB_img, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)

                # Convert to BGR for Ultralytics inference
                batch.append(RGB_img)
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
                    if args.mip_img_size > 0 and (img_uint8.shape[0] != args.mip_img_size or img_uint8.shape[1] != args.mip_img_size):
                        img_uint8 = cv2.resize(img_uint8, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
                    if img_uint8.ndim == 2:
                        # Use BGR for Ultralytics inference
                        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    batch.append(img_uint8)
                    if len(batch) >= args.batch_size:
                        flush_batch(batch)
                        batch.clear()
            flush_batch(batch)

        series_probs[sid] = max_conf_all
        series_pred_counts[sid] = total_dets
        if args.verbose:
            print(f"Series {sid} max_conf_any={max_conf_all:.4f} dets={total_dets} (processed {len(dicoms)} slices)")

        # Labels and metadata for this series
        row = train_df[train_df['SeriesInstanceUID'] == sid].iloc[0]
        label = int(row['Aneurysm Present'])
        cls_labels.append(label)
        
        # Update partial tracking
        partial_series_probs[sid] = max_conf_all
        partial_cls_labels.append(label)
        processed_count += 1
        
        # Print partial AUC every 10 tomograms
        if processed_count % 10 == 0:
            partial_cls_auc = calculate_partial_auc(partial_cls_labels, partial_series_probs)
            print(f"\n--- Partial AUC after {processed_count} tomograms ---")
            print(f"Classification AUC: {partial_cls_auc:.4f}")
            print("---" + "-" * 40 + "\n")

    if not series_probs:
        print("No series processed.")
        # Return empty metrics and df
        return {
            'cls_auc': float('nan'),
            'cls_ap': float('nan'),
            'main_metric': float('nan'),
            'mean_num_classifications': 0.0,
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

    print(f"Classification AUC (aneurysm present): {cls_auc:.4f}")
    print(f"Classification AP (PR AUC): {cls_ap:.4f}")
    
    # For binary classification, use aneurysm present AUC as main metric
    main_metric = cls_auc

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
        out_rows.append(row)
    per_series_df = pd.DataFrame(out_rows)

    metrics = {
        'fold_id': fold_id,
        'cls_auc': float(cls_auc),
        'cls_ap': float(cls_ap) if not (isinstance(cls_ap, float) and math.isnan(cls_ap)) else float('nan'),
        'main_metric': float(main_metric),
        'mean_num_classifications': float(np.mean(list(series_pred_counts.values())) if series_pred_counts else 0.0),
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
        'mip_img_size': args.mip_img_size,
        'mip_no_overlap': args.mip_no_overlap,
        'rgb_mode': args.rgb_mode,
        'rgb_non_overlapping': args.rgb_non_overlapping,
        'batch_size': args.batch_size,
        'series_limit': args.series_limit,
        'max_slices': args.max_slices,
        'use_volume_resize': args.use_volume_resize,
        'volume_target_depth': args.volume_target_depth,
        'volume_target_height': args.volume_target_height,
        'volume_target_width': args.volume_target_width,
        'cupy_available': CUPY_AVAILABLE,
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
        
        # Check if fold_id column exists, if not generate folds using StratifiedKFold
        if 'fold_id' not in train_df.columns:
            print("fold_id column not found in train.csv, generating stratified folds...")
            fold_map = load_folds(data_root)
            # Reload the dataframe with the new fold_id column
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
            # Metrics JSON
            import json
            with open(fold_out_dir / 'metrics.json', 'w') as fjs:
                json.dump(metrics, fjs, indent=2)

        # W&B logging of metrics and tables
        if wandb is not None:
            try:
                # Scalars
                scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                wandb.log(scalars)
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
        agg_keys = ['cls_auc', 'cls_ap', 'main_metric', 'mean_num_classifications']
        summary = {'num_folds': len(all_fold_metrics)}
        for k in agg_keys:
            vals = [m[k] for m in all_fold_metrics if isinstance(m.get(k), (int, float)) and not math.isnan(m.get(k))]
            summary[f'mean_{k}'] = float(np.mean(vals)) if vals else float('nan')
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


# Binary classification validation examples:
# python yolo_multiclass_validation_with_resize_cls.py --weights path/to/weights.pt --val-fold 1 --use-volume-resize
# python yolo_multiclass_validation_with_resize_cls.py --weights path/to/weights.pt --val-fold 1 --use-volume-resize --volume-target-depth 64 --volume-target-height 256 --volume-target-width 256
# python yolo_multiclass_validation_with_resize_cls.py --weights path/to/weights.pt --val-fold 1 --rgb-mode
#
# Note: By default, --volume-target-depth=None preserves the original z-axis dimension.
# OpenCV is used for per-slice resizing (faster when z unchanged); CuPy for 3D resizing when z-axis changes.