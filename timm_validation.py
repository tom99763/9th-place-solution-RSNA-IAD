
import argparse
from pathlib import Path
import sys
import math
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import pydicom
import torch
import timm_3d
from scipy import ndimage
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# Removed MONAI dependencies - using direct full volume inference
import cv2
# Project root & config imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))  # to allow importing configs.data_config
from configs.data_config import data_path  # type: ignore
from typing import Tuple
#try:
#    from ultralytics import YOLO  # type: ignore
#except ImportError as e:  # pragma: no cover
#    raise SystemExit("ultralytics not installed. Install with `pip install ultralytics`.") from e
#

# Binary classification for aneurysm presence
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    ap = argparse.ArgumentParser(description="Series-level validation for timm3d models (full volume inference)")
    ap.add_argument('--weights', type=str, required=True, help='Path to timm3d model checkpoint (.ckpt)')
    ap.add_argument('--val-fold', type=int, default=0, help='Fold id to evaluate (matches train.csv fold_id)')
    ap.add_argument('--series-limit', type=int, default=0, help='Optional limit on number of validation series (debug)')
    ap.add_argument('--save-csv', type=str, default='', help='Optional path to save per-series predictions CSV')
    ap.add_argument('--verbose', default=True, action='store_true')
    # Volume processing settings (simplified for full volume inference)
    ap.add_argument('--target-shape', type=int, nargs=3, default=[128, 384, 384], help='Target volume shape (D H W)')
    ap.add_argument('--model-name', type=str, default='tf_efficientnetv2_s.in21k_ft_in1k', help='timm3d model name')
    ap.add_argument('--normalization', type=str, default='minmax', choices=['minmax'], help='Normalization method (standardized to minmax)')
    # Plot arguments
    ap.add_argument('--save-plot', type=str, default='', help='Path to save AUC evolution plot (e.g., auc_evolution.png)')
    ap.add_argument('--show-plot', action='store_true', help='Display AUC evolution plot during validation')
    ap.add_argument('--plot-interval', type=int, default=10, help='Interval for AUC tracking (every N series)')
    # Memory optimization
    ap.add_argument('--use-int8', action='store_true', help='Use int8 quantization for memory savings (matches training data format)')
    return ap.parse_args()


def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    """Return list of HU frames from a DICOM. Handles 2D, multi-frame, and RGB->grayscale."""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    frames: List[np.ndarray] = []

    # Normalize handling of common pixel array shapes:
    # - 2D: (H, W)
    # - 3D: could be (frames, H, W) or (H, W, channels)
    # - 4D: (frames, H, W, channels)
    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:
        # Distinguish color single-frame (H,W,3/4) vs multi-frame (F,H,W)
        if pix.shape[-1] in (3, 4) and pix.shape[0] not in (3, 4):
            # color image with channels in last dim
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            # assume first dimension is frame index
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    elif pix.ndim == 4:
        # (frames, H, W, channels)
        for i in range(pix.shape[0]):
            frame = pix[i]
            if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                try:
                    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                except Exception:
                    gray = frame[..., 0].astype(np.float32)
                frames.append(gray * slope + intercept)
            else:
                frames.append(frame.astype(np.float32) * slope + intercept)
    else:
        # Unexpected dimensionality: try to coerce to 2D
        try:
            arr = np.asarray(pix)
            if arr.ndim >= 2:
                img = arr.reshape(arr.shape[-2], arr.shape[-1]).astype(np.float32)
                frames.append(img * slope + intercept)
        except Exception:
            pass

    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to [0,1] range - matches training pipeline exactly.
    """
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    norm = (img - mn) / (mx - mn)
    return norm.astype(np.float32)

def simple_normalize_255(img: np.ndarray) -> np.ndarray:
    """
    Legacy normalization - no longer used. Data is now stored as float16 [0,1].
    """
    return (img.astype(np.float32) / 255.0).clip(0, 1)

def load_timm3d_model(weights_path: str, model_name: str) -> torch.nn.Module:
    """Load timm3d model from checkpoint."""
    model = timm_3d.create_model(
        model_name,
        pretrained=False,
        num_classes=1,  # Binary classification
        in_chans=1,
        global_pool='avg',
        drop_path_rate=0.0,
        drop_rate=0.0,
    )
    
    # Load checkpoint
    ckpt_path = Path(weights_path)
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # Strip 'model.' prefix from Lightning checkpoint
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    cleaned[k.replace('model.', '', 1)] = v
                elif not any(k.startswith(p) for p in ['loss_fn', 'pos_weight', 'hparams']):
                    cleaned[k] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Failed to load checkpoint {ckpt_path}: {e}. Proceeding with random weights.")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using randomly initialized weights.")
    
    model.eval().to(DEVICE)
    return model


def collect_series_slices(series_dir: Path) -> List[Path]:
    return sorted(series_dir.glob('*.dcm'))

def resample_to_patch_size_gpu(vol: np.ndarray, target_shape: Tuple[int, int, int] = (128, 384, 384)) -> np.ndarray:
    """Resample volume to target shape using GPU acceleration (Z, Y, X order)."""
    current_shape = vol.shape
    zoom = (target_shape[0] / current_shape[0], 
            target_shape[1] / current_shape[1], 
            target_shape[2] / current_shape[2])
    
    # Move to GPU, resample, then back to CPU
    vol_gpu = cp.asarray(vol)
    vol_resampled_gpu = cpx_ndimage.zoom(vol_gpu, zoom, order=1, prefilter=False)
    vol_resampled = cp.asnumpy(vol_resampled_gpu)
    
    # Free GPU memory
    del vol_gpu, vol_resampled_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return vol_resampled


def load_volume_from_series(series_dir: Path, target_shape: Tuple[int, int, int] = (128, 384, 384), normalization: str = 'minmax', use_int8: bool = False):
    """Load and process volume from DICOM series with isotropic resampling."""
    # Get ordered DICOM files
    files = list(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOMs in {series_dir}")
    
    # Sort by slice location or instance number
    tmp = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True)
            if hasattr(ds, "SliceLocation"):
                key = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                key = float(ds.ImagePositionPatient[-1])
            else:
                key = float(getattr(ds, "InstanceNumber", 0))
        except Exception:
            key = 0.0
        tmp.append((key, fp))
    
    tmp.sort(key=lambda x: x[0])
    paths = [t[1] for t in tmp]
    
    # Load volume without normalization
    slices = []
    spacing = None
    for p in paths:
        frames = read_dicom_frames_hu(p)
        if len(frames) > 1:
            try:
                ds_local = pydicom.dcmread(str(p), stop_before_pixels=True)
                pixel_spacing = getattr(ds_local, "PixelSpacing", [1.0, 1.0])
                slice_thickness = float(getattr(ds_local, "SliceThickness", 1.0))
                spacing_z = float(getattr(ds_local, "SpacingBetweenSlices", slice_thickness))
                if spacing is None:
                    spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)
            except Exception:
                pass
            for f in frames:
                slices.append(f)
        elif len(frames) == 1:
            slices.append(frames[0])
            if spacing is None:
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
                    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
                    spacing_z = float(getattr(ds, "SpacingBetweenSlices", slice_thickness))
                    spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)
                except Exception:
                    spacing = (1.0, 1.0, 1.0)
        else:
            continue
    
    if not slices:
        raise RuntimeError(f"No readable frames in {series_dir}")
    
    vol = np.stack(slices, axis=0).astype(np.float32)
    # Apply min-max normalization to [0,1] to match training pipeline exactly
    vol = min_max_normalize(vol)
    print(f"Loaded volume shape: {vol.shape}")
    
    # Resample and convert to target dtype if needed
    output_dtype = np.uint8 if use_int8 else np.float32
    vol = resample_to_patch_size_gpu(vol, target_shape=target_shape)
    
    # Convert to uint8 if quantization is enabled
    if use_int8:
        vol = (vol * 255).astype(np.uint8)
    
    print(f"Resampled volume shape: {vol.shape}, dtype: {vol.dtype}")
    
    return vol


def full_volume_inference(model: torch.nn.Module, volume: np.ndarray) -> float:
    """
    Direct inference on full volume without sliding window.
    Assumes volume is already resized to (128, 384, 384).
    """
    # Ensure volume has the expected shape
    if volume.shape != (128, 384, 384):
        print(f"Warning: Volume shape {volume.shape} != expected (128, 384, 384)")
    
    # Handle different input formats: uint8 [0,255] or float32 [0,1]
    if volume.dtype == np.uint8:
        # Convert uint8 [0,255] to float32 [0,1] for model inference
        volume_normalized = volume.astype(np.float32) / 255.0
    else:
        # Already normalized float32
        volume_normalized = volume.astype(np.float32)
    
    # Convert to tensor: (C, H, W, D) format for timm_3d
    # volume shape: (128, 384, 384) = (D, H, W)
    volume_transposed = volume_normalized.transpose(1, 2, 0)  # (H, W, D)
    volume_tensor = torch.from_numpy(volume_transposed).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W, D)
    
    with torch.no_grad():
        logits = model(volume_tensor)  # Shape: (1, 1)
        logit = logits.item()
        
    # Convert logit to probability
    prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
    print(f"Volume prob: {prob:.4f}")
    return float(prob)


def create_auc_evolution_plot(auc_tracking: Dict, save_path: str = '', show_plot: bool = False, plane: str = ''):
    """Create and optionally save/show AUC evolution plot."""
    plt.figure(figsize=(12, 8))
    
    series_counts = auc_tracking['series_counts']
    cls_aucs = auc_tracking['cls_aucs']
    
    # Plot AUC curve
    plt.plot(series_counts, cls_aucs, 'b-o', label='Classification AUC', linewidth=2, markersize=6)
    
    # Add horizontal reference lines
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random baseline (0.5)')
    
    # Formatting
    plt.xlabel('Number of Series Processed', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('AUC Evolution During Validation - timm3d Model', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits to focus on relevant range
    all_aucs = [auc for auc in cls_aucs if not math.isnan(auc)]
    if all_aucs:
        min_auc = min(all_aucs)
        max_auc = max(all_aucs)
        margin = (max_auc - min_auc) * 0.1
        plt.ylim(max(0, min_auc - margin), min(1, max_auc + margin))
    else:
        plt.ylim(0, 1)
    
    # Add final values as text annotations
    if cls_aucs and not math.isnan(cls_aucs[-1]):
        plt.annotate(f'Final AUC: {cls_aucs[-1]:.3f}', 
                    xy=(series_counts[-1], cls_aucs[-1]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"AUC evolution plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()



def main():
    args = parse_args()
    
    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_df = pd.read_csv(data_root / 'train.csv')
    if 'Aneurysm Present' not in train_df.columns:
        raise SystemExit("train_df.csv requires 'Aneurysm Present' column for classification label")

    val_series = train_df[train_df['fold_id'] == args.val_fold]['SeriesInstanceUID'].unique().tolist()
    if args.series_limit:
        val_series = val_series[:args.series_limit]

    print(f"Validation fold {args.val_fold}: {len(val_series)} series")
    print(f"Model weights: {args.weights}")
    print(f"Model name: {args.model_name}")
    print(f"Target shape: {args.target_shape}")
    print(f"Normalization: {args.normalization}")


    # Load timm3d model
    model = load_timm3d_model(args.weights, args.model_name)

    series_probs: Dict[str, float] = {}
    scores_list: List[float] = []
    cls_labels: List[int] = []
    
    # For AUC evolution tracking
    auc_tracking = {
        'series_counts': [],
        'cls_aucs': [],
    }

    # For confidence comparison
    pos_confidences: list[float] = []
    neg_confidences: list[float] = []
    pos_modalities: list[str] = []
    neg_modalities: list[str] = []
    # Per-series metadata for optional CSV
    series_true_labels: Dict[str, int] = {}
    series_modalities: Dict[str, str] = {}

    for sid in tqdm(val_series, desc="Validating timm3d", unit="series"):
        series_dir = series_root / sid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] {sid} (no directory)")
            continue
        
        try:
            # Load volume and resize to target shape
            vol = load_volume_from_series(series_dir, tuple(args.target_shape), args.normalization, args.use_int8)
            
            # Check if volume is valid
            if vol is None or vol.size == 0:
                if args.verbose:
                    print(f"[EMPTY] {sid}")
                continue
            
            # Run full volume inference
            prob = full_volume_inference(model, vol)
            
        except Exception as e:
            if args.verbose:
                print(f"[ERROR] {sid}: {e}")
            continue

        series_probs[sid] = prob
        scores_list.append(prob)
        print(f"Series {sid} prob={prob:.4f}")  # Always print this to debug

        # Labels and metadata for this series
        row = train_df[train_df['SeriesInstanceUID'] == sid].iloc[0]
        label = int(row['Aneurysm Present'])
        cls_labels.append(label)
        series_true_labels[sid] = label

        # Periodic partial AUC tracking
        if len(cls_labels) % args.plot_interval == 0 and len(cls_labels) >= args.plot_interval:
            try:
                # Classification partial AUC
                current_y_true = np.array(cls_labels)
                current_y_scores = np.array(scores_list)
                cls_pauc = roc_auc_score(current_y_true, current_y_scores)
            except ValueError:
                cls_pauc = float('nan')
            
            # Track AUC evolution
            auc_tracking['series_counts'].append(len(cls_labels))
            auc_tracking['cls_aucs'].append(cls_pauc)
            
            print(f"Partial AUC after {len(cls_labels)} series -> Classification: {cls_pauc:.4f}")

        # Collect confidences and modality for comparison
        modality = 'Unknown'
        try:
            # Get modality from first DICOM in series
            first_dcm = next(series_dir.glob('*.dcm'), None)
            if first_dcm:
                ds = pydicom.dcmread(str(first_dcm), stop_before_pixels=True)
                modality = getattr(ds, 'Modality', 'Unknown')
        except Exception:
            modality = 'Error'
        series_modalities[sid] = modality
        if label == 1:
            pos_confidences.append(prob)
            pos_modalities.append(modality)
        else:
            neg_confidences.append(prob)
            neg_modalities.append(modality)

    print(f"Debug: Processed {len(series_probs)} series, collected {len(cls_labels)} labels")
    if not series_probs:
        print("No series processed.")
        return

    # Confidence comparison between positive and negative tomograms
    if pos_confidences and neg_confidences:
        pos_arr = np.asarray(pos_confidences, dtype=float)
        neg_arr = np.asarray(neg_confidences, dtype=float)
        def stats(a: np.ndarray) -> dict:
            return {
                'n': int(a.size),
                'mean': float(np.mean(a)),
                'median': float(np.median(a)),
                'std': float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
                'min': float(np.min(a)),
                'q25': float(np.percentile(a, 25)),
                'q75': float(np.percentile(a, 75)),
                'max': float(np.max(a)),
            }
        pos_s = stats(pos_arr)
        neg_s = stats(neg_arr)
        # Cohen's d (pooled std)
        if pos_s['n'] > 1 and neg_s['n'] > 1:
            s_pooled_num = (pos_s['n'] - 1) * (pos_s['std'] ** 2) + (neg_s['n'] - 1) * (neg_s['std'] ** 2)
            s_pooled_den = pos_s['n'] + neg_s['n'] - 2
            s_pooled = math.sqrt(s_pooled_num / s_pooled_den) if s_pooled_den > 0 and s_pooled_num >= 0 else 0.0
            cohend = (pos_s['mean'] - neg_s['mean']) / s_pooled if s_pooled > 0 else float('nan')
        else:
            cohend = float('nan')

        # Determine optimal threshold (Youden's J)
        y_true = np.array(cls_labels)
        y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
        fpr, tpr, thr = roc_curve(y_true, y_scores)
        youden = tpr - fpr
        best_idx = int(np.argmax(youden)) if youden.size else 0
        best_thr = float(thr[best_idx]) if thr.size else 0.5

        # Confusion matrix at best threshold
        y_pred = (y_scores >= best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        print("\n--- Confidence Comparison ---")
        print(f"Positive tomograms (n={pos_s['n']}): mean={pos_s['mean']:.4f}, median={pos_s['median']:.4f}, std={pos_s['std']:.4f}, min={pos_s['min']:.4f}, q25={pos_s['q25']:.4f}, q75={pos_s['q75']:.4f}, max={pos_s['max']:.4f}")
        print(f"Negative tomograms (n={neg_s['n']}): mean={neg_s['mean']:.4f}, median={neg_s['median']:.4f}, std={neg_s['std']:.4f}, min={neg_s['min']:.4f}, q25={neg_s['q25']:.4f}, q75={neg_s['q75']:.4f}, max={neg_s['max']:.4f}")
        print(f"Cohen's d: {cohend:.3f}")

        # Modality breakdown
        if pos_modalities or neg_modalities:
            from collections import Counter
            pos_mod_ct = Counter(pos_modalities)
            neg_mod_ct = Counter(neg_modalities)
            print("\nModalities (top) for positives:")
            for m, c in pos_mod_ct.most_common(5):
                print(f"  {m}: {c}")
            print("Modalities (top) for negatives:")
            for m, c in neg_mod_ct.most_common(5):
                print(f"  {m}: {c}")

        # Threshold summary
        print(f"\nBest threshold by Youden's J: {best_thr:.4f}  (TPR={tpr[best_idx]:.3f}, fpr={fpr[best_idx]:.3f})")
        print(f"Confusion matrix at best threshold: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Precision={prec:.3f}, Recall/Sensitivity={sens:.3f}, Specificity={spec:.3f}")

    # Metrics
    y_true = np.array(cls_labels)
    y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
    cls_auc = roc_auc_score(y_true, y_scores)
    
    print(f"Classification AUC (aneurysm present): {cls_auc:.4f}")
    
    # Print final partial AUC if we have enough samples
    if len(cls_labels) >= args.plot_interval:
        try:
            cls_partial_auc = roc_auc_score(y_true, y_scores)
            print(f"Final partial AUC after {len(cls_labels)} series -> Classification: {cls_partial_auc:.4f}")
        except ValueError as e:
            print(f"Could not compute classification partial AUC: {e}")
    else:
        print(f"Not enough samples for partial AUC calculation (need at least {args.plot_interval}, got {len(cls_labels)})")
    
    # Add final AUC values to tracking
    if len(cls_labels) > 0:
        final_count = len(cls_labels)
        # Only add if it's different from the last tracked point
        if not auc_tracking['series_counts'] or auc_tracking['series_counts'][-1] != final_count:
            auc_tracking['series_counts'].append(final_count)
            auc_tracking['cls_aucs'].append(cls_auc)
    
    # Create AUC evolution plot
    if (args.save_plot or args.show_plot) and len(auc_tracking['series_counts']) > 1:
        create_auc_evolution_plot(auc_tracking, args.save_plot, args.show_plot, "")

    if args.save_csv:
        out_rows = []
        for sid, prob in series_probs.items():
            out_rows.append({
                'SeriesInstanceUID': sid,
                'aneurysm_prob': prob,
                'true_label': series_true_labels.get(sid, None),
                'modality': series_modalities.get(sid, ''),
            })
        pd.DataFrame(out_rows).to_csv(args.save_csv, index=False)
        print(f"Saved per-series predictions to {args.save_csv}")

import time
if __name__ == '__main__':
    main()

# yolo11n.pt 48 bbox fold1
#Classification AUC (aneurysm present): 0.6960
#Location macro AUC: 0.7659
#Combined (mean) metric: 0.7309

#python3 timm_validation.py \
#    --weights rsna-iad-binary/ez8q5xzs/checkpoints/best-fold0-epoch=4-val_weighted_auc=0.723.ckpt \
#    --val-fold 0 \
#    --model-name tf_efficientnetv2_s.in21k_ft_in1k \
#    --target-shape 128 384 384 \
#    --normalization minmax \
#    --save-csv predictions.csv

# Original (potentially inconsistent with training):
#python3 timm_validation.py --weights model.ckpt --val-fold 0 --normalization minmax





