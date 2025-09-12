
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
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from ultralytics import YOLO
from tqdm.auto import tqdm

# Project root & config imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))  # to allow importing configs.data_config
from configs.data_config import data_path  # type: ignore

#try:
#    from ultralytics import YOLO  # type: ignore
#except ImportError as e:  # pragma: no cover
#    raise SystemExit("ultralytics not installed. Install with `pip install ultralytics`.") from e
#

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
    ap = argparse.ArgumentParser(description="Series-level validation for YOLO plane models")
    ap.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt)')
    ap.add_argument('--val-fold', type=int, default=0, help='Fold id to evaluate (matches train.csv fold_id)')
    ap.add_argument('--series-limit', type=int, default=0, help='Optional limit on number of validation series (debug)')
    ap.add_argument('--max-slices', type=int, default=0, help='Optional cap on number of slices per series (debug)')
    ap.add_argument('--save-csv', type=str, default='', help='Optional path to save per-series predictions CSV')
    ap.add_argument('--batch-size', type=int, default=16, help='Batch size for slice inference (higher = faster, more VRAM)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice (default=1)')
    # Plane-specific validation settings
    ap.add_argument('--plane', type=str, choices=['axial', 'coronal', 'sagittal'], default='', help='Specific plane to validate (auto-detected from weights path if not provided)')
    ap.add_argument('--target-spacing', type=float, default=1.0, help='Target isotropic spacing for volume resampling')
    ap.add_argument('--img-size', type=int, default=512, help='Image size for plane slicing')
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
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def collect_series_slices(series_dir: Path) -> List[Path]:
    return sorted(series_dir.glob('*.dcm'))


def load_volume_from_series(series_dir: Path, target_spacing: float = 1.0, img_size: int = 512):
    """Load and process volume from DICOM series for plane-specific slicing.

    Note: We resample to isotropic spacing in 3D using SciPy (order=1), 
    maybe if resize to cube here is faster, but would have a domain shift from data used in training
    we have to check
    """
    from scipy import ndimage
    import pydicom
    
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
    
    # Load volume
    slices = []
    spacing = None
    for p in paths:
        # Read frames and also read dataset metadata locally to avoid using an undefined 'ds'
        frames = read_dicom_frames_hu(p)
        # Multi-frame DICOM: append all frames and capture spacing from this file if available
        if len(frames) > 1:
            try:
                ds_local = pydicom.dcmread(str(p), stop_before_pixels=True)
                pixel_spacing = getattr(ds_local, "PixelSpacing", [1.0, 1.0])
                slice_thickness = float(getattr(ds_local, "SliceThickness", 1.0))
                spacing_z = float(getattr(ds_local, "SpacingBetweenSlices", slice_thickness))
                if spacing is None:
                    # Use spacing from multiframe file (pixel_spacing gives [row, col])
                    spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)
            except Exception:
                print("multiframe detected (could not read spacing metadata)")
            # append every frame from the multi-frame file
            for f in frames:
                slices.append(f)
        # Single-frame DICOM: take the single frame and capture spacing if not set
        elif len(frames) == 1:
            slices.append(frames[0])  # Take first frame
            # Get spacing from first DICOM when not already set
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
            # No frames returned by read_dicom_frames_hu for this file - skip
            continue
    if not slices:
        raise RuntimeError(f"No readable frames in {series_dir}")
    
    vol = np.stack(slices, axis=0).astype(np.float32)
    print(f"Loaded volume shape: {vol.shape}")
    # Resample to isotropic and resize to target size
    if spacing is not None:
        sx, sy, sz = spacing
        zoom = (sz / target_spacing, sy / target_spacing, sx / target_spacing)
        vol = ndimage.zoom(vol, zoom, order=1, prefilter=False)
    
    return vol


def extract_plane_slices(vol: np.ndarray, plane: str) -> List[np.ndarray]:
    """Extract all slices from a volume for a specific plane."""
    Z, H, W = vol.shape
    slices = []
    
    if plane == "axial":
        for z in range(Z):
            slice_img = vol[z, :, :]
            slices.append(slice_img)
    elif plane == "coronal":
        for y in range(H):
            slice_img = np.flipud(vol[:, y, :])
            slices.append(slice_img)
    elif plane == "sagittal":
        for x in range(W):
            slice_img = np.flipud(vol[:, :, x])
            slices.append(slice_img)
    else:
        raise ValueError(f"Unknown plane: {plane}")
    
    return slices


def detect_plane_from_weights_path(weights_path: str) -> str:
    """Auto-detect plane from weights file path."""
    path_lower = weights_path.lower()
    if 'axial' in path_lower:
        return 'axial'
    elif 'coronal' in path_lower:
        return 'coronal'
    elif 'sagittal' in path_lower:
        return 'sagittal'
    else:
        # Default to axial if can't detect
        return 'axial'



def main():
    args = parse_args()
    
    # Determine plane from arguments or weights path
    plane = args.plane
    if not plane:
        plane = detect_plane_from_weights_path(args.weights)
        print(f"Auto-detected plane: {plane}")
    else:
        print(f"Using specified plane: {plane}")
    
    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_df = pd.read_csv(data_root / 'train_df.csv') if (data_root / 'train_df.csv').exists() else pd.read_csv(data_root / 'train.csv')
    if 'Aneurysm Present' not in train_df.columns:
        raise SystemExit("train_df.csv requires 'Aneurysm Present' column for classification label")

    val_series = train_df[train_df['fold_id'] == args.val_fold]['SeriesInstanceUID'].unique().tolist()
    if args.series_limit:
        val_series = val_series[:args.series_limit]

    print(f"Validation fold {args.val_fold}: {len(val_series)} series")
    print(f"Plane: {plane}")
    print(f"Model weights: {args.weights}")
    model = YOLO(args.weights)

    series_probs: Dict[str, float] = {}
    scores_list: List[float] = []
    cls_labels: List[int] = []
    loc_labels: List[np.ndarray] = []
    series_pred_loc_probs: List[np.ndarray] = []
    # For CSV export by series
    series_pred_loc_probs_map: Dict[str, np.ndarray] = {}

    # For confidence comparison
    pos_confidences: list[float] = []
    neg_confidences: list[float] = []
    pos_modalities: list[str] = []
    neg_modalities: list[str] = []
    # Detection counts per class for optional summary
    pos_det_counts: list[int] = []
    neg_det_counts: list[int] = []
    # Per-series metadata for optional CSV
    series_true_labels: Dict[str, int] = {}
    series_modalities: Dict[str, str] = {}
    series_pred_counts: Dict[str, int] = {}

    for sid in tqdm(val_series, desc=f"Validating {plane}", unit="series"):
        series_dir = series_root / sid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] {sid} (no directory)")
            continue
        
        try:
            # Load volume and extract plane slices
            vol = load_volume_from_series(series_dir, args.target_spacing, args.img_size)
            plane_slices = extract_plane_slices(vol, plane)
            
            if args.max_slices > 0:
                plane_slices = plane_slices[:args.max_slices]
            
            # Apply slice step sampling
            if args.slice_step > 1:
                plane_slices = plane_slices[::args.slice_step]
                
        except Exception as e:
            if args.verbose:
                print(f"[ERROR] {sid}: {e}")
            continue
        
        if not plane_slices:
            if args.verbose:
                print(f"[EMPTY] {sid}")
            continue

        # Reset per-class max confidences for this series
        per_class_max = np.zeros(N_LOC, dtype=np.float32)
        max_conf = 0.0
        total_dets = 0
        batch: list[np.ndarray] = []
        
        def flush_batch(batch_imgs: list[np.ndarray]):
            nonlocal max_conf, total_dets, per_class_max
            if not batch_imgs:
                return
            # Ultralytics can take list/np.array. Provide list for variable shapes (should be consistent though).
            results = model.predict(batch_imgs, verbose=False, conf=0.01)
            for r in results:
                if not r or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                    continue
                try:
                    confs = r.boxes.conf
                    clses = r.boxes.cls
                    # Resolve class id -> class name mapping
                    names = getattr(r, 'names', getattr(model, 'names', None))
                    n = len(confs)
                    total_dets += int(n)
                    for i in range(n):
                        c = float(confs[i].item())
                        k_raw = int(clses[i].item())
                        if c > max_conf:
                            max_conf = c
                        # Map to our LOCATION_LABELS order using class name when available
                        idx = None
                        try:
                            class_name = None
                            if isinstance(names, dict):
                                class_name = names.get(k_raw)
                            elif isinstance(names, (list, tuple)):
                                class_name = names[k_raw] if 0 <= k_raw < len(names) else None
                            if class_name is not None and class_name in LABELS_TO_IDX:
                                idx = LABELS_TO_IDX[class_name]
                        except Exception:
                            idx = None
                        # Fallback: assume the training class order already matches LOCATION_LABELS
                        if idx is None and 0 <= k_raw < N_LOC:
                            idx = k_raw
                        if idx is not None:
                            if c > per_class_max[idx]:
                                per_class_max[idx] = c
                except Exception:
                    # Fallback: just use max conf
                    try:
                        batch_max = float(r.boxes.conf.max().item())
                        if batch_max > max_conf:
                            max_conf = batch_max
                        total_dets += int(len(r.boxes))
                    except Exception:
                        pass

        # Process plane slices
        for slice_img in plane_slices:
            img_uint8 = min_max_normalize(slice_img)
            # Resize per-slice with OpenCV to match training data generation
            if args.img_size > 0:
                img_uint8 = cv2.resize(img_uint8, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
            if img_uint8.ndim == 2:
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            batch.append(img_uint8)
            if len(batch) >= args.batch_size:
                flush_batch(batch)
                batch.clear()
        flush_batch(batch)
        series_probs[sid] = max_conf
        scores_list.append(max_conf)
        series_pred_counts[sid] = total_dets
        if args.verbose:
            print(f"Series {sid} max_conf={max_conf:.4f} dets={total_dets}")

        # Labels and metadata for this series
        row = train_df[train_df['SeriesInstanceUID'] == sid].iloc[0]
        label = int(row['Aneurysm Present'])
        cls_labels.append(label)
        series_true_labels[sid] = label
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
        # Store per-series location probabilities
        series_pred_loc_probs.append(per_class_max.copy())
        series_pred_loc_probs_map[sid] = per_class_max.copy()

        # Periodic partial AUC (cls and loc macro) together every 10 series
        if len(cls_labels) % 10 == 0 and len(loc_labels) % 10 == 0:
            try:
                # Classification partial AUC
                current_y_true = np.array(cls_labels)
                current_y_scores = np.array(scores_list)
                cls_pauc = roc_auc_score(current_y_true, current_y_scores)
            except ValueError:
                cls_pauc = float('nan')
            # Location macro partial AUC
            try:
                cur_loc_labels = np.stack(loc_labels)
                cur_loc_preds = np.stack(series_pred_loc_probs)
                per_loc_pauc = []
                for i in range(N_LOC):
                    try:
                        auc_i = roc_auc_score(cur_loc_labels[:, i], cur_loc_preds[:, i])
                    except ValueError:
                        auc_i = float('nan')
                    per_loc_pauc.append(auc_i)
                loc_macro_pauc = float(np.nanmean(per_loc_pauc)) if per_loc_pauc else float('nan')
            except Exception:
                loc_macro_pauc = float('nan')
            print(f"Partial AUC after {len(cls_labels)} series -> cls: {cls_pauc:.4f}, loc_macro: {loc_macro_pauc:.4f}")

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
            pos_confidences.append(max_conf)
            pos_modalities.append(modality)
            pos_det_counts.append(total_dets)
        else:
            neg_confidences.append(max_conf)
            neg_modalities.append(modality)
            neg_det_counts.append(total_dets)

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

        # Effect size via rank-based separation: Cliff's delta approx from AUROC
        # AUROC computed later as cls_auc; delta = 2*AUC - 1
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

        # Detection count summary by class
        if pos_det_counts or neg_det_counts:
            pd_mean = float(np.mean(pos_det_counts)) if pos_det_counts else 0.0
            pd_med = float(np.median(pos_det_counts)) if pos_det_counts else 0.0
            nd_mean = float(np.mean(neg_det_counts)) if neg_det_counts else 0.0
            nd_med = float(np.median(neg_det_counts)) if neg_det_counts else 0.0
            print("\nDetection counts per tomogram:")
            print(f"  Positives: mean={pd_mean:.2f}, median={pd_med:.2f}")
            print(f"  Negatives: mean={nd_mean:.2f}, median={nd_med:.2f}")

    # Metrics
    y_true = np.array(cls_labels)
    y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
    cls_auc = roc_auc_score(y_true, y_scores)
    try:
        cls_partial_auc = roc_auc_score(y_true, y_scores)
        print(f"Classification partial AUC: {cls_partial_auc:.4f}")
    except ValueError as e:
        print(f"Could not compute classification partial AUC: {e}")

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

    # Location partial AUCs (macro)
    per_loc_pauc = []
    for i in range(N_LOC):
        try:
            auc_i = roc_auc_score(loc_labels_arr[:, i], loc_pred_arr[:, i])
        except ValueError:
            auc_i = float('nan')
        per_loc_pauc.append(auc_i)
    loc_macro_pauc = np.nanmean(per_loc_pauc)

    print(f"Classification AUC (aneurysm present): {cls_auc:.4f}")
    print(f"Location macro AUC (constant 0.5 baseline): {loc_macro_auc:.4f}")
    print(f"Combined (mean) CV metric: {(cls_auc + (loc_macro_auc if not math.isnan(loc_macro_auc) else 0))/2:.4f}")
    print(f"Location macro partial AUC: {loc_macro_pauc:.4f}")

    if args.save_csv:
        out_rows = []
        for sid, prob in series_probs.items():
            loc_arr = series_pred_loc_probs_map.get(sid, np.zeros(N_LOC, dtype=np.float32))
            out_rows.append({
                'SeriesInstanceUID': sid,
                'aneurysm_prob': prob,
                'true_label': series_true_labels.get(sid, None),
                'modality': series_modalities.get(sid, ''),
                'num_detections': series_pred_counts.get(sid, 0),
                **{f'loc_prob_{i}': float(loc_arr[i]) for i in range(N_LOC)}
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