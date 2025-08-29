"""Validate a single-class YOLO aneurysm detector at series level.

Preprocessing:
    - DICOM -> HU -> per-slice min-max -> uint8 grayscale for YOLO (auto BGR conversion on call)

Series score:
    - Probability = max detection confidence across all processed slices/MIP windows in a series

Outputs:
    - Prints classification AUC (aneurysm present)
    - Prints location macro AUC (degenerate ~0.5; constant probs)
    - Optional CSV with per-series probabilities

Notes:
    - Requires ultralytics (YOLOv8/11). Single-class model assumed.
    - Use --mip-window > 0 to validate on sliding-window MIPs instead of raw slices.
"""
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
sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

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
    ap = argparse.ArgumentParser(description="Series-level validation for YOLO aneurysm detector")
    #ap.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt)')
    ap.add_argument('--val-fold', type=int, default=0, help='Fold id to evaluate (matches train.csv fold_id)')
    ap.add_argument('--series-limit', type=int, default=0, help='Optional limit on number of validation series (debug)')
    ap.add_argument('--max-slices', type=int, default=0, help='Optional cap on number of slices per series (debug)')
    ap.add_argument('--save-csv', type=str, default='', help='Optional path to save per-series predictions CSV')
    ap.add_argument('--batch-size', type=int, default=16, help='Batch size for slice inference (higher = faster, more VRAM)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice (default=1)')
    # Sliding-window MIP mode
    ap.add_argument('--mip-window', type=int, default=3, help='Half-window (in slices) for sliding MIP across the full tomogram; 0 disables MIP mode')
    ap.add_argument('--mip-img-size', type=int, default=0, help='Optional resize of MIP to this square size before inference (0 keeps original)')
    ap.add_argument('--mip-no-overlap', action='store_true', help='Use non-overlapping MIP windows (stride = 2*w+1 instead of slice_step)')
    return ap.parse_args()


def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    """Return list of HU frames from a DICOM. Handles 2D, multi-frame, and RGB->grayscale."""
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
    print(f"Processing every {args.slice_step} slice(s)")
    model_path = "/home/sersasj/RSNA-IAD-Codebase/runs/yolo_aneurysm/baseline_slice_24bbox2/weights/best.pt"
    model = YOLO(model_path)

    series_probs: Dict[str, float] = {}
    cls_labels: List[int] = []
    loc_labels: List[np.ndarray] = []
    const_loc_probs = np.full(N_LOC, 0.5, dtype=np.float32)
    series_pred_loc_probs: List[np.ndarray] = []

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
        
        # In MIP mode we need the full tomogram; otherwise allow stepping per file
        if args.mip_window <= 0:
            # Apply slice stepping BEFORE max_slices limit
            dicoms = dicoms[::args.slice_step]  # Skip every N slices
            if args.max_slices and len(dicoms) > args.max_slices:
                dicoms = dicoms[:args.max_slices]

        max_conf = 0.0
        total_dets = 0
        batch: list[np.ndarray] = []
        def flush_batch(batch_imgs: list[np.ndarray]):
            nonlocal max_conf, total_dets
            if not batch_imgs:
                return
            # Ultralytics can take list/np.array. Provide list for variable shapes (should be consistent though).
            results = model.predict(batch_imgs, verbose=False, conf=0.01)
            for r in results:
                if not r or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                    continue
                batch_max = float(r.boxes.conf.max().item())
                if batch_max > max_conf:
                    max_conf = batch_max
                # Count detections (single-class)
                try:
                    total_dets += int(len(r.boxes))
                except Exception:
                    pass
        if args.mip_window > 0:
            # Load all frames as HU arrays (keep original shapes)
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
                if args.verbose:
                    print(f"[NO_VALID_SLICES] {sid}")
                series_probs[sid] = 0.0
                series_pred_counts[sid] = 0
                continue
            # Choose a target shape (most common); resize MIPs (or per-slice via cache) to this
            target_shape = max(shapes_count.items(), key=lambda kv: kv[1])[0]
            th, tw = target_shape
            # Window centers
            n = len(slices_hu)
            if args.mip_no_overlap:
                # Non-overlapping windows: stride equals full window size (2w+1)
                w = max(0, int(args.mip_window))
                window_size = 2 * w + 1
                if n <= 0:
                    centers: list[int] = []
                elif n < window_size:
                    centers = [n // 2]
                else:
                    centers = list(range(w, n, window_size))
                    # Ensure tail coverage if remaining slices after last window are significant
                    if centers and centers[-1] + w < n - 1:
                        centers.append(min(n - 1 - w, n - 1))
                if args.verbose:
                    print(f"[MIP] non-overlap mode: window_size={window_size}, centers={len(centers)}")
            else:
                stride = max(1, args.slice_step)
                centers = list(range(0, n, stride))
                if args.verbose:
                    print(f"[MIP] overlap mode: half-window={args.mip_window}, stride={stride}, centers={len(centers)}")
            if args.max_slices and len(centers) > args.max_slices:
                centers = centers[:args.max_slices]
            # Optional cache for resized HU slices to avoid repeated resizing across overlapping windows
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
                # Optional square resize for YOLO
                mip_rgb = cv2.cvtColor(mip_u8, cv2.COLOR_GRAY2BGR)
                batch.append(mip_rgb)
                if len(batch) >= args.batch_size:
                    flush_batch(batch)
                    batch.clear()
            # Flush remaining MIP windows
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
                    if img_uint8.ndim == 2:
                        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    batch.append(img_uint8)
                    if len(batch) >= args.batch_size:
                        flush_batch(batch)
                        batch.clear()
            flush_batch(batch)
        series_probs[sid] = max_conf
        series_pred_counts[sid] = total_dets
        print(f"Series {sid} max_conf={max_conf:.4f} dets={total_dets} (processed {len(dicoms)} slices)")
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
        series_pred_loc_probs.append(const_loc_probs.copy())

        # Collect confidences and modality for comparison
        modality = 'Unknown'
        if dicoms:
            try:
                ds = pydicom.dcmread(str(dicoms[0]), stop_before_pixels=True)
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
        print(f"\nBest threshold by Youden's J: {best_thr:.4f}  (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")
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
                'true_label': series_true_labels.get(sid, None),
                'modality': series_modalities.get(sid, ''),
                'num_detections': series_pred_counts.get(sid, 0),
                **{f'loc_prob_{i}': 0.5 for i in range(N_LOC)}
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