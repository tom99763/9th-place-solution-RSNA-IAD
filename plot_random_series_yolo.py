"""Select a random validation tomogram and plot YOLO inference results.

Features:
  - Picks a random series from the requested validation fold (train_df.csv/train.csv)
  - Runs YOLO on either raw slices or sliding-window MIPs
  - Selects top-K images/windows by max detection confidence
  - Saves a matplotlib grid with drawn boxes and confidences

Notes:
  - Assumes a single-class YOLO aneurysm detector.
  - Requires ultralytics, numpy, pandas, pydicom, opencv-python, matplotlib.
  - Defaults to the same weights path used in yolo_validation.py; override with --weights.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO  # type: ignore

# Project root & config imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))  # to allow importing configs.data_config
from configs.data_config import data_path  # type: ignore


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


def draw_boxes(image_bgr: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray | None = None) -> np.ndarray:
    out = image_bgr.copy()
    color = (0, 255, 0)
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if confs is not None:
            c = float(confs[i])
            label = f"{c:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot YOLO detections for a random validation series")
    ap.add_argument('--val-fold', type=int, default=0, help='Validation fold id to sample from')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for series selection')
    ap.add_argument('--num-panels', type=int, default=12, help='Number of images/windows to show in the grid')
    ap.add_argument('--cols', type=int, default=4, help='Columns in the grid')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice (when not using MIP)')
    ap.add_argument('--max-slices', type=int, default=0, help='Optional cap on processed slices/windows')
    ap.add_argument('--mip-window', type=int, default=0, help='Half-window for sliding MIP; 0 = raw slices')
    ap.add_argument('--mip-no-overlap', action='store_true', help='Use non-overlapping MIP windows')
    ap.add_argument('--mip-img-size', type=int, default=0, help='Optional square resize before inference (0=original)')
    ap.add_argument('--batch-size', type=int, default=16, help='YOLO inference batch size')
    ap.add_argument('--weights', type=str, default=str(ROOT / 'runs/yolo_aneurysm/baseline_one_slice3/weights/best.pt'), help='Path to YOLO weights')
    ap.add_argument('--save', type=str, default=str(ROOT / 'outputs/random_val_series_yolo.png'), help='Output image path')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def pick_random_series(train_df: pd.DataFrame, fold_id: int, seed: int) -> str:
    candidates = train_df[train_df['fold_id'] == fold_id]['SeriesInstanceUID'].dropna().astype(str).unique().tolist()
    if not candidates:
        raise SystemExit(f"No series found for fold_id={fold_id}")
    random.Random(seed).shuffle(candidates)
    return candidates[0]


def run_inference_on_series(series_dir: Path, model: YOLO, args: argparse.Namespace) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Process a series and return list of (rgb_img, boxes_xyxy, confs, max_conf) per item.

    Returns top-K by max_conf later; this returns all items for selection.
    """
    dcm_paths = collect_series_slices(series_dir)
    if not dcm_paths:
        return []

    items_imgs: List[np.ndarray] = []  # BGR images fed to model
    # Build items
    if args.mip_window > 0:
        # Build MIP windows across full tomogram
        slices_hu: list[np.ndarray] = []
        shapes_count: Dict[tuple[int, int], int] = {}
        for p in dcm_paths:
            try:
                frames = read_dicom_frames_hu(p)
            except Exception:
                continue
            for f in frames:
                f = f.astype(np.float32)
                slices_hu.append(f)
                shapes_count[f.shape] = shapes_count.get(f.shape, 0) + 1
        if not slices_hu:
            return []
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
            img_bgr = cv2.cvtColor(mip_u8, cv2.COLOR_GRAY2BGR)
            if args.mip_img_size and args.mip_img_size > 0:
                img_bgr = cv2.resize(img_bgr, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
            items_imgs.append(img_bgr)
    else:
        # Per-slice
        step_paths = dcm_paths[::max(1, args.slice_step)]
        if args.max_slices and len(step_paths) > args.max_slices:
            step_paths = step_paths[:args.max_slices]
        for p in step_paths:
            try:
                frames = read_dicom_frames_hu(p)
            except Exception:
                continue
            for f in frames:
                u8 = min_max_normalize(f)
                if u8.ndim == 2:
                    img_bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = u8
                items_imgs.append(img_bgr)

    # Run model in batches, collect boxes & confidences
    results_boxes: List[np.ndarray] = []
    results_confs: List[np.ndarray] = []
    results_max: List[float] = []

    def process_batch(batch_imgs: List[np.ndarray]):
        if not batch_imgs:
            return
        res = model.predict(batch_imgs, verbose=False, conf=0.01)
        for r in res:
            if r is None or r.boxes is None or len(r.boxes) == 0:
                results_boxes.append(np.zeros((0, 4), dtype=np.float32))
                results_confs.append(np.zeros((0,), dtype=np.float32))
                results_max.append(0.0)
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            results_boxes.append(boxes)
            results_confs.append(confs)
            results_max.append(float(confs.max()))

    batch: List[np.ndarray] = []
    for img in items_imgs:
        batch.append(img)
        if len(batch) >= args.batch_size:
            process_batch(batch)
            batch.clear()
    process_batch(batch)

    items: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for i in range(len(items_imgs)):
        items.append((items_imgs[i], results_boxes[i], results_confs[i], results_max[i]))
    return items


def save_grid(series_id: str, items: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], out_path: Path, num_panels: int, cols: int):
    if not items:
        print("No items to plot.")
        return
    # pick top-K by max_conf
    items_sorted = sorted(items, key=lambda t: t[3], reverse=True)
    top = items_sorted[:num_panels]
    rows = int(np.ceil(len(top) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = np.array([[ax] for ax in axs])

    for idx, (img_bgr, boxes, confs, mx) in enumerate(top):
        r, c = divmod(idx, cols)
        ax = axs[r, c]
        vis = draw_boxes(img_bgr, boxes, confs)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        ax.imshow(vis_rgb)
        ax.set_title(f"max={mx:.2f}")
        ax.axis('off')
    # Hide unused axes
    for idx in range(len(top), rows * cols):
        r, c = divmod(idx, cols)
        axs[r, c].axis('off')

    fig.suptitle(f"Random validation series {series_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved grid to {out_path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_csv = data_root / 'train_df.csv'
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
    else:
        train_df = pd.read_csv(data_root / 'train.csv')
    if 'fold_id' not in train_df.columns or 'SeriesInstanceUID' not in train_df.columns:
        raise SystemExit("train_df/train.csv must contain fold_id and SeriesInstanceUID")

    series_id = pick_random_series(train_df, args.val_fold, args.seed)
    print(f"Selected series (fold {args.val_fold}): {series_id}")

    series_dir = series_root / series_id
    if not series_dir.exists():
        raise SystemExit(f"Series directory missing: {series_dir}")

    model = YOLO(args.weights)

    items = run_inference_on_series(series_dir, model, args)
    if args.verbose:
        print(f"Processed {len(items)} item(s) from series {series_id}")
    if not items:
        print("No images generated for this series; nothing to plot.")
        return

    save_grid(series_id, items, Path(args.save), args.num_panels, args.cols)


if __name__ == '__main__':
    main()
#python3 /home/sersasj/RSNA-IAD-Codebase/plot_random_series_yolo.py \
#  --val-fold 0 \
#  --seed 43 \
#  --mip-window 3 \
#  --num-panels 12 \
#  --cols 4 \
#  --batch-size 16 \
#  --save /home/sersasj/RSNA-IAD-Codebase/outputs/random_val_series_yolo.png