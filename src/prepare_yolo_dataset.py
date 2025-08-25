from __future__ import annotations

import argparse
import ast
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold

# Allow importing project configs
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, N_FOLDS, SEED  # type: ignore


rng = random.Random(SEED)


def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    """Read a DICOM file and return a list of frames in HU (float32).

    Supports:
      - 2D single-slice DICOM -> one frame [H,W]
      - 3D multi-frame -> list of frames [N,H,W]
      - 3-channel RGB DICOM -> converted to single grayscale frame
    """
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:
        # If RGB (H,W,3), take first channel; else assume multi-frame (N,H,W)
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    else:
        # Unsupported layout
        pass
    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def build_box(x: float, y: float, box_size: int, w: int, h: int) -> Tuple[float, float, float, float]:
    """Return normalized (xc, yc, bw, bh) for YOLO given center point and square size in pixels."""
    half = box_size / 2.0
    x0 = max(0.0, x - half)
    y0 = max(0.0, y - half)
    x1 = min(w - 1.0, x + half)
    y1 = min(h - 1.0, y + half)
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    xc = x0 + bw / 2.0
    yc = y0 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def ensure_yolo_dirs(base: Path):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare MIP-based YOLO aneurysm dataset")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    ap.add_argument("--val-fold", type=int, default=0, help="Fold id used for validation")
    ap.add_argument("--box-size", type=int, default=48, help="Square box size in pixels around the point")
    ap.add_argument("--image-ext", type=str, default="png", choices=["png", "jpg", "jpeg"], help="Output image extension")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--verbose", action="store_true")
    # MIP options
    ap.add_argument("--mip-img-size", type=int, default=512, help="Final square size (0 = keep original MIP size)")
    ap.add_argument("--mip-out-name", type=str, default="yolo_dataset", help="Subdirectory under data/ for outputs")
    ap.add_argument("--mip-pos-window", type=int, default=3, help="If >0, build MIPs from +/- window around positive slices")
    return ap.parse_args()


def load_folds(root: Path) -> Dict[str, int]:
    """Map SeriesInstanceUID -> fold_id using stratified folds from train_df.csv.

    Requirements:
      - data/train_df.csv must exist
      - Columns: 'SeriesInstanceUID', 'Aneurysm Present'
      - Will create N_FOLDS stratified splits on the series-level label
    """
    df_path = root / "train_df.csv"

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
    return fold_map


def load_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in label_df.columns or "y" not in label_df.columns:
        label_df["x"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        label_df["y"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    # Standardize dtypes
    label_df["SeriesInstanceUID"] = label_df["SeriesInstanceUID"].astype(str)
    label_df["SOPInstanceUID"] = label_df["SOPInstanceUID"].astype(str)
    return label_df


def ordered_dcm_paths(series_dir: Path) -> Tuple[List[Path], Dict[str, int]]:
    paths = sorted(series_dir.glob("*.dcm"), key=lambda p: p.name)
    sop_to_idx = {p.stem: i for i, p in enumerate(paths)}
    return paths, sop_to_idx


def load_series_slices_selected(paths: List[Path], selected_indices: List[int] | None) -> List[np.ndarray]:
    indices = selected_indices if selected_indices is not None else list(range(len(paths)))
    slices: List[np.ndarray] = []
    for i in indices:
        if 0 <= i < len(paths):
            for arr in read_dicom_frames_hu(paths[i]):
                slices.append(arr)
    return slices


def main():
    args = parse_args()
    global rng
    rng = random.Random(args.seed)

    root = Path(data_path)
    out_base = root / args.mip_out_name
    ensure_yolo_dirs(out_base)

    # Ignore known problematic and multi-frame series
    ignore_uids: Set[str] = set(
        [
            "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
            "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
            "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242",
        ]
    )
    mf_path = root / "multiframe_dicoms.csv"
    if mf_path.exists():
        try:
            mf_df = pd.read_csv(mf_path)
            if "SeriesInstanceUID" in mf_df.columns:
                ignore_uids.update(mf_df["SeriesInstanceUID"].astype(str).tolist())
        except Exception:
            pass

    folds = load_folds(root)
    label_df = load_labels(root)

    # Group labels per series, keep SOP to locate slice indices
    series_to_labels: Dict[str, List[Tuple[str, float, float]]] = {}
    for _, r in label_df.iterrows():
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append(
            (str(r.SOPInstanceUID), float(r.x), float(r.y))
        )

    all_series: List[str] = []
    train_df_path = root / "train_df.csv"
    if train_df_path.exists():
        try:
            train_csv = pd.read_csv(train_df_path)
            if "SeriesInstanceUID" in train_csv.columns:
                all_series = (
                    train_csv["SeriesInstanceUID"].astype(str).unique().tolist()
                )
        except Exception:
            pass
    if not all_series:
        all_series = sorted(series_to_labels.keys())

    total_series = 0
    total_pos = 0
    total_neg = 0
    print(f"Processing {len(all_series)} series for MIP YOLO dataset...")

    for uid in all_series:
        if uid in ignore_uids:
            if args.verbose:
                print(f"[SKIP] ignored series {uid}")
            continue
        split = "val" if folds.get(uid, 0) == args.val_fold else "train"
        series_dir = root / "series" / uid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] series {uid}")
            continue

        paths, sop_to_idx = ordered_dcm_paths(series_dir)
        n_slices = len(paths)
        labels_for_series = series_to_labels.get(uid, [])

        # MIP windows around positives
        if args.mip_pos_window and args.mip_pos_window > 0 and n_slices > 0:
            window = args.mip_pos_window
            if labels_for_series:
                for (sop_c, _x_c, _y_c) in labels_for_series:
                    if sop_c not in sop_to_idx:
                        continue
                    center = sop_to_idx[sop_c]
                    lo = max(0, center - window)
                    hi = min(n_slices - 1, center + window)
                    idxs = list(range(lo, hi + 1))
                    slices = load_series_slices_selected(paths, idxs)
                    if not slices:
                        continue
                    base_shape = slices[0].shape
                    mip = None
                    for s in slices:
                        if s.shape != base_shape:
                            s = cv2.resize(s, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_LINEAR)
                        mip = s if mip is None else np.maximum(mip, s)
                    mip_uint8 = min_max_normalize(mip)
                    if args.mip_img_size > 0 and (
                        mip_uint8.shape[0] != args.mip_img_size
                        or mip_uint8.shape[1] != args.mip_img_size
                    ):
                        mip_uint8 = cv2.resize(
                            mip_uint8,
                            (args.mip_img_size, args.mip_img_size),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        resize_w = resize_h = args.mip_img_size
                    else:
                        resize_h, resize_w = mip_uint8.shape

                    stem = f"{uid}_{sop_c}_w{lo}-{hi}"
                    img_path = out_base / "images" / split / f"{stem}.{args.image_ext}"
                    label_path = out_base / "labels" / split / f"{stem}.txt"
                    if img_path.exists() and label_path.exists() and not args.overwrite:
                        continue
                    cv2.imwrite(str(img_path), mip_uint8)

                    # Points within this window
                    pts_in_win: List[Tuple[float, float]] = []
                    for (sop, x, y) in labels_for_series:
                        idx = sop_to_idx.get(sop)
                        if idx is not None and lo <= idx <= hi:
                            pts_in_win.append((x, y))
                    with open(label_path, "w") as f:
                        for (x, y) in pts_in_win:
                            orig_h, orig_w = base_shape
                            if (resize_w, resize_h) != (orig_w, orig_h):
                                x_scaled = x * (resize_w / orig_w)
                                y_scaled = y * (resize_h / orig_h)
                            else:
                                x_scaled, y_scaled = x, y
                            xc, yc, bw, bh = build_box(
                                x_scaled, y_scaled, args.box_size, resize_w, resize_h
                            )
                            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    total_pos += 1
                    total_series += 1
            else:
                # Negative: random window with empty label
                center = rng.randint(0, n_slices - 1)
                lo = max(0, center - window)
                hi = min(n_slices - 1, center + window)
                idxs = list(range(lo, hi + 1))
                slices = load_series_slices_selected(paths, idxs)
                if not slices:
                    continue
                base_shape = slices[0].shape
                mip = None
                for s in slices:
                    if s.shape != base_shape:
                        s = cv2.resize(s, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_LINEAR)
                    mip = s if mip is None else np.maximum(mip, s)
                mip_uint8 = min_max_normalize(mip)
                if args.mip_img_size > 0 and (
                    mip_uint8.shape[0] != args.mip_img_size or mip_uint8.shape[1] != args.mip_img_size
                ):
                    mip_uint8 = cv2.resize(
                        mip_uint8, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR
                    )
                stem = f"{uid}_neg_w{lo}-{hi}"
                img_path = out_base / "images" / split / f"{stem}.{args.image_ext}"
                label_path = out_base / "labels" / split / f"{stem}.txt"
                if not (img_path.exists() and label_path.exists()) or args.overwrite:
                    cv2.imwrite(str(img_path), mip_uint8)
                    open(label_path, "w").close()
                    total_neg += 1
                    total_series += 1
            # Windowed case handled; continue to next series
            continue

        # Whole-series MIP (no window requested)
        slices = load_series_slices_selected(paths, None)
        if not slices:
            if args.verbose:
                print(f"[EMPTY] series {uid}")
            continue
        base_shape = slices[0].shape
        mip = None
        for s in slices:
            if s.shape != base_shape:
                s = cv2.resize(s, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_LINEAR)
            mip = s if mip is None else np.maximum(mip, s)
        mip_uint8 = min_max_normalize(mip)
        if args.mip_img_size > 0 and (
            mip_uint8.shape[0] != args.mip_img_size or mip_uint8.shape[1] != args.mip_img_size
        ):
            mip_uint8 = cv2.resize(
                mip_uint8, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR
            )
            resize_w = resize_h = args.mip_img_size
        else:
            resize_h, resize_w = mip_uint8.shape

        img_path = out_base / "images" / split / f"{uid}.{args.image_ext}"
        label_path = out_base / "labels" / split / f"{uid}.txt"
        if not (img_path.exists() and label_path.exists()) or args.overwrite:
            cv2.imwrite(str(img_path), mip_uint8)
            pts = [(x, y) for (_sop, x, y) in labels_for_series]
            if pts:
                with open(label_path, "w") as f:
                    for (x, y) in pts:
                        orig_h, orig_w = base_shape
                        if (resize_w, resize_h) != (orig_w, orig_h):
                            x_scaled = x * (resize_w / orig_w)
                            y_scaled = y * (resize_h / orig_h)
                        else:
                            x_scaled, y_scaled = x, y
                        xc, yc, bw, bh = build_box(x_scaled, y_scaled, args.box_size, resize_w, resize_h)
                        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                total_pos += 1
            else:
                open(label_path, "w").close()
                total_neg += 1
            total_series += 1

    print(f"Series processed: {total_series}  (positive label files: {total_pos}, negative: {total_neg})")
    print(f"MIP YOLO dataset root: {out_base}")
    for split in ["train", "val"]:
        n_imgs = len(list((out_base / "images" / split).glob("*")))
        n_lbls = len(list((out_base / "labels" / split).glob("*.txt")))
        print(f"{split}: images={n_imgs} labels={n_lbls}")


if __name__ == "__main__":
    main()