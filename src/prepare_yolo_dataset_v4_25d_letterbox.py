
import argparse
import ast
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold
# Multilabel stratification for better balance across modality and classes
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # type: ignore

# Allow importing project configs
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, N_FOLDS, SEED, LABELS_TO_IDX  # type: ignore

N_FOLDS = 5
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


def letterbox(
    img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resize and pad image while maintaining aspect ratio.

    Args:
        img (np.ndarray): Input image with shape (H, W, C).
        new_shape (Tuple[int, int]): Target shape (height, width).

    Returns:
        (np.ndarray): Resized and padded image.
        (Tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
    """
    shape = img.shape[:2]  # Current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

    if shape[::-1] != new_unpad:  # Resize if needed
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, (top / img.shape[0], left / img.shape[1])


def create_rgb_from_slices(slice_prev: np.ndarray, slice_curr: np.ndarray, slice_next: np.ndarray) -> np.ndarray:
    """Create RGB image from three adjacent slices, normalizing each channel individually."""
    # Normalize each slice individually
    r_channel = min_max_normalize(slice_prev)
    g_channel = min_max_normalize(slice_curr)
    b_channel = min_max_normalize(slice_next)
    
    # Stack as RGB channels
    rgb_img = np.stack([r_channel, g_channel, b_channel], axis=-1)
    return rgb_img


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
    ap = argparse.ArgumentParser(description="Prepare per-slice YOLO aneurysm dataset (no MIP/BGR)")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    ap.add_argument("--val-fold", type=int, default=0, help="Fold id used for validation (ignored if --generate-all-folds)")
    ap.add_argument("--generate-all-folds", action="store_true", help="Generate one dataset per fold and write per-fold YAMLs")
    ap.add_argument("--box-size", type=int, default=24, help="Square box size in pixels around the point")
    ap.add_argument("--image-ext", type=str, default="png", choices=["png", "jpg", "jpeg"], help="Output image extension")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--img-size", type=int, default=512, help="Final square image size for saved samples (0 = keep original)")
    ap.add_argument("--out-name", type=str, default="yolo_dataset", help="Base subdirectory under data/ for outputs; when --generate-all-folds, becomes {base}_fold{fold}")
    # Labeling scheme
    ap.add_argument(
        "--label-scheme",
        type=str,
        choices=["locations", "aneurysm_present"],
        default="locations",
        help="Use location-specific classes (from LABELS_TO_IDX) or single binary class 'aneurysm_present'",
    )
    # Negative sampling controls
    ap.add_argument(
        "--neg-per-series",
        type=int,
        default=10,
        help="Number of negative slices to sample per series without positives.",
    )
    # YAML outputs
    ap.add_argument("--yaml-out-dir", type=str, default=str(ROOT / "configs"), help="Directory to write per-fold YOLO YAMLs")
    ap.add_argument("--yaml-name-template", type=str, default="yolo_fold{fold}.yaml", help="YAML filename template with {fold}")
    # RGB mode
    ap.add_argument("--rgb-mode", action="store_true", help="Use RGB mode: combine selected-1, selected, selected+1 slices as RGB channels")
    return ap.parse_args()


def load_folds(root: Path) -> Dict[str, int]:
    """Map SeriesInstanceUID -> fold_id using MultilabelStratifiedKFold.

    Uses MultilabelStratifiedKFold over [Aneurysm Present] + all location columns + Modality one-hot
    for optimal balance across modality and classes.
    """
    df_path = root / "train.csv"

    df = pd.read_csv(df_path)

    base_cols = ["SeriesInstanceUID", "Modality", "Aneurysm Present"]
    ignore_cols = set(["PatientAge", "PatientSex", "fold_id"]) | set(base_cols)
    location_cols = [c for c in df.columns if c not in ignore_cols]
    series_df = df[base_cols + location_cols].drop_duplicates(subset=["SeriesInstanceUID"]).reset_index(drop=True)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    fold_map: Dict[str, int] = {}

    modality_onehot = pd.get_dummies(series_df["Modality"], prefix="mod").astype(int)
    y_df = pd.concat(
        [
            series_df[["Aneurysm Present"]].astype(int),
            series_df[location_cols].fillna(0).astype(int) if location_cols else pd.DataFrame(index=series_df.index),
            modality_onehot,
        ],
        axis=1,
    )
    y = y_df.values
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for i, (_, test_idx) in enumerate(mskf.split(np.zeros(len(series_df)), y)):
        for uid in series_df.loc[test_idx, "SeriesInstanceUID"].tolist():
            fold_map[uid] = i

    df["fold_id"] = df["SeriesInstanceUID"].astype(str).map(fold_map)
    df.to_csv(df_path, index=False)
    print(f"Updated {df_path} with new fold_id column based on N_FOLDS={N_FOLDS}")

    return fold_map


def load_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in label_df.columns or "y" not in label_df.columns:
        label_df["x"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        label_df["y"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    # Parse frame index 'f' from coordinates when present
    if "f" not in label_df.columns and "coordinates" in label_df.columns:
        label_df["f"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s).get("f"))  # type: ignore[arg-type]
    # Coerce 'f' to nullable integer where possible
    if "f" in label_df.columns:
        try:
            label_df["f"] = pd.to_numeric(label_df["f"], errors="coerce").astype("Int64")
        except Exception:
            # If coercion fails, leave as-is; downstream will handle None/NA
            pass
    # Standardize dtypes
    label_df["SeriesInstanceUID"] = label_df["SeriesInstanceUID"].astype(str)
    label_df["SOPInstanceUID"] = label_df["SOPInstanceUID"].astype(str)
    return label_df


def ordered_dcm_paths(series_dir: Path) -> Tuple[List[Path], Dict[str, int]]:
    """Collect all DICOM files in series directory and sort by spatial position."""
    dicom_files = list(series_dir.glob("*.dcm"))
    
    if not dicom_files:
        return [], {}
    
    # First pass: collect all slices with their spatial information
    temp_slices = []
    for filepath in dicom_files:
        try:
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
            
            # Priority order for sorting: SliceLocation > ImagePositionPatient > InstanceNumber
            if hasattr(ds, "SliceLocation"):
                # SliceLocation is the most reliable for slice ordering
                sort_val = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                # Fallback to z-coordinate from ImagePositionPatient
                sort_val = float(ds.ImagePositionPatient[-1])
            else:
                # Final fallback to InstanceNumber
                sort_val = float(getattr(ds, "InstanceNumber", 0))
            
            # Store filepath with its sort value
            temp_slices.append((sort_val, filepath))
            
        except Exception as e:
            # Fallback: use filename as last resort
            temp_slices.append((str(filepath.name), filepath))
            continue
    
    if not temp_slices:
        return [], {}
    
    # Sort slices by the determined sort value (spatial order)
    temp_slices.sort(key=lambda x: x[0])
    
    # Extract the sorted filepaths
    sorted_files = [item[1] for item in temp_slices]
    sop_to_idx = {p.stem: i for i, p in enumerate(sorted_files)}
    return sorted_files, sop_to_idx


def load_series_slices_selected(paths: List[Path], selected_indices: List[int] | None) -> List[np.ndarray]:
    indices = selected_indices if selected_indices is not None else list(range(len(paths)))
    slices: List[np.ndarray] = []
    for i in indices:
        if 0 <= i < len(paths):
            for arr in read_dicom_frames_hu(paths[i]):
                slices.append(arr)
    return slices


def get_adjacent_slices_for_rgb(paths: List[Path], target_sop: str, sop_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Get three adjacent slices (prev, curr, next) for RGB mode from a series."""
    if target_sop not in sop_to_idx:
        return None
    
    curr_idx = sop_to_idx[target_sop]
    prev_idx = curr_idx - 1
    next_idx = curr_idx + 1
    
    # Get the three slices
    slices = []
    for idx in [prev_idx, curr_idx, next_idx]:
        if 0 <= idx < len(paths):
            frames = read_dicom_frames_hu(paths[idx])
            if frames:
                # Take the first frame if multi-frame DICOM
                slices.append(frames[0])
            else:
                return None
        else:
            # If we're at the boundary, duplicate the available slice
            if idx < 0:
                # Use the first slice for prev
                frames = read_dicom_frames_hu(paths[0])
                if frames:
                    slices.append(frames[0])
                else:
                    return None
            else:
                # Use the last slice for next
                frames = read_dicom_frames_hu(paths[-1])
                if frames:
                    slices.append(frames[0])
                else:
                    return None
    
    if len(slices) == 3:
        return slices[0], slices[1], slices[2]
    return None


def generate_for_fold(val_fold: int, args) -> Tuple[Path, Dict[str, int]]:
    """Generate per-slice dataset for a single fold and return (out_base, fold_map)."""
    global rng
    root = Path(data_path)

    # Resolve output base for this fold
    out_dir_name = args.out_name if not hasattr(args, 'generate_all_folds') or not args.generate_all_folds else f"{args.out_name}_fold{val_fold}"
    out_base = root / out_dir_name
    ensure_yolo_dirs(out_base)

    # Ignore only known problematic series
    ignore_uids: Set[str] = set(
        [
            "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
            "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
            "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242",
        ]
    )

    folds = load_folds(root)
    label_df = load_labels(root)

    # Group labels per series, keep SOP to locate slice indices
    # Build map of series -> list of (SOPInstanceUID, x, y, cls_id, frame_idx)
    series_to_labels: Dict[str, List[Tuple[str, float, float, int, int | None]]] = {}
    for _, r in label_df.iterrows():
        if args.label_scheme == "locations":
            loc = r.get("location", None)
            if isinstance(loc, str) and loc in LABELS_TO_IDX:
                cls_id = int(LABELS_TO_IDX[loc])
            else:
                # Skip entries without a recognized location label
                continue
        else:
            # Binary: any annotation is class 0
            cls_id = 0
        # Frame index can be NA/None; convert to Python int or None
        f_val = r.get("f", None)
        try:
            f_idx = int(f_val) if pd.notna(f_val) else None
        except Exception:
            f_idx = None
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append(
            (str(r.SOPInstanceUID), float(r.x), float(r.y), cls_id, f_idx)
        )

    all_series: List[str] = []
    train_df_path = root / "train.csv"
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
    print(f"Processing {len(all_series)} series for per-slice YOLO dataset (val fold = {val_fold})...")

    for uid in all_series:
        if uid in ignore_uids:
            if args.verbose:
                print(f"[SKIP] ignored series {uid}")
            continue
        split = "val" if folds.get(uid, 0) == val_fold else "train"
        series_dir = root / "series" / uid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] series {uid}")
            continue

        paths, sop_to_idx = ordered_dcm_paths(series_dir)
        n_slices = len(paths)
        labels_for_series = series_to_labels.get(uid, [])

        # Per-slice generation (no MIP/BGR)
        if labels_for_series:
            pos_saved = 0
            # Track positive frames used per SOP to enable intra-DICOM adjacent negatives
            sop_to_pos_frames: Dict[str, Set[int]] = {}
            for (sop, x, y, cls_id, f_idx) in labels_for_series:
                dcm_path = root / "series" / uid / f"{sop}.dcm"
                if not dcm_path.exists():
                    continue
                
                if args.rgb_mode:
                    # RGB mode: get three adjacent slices
                    adjacent_slices = get_adjacent_slices_for_rgb(paths, sop, sop_to_idx)
                    if adjacent_slices is None:
                        continue
                    slice_prev, slice_curr, slice_next = adjacent_slices
                    img = create_rgb_from_slices(slice_prev, slice_curr, slice_next)
                    # For RGB, we need to handle 3D array (H, W, C)
                    if args.img_size > 0:
                        img_resized, (pad_h_ratio, pad_w_ratio) = letterbox(img, (args.img_size, args.img_size))
                        resize_w = resize_h = args.img_size
                    else:
                        img_resized = img
                        resize_h, resize_w = img_resized.shape[:2]
                        pad_h_ratio = pad_w_ratio = 0.0
                else:
                    # Original mode: single slice
                    frames = read_dicom_frames_hu(dcm_path)
                    if not frames:
                        continue
                    # Select correct frame for positives. Treat 'f' as 0-based; fall back to 1-based if needed.
                    frame_index = 0
                    if len(frames) > 1 and f_idx is not None:
                        if 0 <= f_idx < len(frames):
                            frame_index = f_idx
                        elif 1 <= f_idx <= len(frames):
                            frame_index = f_idx - 1
                    img = min_max_normalize(frames[frame_index])
                    if args.img_size > 0:
                        img_resized, (pad_h_ratio, pad_w_ratio) = letterbox(img, (args.img_size, args.img_size))
                        resize_w = resize_h = args.img_size
                    else:
                        img_resized = img
                        resize_h, resize_w = img_resized.shape
                        pad_h_ratio = pad_w_ratio = 0.0

                # Name images to include frame index for multi-frame DICOMs to avoid collisions
                if args.rgb_mode:
                    stem = f"{uid}_{sop}_rgb"
                elif len(frames) > 1:
                    stem = f"{uid}_{sop}_frame{frame_index}"
                else:
                    stem = f"{uid}_{sop}_slice"
                img_path = out_base / "images" / split / f"{stem}.{args.image_ext}"
                label_path = out_base / "labels" / split / f"{stem}.txt"
                if img_path.exists() and label_path.exists() and not args.overwrite:
                    # Still count as existing positive example for balancing if files already present
                    pos_saved += 1
                    continue
                cv2.imwrite(str(img_path), img_resized)

                # Scale point for letterbox resize
                if args.rgb_mode:
                    # For RGB mode, use the current slice shape as reference
                    orig_h, orig_w = slice_curr.shape
                else:
                    orig_h, orig_w = frames[frame_index].shape

                if args.img_size > 0:
                    # Calculate scaling factor to maintain aspect ratio
                    scale_ratio = min(args.img_size / orig_h, args.img_size / orig_w)
                    x_scaled = x * scale_ratio + pad_w_ratio * args.img_size
                    y_scaled = y * scale_ratio + pad_h_ratio * args.img_size
                else:
                    x_scaled, y_scaled = x, y
                xc, yc, bw, bh = build_box(x_scaled, y_scaled, args.box_size, resize_w, resize_h)
                with open(label_path, "w") as f:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                total_pos += 1
                total_series += 1
                pos_saved += 1
                # Track used positive frame indices per SOP
                if not args.rgb_mode and len(frames) > 1:
                    sop_to_pos_frames.setdefault(sop, set()).add(frame_index)

        else:
            # Negative series: choose evenly sampled slices and write empty labels
            if n_slices == 0:
                continue
            need = max(0, int(args.neg_per_series))
            
            # Collect all available frames from all DICOMs for proper even sampling
            all_frames_info = []  # List of (dcm_path, frame_idx, total_frames_in_dcm)
            for dcm_path in paths:
                frames = read_dicom_frames_hu(dcm_path)
                if frames:
                    for frame_idx in range(len(frames)):
                        all_frames_info.append((dcm_path, frame_idx, len(frames)))
            
            total_frames = len(all_frames_info)
            if total_frames == 0:
                continue
                
            # Sample up to 'need' evenly spaced frames across all available frames
            if need >= total_frames:
                # If we need more frames than available, take all
                pick = list(range(total_frames))
            else:
                pick = np.linspace(1, total_frames - 2, need, dtype=int).tolist()
                
            for frame_idx in pick:
                dcm_path, frame_num, total_frames_in_dcm = all_frames_info[frame_idx]
                
                if args.rgb_mode:
                    # RGB mode: get three adjacent slices for negative samples
                    # Find the index of this DICOM in the sorted paths
                    dcm_idx = None
                    for i, path in enumerate(paths):
                        if path == dcm_path:
                            dcm_idx = i
                            break
                    
                    if dcm_idx is None:
                        continue
                    
                    # Get adjacent slices
                    prev_idx = dcm_idx - 1
                    curr_idx = dcm_idx
                    next_idx = dcm_idx + 1
                    
                    slices = []
                    for idx in [prev_idx, curr_idx, next_idx]:
                        if 0 <= idx < len(paths):
                            frames = read_dicom_frames_hu(paths[idx])
                            if frames:
                                slices.append(frames[0])  # Take first frame
                            else:
                                break
                        else:
                            # Handle boundary cases
                            if idx < 0:
                                frames = read_dicom_frames_hu(paths[0])
                                if frames:
                                    slices.append(frames[0])
                                else:
                                    break
                            else:
                                frames = read_dicom_frames_hu(paths[-1])
                                if frames:
                                    slices.append(frames[0])
                                else:
                                    break
                    
                    if len(slices) == 3:
                        img = create_rgb_from_slices(slices[0], slices[1], slices[2])
                        if args.img_size > 0:
                            img, _ = letterbox(img, (args.img_size, args.img_size))
                        stem = f"{uid}_{dcm_path.stem}_rgb_neg"
                    else:
                        continue
                else:
                    # Original mode: single slice
                    frames = read_dicom_frames_hu(dcm_path)
                    if not frames or frame_num >= len(frames):
                        continue

                    img = min_max_normalize(frames[frame_num])
                    if args.img_size > 0:
                        img, _ = letterbox(img, (args.img_size, args.img_size))

                    # Name images to include frame index for multi-frame DICOMs to avoid collisions
                    if total_frames_in_dcm > 1:
                        stem = f"{uid}_{dcm_path.stem}_frame{frame_num}_neg"
                    else:
                        stem = f"{uid}_{dcm_path.stem}_slice_neg"
                    
                img_path = out_base / "images" / split / f"{stem}.{args.image_ext}"
                label_path = out_base / "labels" / split / f"{stem}.txt"
                if not (img_path.exists() and label_path.exists()) or args.overwrite:
                    cv2.imwrite(str(img_path), img)
                    open(label_path, "w").close()
                    total_neg += 1
                    total_series += 1

    print(f"Series processed: {total_series}  (positive label files: {total_pos}, negative: {total_neg})")
    print(f"YOLO dataset root: {out_base}")
    stats = {}
    for split in ["train", "val"]:
        n_imgs = len(list((out_base / "images" / split).glob("*")))
        n_lbls = len(list((out_base / "labels" / split).glob("*.txt")))
        print(f"{split}: images={n_imgs} labels={n_lbls}")
        stats[split] = {"images": n_imgs, "labels": n_lbls}
    return out_base, folds


def write_yolo_yaml(yaml_dir: Path, yaml_name: str, dataset_root: Path, label_scheme: str):
    yaml_dir.mkdir(parents=True, exist_ok=True)
    # Make paths relative to the workspace root
    relative_path = dataset_root.resolve().relative_to(ROOT)
    # Names section depends on labeling scheme
    names_lines: List[str] = []
    if label_scheme == "aneurysm_present":
        names_lines.append("  0: aneurysm_present")
    else:
        # Invert LABELS_TO_IDX to ensure correct index order (locations)
        idx_to_label = {idx: name for name, idx in LABELS_TO_IDX.items()}
        max_idx = max(idx_to_label.keys()) if idx_to_label else -1
        for i in range(max_idx + 1):
            label = idx_to_label.get(i, f"class_{i}")
            names_lines.append(f"  {i}: {label}")
    yaml_text = "\n".join(
        [
            f"path: {relative_path}",
            f"train: images/train",
            f"val: images/val",
            "",
            "names:",
            *names_lines,
            "",
        ]
    )
    with open(yaml_dir / yaml_name, "w") as f:
        f.write("# Auto-generated by prepare_yolo_dataset_v2.py\n")
        f.write(yaml_text)


if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(args.seed)
    if args.generate_all_folds:
        # Generate per-fold datasets and YAMLs
        for f in range(N_FOLDS):
            out_base, _ = generate_for_fold(f, args)
            yaml_dir = Path(args.yaml_out_dir)
            yaml_name = args.yaml_name_template.format(fold=f)
            write_yolo_yaml(yaml_dir, yaml_name, out_base, args.label_scheme)
        print(f"Generated datasets and YAMLs for folds 0..{N_FOLDS-1}.")
        print(f"YAMLs written under: {args.yaml_out_dir}")
    else:
        out_base, _ = generate_for_fold(args.val_fold, args)
        print("Done. To train, point Ultralytics to a YAML like:")
        print(f"  path: {out_base}")
        print("  train: images/train\n  val: images/val")


#Locations (multiclass, with letterbox resize):
#Example: python3 -m src.prepare_yolo_dataset_v4_25d_letterbox --generate-all-folds --out-name yolo_dataset --img-size 512 --label-scheme locations --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite

#RGB mode (3-channel from adjacent slices, with letterbox resize):
#Example: python3 -m src.prepare_yolo_dataset_v4_25d_letterbox --generate-all-folds --out-name yolo_dataset_rgb --img-size 512 --label-scheme locations --rgb-mode --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite
