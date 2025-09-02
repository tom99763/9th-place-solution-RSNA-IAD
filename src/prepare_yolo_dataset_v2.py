
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
    # Image/output options (kept name for backward compatibility)
    ap.add_argument("--mip-img-size", type=int, default=512, help="Final square image size for saved samples (0 = keep original)")
    ap.add_argument("--mip-out-name", type=str, default="yolo_dataset", help="Base subdirectory under data/ for outputs; when --generate-all-folds, becomes {base}_fold{fold}")
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
        default=1,
        help="Number of negative slices to sample per series without positives (fallback when pos-neg-ratio=0).",
    )
    ap.add_argument(
        "--pos-neg-ratio",
        type=float,
        default=0,
        help=(
            "Negatives per positive for positive series (sampled from the same series). "
            "E.g., 2.0 adds ~2 negative slices for each positive slice (capped by available slices)."
        ),
    )
    ap.add_argument(
        "--use-adjacent-negatives",
        action="store_true",
        help="Add adjacent slices (positive ±2) as negative examples to help model distinguish aneurysms from normal vessels",
    )
    ap.add_argument(
        "--adjacent-offset",
        type=int,
        default=0,
        help="Offset for adjacent negative slices (default: ±2 slices from positive)",
    )
    # YAML outputs
    ap.add_argument("--yaml-out-dir", type=str, default=str(ROOT / "configs"), help="Directory to write per-fold YOLO YAMLs")
    ap.add_argument("--yaml-name-template", type=str, default="yolo_fold{fold}.yaml", help="YAML filename template with {fold}")
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
    
    # Update train_df.csv with the new folds
    df["fold_id"] = df["SeriesInstanceUID"].astype(str).map(fold_map)
    df.to_csv(df_path, index=False)
    print(f"Updated {df_path} with new fold_id column based on N_FOLDS={N_FOLDS}")
    
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


def generate_for_fold(val_fold: int, args) -> Tuple[Path, Dict[str, int]]:
    """Generate per-slice dataset for a single fold and return (out_base, fold_map)."""
    global rng
    root = Path(data_path)

    # Resolve output base for this fold
    out_dir_name = args.mip_out_name if not hasattr(args, 'generate_all_folds') or not args.generate_all_folds else f"{args.mip_out_name}_fold{val_fold}"
    out_base = root / out_dir_name
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
    # Build map of series -> list of (SOPInstanceUID, x, y, cls_id)
    series_to_labels: Dict[str, List[Tuple[str, float, float, int]]] = {}
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
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append(
            (str(r.SOPInstanceUID), float(r.x), float(r.y), cls_id)
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
            for (sop, x, y, cls_id) in labels_for_series:
                dcm_path = root / "series" / uid / f"{sop}.dcm"
                if not dcm_path.exists():
                    continue
                frames = read_dicom_frames_hu(dcm_path)
                if not frames:
                    continue
                img = min_max_normalize(frames[0])
                if args.mip_img_size > 0 and (
                    img.shape[0] != args.mip_img_size or img.shape[1] != args.mip_img_size
                ):
                    img_resized = cv2.resize(img, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
                    resize_w = resize_h = args.mip_img_size
                else:
                    img_resized = img
                    resize_h, resize_w = img_resized.shape

                stem = f"{uid}_{sop}_slice"
                img_path = out_base / "images" / split / f"{stem}.{args.image_ext}"
                label_path = out_base / "labels" / split / f"{stem}.txt"
                if img_path.exists() and label_path.exists() and not args.overwrite:
                    # Still count as existing positive example for balancing if files already present
                    pos_saved += 1
                    continue
                cv2.imwrite(str(img_path), img_resized)

                # Scale point if resized
                orig_h, orig_w = frames[0].shape
                if (resize_w, resize_h) != (orig_w, orig_h):
                    x_scaled = x * (resize_w / orig_w)
                    y_scaled = y * (resize_h / orig_h)
                else:
                    x_scaled, y_scaled = x, y
                xc, yc, bw, bh = build_box(x_scaled, y_scaled, args.box_size, resize_w, resize_h)
                with open(label_path, "w") as f:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                total_pos += 1
                total_series += 1
                pos_saved += 1

            # Sample adjacent negatives if requested
            adjacent_neg_saved = 0
            if args.use_adjacent_negatives and pos_saved > 0:
                pos_indices: Set[int] = set()
                for (sop, _x, _y, _cid) in labels_for_series:
                    idx = sop_to_idx.get(sop)
                    if idx is not None:
                        pos_indices.add(idx)
                
                # For each positive slice, add adjacent slices as negatives
                for pos_idx in pos_indices:
                    for offset in [-args.adjacent_offset, args.adjacent_offset]:
                        adj_idx = pos_idx + offset
                        # Check bounds and ensure not a positive slice
                        if 0 <= adj_idx < n_slices and adj_idx not in pos_indices:
                            dcm_path = paths[adj_idx]
                            try:
                                frames = read_dicom_frames_hu(dcm_path)
                                if not frames:
                                    continue
                                img = min_max_normalize(frames[0])
                                if args.mip_img_size > 0 and (
                                    img.shape[0] != args.mip_img_size or img.shape[1] != args.mip_img_size
                                ):
                                    img = cv2.resize(img, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
                                stem = f"{uid}_{dcm_path.stem}_slice_adj_neg_{offset:+d}"
                                img_path = out_base / "images" / split / f"{stem}.{args.image_ext}"
                                label_path = out_base / "labels" / split / f"{stem}.txt"
                                if (img_path.exists() and label_path.exists()) and not args.overwrite:
                                    continue
                                cv2.imwrite(str(img_path), img)
                                open(label_path, "w").close()
                                total_neg += 1
                                total_series += 1
                                adjacent_neg_saved += 1
                                if args.verbose:
                                    print(f"  Added adjacent negative: slice {adj_idx} (offset {offset:+d} from positive {pos_idx})")
                            except Exception as e:
                                if args.verbose:
                                    print(f"  Failed to process adjacent slice {adj_idx}: {e}")
                                continue

            # Sample additional random negatives from this positive series if requested
            if args.pos_neg_ratio > 0 and n_slices > 0 and pos_saved > 0:
                # Candidate indices are slices without a positive label
                pos_indices: Set[int] = set()
                for (sop, _x, _y, _cid) in labels_for_series:
                    idx = sop_to_idx.get(sop)
                    if idx is not None:
                        pos_indices.add(idx)
                
                # Exclude already used adjacent negatives if applicable
                used_adj_indices: Set[int] = set()
                if args.use_adjacent_negatives:
                    for pos_idx in pos_indices:
                        for offset in [-args.adjacent_offset, args.adjacent_offset]:
                            adj_idx = pos_idx + offset
                            if 0 <= adj_idx < n_slices:
                                used_adj_indices.add(adj_idx)
                
                candidates = [i for i in range(n_slices) if i not in pos_indices and i not in used_adj_indices]
                if candidates:
                    need = int(math.ceil(args.pos_neg_ratio * pos_saved))
                    # Avoid oversampling beyond available slices
                    k = min(need, len(candidates))
                    rng.shuffle(candidates)
                    pick = candidates[:k]
                    for idx in pick:
                        dcm_path = paths[idx]
                        frames = read_dicom_frames_hu(dcm_path)
                        if not frames:
                            continue
                        img = min_max_normalize(frames[0])
                        if args.mip_img_size > 0 and (
                            img.shape[0] != args.mip_img_size or img.shape[1] != args.mip_img_size
                        ):
                            img = cv2.resize(img, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
                        stem = f"{uid}_{dcm_path.stem}_slice_neg"
                        img_path = out_base / "images" / split / f"{stem}.{args.image_ext}"
                        label_path = out_base / "labels" / split / f"{stem}.txt"
                        if (img_path.exists() and label_path.exists()) and not args.overwrite:
                            continue
                        cv2.imwrite(str(img_path), img)
                        open(label_path, "w").close()
                        total_neg += 1
                        total_series += 1
            
            if args.verbose and adjacent_neg_saved > 0:
                print(f"  Series {uid}: Added {adjacent_neg_saved} adjacent negative slices")
        else:
            # Negative series: choose one random slice and write empty label
            if n_slices == 0:
                continue
            need = max(0, int(args.neg_per_series))
            # Sample up to 'need' unique slices
            indices = list(range(n_slices))
            rng.shuffle(indices)
            pick = indices[: min(need, n_slices)]
            for idx in pick:
                dcm_path = paths[idx]
                frames = read_dicom_frames_hu(dcm_path)
                if not frames:
                    continue
                img = min_max_normalize(frames[0])
                if args.mip_img_size > 0 and (
                    img.shape[0] != args.mip_img_size or img.shape[1] != args.mip_img_size
                ):
                    img = cv2.resize(img, (args.mip_img_size, args.mip_img_size), interpolation=cv2.INTER_LINEAR)
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
            f"train: {relative_path}/images/train",
            f"val: {relative_path}/images/val",
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


#Locations (multiclass, original behavior):
#Example: python3 -m src.prepare_yolo_dataset_v2 --generate-all-folds --mip-out-name yolo_dataset --mip-img-size 512 --label-scheme locations --yaml-out-dir configs --yaml-name-template yolo_fold{fold}.yaml --overwrite
#Binary (single class "aneurysm_present"):
#Example: python3 -m src.prepare_yolo_dataset_v2 --generate-all-folds --mip-out-name yolo_dataset_bin --mip-img-size 512 --label-scheme aneurysm_present --yaml-out-dir configs --yaml-name-template yolo_bin_fold{fold}.yaml --overwrite
#Binary with adjacent negatives (helps distinguish aneurysms from normal vessels):
#Example: python3 -m src.prepare_yolo_dataset_v2 --generate-all-folds --mip-out-name yolo_dataset_bin_adj --mip-img-size 512 --label-scheme aneurysm_present --use-adjacent-negatives --adjacent-offset 2 --yaml-out-dir configs --yaml-name-template yolo_bin_fold{fold}.yaml --overwrite

#python3 -m src.prepare_yolo_dataset_v2 --val-fold 1 --mip-out-name yolo_dataset_hard_negative_fold_1 --label-scheme aneurysm_present --yaml-out-dir configs --yaml-name-template yolo_bin_fold1_hard_negatives.yaml --overwrite --neg-per-series 3 --pos-neg-ratio 3.0 --use-adjacent-negativ
#es --adjacent-offset 2 --mip-img-size 512 --image-ext png --verbose


#python3 -m src.prepare_yolo_dataset_v2 \
#  --generate-all-folds \
#  --mip-out-name yolo_dataset_positive_adj_neg \
#  --mip-img-size 512 \
#  --label-scheme aneurysm_present \
#  --use-adjacent-negatives \
#  --adjacent-offset 2 \
#  --neg-per-series 0 \
#  --pos-neg-ratio 0 \
#  --yaml-out-dir configs \
#  --yaml-name-template yolo_bin_fold{fold}.yaml \
#  --overwrite \
#  --verbose

