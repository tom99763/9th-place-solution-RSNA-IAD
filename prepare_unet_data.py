import argparse
import ast
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import multiprocessing

import cv2
import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold

# Allow importing project configs
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, N_FOLDS, SEED, LABELS_TO_IDX  # type: ignore

N_FOLDS = 5
rng = random.Random(SEED)


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float16)
    norm = (img - mn) / (mx - mn)
    return norm.astype(np.float16)


def create_hard_ball(shape: Tuple[int, int, int], center_z: int, center_y: int, center_x: int, 
                    original_shape: Tuple[int, int, int]) -> np.ndarray:
    """Create a hard 3D ball mask with dynamic sizing.
    
    Args:
        shape: (depth, height, width) of the volume
        center_z, center_y, center_x: Center coordinates of the ball
        original_shape: Original volume shape before any resizing for dynamic scaling
    
    Returns:
        3D binary mask with 0/1 values
    """
    mask = np.zeros(shape, dtype=np.float32)
    depth, height, width = shape
    orig_depth, orig_height, orig_width = original_shape
    
    # Dynamic radius based on original image size
    # Scale radius proportionally to image dimensions (you can adjust these factors)
    base_radius = 6  # Base radius for reference
    radius_x = max(6, int(base_radius * orig_width / 512))  # Assuming 512 as reference width
    radius_y = max(6, int(base_radius * orig_height / 512))  # Assuming 512 as reference height
    
    # Dynamic z-extent based on original depth
    # For shallow volumes, use fewer slices; for deep volumes, use more
    if orig_depth <= 10:
        z_extent = 1  # Only current slice
    elif orig_depth <= 30:
        z_extent = 2  # ±1 slice
    elif orig_depth <= 90:
        z_extent = 3  # ±1 slice
    else:
        z_extent = 4  # Scale with depth

    # Define the extent of the ball
    z_min = max(0, center_z - z_extent)
    z_max = min(depth, center_z + z_extent + 1)
    
    y_min = max(0, center_y - radius_y)
    y_max = min(height, center_y + radius_y + 1)
    
    x_min = max(0, center_x - radius_x)
    x_max = min(width, center_x + radius_x + 1)
    
    # Create hard ball mask
    for z in range(z_min, z_max):
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # Calculate 3D distance from center
                dz = z - center_z
                dy = y - center_y  
                dx = x - center_x
                
                # Use ellipsoidal distance with different scaling for each axis
                # Normalize by the respective radii
                distance = np.sqrt((dx/radius_x)**2 + (dy/radius_y)**2 + (dz/z_extent)**2)
                
                # Hard threshold: inside the ball = 1, outside = 0
                if distance <= 1.0:
                    mask[z, y, x] = 1.0
    
    return mask


def load_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in label_df.columns or "y" not in label_df.columns:
        label_df["x"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        label_df["y"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    # Standardize dtypes
    label_df["SeriesInstanceUID"] = label_df["SeriesInstanceUID"].astype(str)
    label_df["SOPInstanceUID"] = label_df["SOPInstanceUID"].astype(str)
    return label_df

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


def ensure_unet_dirs(base: Path):
    for sub in ["train", "val"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def process_series(args_tuple):
    uid, root, out_base, folds, series_to_labels, ignore_uids, args = args_tuple
    if uid in ignore_uids:
        if args.verbose:
            print(f"[SKIP] ignored series {uid}")
        return 0
    split = "val" if folds.get(uid, 0) == args.val_fold else "train"
    series_dir = root / "series" / uid
    if not series_dir.exists():
        if args.verbose:
            print(f"[MISS] series {uid}")
        return 0

    paths, sop_to_idx = ordered_dcm_paths(series_dir)
    n_slices = len(paths)
    if n_slices == 0:
        return 0

    # Load all slices
    slices = []
    for path in paths:
        frames = read_dicom_frames_hu(path)
        if frames:
            slices.append(min_max_normalize(frames[0]))

    if not slices:
        return 0

    # Stack into 3D volume
    volume = np.stack(slices, axis=0)  # (n_slices, h, w)
    volume = volume.astype(np.float32)
    
    # Store original shape for dynamic sizing
    original_shape = volume.shape  # (n_slices, h, w)

    # Create 3D mask with hard balls
    h, w = volume.shape[1], volume.shape[2]
    mask = np.zeros((n_slices, h, w), dtype=np.float32)

    if not args.no_masks:
        labels_for_series = series_to_labels.get(uid, [])
        for sop, x, y in labels_for_series:
            idx = sop_to_idx.get(sop)
            if idx is not None:
                # Ensure coordinates are within bounds
                y_int, x_int = int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))
                
                # Create hard ball centered at this point with dynamic sizing
                ball_mask = create_hard_ball(
                    shape=(n_slices, h, w),
                    center_z=idx,
                    center_y=y_int, 
                    center_x=x_int,
                    original_shape=original_shape  # Pass original shape for dynamic sizing
                )
                
                # Add to existing mask (in case of overlapping annotations)
                mask = np.maximum(mask, ball_mask)

    # Resize to 384x384x32
    target_shape = (32, 384, 384)
    volume_resized = zoom(volume, (
        target_shape[0] / volume.shape[0], 
        target_shape[1] / volume.shape[1], 
        target_shape[2] / volume.shape[2]
    ), order=1)
    mask_resized = zoom(mask, (
        target_shape[0] / mask.shape[0], 
        target_shape[1] / mask.shape[1], 
        target_shape[2] / mask.shape[2]
    ), order=1)
    
    # For hard labels, threshold the resized mask to maintain binary nature
    mask_resized = (mask_resized > 0.5).astype(np.float32)

    # Convert back to appropriate dtypes for saving
    volume_resized = (volume_resized * 255).astype(np.uint8)
    mask_resized = mask_resized.astype(np.float16)

    # Save as NPZ
    out_path = out_base / split / f"{uid}.npz"
    if out_path.exists() and not args.overwrite:
        if args.verbose:
            print(f"[SKIP] {out_path} exists")
        return 0
    
    np.savez_compressed(out_path, volume=volume_resized, mask=mask_resized)
    if args.verbose:
        print(f"Saved {uid} to {split}")
    return 1

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare U-Net dataset with 3D volumes and simple Gaussian ball masks")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    ap.add_argument("--val-fold", type=int, default=0, help="Fold id used for validation (ignored if --generate-all-folds)")
    ap.add_argument("--generate-all-folds", action="store_true", help="Generate one dataset per fold")
    ap.add_argument("--unet-out-name", type=str, default="unet_dataset_v2", help="Base subdirectory under data/ for outputs; when --generate-all-folds, becomes {base}_fold{fold}")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--verbose", default=True, action="store_true")
    ap.add_argument("--no-masks", default=True, action="store_true", help="Skip generating mask annotations")
    return ap.parse_args()


def generate_for_fold(val_fold: int, args) -> Tuple[Path, Dict[str, int]]:
    """Generate U-Net dataset for a single fold and return (out_base, fold_map)."""
    global rng
    root = Path(data_path)

    # Resolve output base for this fold
    out_dir_name = args.unet_out_name if not hasattr(args, 'generate_all_folds') or not args.generate_all_folds else f"{args.unet_out_name}_fold{val_fold}"
    out_base = root / out_dir_name
    ensure_unet_dirs(out_base)

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

    # Group labels per series
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
    print(f"Processing {len(all_series)} series for U-Net dataset (val fold = {val_fold})...")

    # Filter series to process
    series_to_process = [uid for uid in all_series if uid not in ignore_uids]

    # Prepare arguments for multiprocessing
    args_tuples = [(uid, root, out_base, folds, series_to_labels, ignore_uids, args) for uid in series_to_process]

    # Use multiprocessing to process series in parallel
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_series, args_tuples)

    total_series = sum(results)

    print(f"Series processed: {total_series}")
    print(f"U-Net dataset root: {out_base}")
    return out_base, folds

if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(args.seed)
    if args.generate_all_folds:
        for f in range(N_FOLDS):
            out_base, _ = generate_for_fold(f, args)
        print(f"Generated datasets for folds 0..{N_FOLDS-1}.")
    else:
        out_base, _ = generate_for_fold(args.val_fold, args)
        print("Done.")