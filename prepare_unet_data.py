import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold

# Allow importing project configs
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, SEED


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float16)
    norm = (img - mn) / (mx - mn)
    return norm.astype(np.float16)


def create_ball_mask(shape: Tuple[int, int, int], center_z: int, center_y: int, center_x: int) -> np.ndarray:
    """Create a simple 3D ball mask."""
    mask = np.zeros(shape, dtype=np.float32)
    depth, height, width = shape
    
    radius = 6
    z_extent = 2
    
    for z in range(max(0, center_z - z_extent), min(depth, center_z + z_extent + 1)):
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                dz, dy, dx = z - center_z, y - center_y, x - center_x
                if np.sqrt(dx**2 + dy**2 + (dz*2)**2) <= radius:
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

def load_folds(root: Path, n_folds: int) -> Dict[str, int]:
    """Create stratified folds from train_df.csv."""
    df = pd.read_csv(root / "train_df.csv")
    series_df = df[["SeriesInstanceUID", "Aneurysm Present"]].drop_duplicates()
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_map = {}
    for i, (_, test_idx) in enumerate(skf.split(series_df["SeriesInstanceUID"], series_df["Aneurysm Present"])):
        for uid in series_df.loc[test_idx, "SeriesInstanceUID"]:
            fold_map[uid] = i
    
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


def ensure_dirs(base: Path):
    for sub in ["train", "test"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def process_series(uid: str, root: Path, out_base: Path, folds: Dict[str, int], 
                  series_to_labels: Dict[str, List[Tuple[str, float, float]]], 
                  test_fold: int, target_shape: Tuple[int, int, int], overwrite: bool) -> int:
    """Process a single series."""
    split = "test" if folds.get(uid, 0) == test_fold else "train"
    series_dir = root / "series" / uid
    
    if not series_dir.exists():
        return 0

    paths, sop_to_idx = ordered_dcm_paths(series_dir)
    if not paths:
        return 0

    # Load slices
    slices = []
    for path in paths:
        frames = read_dicom_frames_hu(path)
        if frames:
            slices.append(min_max_normalize(frames[0]))
    
    if not slices:
        return 0

    volume = np.stack(slices, axis=0).astype(np.float32)
    n_slices, h, w = volume.shape
    mask = np.zeros((n_slices, h, w), dtype=np.float32)

    # Add annotations
    for sop, x, y in series_to_labels.get(uid, []):
        idx = sop_to_idx.get(sop)
        if idx is not None:
            y_int, x_int = int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))
            ball_mask = create_ball_mask((n_slices, h, w), idx, y_int, x_int)
            mask = np.maximum(mask, ball_mask)

    # Resize
    zoom_factors = (target_shape[0]/n_slices, target_shape[1]/h, target_shape[2]/w)
    volume_resized = zoom(volume, zoom_factors, order=1)
    mask_resized = zoom(mask, zoom_factors, order=1)
    mask_resized = (mask_resized > 0.5).astype(np.float32)

    # Convert types
    volume_resized = (volume_resized * 255).astype(np.uint8)
    mask_resized = mask_resized.astype(np.float16)

    # Save
    out_path = out_base / split / f"{uid}.npz"
    if out_path.exists() and not overwrite:
        return 0
    
    np.savez_compressed(out_path, volume=volume_resized, mask=mask_resized)
    print(f"Saved {uid} to {split}")
    return 1

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare nnU-Net dataset")
    ap.add_argument("--dataset-id", type=int, required=True, help="nnU-Net dataset ID")
    ap.add_argument("--dataset-name", type=str, required=True, help="nnU-Net dataset name")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--workers", type=int, default=4, help="Number of workers")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[48, 384, 384], help="Target volume shape (D H W)")
    ap.add_argument("--fold-as-test", type=int, default=0, help="Fold to use as test set")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(data_path)
    out_base = root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    ensure_dirs(out_base)
    
    # Load data
    folds = load_folds(root, 5)
    label_df = load_labels(root)
    
    # Group labels
    series_to_labels = {}
    for _, r in label_df.iterrows():
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append(
            (str(r.SOPInstanceUID), float(r.x), float(r.y))
        )
    
    # Get all series
    train_csv = pd.read_csv(root / "train_df.csv")
    all_series = train_csv["SeriesInstanceUID"].astype(str).unique().tolist()
    
    print(f"Processing {len(all_series)} series...")
    
    # Process series
    total = 0
    for uid in all_series:
        total += process_series(uid, root, out_base, folds, series_to_labels, 
                              args.fold_as_test, tuple(args.target_shape), args.overwrite)
    
    print(f"Processed {total} series")
    print(f"Output: {out_base}")

if __name__ == "__main__":
    main()