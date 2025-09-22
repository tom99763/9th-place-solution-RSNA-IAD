import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import gc
import pandas as pd
import pydicom
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def create_gaussian_ball_mask(shape: Tuple[int, int, int], center: Tuple[int, int, int], radius: int = 6) -> np.ndarray:
    """Create a 3D Gaussian ball mask."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    
    # Create distance array from center
    distance = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    
    # Gaussian mask with sigma = radius/3 for smooth falloff
    sigma = radius / 3.0
    mask = np.exp(-(distance**2) / (2 * sigma**2))
    
    # Clip to radius
    mask[distance > radius] = 0
    
    return mask.astype(np.float32)


def load_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in label_df.columns or "y" not in label_df.columns:
        label_df["x"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        label_df["y"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    # Standardize dtypes
    label_df["SeriesInstanceUID"] = label_df["SeriesInstanceUID"].astype(str)
    label_df["SOPInstanceUID"] = label_df["SOPInstanceUID"].astype(str)
    return label_df

def load_folds(root: Path, n_folds: int = 5) -> Dict[str, int]:
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

def plot_examples(volumes: List[np.ndarray], masks: List[np.ndarray], series_ids: List[str], labels: List[int], output_dir: Path):
    """Plot some example cubes with masks."""
    output_dir.mkdir(exist_ok=True)
    
    for i, (vol, mask, sid, label) in enumerate(zip(volumes[:20], masks[:20], series_ids[:20], labels[:20])):
        fig = plt.figure(figsize=(15, 5))
        
        # Central slice views
        mid_z = vol.shape[0] // 2
        
        # Volume slice
        ax1 = fig.add_subplot(131)
        ax1.imshow(vol[mid_z], cmap='gray')
        ax1.set_title(f'{"Positive" if label else "Negative"} {sid}\nSlice {mid_z}')
        ax1.axis('off')
        
        # Mask slice
        ax2 = fig.add_subplot(132)
        ax2.imshow(mask[mid_z], cmap='hot', alpha=0.7)
        ax2.set_title(f'Mask\nSlice {mid_z}')
        ax2.axis('off')
        
        # Overlay
        ax3 = fig.add_subplot(133)
        ax3.imshow(vol[mid_z], cmap='gray')
        ax3.imshow(mask[mid_z], cmap='hot', alpha=0.5)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'example_{i:02d}_{sid}_{"pos" if label else "neg"}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved example plots to {output_dir}")


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




def resample_isotropic(vol: np.ndarray, spacing: Tuple[float, float, float], target: float = 1.0) -> np.ndarray:
    """Resample volume to isotropic voxels."""
    sx, sy, sz = spacing
    zoom = (sz / target, sy / target, sx / target)  # Z, Y, X order
    return ndimage.zoom(vol, zoom, order=1, prefilter=False)

def resample_to_patch_size(vol: np.ndarray, target_shape: Tuple[int, int, int] = (128, 384, 384)) -> np.ndarray:
    """Resample volume to target shape (Z, Y, X order)."""
    current_shape = vol.shape
    zoom = (target_shape[0] / current_shape[0], 
            target_shape[1] / current_shape[1], 
            target_shape[2] / current_shape[2])
    return ndimage.zoom(vol, zoom, order=1, prefilter=False)


def get_spacing_from_paths(paths: List[Path]) -> Tuple[float, float, float]:
    """Extract spacing information from DICOM files."""
    if not paths:
        return (1.0, 1.0, 1.0)
    
    try:
        ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
        pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
        spacing_z = float(getattr(ds, "SpacingBetweenSlices", slice_thickness))
        return (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)
    except Exception:
        return (1.0, 1.0, 1.0)


def extract_aneurysm_cubes(uid: str, root: Path, series_to_labels: Dict[str, List[Tuple[str, float, float]]], 
                          cube_shape: Tuple[int, int, int] = (32, 32, 32)) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract cubes of specified shape centered on aneurysms."""
    series_dir = root / "series" / uid
    if not series_dir.exists() or uid not in series_to_labels:
        return []

    paths, sop_to_idx = ordered_dcm_paths(series_dir)
    if not paths:
        return []

    # Load and normalize slices
    slices = []
    for path in paths:
        frames = read_dicom_frames_hu(path)
        if frames:
            slices.append(min_max_normalize(frames[0]))
    
    if not slices:
        return []

    volume = np.stack(slices, axis=0).astype(np.float32)
    # Free slice list ASAP
    del slices
    spacing = get_spacing_from_paths(paths)
    
    # Store original shape for scaling calculations
    orig_shape = volume.shape
    
    # Resample to isotropic (1mm) voxels
    volume_iso = resample_to_patch_size(volume, target_shape=(128, 384, 384))
    n_slices_iso, h_iso, w_iso = volume_iso.shape
    
    # Free original volume memory immediately
    del volume
    
    # Calculate scaling factors
    sx, sy, sz = spacing
    scale_x = w_iso / orig_shape[2]
    scale_y = h_iso / orig_shape[1]  
    scale_z = n_slices_iso / orig_shape[0]
    
    cubes = []
    d, h, w = cube_shape
    half_d, half_h, half_w = d // 2, h // 2, w // 2
    
    # Extract cube for each aneurysm
    for sop, x, y in series_to_labels[uid]:
        idx = sop_to_idx.get(sop)
        if idx is None:
            continue
            
        # Transform to isotropic space
        x_iso = int(x * scale_x)
        y_iso = int(y * scale_y)
        z_iso = int(idx * scale_z)
        
        # Extract cube bounds
        z_start = max(0, z_iso - half_d)
        z_end = min(n_slices_iso, z_iso + half_d)
        y_start = max(0, y_iso - half_h)
        y_end = min(h_iso, y_iso + half_h)
        x_start = max(0, x_iso - half_w)
        x_end = min(w_iso, x_iso + half_w)
        
        # Extract and pad to exact size
        cube = volume_iso[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if needed
        pad_z = (max(0, half_d - z_iso), max(0, z_iso + half_d - n_slices_iso))
        pad_y = (max(0, half_h - y_iso), max(0, y_iso + half_h - h_iso))
        pad_x = (max(0, half_w - x_iso), max(0, x_iso + half_w - w_iso))
        
        if any(p[0] > 0 or p[1] > 0 for p in [pad_z, pad_y, pad_x]):
            cube = np.pad(cube, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
        
        # Ensure exact size
        cube = cube[:d, :h, :w]
        # IMPORTANT: Detach from volume_iso to avoid keeping large base array in memory
        cube = np.ascontiguousarray(cube).astype(np.float16, copy=False)
        
        # Create Gaussian mask at center
        center = (d//2, h//2, w//2)
        mask = create_gaussian_ball_mask(cube_shape, center, radius=6).astype(np.float16)
        
        cubes.append((cube, mask))
    
    # Free large resampled volume before returning
    del volume_iso
    gc.collect()
    return cubes

def extract_negative_cubes(uid: str, root: Path, cube_shape: Tuple[int, int, int] = (32, 128, 128), 
                          num_patches: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract random patches from negative series (no aneurysms)."""
    series_dir = root / "series" / uid
    if not series_dir.exists():
        return []

    paths, _ = ordered_dcm_paths(series_dir)
    if not paths:
        return []

    # Load and normalize slices
    slices = []
    for path in paths:
        frames = read_dicom_frames_hu(path)
        if frames:
            slices.append(min_max_normalize(frames[0]))
    
    if not slices:
        return []

    volume = np.stack(slices, axis=0).astype(np.float32)
    # Free slice list ASAP
    del slices
    spacing = get_spacing_from_paths(paths)
    
    # Store original shape for scaling calculations  
    orig_shape = volume.shape
    
    # Resample to isotropic (1mm) voxels
    volume_iso = resample_to_patch_size(volume, target_shape=(128, 384, 384))
    n_slices_iso, h_iso, w_iso = volume_iso.shape
    
    # Free original volume memory immediately
    del volume
    
    cubes = []
    d, h, w = cube_shape
    half_d, half_h, half_w = d // 2, h // 2, w // 2
    
    # Extract random patches
    np.random.seed(SEED + hash(uid) % 10000)  # Consistent but different per series
    
    for _ in range(num_patches):
        # Random center coordinates (avoid edges)
        z_center = np.random.randint(half_d, n_slices_iso - half_d) if n_slices_iso > d else n_slices_iso // 2
        y_center = np.random.randint(half_h, h_iso - half_h) if h_iso > h else h_iso // 2
        x_center = np.random.randint(half_w, w_iso - half_w) if w_iso > w else w_iso // 2
        
        # Extract cube bounds
        z_start = max(0, z_center - half_d)
        z_end = min(n_slices_iso, z_center + half_d)
        y_start = max(0, y_center - half_h)
        y_end = min(h_iso, y_center + half_h)
        x_start = max(0, x_center - half_w)
        x_end = min(w_iso, x_center + half_w)
        
        # Extract and pad to exact size
        cube = volume_iso[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if needed
        pad_z = (max(0, half_d - z_center), max(0, z_center + half_d - n_slices_iso))
        pad_y = (max(0, half_h - y_center), max(0, y_center + half_h - h_iso))
        pad_x = (max(0, half_w - x_center), max(0, x_center + half_w - w_iso))
        
        if any(p[0] > 0 or p[1] > 0 for p in [pad_z, pad_y, pad_x]):
            cube = np.pad(cube, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
        
        # Ensure exact size
        cube = cube[:d, :h, :w]
        # IMPORTANT: Detach from volume_iso to avoid keeping large base array in memory
        cube = np.ascontiguousarray(cube).astype(np.float16, copy=False)
        
        # Create empty mask for negative samples
        mask = np.zeros(cube_shape, dtype=np.float16)
        
        cubes.append((cube, mask))
    
    # Free large resampled volume before returning
    del volume_iso
    gc.collect()
    return cubes

def parse_args():
    ap = argparse.ArgumentParser(description="Extract aneurysm cubes with folds")
    ap.add_argument("--output-dir", default="aneurysm_cubes_v2", help="Output directory")
    ap.add_argument("--cube-shape", nargs=3, type=int, default=[128, 384, 384], 
                    help="Cube dimensions as D H W (default: 128 384 384)")
    ap.add_argument("--max-positive", type=int, default=3000, help="Max positive samples to process")
    ap.add_argument("--max-negative", type=int, default=3000, help="Max negative samples to process")
    ap.add_argument("--neg-patches-per-series", type=int, default=1, help="Number of random patches per negative series")
    ap.add_argument("--no-examples", action="store_true", help="Skip generating example plots to save memory")
    ap.add_argument("--batch-size", type=int, default=100, help="Process in batches to control memory usage")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create fold directories
    for fold in range(5):
        (output_dir / f"fold_{fold}").mkdir(exist_ok=True)
    
    # Load folds and train data
    folds = load_folds(root, n_folds=5)
    train_df = pd.read_csv(root / "train_df.csv")
    train_df["SeriesInstanceUID"] = train_df["SeriesInstanceUID"].astype(str)
    # Get positive and negative series
    positive_series = train_df[train_df["Aneurysm Present"] == 1]["SeriesInstanceUID"].unique()
    negative_series = train_df[train_df["Aneurysm Present"] == 0]["SeriesInstanceUID"].unique()
    
    # Load labels for positive series
    label_df = load_labels(root)
    series_to_labels = {}
    for _, r in label_df.iterrows():
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append(
            (str(r.SOPInstanceUID), float(r.x), float(r.y))
        )
    
    print(f"Found {len(positive_series)} positive series, {len(negative_series)} negative series")
    
    # Process samples
    all_volumes, all_masks, all_series_ids, all_labels = [], [], [], []
    cube_count = 0
    
    # Memory management settings
    store_examples = not args.no_examples
    MAX_EXAMPLES = 50 if store_examples else 0
    
    # Process positive samples (all aneurysms)
    processed_positive = 0
    for uid in positive_series:
        if processed_positive >= args.max_positive:
            break
            
        if uid not in series_to_labels:
            continue
            
        # Determine fold first for early skip check
        fold = folds.get(uid, 0)
        # Early skip: if all expected files for this series already exist, avoid heavy processing
        expected_pos = len(series_to_labels.get(uid, []))
        if expected_pos > 0:
            fold_dir = output_dir / f"fold_{fold}"
            existing = list(fold_dir.glob(f"pos_{uid}_*.npz"))
            if len(existing) >= expected_pos:
                # Already processed this series
                continue

        cubes = extract_aneurysm_cubes(uid, root, series_to_labels, tuple(args.cube_shape))
        
        # Skip if no cubes extracted
        if not cubes:
            continue
        
        for i, (volume, mask) in enumerate(cubes):
            cube_id = f"pos_{uid}_{i:02d}"
            output_file = output_dir / f"fold_{fold}" / f"{cube_id}.npz"
            
            # Skip if file already exists
            if output_file.exists():
                print(f"Skipping existing positive file: {cube_id}")
                continue
            
            # Save to fold directory
            np.savez_compressed(
                output_file, 
                volume=(volume * 255).astype(np.uint8), 
                mask=mask.astype(np.float16),
                label=1,
                fold=fold
            )
            
            # Store for plotting (limited to save memory)
            if store_examples and len(all_volumes) < MAX_EXAMPLES:
                all_volumes.append((volume * 255).astype(np.uint8, copy=False))
                all_masks.append((mask * 255).astype(np.uint8, copy=False))
                all_series_ids.append(cube_id)
                all_labels.append(1)
            cube_count += 1
            processed_positive += 1
            # Drop references promptly
            cubes[i] = None
            del volume, mask
            if cube_count % 25 == 0:
                gc.collect()
            
        if cube_count % 50 == 0:
            print(f"Extracted {cube_count} cubes ({processed_positive} positive)...")
    
    # Process negative samples (random patches)
    processed_negative = 0
    neg_series_needed = min(len(negative_series), args.max_negative // args.neg_patches_per_series)
    
    for uid in negative_series[:neg_series_needed]:
        if processed_negative >= args.max_negative:
            break
            
        # Determine fold first for early skip check
        fold = folds.get(uid, 0)
        # Early skip: if all expected negative patches for this series already exist
        expected_neg = args.neg_patches_per_series
        fold_dir = output_dir / f"fold_{fold}"
        existing = list(fold_dir.glob(f"neg_{uid}_*.npz"))
        if len(existing) >= expected_neg:
            continue

        cubes = extract_negative_cubes(uid, root, tuple(args.cube_shape), args.neg_patches_per_series)
        
        # Skip if no cubes extracted
        if not cubes:
            continue
        
        for i, (volume, mask) in enumerate(cubes):
            cube_id = f"neg_{uid}_{i:02d}"
            output_file = output_dir / f"fold_{fold}" / f"{cube_id}.npz"
            
            # Skip if file already exists
            if output_file.exists():
                print(f"Skipping existing negative file: {cube_id}")
                continue
            
            # Save to fold directory
            np.savez_compressed(
                output_file, 
                volume=(volume * 255).astype(np.uint8), 
                mask=mask.astype(np.float16),
                label=0,
                fold=fold
            )
            
            # Store for plotting (limited to save memory)
            if store_examples and len(all_volumes) < MAX_EXAMPLES:
                all_volumes.append((volume * 255).astype(np.uint8, copy=False))
                all_masks.append((mask * 255).astype(np.uint8, copy=False))
                all_series_ids.append(cube_id)
                all_labels.append(0)
            cube_count += 1
            processed_negative += 1
            # Drop references promptly
            cubes[i] = None
            del volume, mask
            if cube_count % 25 == 0:
                gc.collect()
            
        if cube_count % 50 == 0:
            print(f"Extracted {cube_count} cubes ({processed_positive} positive, {processed_negative} negative)...")
    
    print(f"Total cubes extracted: {cube_count}")
    print(f"Positive samples: {processed_positive}")
    print(f"Negative samples: {processed_negative}")
    
    # Print fold distribution
    for fold in range(5):
        fold_files = list((output_dir / f"fold_{fold}").glob("*.npz"))
        print(f"Fold {fold}: {len(fold_files)} samples")
    
    # Plot examples (only if enabled and we have examples)
    if store_examples and all_volumes:
        print(f"Generating {len(all_volumes)} example plots...")
        plot_examples(all_volumes, all_masks, all_series_ids, all_labels, output_dir / "examples")
    elif args.no_examples:
        print("Example plotting disabled to save memory.")
    else:
        print("No examples to plot.")
    
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
