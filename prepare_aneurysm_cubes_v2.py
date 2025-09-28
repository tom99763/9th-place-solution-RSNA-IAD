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
# Multilabel stratification for better balance across modality and classes
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # type: ignore
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from tqdm import tqdm

# Allow importing project configs
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, SEED, LABELS_TO_IDX


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

def load_folds(root: Path, n_folds: int = 5) -> Dict[str, int]:
    """Map SeriesInstanceUID -> fold_id using MultilabelStratifiedKFold.

    Uses MultilabelStratifiedKFold over [Aneurysm Present] + all location columns + Modality one-hot
    for optimal balance across modality and classes (same as YOLO stratification).
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
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for i, (_, test_idx) in enumerate(mskf.split(np.zeros(len(series_df)), y)):
        for uid in series_df.loc[test_idx, "SeriesInstanceUID"].tolist():
            fold_map[uid] = i

    df["fold_id"] = df["SeriesInstanceUID"].astype(str).map(fold_map)
    df.to_csv(df_path, index=False)
    print(f"Updated {df_path} with new fold_id column based on N_FOLDS={n_folds}")

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
        print(f"No DICOM files found in series directory: {series_dir}")
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



def resample_to_patch_size_gpu(vol: np.ndarray, target_shape: Tuple[int, int, int] = (128, 384, 384), output_dtype: np.dtype = np.float16) -> np.ndarray:
    """Resample volume to target shape using GPU acceleration (Z, Y, X order)."""
    current_shape = vol.shape
    zoom = (target_shape[0] / current_shape[0], 
            target_shape[1] / current_shape[1], 
            target_shape[2] / current_shape[2])
    
    # Move to GPU, resample, then back to CPU with direct dtype conversion
    vol_gpu = cp.asarray(vol, dtype=cp.float32)  # Ensure float32 for processing
    vol_resampled_gpu = cpx_ndimage.zoom(vol_gpu, zoom, order=1)
    
    # Handle uint8 conversion properly by scaling to 0-255 range
    if output_dtype == np.uint8:
        # Scale from 0-1 to 0-255 for uint8
        vol_resampled = cp.asnumpy((vol_resampled_gpu * 255).astype(cp.uint8))
    else:
        vol_resampled = cp.asnumpy(vol_resampled_gpu).astype(output_dtype)
    
    # Free GPU memory
    del vol_gpu, vol_resampled_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return vol_resampled


def get_spacing_from_paths(paths: List[Path]) -> Tuple[float, float, float]:
    """Extract spacing information from DICOM files."""
    if not paths:
        print(f"No paths found for series: {paths}")
        return (1.0, 1.0, 1.0)
    
    try:
        ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
        pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
        spacing_z = float(getattr(ds, "SpacingBetweenSlices", slice_thickness))
        return (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)
    except Exception:
        return (1.0, 1.0, 1.0)


def extract_aneurysm_volumes(uid: str, root: Path, series_to_labels: Dict[str, List[Tuple[str, float, float, int | None]]], 
                            target_shape: Tuple[int, int, int] = (128, 384, 384), use_int8: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract full resized volumes for aneurysm series with masks."""
    series_dir = root / "series" / uid
    if not series_dir.exists() or uid not in series_to_labels:
        print(f"Series directory does not exist or uid not in series_to_labels: {series_dir}, {uid}")
        return []

    paths, sop_to_idx = ordered_dcm_paths(series_dir)
    if not paths:
        print(f"No paths found for series: {series_dir}")
        return []

    # Load slices without normalization
    # For multiframe DICOMs, we need to handle frame indices properly
    # Create a mapping from SOP to frame data for proper handling
    sop_to_frames = {}
    for path in paths:
        frames = read_dicom_frames_hu(path)
        if frames:
            sop_to_frames[path.stem] = frames
    
    # Build volume by taking the appropriate frame from each SOP
    # Check if this SOP has frame-specific annotations
    sop_to_annotated_frame = {}
    for sop, x, y, f_idx in series_to_labels[uid]:
        if f_idx is not None:
            sop_to_annotated_frame[sop] = f_idx
    
    slices = []
    for path in paths:
        sop_id = path.stem
        frames = sop_to_frames.get(sop_id, [])
        if frames:
            # Use annotated frame if available, otherwise use first frame
            frame_idx = 0
            if sop_id in sop_to_annotated_frame:
                annotated_frame = sop_to_annotated_frame[sop_id]
                # Handle both 0-based and 1-based indexing
                if 0 <= annotated_frame < len(frames):
                    frame_idx = annotated_frame
                elif 1 <= annotated_frame <= len(frames):
                    frame_idx = annotated_frame - 1
                    
            slices.append(frames[frame_idx])
    
    if not slices:
        print(f"No slices found for series: {uid}")
        return []

    volume = np.stack(slices, axis=0).astype(np.float32)
    # Apply min-max normalization at volume level
    volume = min_max_normalize(volume)
    # Free slice list ASAP
    del slices
    spacing = get_spacing_from_paths(paths)
    
    # Store original shape for scaling calculations
    orig_shape = volume.shape
    
    # Determine output dtype based on quantization option
    output_dtype = np.uint8 if use_int8 else np.float16
    
    # Resample using GPU acceleration to target shape with direct dtype conversion
    volume_resized = resample_to_patch_size_gpu(volume, target_shape=target_shape, output_dtype=output_dtype)
    d, h, w = volume_resized.shape
    
    # Free original volume memory immediately
    del volume
    
    # Calculate scaling factors for annotation coordinates
    scale_x = w / orig_shape[2]
    scale_y = h / orig_shape[1]  
    scale_z = d / orig_shape[0]
    
    # Create combined mask for all aneurysms in this volume with target dtype
    mask_dtype = np.uint8 if use_int8 else np.float16
    combined_mask = np.zeros(target_shape, dtype=mask_dtype)
    
    # Add Gaussian balls for each aneurysm location
    for sop, x, y, f_idx in series_to_labels[uid]:
        idx = sop_to_idx.get(sop)
        if idx is None:
            continue
            
        # Frame handling is now done during volume construction above
            
        # Transform annotation coordinates to resized space
        x_resized = int(x * scale_x)
        y_resized = int(y * scale_y)
        z_resized = int(idx * scale_z)
        
        # Ensure coordinates are within bounds
        x_resized = max(0, min(w - 1, x_resized))
        y_resized = max(0, min(h - 1, y_resized))
        z_resized = max(0, min(d - 1, z_resized))
        
        # Create Gaussian mask at aneurysm location
        center = (z_resized, y_resized, x_resized)
        mask = create_gaussian_ball_mask(target_shape, center, radius=6)
        
        # Convert mask to target dtype and scale if needed
        if use_int8:
            mask_scaled = (mask * 255).astype(np.uint8)
        else:
            mask_scaled = mask.astype(np.float16)
        
        # Add to combined mask (max to handle overlaps)
        combined_mask = np.maximum(combined_mask, mask_scaled)
    
    # Volume is already in target dtype from resample function
    volume_final = volume_resized
    mask_final = combined_mask
    
    # Free GPU memory and intermediate arrays
    del volume_resized, combined_mask
    gc.collect()
    
    # Return single volume-mask pair
    return [(volume_final, mask_final)]

def extract_negative_volumes(uid: str, root: Path, target_shape: Tuple[int, int, int] = (128, 384, 384),
                           use_int8: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract full resized volumes from negative series (no aneurysms)."""
    series_dir = root / "series" / uid
    if not series_dir.exists():
        return []

    paths, _ = ordered_dcm_paths(series_dir)
    if not paths:
        return []

    # Load slices without normalization
    slices = []
    for path in paths:
        frames = read_dicom_frames_hu(path)
        if frames:
            slices.append(frames[0])
    
    if not slices:
        return []

    volume = np.stack(slices, axis=0).astype(np.float32)
    # Apply min-max normalization at volume level
    volume = min_max_normalize(volume)
    # Free slice list ASAP
    del slices
    
    # Determine output dtype based on quantization option
    output_dtype = np.uint8 if use_int8 else np.float16
    
    # Resample using GPU acceleration to target shape with direct dtype conversion
    volume_resized = resample_to_patch_size_gpu(volume, target_shape=target_shape, output_dtype=output_dtype)
    
    # Free original volume memory immediately
    del volume
    
    # Volume is already in target dtype from resample function
    volume_final = volume_resized
    
    # Create empty mask for negative samples with target dtype
    mask_dtype = np.uint8 if use_int8 else np.float16
    mask = np.zeros(target_shape, dtype=mask_dtype)
    
    # Free GPU memory
    del volume_resized
    gc.collect()
    
    # Return single volume-mask pair
    return [(volume_final, mask)]

def parse_args():
    ap = argparse.ArgumentParser(description="Extract aneurysm volumes with folds (no cropping, just resize to target shape)")
    ap.add_argument("--output-dir", default="aneurysm_cubes_v2", help="Output directory")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[128, 384, 384], 
                    help="Target volume dimensions as D H W (default: 128 384 384)")
    ap.add_argument("--max-positive", type=int, default=3000, help="Max positive series to process")
    ap.add_argument("--max-negative", type=int, default=3000, help="Max negative series to process")
    ap.add_argument("--no-examples", action="store_true", help="Skip generating example plots to save memory")
    ap.add_argument("--batch-size", type=int, default=1, help="Process in batches to control memory usage")
    ap.add_argument("--use-int8", action="store_true", 
                    help="Use int8 quantization (0-255) instead of float16 for 50%% space savings")
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
    train_df = pd.read_csv(root / "train.csv")
    train_df["SeriesInstanceUID"] = train_df["SeriesInstanceUID"].astype(str)
    # Get positive and negative series
    positive_series = train_df[train_df["Aneurysm Present"] == 1]["SeriesInstanceUID"].unique()
    negative_series = train_df[train_df["Aneurysm Present"] == 0]["SeriesInstanceUID"].unique()
    
    # Load labels for positive series
    label_df = load_labels(root)
    series_to_labels = {}
    for _, r in label_df.iterrows():
        # Frame index can be NA/None; convert to Python int or None
        f_val = r.get("f", None)
        try:
            f_idx = int(f_val) if pd.notna(f_val) else None
        except Exception:
            f_idx = None
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append(
            (str(r.SOPInstanceUID), float(r.x), float(r.y), f_idx)
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
    positive_to_process = positive_series[:args.max_positive]
    
    print("Processing positive series...")
    for uid in tqdm(positive_to_process, desc="Positive series"):
        if processed_positive >= args.max_positive:
            break
            
        if uid not in series_to_labels:
            continue
            
        # Determine fold first for early skip check
        fold = folds.get(uid, 0)
        # Early skip: if file for this series already exists, avoid heavy processing
        fold_dir = output_dir / f"fold_{fold}"
        existing_file = fold_dir / f"pos_{uid}.npz"
        if existing_file.exists():
            # Already processed this series
            print(f"Skipping existing positive file: {uid}")
            continue

        volumes = extract_aneurysm_volumes(uid, root, series_to_labels, tuple(args.target_shape), 
                                          args.use_int8)
        
        # Skip if no volumes extracted
        if not volumes:
            continue
        
        # Each series produces one volume (no cropping, just resizing)
        volume, mask = volumes[0]  # Only one volume per series now
        volume_id = f"pos_{uid}"
        output_file = output_dir / f"fold_{fold}" / f"{volume_id}.npz"
        
        # Skip if file already exists
        if output_file.exists():
            print(f"Skipping existing positive file: {volume_id}")
            continue
        
        # Save to fold directory with optimized compression
        save_kwargs = {
            'volume': volume,  # Already in target dtype
            'mask': mask,      # Already in target dtype  
            'label': np.uint8(1),
            'fold': np.uint8(fold)
        }
        
        # Use compressed format for better space savings
        np.savez_compressed(output_file, **save_kwargs)
        
        # Store for plotting (limited to save memory)
        if store_examples and len(all_volumes) < MAX_EXAMPLES:
            # Convert to uint8 for plotting, handling different input dtypes
            if args.use_int8:
                plot_volume = volume.copy()  # Already uint8
                plot_mask = mask.copy()      # Already uint8
            else:
                plot_volume = (volume * 255).astype(np.uint8, copy=False)
                plot_mask = (mask * 255).astype(np.uint8, copy=False)
            all_volumes.append(plot_volume)
            all_masks.append(plot_mask)
            all_series_ids.append(volume_id)
            all_labels.append(1)
        cube_count += 1
        processed_positive += 1
        # Drop references promptly
        del volume, mask, volumes
        if cube_count % 25 == 0:
            gc.collect()
            
        # Progress is handled by tqdm
    
    # Process negative samples
    processed_negative = 0
    neg_series_needed = min(len(negative_series), args.max_negative)
    negative_to_process = negative_series[:neg_series_needed]
    
    print("Processing negative series...")
    for uid in tqdm(negative_to_process, desc="Negative series"):
        if processed_negative >= args.max_negative:
            break
            
        # Determine fold first for early skip check
        fold = folds.get(uid, 0)
        # Early skip: if file for this series already exists
        fold_dir = output_dir / f"fold_{fold}"
        existing_file = fold_dir / f"neg_{uid}.npz"
        if existing_file.exists():
            continue

        volumes = extract_negative_volumes(uid, root, tuple(args.target_shape), 
                                          args.use_int8)
        
        # Skip if no volumes extracted
        if not volumes:
            print(f"Skipping existing negative file: {volume_id}")
            continue
        
        # Each series produces one volume (no cropping, just resizing)
        volume, mask = volumes[0]  # Only one volume per series now
        volume_id = f"neg_{uid}"
        output_file = output_dir / f"fold_{fold}" / f"{volume_id}.npz"
        
        # Skip if file already exists
        if output_file.exists():
            print(f"Skipping existing negative file: {volume_id}")
            continue
        
        # Save to fold directory with optimized compression
        save_kwargs = {
            'volume': volume,  # Already in target dtype
            'mask': mask,      # Already in target dtype
            'label': np.uint8(0),
            'fold': np.uint8(fold)
        }
        
        # Use compressed format for better space savings
        np.savez_compressed(output_file, **save_kwargs)
        
        # Store for plotting (limited to save memory)
        if store_examples and len(all_volumes) < MAX_EXAMPLES:
            # Convert to uint8 for plotting, handling different input dtypes
            if args.use_int8:
                plot_volume = volume.copy()  # Already uint8
                plot_mask = mask.copy()      # Already uint8
            else:
                plot_volume = (volume * 255).astype(np.uint8, copy=False)
                plot_mask = (mask * 255).astype(np.uint8, copy=False)
            all_volumes.append(plot_volume)
            all_masks.append(plot_mask)
            all_series_ids.append(volume_id)
            all_labels.append(0)
        cube_count += 1
        processed_negative += 1
        # Drop references promptly
        del volume, mask, volumes
        if cube_count % 25 == 0:
            gc.collect()
            
        # Progress is handled by tqdm
    
    print(f"Total volumes extracted: {cube_count}")
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
