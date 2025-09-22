
import argparse
import ast
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

import cv2
import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold

# Try to import CuPy for GPU acceleration, fallback to CPU if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available - will use for 3D resizing when z-axis changes")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - using CPU/OpenCV processing")

# Allow importing project configs
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path as _data_path, N_FOLDS, SEED, LABELS_TO_IDX  # type: ignore

# Resolve data_path relative to ROOT to handle execution from different directories
data_path = str(ROOT / _data_path) if not Path(_data_path).is_absolute() else _data_path

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


def min_max_normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize entire volume consistently to uint8."""
    mn, mx = float(volume.min()), float(volume.max())
    if mx - mn < 1e-6:
        return np.zeros_like(volume, dtype=np.uint8)
    norm = (volume - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def load_and_process_volume(paths: List[Path], target_shape: Tuple[int | None, int, int] = (None, 512, 512), verbose: bool = False) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[Tuple[int, int], int]]:
    """Load DICOM slices, sort, stack, normalize and resize to target shape.
    
    Args:
        paths: List of DICOM file paths
        target_shape: (z, y, x) target shape. If z is None, keep original z dimension
        verbose: Whether to print debug information
    
    Returns:
        volume: Processed volume of shape target_shape as uint8
        zoom_factors: (z_factor, y_factor, x_factor) used for resizing
        origin_to_resized: Dict mapping (original_file_index, frame_index_in_file) -> resized z-slice index
    """
    all_slices = []
    slice_positions = []
    
    # Load all slices and collect spatial positions for sorting
    for i, path in enumerate(paths):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            
            # Get slice position for sorting
            if hasattr(ds, "SliceLocation"):
                position = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                position = float(ds.ImagePositionPatient[-1])
            else:
                position = float(getattr(ds, "InstanceNumber", 0))
            
            slice_positions.append((position, path, i))  # Include original index
        except Exception:
            # Skip problematic files
            continue
    
    # Sort by spatial position
    slice_positions.sort(key=lambda x: x[0])
    
    # Load sorted slices and track origin (file idx, frame idx)
    frame_origins: List[Tuple[int, int]] = []
    for position, path, orig_idx in slice_positions:
        frames = read_dicom_frames_hu(path)
        for frame_idx, frame in enumerate(frames):
            all_slices.append(frame)
            frame_origins.append((orig_idx, frame_idx))
    
    if not all_slices:
        # Return empty volume if no slices loaded
        return np.zeros(target_shape, dtype=np.uint8), (1.0, 1.0, 1.0), {}
    
    # Stack slices into volume (keep as float32 for now)
    volume = np.stack(all_slices, axis=0)  # Shape: (n_slices, height, width)
    
    # Clear the list to free memory
    del all_slices
    
    # Calculate zoom factors for resizing
    original_shape = volume.shape
    if target_shape[0] is None:
        # Keep original z dimension
        final_target_shape = (original_shape[0], target_shape[1], target_shape[2])
    else:
        final_target_shape = target_shape
    
    zoom_factors = (
        final_target_shape[0] / original_shape[0],  # z factor
        final_target_shape[1] / original_shape[1],  # y factor  
        final_target_shape[2] / original_shape[2]   # x factor
    )
    
    # Choose optimal resizing method based on whether z-axis needs resizing
    z_needs_resize = abs(zoom_factors[0] - 1.0) > 1e-6
    
    if z_needs_resize:
        # Always use CuPy for GPU processing to match inference - no CPU fallback
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU processing but not available. Please install CuPy.")
        
        
        # Import CuPy's scipy-compatible ndimage module
        from cupyx.scipy import ndimage as cp_ndimage
        # Move volume to GPU
        volume_gpu = cp.asarray(volume)
        # Use CuPy's zoom function
        resized_volume_gpu = cp_ndimage.zoom(volume_gpu, zoom_factors, order=1)
        # Move back to CPU
        resized_volume = cp.asnumpy(resized_volume_gpu)
        # Clean up GPU memory immediately
        del volume_gpu, resized_volume_gpu
        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f"[DEBUG] Used CuPy for 3D resize with z-factor={zoom_factors[0]:.3f}")
    else:
        # Use OpenCV for per-slice 2D resizing when z-axis unchanged (faster per experiments)
        target_h, target_w = final_target_shape[1], final_target_shape[2]
        if verbose:
            print(f"[DEBUG] Using OpenCV for per-slice resize to {target_h}x{target_w} (z unchanged)")
        
        resized_slices = []
        for slice_idx in range(volume.shape[0]):
            slice_2d = volume[slice_idx]  # Shape: (H, W)
            resized_slice = cv2.resize(slice_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(resized_slice)
        
        resized_volume = np.stack(resized_slices, axis=0)
        del resized_slices
    
    # Clear original volume to free memory
    del volume
    
    # Ensure exact target shape (zoom might have slight rounding differences)
    if resized_volume.shape != final_target_shape:
        # Pad or crop to exact target shape
        final_volume = np.zeros(final_target_shape, dtype=np.float32)
        min_z = min(resized_volume.shape[0], final_target_shape[0])
        min_y = min(resized_volume.shape[1], final_target_shape[1]) 
        min_x = min(resized_volume.shape[2], final_target_shape[2])
        final_volume[:min_z, :min_y, :min_x] = resized_volume[:min_z, :min_y, :min_x]
        resized_volume = final_volume
    
    # Now normalize the entire volume consistently
    normalized_volume = min_max_normalize_volume(resized_volume)
    
    # Create mapping from (original file index, frame index) to resized slice index
    origin_to_resized: Dict[Tuple[int, int], int] = {}
    for sorted_position, origin in enumerate(frame_origins):
        resized_position = round(sorted_position * zoom_factors[0])
        resized_position = max(0, min(final_target_shape[0] - 1, resized_position))
        origin_to_resized[origin] = resized_position
    
    if verbose:
        print(f"[DEBUG] original volume shape: {original_shape}")
        print(f"[DEBUG] resized volume shape: {resized_volume.shape}")
        print(f"[DEBUG] target shape: {final_target_shape}")
        print(f"[DEBUG] zoom factors: {zoom_factors}")
        print(f"[DEBUG] using CuPy: {CUPY_AVAILABLE}")
    
    return normalized_volume, zoom_factors, origin_to_resized


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
    for sub in ["train", "val"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare per-slice classification dataset from DICOM volumes (no boxes)")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    ap.add_argument("--val-fold", type=int, default=0, help="Fold id used for validation (ignored if --generate-all-folds)")
    ap.add_argument("--generate-all-folds", action="store_true", help="Generate one dataset per fold and write per-fold YAMLs")
    ap.add_argument("--box-size", type=int, default=24, help="Square box size in pixels around the point")
    ap.add_argument("--image-ext", type=str, default="png", choices=["png", "jpg", "jpeg"], help="Output image extension")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of worker processes for parallel processing")
    ap.add_argument("--mip-img-size", type=int, default=512, help="Final square image size for saved samples (0 = keep original)")
    ap.add_argument("--z-axis-size", type=int, default=32, help="Final z-axis size for 3D volumes (None = keep original)")
    ap.add_argument("--fold-output-name", type=str, default="yolo_dataset", help="Base subdirectory under data/ for outputs; when --generate-all-folds, becomes {base}_fold{fold}")
    ap.add_argument("--rgb", action="store_true", help="Save 3-channel rgb slices: B=next slice, G=current, R=previous")
    # Labeling scheme
    ap.add_argument(
        "--label-scheme",
        type=str,
        choices=["locations", "aneurysm_present"],
        default="locations",
        help="Use location-specific classes (from LABELS_TO_IDX) or single binary class 'aneurysm_present'",
    )
    # baseline = 10
    # Negative sampling controls
    ap.add_argument(
        "--neg-per-series",
        type=int,
        default=2,
        help="Number of negative slices to sample per series without positives (fallback when pos-neg-ratio=0).",
    )
    # baseline = 5
    ap.add_argument(
        "--pos-neg-ratio",
        type=float,
        default=1,
        help=(
            "Negatives per positive for positive series (sampled from the same series). "
            "E.g., 2.0 adds ~2 negative slices for each positive slice (capped by available slices)."
        ),
    )
    # YAML outputs
    ap.add_argument("--yaml-out-dir", type=str, default=str(ROOT / "configs"), help="Directory to write per-fold YOLO YAMLs")
    ap.add_argument("--yaml-name-template", type=str, default="yolo_fold{fold}.yaml", help="YAML filename template with {fold}")
    return ap.parse_args()


def load_folds(root: Path) -> Dict[str, int]:
    """Map SeriesInstanceUID -> fold_id using stratified folds from train.csv.

    Requirements:
      - data/train.csv must exist
      - Columns: 'SeriesInstanceUID', 'Aneurysm Present'
      - Will create N_FOLDS stratified splits on the series-level label
    """
    df_path = root / "train.csv"

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
    
    # Update train.csv with the new folds
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


def process_series_worker(args_tuple):
    """Fixed worker function for processing a single series in parallel."""
    (
        uid,
        series_dir,
        split,
        labels_for_series,
        sop_to_idx,
        out_base,
        label_scheme,
        box_size,
        image_ext,
        overwrite,
        verbose,
        pos_neg_ratio,
        neg_per_series,
        seed,
        use_rgb,
        z_axis_size
    ) = args_tuple
    
    # Create local random generator for this worker
    local_rng = random.Random(seed + hash(uid) % 1000000)
    # Build reverse mapping for location labels when needed
    idx_to_label = {idx: name for name, idx in LABELS_TO_IDX.items()}
    
    try:
        if verbose:
            print(f"[DEBUG] Starting processing series {uid}")
        
        paths, sop_to_idx_ordered = ordered_dcm_paths(series_dir)
        if not paths:
            if verbose:
                print(f"[DEBUG] no paths found for series {uid}")
            return {"uid": uid, "pos": 0, "neg": 0, "error": "No DICOM files found"}
        
        if verbose:
            print(f"[DEBUG] Loading and processing volume for series {uid} with {len(paths)} files")
        
        # Process entire volume using configurable z-axis size
        volume, zoom_factors, origin_to_resized = load_and_process_volume(
            paths, target_shape=(z_axis_size, 512, 512), verbose=verbose
        )
        
        if verbose:
            print(f"[DEBUG] Volume loaded for series {uid}, shape: {volume.shape}")
            print(f"[DEBUG] zoom factors: {zoom_factors}")
        
        pos_saved = 0
        neg_saved = 0

        # Helper to resolve class folder name by scheme
        def class_name_for_id(class_id: int, is_negative: bool = False) -> str:
            if label_scheme == "aneurysm_present":
                return "no_aneurysm" if is_negative else "aneurysm_present"
            # locations scheme
            if is_negative:
                return "background"
            return idx_to_label.get(class_id, f"class_{class_id}")
        
        if labels_for_series:
            # Process positive series with labels
            for (sop, x, y, cls_id, f_idx) in labels_for_series:
                # Find original slice index for this SOP
                original_sop_idx = sop_to_idx_ordered.get(sop)
                if original_sop_idx is None:
                    if verbose:
                        print(f"[DEBUG] SOP {sop} not found in ordered paths")
                    continue
                
                # Map to resized volume slice index using frame index when available
                frame_index_candidate = int(f_idx) if f_idx is not None else 0
                origin_key = (original_sop_idx, frame_index_candidate)
                if origin_key in origin_to_resized:
                    resized_slice_idx = origin_to_resized[origin_key]
                else:
                    # Try 1-based -> 0-based fallback when frame indices are 1-based in labels
                    if f_idx is not None and frame_index_candidate > 0:
                        alt_key = (original_sop_idx, frame_index_candidate - 1)
                        if alt_key in origin_to_resized:
                            resized_slice_idx = origin_to_resized[alt_key]
                        else:
                            resized_slice_idx = round(original_sop_idx * zoom_factors[0])
                            resized_slice_idx = max(0, min(127, resized_slice_idx))
                    else:
                        # Fallback to proportional mapping by slice index
                        resized_slice_idx = round(original_sop_idx * zoom_factors[0])
                        resized_slice_idx = max(0, min(127, resized_slice_idx))
                
                # Extract the positive slice from the processed volume
                positive_slice = volume[resized_slice_idx]  # Shape: (512, 512)
                if use_rgb:
                    prev_idx = max(0, resized_slice_idx - 1)
                    next_idx = min(volume.shape[0] - 1, resized_slice_idx + 1)
                    r = volume[prev_idx]
                    g = volume[resized_slice_idx]
                    b = volume[next_idx]
                    positive_slice = np.stack([r, g, b], axis=-1)
                
                # Scale coordinates to the resized image space
                # Original coordinates are in the original slice dimensions
                # We need to scale them to 512x512
                
                # Get original image dimensions by reading one slice
                try:
                    temp_frames = read_dicom_frames_hu(paths[original_sop_idx])
                    if temp_frames:
                        print(f"[DEBUG] temp_frames: {temp_frames[0].shape}")
                        original_height, original_width = temp_frames[0].shape
                    else:
                        print(f"[DEBUG] else no temp_frames")
                        original_height, original_width = 512, 512
                except:
                    original_height, original_width = 512, 512
                
                # Scale coordinates from original image size to 512x512
                scaled_x = x * (512.0 / original_width)
                scaled_y = y * (512.0 / original_height)
                
                # Create naming for positive slice and destination class directory
                slice_stem = f"{uid}_{sop}_slice{resized_slice_idx}"
                class_dir = out_base / split / class_name_for_id(cls_id, is_negative=False)
                img_path = class_dir / f"{slice_stem}.{image_ext}"
                
                if img_path.exists() and not overwrite:
                    pos_saved += 1
                    continue
                
                # Ensure directory exists
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the positive slice as image
                cv2.imwrite(str(img_path), positive_slice)
                if verbose:
                    print(f"[DEBUG] saved positive slice to {img_path}")
                    print(f"[DEBUG] original coords: ({x}, {y}), scaled coords: ({scaled_x}, {scaled_y})")
                
                pos_saved += 1
            
            # Sample additional random negatives from this positive series if requested
            if pos_neg_ratio > 0 and pos_saved > 0:
                # Calculate which slices are positive in the processed volume
                pos_slice_indices = set()
                for (sop, _x, _y, _cls_id, _f_idx) in labels_for_series:
                    original_sop_idx = sop_to_idx_ordered.get(sop)
                    if original_sop_idx is None:
                        continue
                    frame_index_candidate = int(_f_idx) if _f_idx is not None else 0
                    origin_key = (original_sop_idx, frame_index_candidate)
                    if origin_key in origin_to_resized:
                        resized_slice_idx = origin_to_resized[origin_key]
                    else:
                        if _f_idx is not None and frame_index_candidate > 0:
                            alt_key = (original_sop_idx, frame_index_candidate - 1)
                            if alt_key in origin_to_resized:
                                resized_slice_idx = origin_to_resized[alt_key]
                            else:
                                resized_slice_idx = round(original_sop_idx * zoom_factors[0])
                                resized_slice_idx = max(0, min(127, resized_slice_idx))
                        else:
                            resized_slice_idx = round(original_sop_idx * zoom_factors[0])
                            resized_slice_idx = max(0, min(127, resized_slice_idx))
                    pos_slice_indices.add(resized_slice_idx)
                
                # Sample negative slices from processed volume
                n_slices = volume.shape[0]
                candidates = [i for i in range(n_slices) if i not in pos_slice_indices]
                # In RGB mode, ensure neither neighbor channel includes a positive slice
                if use_rgb:
                    candidates = [
                        i for i in candidates
                        if (max(0, i - 1) not in pos_slice_indices)
                        and (min(n_slices - 1, i + 1) not in pos_slice_indices)
                    ]
                # Further reduce to non-overlapping RGB triplets: use only centers with valid neighbors, spaced by 3
                if use_rgb:
                    candidates = [i for i in candidates if 0 < i < n_slices - 1 and (i % 3 == 1)]
                if candidates:
                    need = int(math.ceil(pos_neg_ratio * pos_saved))
                    k = min(need, len(candidates))
                    local_rng.shuffle(candidates)
                    pick = candidates[:k]
                    
                    for slice_idx in pick:
                        neg_slice = volume[slice_idx]
                        if use_rgb:
                            prev_idx = max(0, slice_idx - 1)
                            next_idx = min(volume.shape[0] - 1, slice_idx + 1)
                            r = volume[prev_idx]
                            g = volume[slice_idx]
                            b = volume[next_idx]
                            neg_slice = np.stack([r, g, b], axis=-1)
                        slice_stem = f"{uid}_slice{slice_idx}_neg"
                        class_dir = out_base / split / class_name_for_id(0, is_negative=True)
                        img_path = class_dir / f"{slice_stem}.{image_ext}"
                        
                        if img_path.exists() and not overwrite:
                            continue
                        
                        # Ensure directory exists
                        class_dir.mkdir(parents=True, exist_ok=True)
                        
                        cv2.imwrite(str(img_path), neg_slice)
                        neg_saved += 1
        else:
            # Negative series - sample random slices from processed volume
            need = max(0, int(neg_per_series))
            if need > 0:
                # Sample random slice indices from processed volume
                n_slices = volume.shape[0]
                if use_rgb:
                    # Non-overlapping RGB triplets: centers at every 3rd slice with valid neighbors
                    indices = [i for i in range(1, n_slices - 1) if (i % 3 == 1)]
                else:
                    indices = list(range(n_slices))
                local_rng.shuffle(indices)
                pick = indices[:min(need, len(indices))]
                
                for slice_idx in pick:
                    neg_slice = volume[slice_idx]
                    if use_rgb:
                        prev_idx = max(0, slice_idx - 1)
                        next_idx = min(volume.shape[0] - 1, slice_idx + 1)
                        r = volume[prev_idx]
                        g = volume[slice_idx]
                        b = volume[next_idx]
                        neg_slice = np.stack([r, g, b], axis=-1)
                    slice_stem = f"{uid}_slice{slice_idx}_neg"
                    class_dir = out_base / split / class_name_for_id(0, is_negative=True)
                    img_path = class_dir / f"{slice_stem}.{image_ext}"
                    
                    if img_path.exists() and not overwrite:
                        continue
                    
                    # Ensure directory exists
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    cv2.imwrite(str(img_path), neg_slice)
                    neg_saved += 1
        
        return {
            "uid": uid,
            "pos": pos_saved,
            "neg": neg_saved,
            "error": None
        }
        
    except Exception as e:
        return {
            "uid": uid,
            "pos": 0,
            "neg": 0,
            "error": str(e)
        }


def generate_for_fold(val_fold: int, args) -> Tuple[Path, Dict[str, int]]:
    """Generate per-slice dataset for a single fold and return (out_base, fold_map)."""
    global rng
    root = Path(data_path)
    # Helpful debug to confirm resolved data root
    if args.verbose:
        print(f"[DEBUG] data root: {root}")

    # Resolve output base for this fold
    out_dir_name = args.fold_output_name if not hasattr(args, 'generate_all_folds') or not args.generate_all_folds else f"{args.fold_output_name}_fold{val_fold}"
    out_base = root / out_dir_name
    ensure_yolo_dirs(out_base)
    if args.verbose:
        print(f"[DEBUG] output base: {out_base}")
        print(f"[DEBUG] train dir: {out_base / 'train'} | val dir: {out_base / 'val'}")

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
            # Binary classification: 1 = aneurysm_present (0 reserved for no_aneurysm)
            cls_id = 1
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
    train_path = root / "train.csv"
    if train_path.exists():
        try:
            train_csv = pd.read_csv(train_path)
            if "SeriesInstanceUID" in train_csv.columns:
                all_series = (
                    train_csv["SeriesInstanceUID"].astype(str).unique().tolist()
                )
        except Exception:
            pass
    if not all_series:
        all_series = sorted(series_to_labels.keys())

    print(f"Processing {len(all_series)} series for per-slice YOLO dataset (val fold = {val_fold}) with {args.workers} workers...")

    # Prepare work items for parallel processing
    work_items = []
    for uid in all_series:
        if args.verbose:
            print(f"[DEBUG] processing series {uid}")
        # Ensure UID strings are clean (no stray whitespace)
        uid = str(uid).strip()
        if uid in ignore_uids:
            if args.verbose:
                print(f"[SKIP] ignored series {uid}")
            continue
        split = "val" if folds.get(uid, 0) == val_fold else "train"
        series_dir = root / "series" / uid
        if not series_dir.exists():
            if args.verbose:
                print(f"[MISS] series {uid} {series_dir}")
            continue

        paths, sop_to_idx = ordered_dcm_paths(series_dir)
        if not paths:
            if args.verbose:
                print(f"[DEBUG] no paths found for series {uid}")
            continue
            
        labels_for_series = series_to_labels.get(uid, [])
        work_items.append((
            uid,
            series_dir,
            split,
            labels_for_series,
            sop_to_idx,
            out_base,
            args.label_scheme,
            args.box_size,
            args.image_ext,
            args.overwrite,
            args.verbose,
            args.pos_neg_ratio,
            args.neg_per_series,
            args.seed,
            args.rgb,
            args.z_axis_size
        ))

    # Process series in parallel with chunking for better memory management
    total_series = 0
    total_pos = 0
    total_neg = 0
    
    # Ensure CuPy is available for mandatory GPU processing
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is required for GPU processing but not available. Please install CuPy with: pip install cupy-cuda11x (or cupy-cuda12x)")
    
    # Limit workers when using GPU to avoid GPU memory contention
    effective_workers = min(args.workers, 2)  # Always limit to max 2 workers for GPU processing
    if args.workers > 2:
        print(f"[INFO] Limiting to {effective_workers} workers for GPU processing (was {args.workers})")
    
    print(f"Starting processing of {len(work_items)} series with {effective_workers} workers...")
    
    # Process in chunks to avoid memory issues
    chunk_size = max(1, min(50, len(work_items) // max(1, effective_workers)))
    
    if args.workers == 1:
        # Single-threaded processing for debugging
        results = []
        for i, item in enumerate(work_items):
            if i % 10 == 0:
                print(f"Processing series {i+1}/{len(work_items)}")
            result = process_series_worker(item)
            results.append(result)
    else:
        # Multi-threaded processing with chunking and progress reporting
        results = []
        try:
            # Process in chunks to manage memory and provide progress feedback
            for chunk_start in range(0, len(work_items), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(work_items))
                chunk = work_items[chunk_start:chunk_end]
                
                print(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(work_items) + chunk_size - 1)//chunk_size} ({len(chunk)} series)")
                
                with mp.Pool(processes=effective_workers) as pool:
                    chunk_results = pool.map(process_series_worker, chunk)
                    results.extend(chunk_results)
                
                # Show progress
                processed_so_far = len(results)
                print(f"Completed {processed_so_far}/{len(work_items)} series")
        except Exception as e:
            print(f"Error in multiprocessing: {e}")
            print("Falling back to single-threaded processing...")
            results = []
            for i, item in enumerate(work_items):
                if i % 10 == 0:
                    print(f"Processing series {i+1}/{len(work_items)}")
                try:
                    result = process_series_worker(item)
                    results.append(result)
                except Exception as worker_e:
                    print(f"Error processing series {item[0]}: {worker_e}")
                    results.append({"uid": item[0], "pos": 0, "neg": 0, "error": str(worker_e)})
    
    # Collect results
    for result in results:
        if result["error"]:
            if args.verbose:
                print(f"[ERROR] {result['uid']}: {result['error']}")
        else:
            total_pos += result["pos"]
            total_neg += result["neg"]
            if result["pos"] > 0 or result["neg"] > 0:
                total_series += 1
                if args.verbose:
                    pos_neg_str = f"pos={result['pos']}, neg={result['neg']}"
                    print(f"  [DONE] {result['uid']} ({pos_neg_str})")

    print(f"Series processed: {total_series}  (pos images: {total_pos}, neg images: {total_neg})")
    print(f"YOLO-cls dataset root: {out_base}")
    stats = {}
    for split in ["train", "val"]:
        # Count all images recursively under class subfolders
        n_imgs = sum(1 for _ in (out_base / split).rglob(f"*.{args.image_ext}"))
        print(f"{split}: images={n_imgs}")
        stats[split] = {"images": n_imgs}
    return out_base, folds


def write_yolo_yaml(yaml_dir: Path, yaml_name: str, dataset_root: Path, label_scheme: str):
    yaml_dir.mkdir(parents=True, exist_ok=True)
    # Make paths relative to the workspace root
    relative_path = dataset_root.resolve().relative_to(ROOT)
    # Names section depends on labeling scheme
    names_lines: List[str] = []
    if label_scheme == "aneurysm_present":
        # Binary classification: infer from folders is possible, but provide names
        names_lines.append("  0: no_aneurysm")
        names_lines.append("  1: aneurysm_present")
    else:
        # Invert LABELS_TO_IDX to ensure correct index order (locations) + background
        idx_to_label = {idx: name for name, idx in LABELS_TO_IDX.items()}
        max_idx = max(idx_to_label.keys()) if idx_to_label else -1
        for i in range(max_idx + 1):
            label = idx_to_label.get(i, f"class_{i}")
            names_lines.append(f"  {i}: {label}")
        names_lines.append(f"  {max_idx + 1}: background")
    yaml_text = "\n".join(
        [
            f"path: {relative_path}",
            f"train: train",
            f"val: val",
            "",
            "names:",
            *names_lines,
            "",
        ]
    )
    with open(yaml_dir / yaml_name, "w") as f:
        f.write("# Auto-generated by prepare_yolo_dataset_v3_cls.py\n")
        f.write("# NOTE: YOLO classification format: train/val/<class_name>/*.{jpg|png} with 512x512 slices.\n")
        f.write(yaml_text)


if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(args.seed)
    if args.generate_all_folds:
        # Generate per-fold datasets and YAMLs
        for f in range(0,1):
            out_base, _ = generate_for_fold(f, args)
            yaml_dir = Path(args.yaml_out_dir)
            yaml_name = args.yaml_name_template.format(fold=f)
            write_yolo_yaml(yaml_dir, yaml_name, out_base, args.label_scheme)
        print(f"Generated datasets and YAMLs for folds 0..{N_FOLDS-1}.")
        print(f"YAMLs written under: {args.yaml_out_dir}")
    else:
        out_base, _ = generate_for_fold(args.val_fold, args)
        print("Done. To train (YOLO-cls), point Ultralytics to a YAML like:")
        print(f"  path: {out_base}")
        print("  train: train\n  val: val")


#python3 -m src.prepare_yolo_dataset_v3_cls \
#  --generate-all-folds \
#  --label-scheme aneurysm_present \
#  --fold-output-name yolo_cls_bgr \
#  --yaml-out-dir configs \
#  --yaml-name-template yolo_cls_bgr_fold{fold}.yaml \
#  --rgb --overwrite --workers 2 \
#  --z-axis-size 64
#
# Note: This script supports configurable z-axis resizing and uses optimized resizing:
# - When --z-axis-size is not specified: keeps original z-axis dimension
# - When --z-axis-size is specified: resizes to the target z dimension
# - OpenCV for per-slice resizing when z-axis unchanged (faster per experiments)
# - CuPy for 3D resizing when z-axis changes (better for 3D operations)
# CuPy will automatically fallback to CPU if not available.



#  python3 -m src.prepare_yolo_dataset_v3_cls \
#    --workers 2 \
#    --rgb \
#    --z-axis-size 32 \
#    --label-scheme aneurysm_present \
#    --generate-all-folds