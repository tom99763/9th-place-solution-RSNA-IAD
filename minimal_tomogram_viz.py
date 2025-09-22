"""Minimal script to load tomogram, predict with YOLO, and plot every slice with predictions."""
import sys
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

# Add ultralytics path
sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO
from scipy import ndimage
# Ensure project root is importable to reuse preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.prepare_yolo_dataset_v3_cls import ordered_dcm_paths, load_and_process_volume

def load_dicom_series(series_dir):
    """Load and sort DICOM files from directory."""
    series_path = Path(series_dir)
    dicom_files = list(series_path.glob("*.dcm"))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    
    # Sort by SliceLocation > ImagePositionPatient > InstanceNumber
    temp = []
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            if hasattr(ds, "SliceLocation"):
                sort_val = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                sort_val = float(ds.ImagePositionPatient[-1])
            else:
                sort_val = float(getattr(ds, "InstanceNumber", 0))
            temp.append((sort_val, path))
        except Exception:
            temp.append((float('inf'), path))
    
    # Sort and extract paths
    temp.sort(key=lambda x: x[0])
    sorted_paths = [path for _, path in temp]
    
    # Load frames and convert to HU
    frames = []
    for path in sorted_paths:
        ds = pydicom.dcmread(str(path), force=True)
        frame = ds.pixel_array.astype(np.float32)
        
        # Handle missing RescaleSlope/Intercept
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        hu_frame = frame * slope + intercept
        frames.append(hu_frame)
    
    return frames




# --- Volume utilities for resize + normalization ---
def min_max_normalize_volume(volume: np.ndarray) -> np.ndarray:
    mn, mx = float(volume.min()), float(volume.max())
    if mx - mn < 1e-6:
        return np.zeros_like(volume, dtype=np.uint8)
    norm = (volume - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def load_series_as_volume(series_dir: str | Path) -> np.ndarray:
    """Load entire series as a sorted 3D volume (D,H,W) in HU."""
    series_path = Path(series_dir)
    dicom_files = list(series_path.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    # Sort by SliceLocation > ImagePositionPatient[-1] > InstanceNumber
    temp: List[Tuple[float, Path]] = []
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            if hasattr(ds, "SliceLocation"):
                sort_val = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                sort_val = float(ds.ImagePositionPatient[-1])
            else:
                sort_val = float(getattr(ds, "InstanceNumber", 0))
            temp.append((sort_val, path))
        except Exception:
            temp.append((float('inf'), path))
    temp.sort(key=lambda x: x[0])
    sorted_paths = [p for _, p in temp]
    # Read first frame per DICOM and convert to HU
    slices: List[np.ndarray] = []
    for path in sorted_paths:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] != 3:
            arr = arr[..., 0].astype(np.float32)
        elif arr.ndim == 3:
            arr = arr[0].astype(np.float32)
        slices.append(arr * slope + intercept)
    volume = np.stack(slices, axis=0)
    return volume


def resize_volume_to_target(volume: np.ndarray, target_shape: Tuple[int, int, int] = (128, 512, 512)) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    d, h, w = volume.shape
    zf = target_shape[0] / d
    yf = target_shape[1] / h
    xf = target_shape[2] / w
    return ndimage.zoom(volume, (zf, yf, xf), order=1)


def build_rgb_from_volume(volume_u8: np.ndarray, index: int) -> np.ndarray:
    d = volume_u8.shape[0]
    prev_idx = max(0, index - 1)
    next_idx = min(d - 1, index + 1)
    print(f"prev_idx: {prev_idx}, index: {index}, next_idx: {next_idx}")
    r = volume_u8[prev_idx]
    g = volume_u8[index]
    b = volume_u8[next_idx]
    return np.stack([r, g, b], axis=-1)


def plot_tomogram_with_yolo(dicom_path, model_path, max_slices=50, batch_size=1, verbose=False):
    """Load tomogram, preprocess exactly like dataset creation, run YOLO-cls, and plot per-slice probs."""
    model = YOLO(model_path)
    series_dir = Path(dicom_path)
    paths, _ = ordered_dcm_paths(series_dir)
    if not paths:
        raise ValueError(f"No DICOM files found in {series_dir}")
    # Use the same preprocessing as data creation
    volume_u8, zoom_factors, _ = load_and_process_volume(paths, target_shape=(32, 512, 512), verbose=verbose)
    if verbose:
        print(f"Preprocessed volume (uint8) shape: {volume_u8.shape}; zoom_factors={zoom_factors}")
    total = int(volume_u8.shape[0])
    n_use = min(int(max_slices), total) if max_slices is not None else total
    indices = list(range(n_use))

    # Prepare RGB stacks using R=prev, G=current, B=next (same as --rgb in data creation)
    imgs: List[np.ndarray] = [build_rgb_from_volume(volume_u8, i) for i in indices]

    # Inference in batches
    probs: List[float] = []
    for i in range(0, len(imgs), batch_size):
        part = imgs[i:i+batch_size]
        results = model.predict(part, verbose=False)
        for r in results:
            p = 0.0
            if hasattr(r, 'probs') and r.probs is not None:
                try:
                    if len(r.probs.data) >= 2:
                        print(f"r.probs.data: {r.probs.data}")
                        p = float(r.probs.data[0].item())
                    else:
                        p = float(np.max(r.probs.data).item())
                except Exception:
                    p = 0.0
            probs.append(p)

    # Plot grid
    n_slices = len(indices)
    cols = min(8, n_slices)
    rows = (n_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [[ax] for ax in axes]
    for i, (idx, p) in enumerate(zip(indices, probs)):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.imshow(imgs[idx])
        ax.set_title(f"Slice {idx}  p={p:.2f}", fontsize=8)
        ax.axis('off')
    for i in range(n_slices, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("grid.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    # Example usage
    dicom_path = "/home/sersasj/RSNA-IAD-Codebase/data/series/1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636"
    model_path = "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_classification/cv_y11n_cls_binary_fold0/weights/best.pt"
    
    plot_tomogram_with_yolo(dicom_path, model_path, max_slices=32)