import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import pydicom
from concurrent.futures import ThreadPoolExecutor

# If you use MAX_WORKERS in process_dicom_for_yolo, define it somewhere:
MAX_WORKERS = os.cpu_count() or 4
def load_dicom_series(series_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a DICOM series into a 3D HU volume with affine matrix.

    Args:
        series_dir (Path): directory containing DICOM files

    Returns:
        volume (np.ndarray): 3D array (Z, Y, X) in HU
        affine (np.ndarray): 4x4 voxel-to-world affine
    """
    # Collect DICOMs
    dcm_paths = [Path(series_dir) / f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    slices = [pydicom.dcmread(str(p), force=True) for p in dcm_paths]

    # Orientation & sorting
    orientation = np.array(slices[0].ImageOrientationPatient).reshape(2, 3)
    row_cos, col_cos = orientation
    normal = np.cross(row_cos, col_cos)
    slices.sort(key=lambda ds: np.dot(ds.ImagePositionPatient, normal))

    # HU scaling
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    volume = np.stack([ds.pixel_array for ds in slices]).astype(np.float32)
    volume = volume * slope + intercept
    return volume


def normalize_vol(vol):
    p2, p98 = np.percentile(vol, (2, 98))
    mask = (vol >= p2) & (vol <= p98)
    mean = np.mean(vol[mask])
    std = np.std(vol[mask]) + 1e-6

    vol = (vol - mean) / std
    vol = np.clip((vol - vol.min()) / (vol.max() - vol.min() + 1e-6), 0, 1)
    return vol


def read_dicom_frames_hu(path: Path) -> List[Tuple[float, np.ndarray]]:
    """Read DICOM file and return list of (slice_position, HU frame)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

    # Compute slice location using orientation + position
    try:
        orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
        row_cos, col_cos = orientation
        normal = np.cross(row_cos, col_cos)  # slice normal vector
        position = np.array(ds.ImagePositionPatient)
        slice_loc = float(np.dot(position, normal))  # projection along normal
    except Exception:
        # Fallback: SliceLocation / InstanceNumber
        slice_loc = float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0.0)))

    frames: List[Tuple[float, np.ndarray]] = []

    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append((slice_loc, img * slope + intercept))
    elif pix.ndim == 3:
        # RGB or multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append((slice_loc, gray * slope + intercept))
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                # tiny offset ensures consistent ordering for multi-frame
                frames.append((slice_loc + i * 1e-3, frm * slope + intercept))
    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255 with optional flipping"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        norm = np.zeros_like(img, dtype=np.uint8)
    else:
        norm = (img - mn) / (mx - mn)
        norm = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return norm


def process_dicom_file(dcm_path: Path) -> List[Tuple[float, np.ndarray]]:
    """Process single DICOM file -> list of (slice_loc, image) tuples"""
    try:
        frames = read_dicom_frames_hu(dcm_path)
        processed_slices = []
        for loc, f in frames:
            img_u8 = min_max_normalize(f)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            processed_slices.append((loc, img_u8))
        return processed_slices
    except Exception as e:
        print(f"Failed processing {dcm_path.name}: {e}")
        return []


def collect_series_slices(series_dir: Path) -> List[Path]:
    """Collect all DICOM files in a series directory (recursively)."""
    dcm_paths: List[Path] = []
    try:
        for root, _, files in os.walk(series_dir):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_paths.append(Path(root) / f)
    except Exception as e:
        print(f"Failed to walk series dir {series_dir}: {e}")
    return dcm_paths


def slice_sort_key(path: Path) -> float:
    """Compute a robust slice sort key (orientation + position) for a single DICOM file"""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
        row_cos, col_cos = orientation
        normal = np.cross(row_cos, col_cos)
        position = np.array(ds.ImagePositionPatient)
        return float(np.dot(position, normal))
    except Exception:
        # fallback
        try:
            return float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0.0)))
        except:
            return 0.0


def process_dicom_for_yolo(series_path):
    series_path = Path(series_path)
    dicom_files = collect_series_slices(series_path)

    # Sort DICOM files by orientation+position before processing
    dicom_files.sort(key=slice_sort_key)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_dicom_file, dicom_files))

    # Flatten into (loc, img)
    all_slices_with_loc = [item for sublist in results for item in sublist]

    # Already sorted by dicom_files order, but double-check (safe)
    all_slices_with_loc.sort(key=lambda x: x[0])

    # Extract just the images
    all_slices = [img for _, img in all_slices_with_loc]

    # Now dicom_files matches the sorted slices
    dcm_list = [f.stem for f in dicom_files]
    return all_slices