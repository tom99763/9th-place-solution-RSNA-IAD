import torch
import numpy as np
import pandas as pd
import random
import polars as pl
import pydicom
from pathlib import Path
import os
import cv2
from typing import List, Dict, Optional, Tuple


def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    """Read DICOM file and return raw frames (no HU conversion)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    frames: List[np.ndarray] = []

    if pix.ndim == 2:
        frames.append(pix.astype(np.float32))
    elif pix.ndim == 3:
        # RGB or multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append(gray)
        else:
            for i in range(pix.shape[0]):
                frames.append(pix[i].astype(np.float32))
    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def process_dicom_file(dcm_path: Path) -> List[np.ndarray]:
    """Process single DICOM file - for parallel processing"""
    try:
        frames = read_dicom_frames_hu(dcm_path)
        processed_slices = []
        for f in frames:
            img_u8 = min_max_normalize(f)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            processed_slices.append(img_u8)
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
    dcm_paths.sort()
    return dcm_paths