"""Simple utility to visualize all CT/MRI slices from a random sample.

Just run the script - no arguments needed.
Shows all slices from one random SeriesInstanceUID.

Requires: pydicom, numpy, pandas, matplotlib, opencv-python
"""
from __future__ import annotations
import os
import sys
import ast
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
# Data path setup
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
try:
    from configs.data_config import data_path  # type: ignore
except Exception:
    data_path = "data"  # fallback

# ---------------------- Preprocessing helpers ---------------------- #

def window_ct(image: np.ndarray) -> np.ndarray:
    low = 40
    high = 80
    #if high <= low:
    #    high = low + 1.0
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def norm_mri(image: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(image)), float(np.max(image))
    if mx <= mn:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - mn) / (mx - mn)
    return (image * 255.0).clip(0, 255).astype(np.uint8)

# ---------------------- DICOM loading ---------------------- #

def load_series(uid: str, series_root: Path) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Load all frames/slices for a series.

    Returns:
        slices_uint8: list of 2D uint8 arrays
        instance_numbers: list parallel to slices
        z_positions: list parallel (float) for ordering fallback
    """
    series_dir = series_root / uid
    if not series_dir.is_dir():
        raise FileNotFoundError(f"Series directory not found: {series_dir}")

    dcm_files = sorted([f for f in series_dir.glob("*.dcm")])
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")

    slices: List[np.ndarray] = []
    instances: List[int] = []
    z_positions: List[float] = []

    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            arr = ds.pixel_array
            # Convert to float
            arr = arr.astype(np.float32)
            # HU conversion if possible
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            if slope != 1 or intercept != 0:
                arr = arr * slope + intercept

            # Multi-frame handling
            if arr.ndim == 3:
                if arr.shape[-1] == 3:  # RGB -> grayscale
                    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                    arr_frames = [gray]
                else:  # treat as frames
                    arr_frames = [arr[i].astype(np.float32) for i in range(arr.shape[0])]
            else:
                arr_frames = [arr]

            inst_num = int(getattr(ds, "InstanceNumber", len(slices) + 1))
            z = None
            ipp = getattr(ds, "ImagePositionPatient", None)
            if isinstance(ipp, (list, tuple)) and len(ipp) == 3:
                z = float(ipp[2])

            for frame_idx, frame in enumerate(arr_frames):
                slices.append(frame)
                instances.append(inst_num + frame_idx)
                z_positions.append(z if z is not None else inst_num + frame_idx)
        except Exception as e:
            print(f"Warning: failed reading {fp.name}: {e}")
            continue

    # Order by z (then instance) to ensure anatomical order
    order = sorted(range(len(slices)), key=lambda i: (z_positions[i], instances[i]))
    slices = [slices[i] for i in order]
    instances = [instances[i] for i in order]
    z_positions = [z_positions[i] for i in order]

    return slices, instances, z_positions

# ---------------------- Label overlay ---------------------- #

def load_localizer_points(uid: str, loc_df: pd.DataFrame) -> Dict[int, List[Tuple[float, float, str]]]:
    """Map InstanceNumber to list of (x,y,location)."""
    sub = loc_df[loc_df["SeriesInstanceUID"] == uid]
    mapping: Dict[int, List[Tuple[float, float, str]]] = {}
    for _, r in sub.iterrows():
        try:
            coords = ast.literal_eval(r["coordinates"]) if "coordinates" in r else {}
            x, y = coords.get("x"), coords.get("y")
            if x is None or y is None:
                continue
            inst = int(r.get("InstanceNumber", r.get("instance_number", -1)))
            if inst == -1:
                continue
            mapping.setdefault(inst, []).append((float(x), float(y), str(r.get("location", "loc"))))
        except Exception:
            continue
    return mapping

# ---------------------- Main visualization ---------------------- #

def main():
    # Setup paths
    root = Path(data_path)
    series_root = root / "series"
    train_csv = root / "train.csv"
    localizers_csv = root / "train_localizers.csv"

    if not train_csv.exists():
        raise FileNotFoundError("train.csv not found")
    
    # Load data
    train_df = pd.read_csv(train_csv)
    modality_map = dict(zip(train_df["SeriesInstanceUID"], train_df.get("Modality", ["CT"])))
    loc_df = pd.read_csv(localizers_csv) if localizers_csv.exists() else pd.DataFrame()

    # Sample random UID
    unique_uids = train_df['SeriesInstanceUID'].unique().tolist()
    if len(unique_uids) == 0:
        raise ValueError('No SeriesInstanceUID entries found in train.csv')
    
    rng = np.random.default_rng(42)
    #uid = rng.choice(unique_uids)
    uid='1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381'
    print(f"Selected random UID: {uid}")

    # Load and process all slices
    try:
        modality = str(modality_map.get(uid, "CT")).upper()
        raw_slices, inst_nums, _ = load_series(uid, series_root)
        if not raw_slices:
            print(f"No slices for {uid}")
            return

        print(f"Found {len(raw_slices)} slices for {modality} series")

        # Preprocess each slice
        proc_slices = []
        for s in raw_slices:
            if modality == "CT":
                proc_slices.append(window_ct(s))
            else:
                proc_slices.append(norm_mri(s))

        # Load localizer points if available
        inst_to_labels = load_localizer_points(uid, loc_df) if not loc_df.empty else {}

        # Prepare images for display
        fig_images = []
        titles = []
        
        for idx, (img, inst) in enumerate(zip(proc_slices, inst_nums)):
            # Overlay labels if available
            if inst in inst_to_labels:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for (x, y, loc_name) in inst_to_labels[inst]:
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.drawMarker(img_color, (int(x), int(y)), (0, 0, 255), 
                                     markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
                display_img = img_color
            else:
                display_img = img
            
            fig_images.append(display_img)
            titles.append(f"Slice {idx+1}/{len(proc_slices)} - Inst:{inst}")

        # Create visualization
        n = len(fig_images)
        cols = min(6, n)  # Max 6 columns
        rows = int(np.ceil(n / cols))

        # Dynamic figure sizing
        scale = 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale))
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        # Plot each image
        for i, img in enumerate(fig_images):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:  # BGR with overlays
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            ax.set_title(titles[i], fontsize=8)
            ax.axis('off')

        # Hide unused subplots
        for j in range(n, rows * cols):
            r = j // cols
            c = j % cols
            axes[r][c].axis('off')

        plt.suptitle(f"{modality} Series: {uid[:20]}...", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{uid}_slices.png", dpi=150, bbox_inches='tight')
        print(f"Saved visualization as: {uid}_slices.png")

    except Exception as e:
        print(f"Failed to process UID {uid}: {e}")


if __name__ == '__main__':
    main()