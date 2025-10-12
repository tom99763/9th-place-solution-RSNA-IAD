#!/usr/bin/env python3
"""
Test script to verify YOLO model loading and prediction functionality
Tests fold 0 EfficientNet model on DICOM series: 1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317
"""

import os
import sys
import numpy as np
import pydicom
import cv2
from pathlib import Path
from typing import List
import warnings
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

warnings.filterwarnings('ignore')

# Add ultralytics-timm to path (same as in main script)
sys.path.insert(0, "/home/sersasj/RSNA-IAD-Codebase/ultralytics-timm")

# YOLO
from ultralytics import YOLO

# Set device and deterministic settings
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# YOLO label mappings (same as main script)
YOLO_LABELS_TO_IDX = {
    'Anterior Communicating Artery': 0,
    'Basilar Tip': 1,
    'Left Anterior Cerebral Artery': 2,
    'Left Infraclinoid Internal Carotid Artery': 3,
    'Left Middle Cerebral Artery': 4,
    'Left Posterior Communicating Artery': 5,
    'Left Supraclinoid Internal Carotid Artery': 6,
    'Other Posterior Circulation': 7,
    'Right Anterior Cerebral Artery': 8,
    'Right Infraclinoid Internal Carotid Artery': 9,
    'Right Middle Cerebral Artery': 10,
    'Right Posterior Communicating Artery': 11,
    'Right Supraclinoid Internal Carotid Artery': 12
}

YOLO_LABELS = sorted(list(YOLO_LABELS_TO_IDX.keys()))

# Configuration - match validation script defaults
IMG_SIZE = 512
BATCH_SIZE = 16  # Match validation script default
MAX_WORKERS = 4

def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    """Read DICOM file and return HU frames (with slope/intercept conversion)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    frames: List[np.ndarray] = []

    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    return frames

def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)

def process_dicom_file_yolo(dcm_path: Path, keep_grayscale: bool = False) -> List[np.ndarray]:
    """Process single DICOM file for YOLO - for parallel processing"""
    try:
        frames = read_dicom_frames_hu(dcm_path)
        processed_slices = []
        for f in frames:
            img_u8 = min_max_normalize(f)
            if img_u8.ndim == 2:
                if not keep_grayscale:  # For 2D mode, convert to RGB
                    img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
                # For 2.5D mode, keep as grayscale (single channel)
            processed_slices.append(img_u8)
        return processed_slices
    except Exception as e:
        print(f"Error processing {dcm_path}: {e}")
        return []

def collect_series_slices_sorted(series_dir: Path) -> List[Path]:
    """Collect all DICOM files in series directory and sort by spatial position (match validation script)."""
    dicom_files = list(series_dir.glob("*.dcm"))

    if not dicom_files:
        return []

    # First pass: collect all slices with their spatial information (match validation script)
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
        return []

    # Sort slices by the determined sort value (spatial order)
    temp_slices.sort(key=lambda x: x[0])

    # Extract the sorted filepaths
    sorted_files = [item[1] for item in temp_slices]
    return sorted_files

def process_dicom_for_yolo(series_path: str, mode: str = "2.5D") -> List[np.ndarray]:
    """Process DICOM for YOLO with parallel processing and mode support"""
    series_path = Path(series_path)
    dicom_files = collect_series_slices_sorted(series_path)

    # Helper function for parallel processing that returns (index, slices)
    def process_with_index(index_dcm_tuple):
        index, dcm_path = index_dcm_tuple
        keep_grayscale = (mode == "2.5D")  # For 2.5D, keep as grayscale
        slices = process_dicom_file_yolo(dcm_path, keep_grayscale)
        return index, slices

    # Helper function for 2D mode processing
    def process_2d_with_index(index_dcm_tuple):
        index, dcm_path = index_dcm_tuple
        slices = process_dicom_file_yolo(dcm_path, False)  # Convert to RGB for 2D
        return index, slices

    if mode == "2D":
        # For 2D mode, process each DICOM file individually (convert to RGB)
        # Process in parallel but maintain deterministic ordering
        all_slices = [None] * len(dicom_files)  # Pre-allocate with correct size
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {executor.submit(process_2d_with_index, (i, dcm_path)): i
                              for i, dcm_path in enumerate(dicom_files)}

            for future in as_completed(future_to_index):
                try:
                    original_index, slices = future.result()
                    all_slices[original_index] = slices
                except Exception as e:
                    original_index = future_to_index[future]
                    all_slices[original_index] = []

        # Flatten the list and remove empty results
        result_slices = []
        for slices in all_slices:
            if slices:
                result_slices.extend(slices)
        return result_slices

    elif mode == "2.5D":
        # For 2.5D mode, load all DICOMs first, then create triplets
        if len(dicom_files) < 3:
            # Fallback to 2D mode if insufficient slices
            return process_dicom_for_yolo(series_path, "2D")

        # Load all frames first (keep as grayscale for 2.5D stacking)
        # Process in parallel but maintain deterministic ordering
        all_frames = [None] * len(dicom_files)  # Pre-allocate with correct size
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {executor.submit(process_with_index, (i, dcm_path)): i
                              for i, dcm_path in enumerate(dicom_files)}

            for future in as_completed(future_to_index):
                try:
                    original_index, slices = future.result()
                    all_frames[original_index] = slices[0] if slices else None  # Take first frame from each DICOM
                except Exception as e:
                    original_index = future_to_index[future]
                    all_frames[original_index] = None

        # Remove None values and filter out failed processing
        all_frames = [frame for frame in all_frames if frame is not None]

        # Validation check
        if len(all_frames) < 3:
            print(f"Warning: Only {len(all_frames)} frames available, need at least 3 for 2.5D")
            return all_frames if all_frames else []

        # Create RGB triplets for 2.5D (match validation script RGB mode)
        # Use stride=1 for dense sliding window (overlapping triplets)
        rgb_slices = []
        for i in range(1, len(all_frames)-1):
            try:
                prev_frame = all_frames[i-1]
                curr_frame = all_frames[i]
                next_frame = all_frames[i+1]

                # Ensure all frames are valid and 2D
                if prev_frame is None or curr_frame is None or next_frame is None:
                    continue

                if not (prev_frame.shape == curr_frame.shape == next_frame.shape):
                    print(f"Warning: Frame shape mismatch at index {i}")
                    continue

                # Stack as RGB channels
                rgb_img = np.stack([prev_frame, curr_frame, next_frame], axis=-1)

                # Validate output shape
                if rgb_img.shape[-1] != 3 or rgb_img.ndim != 3:
                    print(f"Warning: Invalid RGB shape {rgb_img.shape} at index {i}")
                    continue

                rgb_slices.append(rgb_img)

            except Exception as e:
                print(f"Error creating RGB triplet at index {i}: {e}")
                continue

        print(f"Created {len(rgb_slices)} valid 2.5D slices from {len(all_frames)} frames")
        return rgb_slices if rgb_slices else all_frames

    else:
        raise ValueError(f"Unsupported YOLO mode: {mode}. Use '2D' or '2.5D'")

def predict_yolo_model(slices: List[np.ndarray], model_path: str) -> tuple:
    """Run YOLO inference using a specific model"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1

    # Load model
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path, task='detect')

    max_conf_all = 0.0
    per_class_max = np.zeros(len(YOLO_LABELS), dtype=np.float32)

    # Process in batches (match validation script - no padding, variable batch sizes)
    for i in range(0, len(slices), BATCH_SIZE):
        batch_slices = slices[i:i+BATCH_SIZE]
        #if (len(batch_slices) < 32): #and (model_name == 'effv2s'):
        #    batch_slices += [batch_slices[0]] * (32 - len(batch_slices))
                
        with torch.no_grad():
            results = model.predict(
                batch_slices,
                verbose=False,
                conf=0.01,
                imgsz=512,
            
                # Remove imgsz and explicit batch params to match validation script exactly
            )

        for r in results:
            if r is None or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                continue
            try:
                confs = r.boxes.conf
                clses = r.boxes.cls
                for j in range(len(confs)):
                    c = float(confs[j].item())
                    k = int(clses[j].item())
                    if 0 <= k < len(YOLO_LABELS) and c > per_class_max[k]:
                        per_class_max[k] = c
            except Exception:
                pass

    print(f"YOLO predictions: {per_class_max}")

    # Clear cache after each model to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_cls_pred = float(per_class_max.max())
    print(f"Max confidence: {final_cls_pred}")

    return final_cls_pred, per_class_max

def main():
    """Main test function"""
    print("=== YOLO Model Test ===")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # DICOM series path
    #series_uid = "1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317"
    series_uid = "1.2.826.0.1.3680043.8.498.10009383108068795488741533244914370182"
    #series_uid = "1.2.826.0.1.3680043.8.498.10022688097731894079510930966432818105"
    series_path = f"/home/sersasj/RSNA-IAD-Codebase/data/series/{series_uid}"

    # Model path
    model_path = "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/cv_effnetv2_s_drop_path_25d_fold0/weights/best.pt"

    # Check if paths exist
    if not os.path.exists(series_path):
        print(f"ERROR: Series path does not exist: {series_path}")
        return

    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        return

    print(f"Series path: {series_path}")
    print(f"Model path: {model_path}")

    # Process DICOM
    print("\n=== Processing DICOM ===")
    start_time = time.time()
    try:
        yolo_slices = process_dicom_for_yolo(series_path, mode="2.5D")
        print(f"DICOM processing time: {time.time() - start_time:.2f}s")
        print(f"Number of slices created: {len(yolo_slices)}")

        if not yolo_slices:
            print("ERROR: No slices created from DICOM series")
            return

        print(f"Slice shape: {yolo_slices[0].shape}")

    except Exception as e:
        print(f"ERROR processing DICOM: {e}")
        return

    # Run prediction
    print("\n=== Running YOLO Prediction ===")
    start_time = time.time()
    try:
        final_cls_pred, yolo_loc_preds = predict_yolo_model(yolo_slices, model_path)
        prediction_time = time.time() - start_time

        print(f"Prediction time: {prediction_time:.2f}s")
        print(f"Final class prediction: {final_cls_pred}")
        print("Location predictions:")
        for i, (label, pred) in enumerate(zip(YOLO_LABELS, yolo_loc_preds)):
            print(f"  {label}: {pred:.4f}")

    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return

    # Summary
    print("\n=== Test Summary ===")
    print(f"✅ Model loaded successfully: {os.path.basename(model_path)}")
    print(f"✅ DICOM processed successfully: {len(yolo_slices)} slices")
    print(f"✅ Prediction completed in {prediction_time:.2f}s")
    print(f"✅ Max confidence: {final_cls_pred:.4f}")

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
