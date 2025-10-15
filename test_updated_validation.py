#!/usr/bin/env python3
"""
Test updated validation script logic on specific series
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

# Import from validation script
from yolo_multiclass_validation_old_copy import read_dicom_frames_hu, min_max_normalize, ordered_dcm_paths

# Configuration
TARGET_SERIES = "1.2.826.0.1.3680043.8.498.10009383108068795488741533244914370182"
MODEL_PATH = "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/cv_effnetv2_s_drop_path_25d_fold0/weights/best.pt"
SERIES_PATH = f"/home/sersasj/RSNA-IAD-Codebase/data/series/{TARGET_SERIES}"
BATCH_SIZE = 16

def test_updated_preprocessing():
    print(f"Testing updated preprocessing on series: {TARGET_SERIES}")
    
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Get DICOM files
    series_dir = Path(SERIES_PATH)
    dicoms, _ = ordered_dcm_paths(series_dir)
    print(f"Found {len(dicoms)} DICOM files")
    
    # Updated RGB mode processing (match data preparation approach)
    slices_hu = []
    for dcm_path in dicoms:
        try:
            frames = read_dicom_frames_hu(dcm_path)
        except Exception as e:
            print(f"[SKIP] {dcm_path.name}: {e}")
            continue
        for f in frames:
            # Keep as HU values, don't normalize yet (match data prep approach)
            slices_hu.append(f.astype(np.float32))
    
    print(f"Loaded {len(slices_hu)} slices (as HU values)")
    
    if len(slices_hu) < 3:
        print("Not enough slices for RGB mode")
        return
    
    # Create RGB triplets (match updated validation script approach)
    slice_indices = list(range(1, len(slices_hu)-1))
    print(f"Will create {len(slice_indices)} RGB triplets")
    
    # Process in batches
    batch = []
    max_conf_all = 0.0
    total_dets = 0
    
    for i in slice_indices:
        try:
            # Get 3 consecutive slices
            prev_frame = slices_hu[i-1]
            curr_frame = slices_hu[i]
            next_frame = slices_hu[i+1]
            
            # Ensure all frames are valid and same shape
            if prev_frame is None or curr_frame is None or next_frame is None:
                continue
                
            if not (prev_frame.shape == curr_frame.shape == next_frame.shape):
                print(f"Warning: Frame shape mismatch at index {i}")
                continue
            
            # Create RGB using individual normalization per channel (match data prep)
            r_channel = min_max_normalize(prev_frame)
            g_channel = min_max_normalize(curr_frame)
            b_channel = min_max_normalize(next_frame)
            rgb_img = np.stack([r_channel, g_channel, b_channel], axis=-1)
            
            # Validate output shape
            if rgb_img.shape[-1] != 3 or rgb_img.ndim != 3:
                print(f"Warning: Invalid RGB shape {rgb_img.shape} at index {i}")
                continue
            
            # Apply resizing (match data prep behavior)
            import cv2
            IMG_SIZE = 512
            if IMG_SIZE > 0 and (rgb_img.shape[0] != IMG_SIZE or rgb_img.shape[1] != IMG_SIZE):
                rgb_img = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            
            batch.append(rgb_img)
            
            if len(batch) >= BATCH_SIZE:
                # Process batch
                results = model.predict(batch, verbose=False, conf=0.01, imgsz=512)
                for r in results:
                    if not r or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                        continue
                    try:
                        batch_max = float(r.boxes.conf.max().item())
                        if batch_max > max_conf_all:
                            max_conf_all = batch_max
                        total_dets += int(len(r.boxes))
                    except Exception:
                        pass
                batch.clear()
                
        except Exception as e:
            print(f"Error creating RGB triplet at index {i}: {e}")
            continue
    
    # Process remaining batch
    if batch:
        results = model.predict(batch, verbose=False, conf=0.01, imgsz=512)
        for r in results:
            if not r or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                continue
            try:
                batch_max = float(r.boxes.conf.max().item())
                if batch_max > max_conf_all:
                    max_conf_all = batch_max
                total_dets += int(len(r.boxes))
            except Exception:
                pass
    
    print(f"Max confidence: {max_conf_all:.6f}")
    print(f"Total detections: {total_dets}")
    print(f"Created {len(slice_indices)} slices")
    
    if len(slice_indices) > 0:
        print(f"Final slice shape: {rgb_img.shape}")

if __name__ == "__main__":
    test_updated_preprocessing()