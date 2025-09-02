import torch
import numpy as np
import pandas as pd
import random

import polars as pl

from pathlib import Path

import ast
import pydicom

from pathlib import Path
import os
import cv2
import multiprocessing
from tqdm import tqdm
import torch.cuda.amp as amp
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

import kaggle_evaluation.rsna_inference_server
import shutil
import gc
import time

# Optimization settings
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

# Constants
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = int(os.getenv("MAX_DICOM_WORKERS", "8"))  # Configurable worker count

# Labels
LABELS = [
    'Anterior Communicating Artery', 'Basilar Tip', 'Left Anterior Cerebral Artery',
    'Left Infraclinoid Internal Carotid Artery', 'Left Middle Cerebral Artery',
    'Left Posterior Communicating Artery', 'Left Supraclinoid Internal Carotid Artery',
    'Other Posterior Circulation', 'Right Anterior Cerebral Artery',
    'Right Infraclinoid Internal Carotid Artery', 'Right Middle Cerebral Artery',
    'Right Posterior Communicating Artery', 'Right Supraclinoid Internal Carotid Artery'
]
LABEL_COLS = LABELS + ['Aneurysm Present']

# Model configurations
ANEURYSM_MODELS = [
    "/kaggle/input/rsna-sergio-models/cv_y11s_positive_only_pretrain_hard_negatives_fold0/weights/best.pt",
    "/kaggle/input/rsna-sergio-models/cv_y11s_positive_only_pretrain_hard_negatives_fold1/weights/best.pt",
]

LOCATION_MODELS = [
    "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold02/weights/best.pt",
    "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold12/weights/best.pt",
    "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold22/weights/best.pt"
]

def load_models(model_paths):
    """Load models on GPU"""
    models = []
    for path in model_paths:
        model = YOLO(path)
        model.to("cuda:0")
        models.append(model)
    return models

# Load models
models_aneurysm = load_models(ANEURYSM_MODELS)
models_location = load_models(LOCATION_MODELS)

def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)

def process_dicom(dcm_path: Path) -> Tuple[int, Optional[np.ndarray]]:
    """Process single DICOM file to normalized BGR image with index"""
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        pix = ds.pixel_array
        
        # Apply HU conversion
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        img = pix.astype(np.float32) * slope + intercept
        
        # Handle multi-frame or RGB
        if img.ndim == 3:
            if img.shape[-1] == 3:  # RGB
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:  # Multi-frame, take first
                img = img[0]
        
        # Normalize to 0-255 using the original function
        img_u8 = min_max_normalize(img)
        
        # Convert to BGR for YOLO
        if img_u8.ndim == 2:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            
        # Return with original index from filename for sorting
        return (int(dcm_path.stem) if dcm_path.stem.isdigit() else 0, img_u8)
    except Exception as e:
        print(f"Failed processing {dcm_path.name}: {e}")
        return (0, None)

def process_dicom_batch(dicom_files: List[Path], max_workers: int = None) -> List[np.ndarray]:
    """Process multiple DICOM files in parallel with ThreadPoolExecutor"""
    if max_workers is None:
        max_workers = min(MAX_WORKERS, len(dicom_files))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_dicom, dcm_file): i 
            for i, dcm_file in enumerate(dicom_files)
        }
        
        # Collect results maintaining order
        indexed_results = [None] * len(dicom_files)
        
        for future in as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                file_index, img = future.result()
                if img is not None:
                    indexed_results[original_index] = img
            except Exception as e:
                print(f"Error processing DICOM {original_index}: {e}")
                indexed_results[original_index] = None
    
    # Filter out None results
    return [img for img in indexed_results if img is not None]

def get_dicom_files(series_dir: Path) -> List[Path]:
    """Get all DICOM files from directory"""
    files = []
    for root, _, filenames in os.walk(series_dir):
        for f in filenames:
            if f.lower().endswith('.dcm'):
                files.append(Path(root) / f)
    return sorted(files)

def predict_batch(models, images):
    """Run ensemble prediction on batch of images"""
    if not images:
        return np.array([0.1])
    
    all_probs = []
    
    for model in models:
        try:
            # Process in batches
            batch_probs = []
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE]
                results = model.predict(batch, verbose=False, device="cuda:0", conf=0.01)
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Get max confidence per class for this image
                        confs = result.boxes.conf.cpu().numpy()
                        if hasattr(result.boxes, 'cls'):
                            clses = result.boxes.cls.cpu().numpy().astype(int)
                            class_probs = np.zeros(len(LABELS))
                            for conf, cls in zip(confs, clses):
                                if 0 <= cls < len(LABELS):
                                    class_probs[cls] = max(class_probs[cls], conf)
                            batch_probs.append(class_probs)
                        else:
                            # No class info, use max confidence for all classes
                            batch_probs.append(np.full(len(LABELS), confs.max()))
                    else:
                        batch_probs.append(np.zeros(len(LABELS)))
            
            if batch_probs:
                # Take max across all images for this model
                all_probs.append(np.max(batch_probs, axis=0))
            else:
                all_probs.append(np.zeros(len(LABELS)))
                
        except Exception as e:
            print(f"Model prediction error: {e}")
            all_probs.append(np.full(len(LABELS), 0.1))
    
    # Average across models
    return np.mean(all_probs, axis=0) if all_probs else np.full(len(LABELS), 0.1)

@torch.no_grad()
def predict(series_path: str):
    """Main prediction function with parallel DICOM processing"""
    try:
        series_path = Path(series_path)
        
        # Get all DICOM files
        dicom_files = get_dicom_files(series_path)
        
        if not dicom_files:
            return pl.DataFrame(
                data=[[0.1] * len(LABEL_COLS)],
                schema=LABEL_COLS
            )
        
        # Process DICOM files in parallel
        print(f"Processing {len(dicom_files)} DICOM files with {MAX_WORKERS} workers...")
        start_time = time.time()
        
        images = process_dicom_batch(dicom_files, MAX_WORKERS)
        
        processing_time = time.time() - start_time
        print(f"DICOM processing completed in {processing_time:.2f}s ({len(images)} valid images)")
        
        if not images:
            # No valid images
            return pl.DataFrame(
                data=[[0.1] * len(LABEL_COLS)],
                schema=LABEL_COLS
            )
        
        # Single pass aneurysm detection - collect both probabilities and slice scores
        aneurysm_probs = []
        slice_scores = np.zeros(len(images))
        
        for model in models_aneurysm:
            model_slice_probs = []
            
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE]
                results = model.predict(batch, verbose=False, device="cuda:0", conf=0.01)
                
                for j, result in enumerate(results):
                    slice_idx = i + j
                    if result.boxes is not None and len(result.boxes) > 0:
                        conf = result.boxes.conf.max().item()
                        model_slice_probs.append(conf)
                        slice_scores[slice_idx] += conf  # Accumulate for slice selection
                    else:
                        model_slice_probs.append(0.0)
            
            # Max slice probability for this model
            aneurysm_probs.append(max(model_slice_probs) if model_slice_probs else 0.1)
        
        # Average aneurysm probability across models
        aneurysm_prob = np.mean(aneurysm_probs)
        
        # Select top slices based on accumulated scores (no additional model calls)
        if len(images) > 3:
            top_indices = np.argsort(-slice_scores)[:3]
            top_images = [images[i] for i in top_indices]
        else:
            top_images = images
        
        # Location prediction on selected slices
        location_probs = predict_batch(models_location, top_images)
        
        # Combine results
        final_probs = list(location_probs) + [aneurysm_prob]
        
        return pl.DataFrame(
            data=[final_probs],
            schema=LABEL_COLS
        )
        
    except Exception as e:
        print(f"Prediction error for {os.path.basename(series_path)}: {e}")
        return pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS
        )
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Clean shared directory
        shared_dir = '/kaggle/shared' if os.path.exists('/kaggle') else 'kaggle_shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)