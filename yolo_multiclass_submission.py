# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-28T14:35:31.836580Z","iopub.execute_input":"2025-08-28T14:35:31.836882Z","iopub.status.idle":"2025-08-28T14:35:32.182084Z","shell.execute_reply.started":"2025-08-28T14:35:31.836855Z","shell.execute_reply":"2025-08-28T14:35:32.181243Z"}}
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
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

# Constants
IMG_SIZE = 512
FACTOR = 1
# Batch size for batched YOLO inference
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4  # For parallel DICOM reading

# Label mappings
LABELS_TO_IDX = {
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

LABELS = sorted(list(LABELS_TO_IDX.keys()))
LABEL_COLS = LABELS + ['Aneurysm Present']

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

# Model configurations - Add your models here
MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold02/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold12/weights/best.pt",
        "fold": "1",
        "weight": 1.0,
        "name": "YOLOv11n_fold1"
    },  
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold22/weights/best.pt",
        "fold": "2",
        "weight": 1.0,
        "name": "YOLOv11n_fold2"
    },
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold32/weights/best.pt",
        "fold": "3",
        "weight": 1.0,
        "name": "YOLOv11n_fold3"
    },
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold42/weights/best.pt",
        "fold": "4",
        "weight": 1.0,
        "name": "YOLOv11n_fold4"
    }
]

def load_models():
    """Load all models on single GPU (cuda:0)"""
    models = []
    for config in MODEL_CONFIGS:
        print(f"Loading model: {config['name']} on cuda:0")
        
        model = YOLO(config["path"])
        model.to("cuda:0")
        
        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"]
        }
        models.append(model_dict)
    return models

# Load all models
models = load_models()
print(f"Loaded {len(models)} models on single GPU")

@torch.no_grad()
def eval_one_series_ensemble(slices: List[np.ndarray]):
    """Run inference using all models on single GPU"""
    if not slices:
        return 0.1, np.ones(len(LABELS)) * 0.1
    
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    for model_dict in models:
        model = model_dict["model"]
        weight = model_dict["weight"]
        
        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(LABELS), dtype=np.float32)
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                
                results = model.predict(
                    batch_slices, 
                    verbose=False, 
                    batch=len(batch_slices), 
                    device="cuda:0", 
                    conf=0.01
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
                            if c > max_conf_all:
                                max_conf_all = c
                            if 0 <= k < len(LABELS) and c > per_class_max[k]:
                                per_class_max[k] = c
                    except Exception:
                        try:
                            batch_max = float(r.boxes.conf.max().item())
                            if batch_max > max_conf_all:
                                max_conf_all = batch_max
                        except Exception:
                            pass
            
            ensemble_cls_preds.append(max_conf_all * weight)
            ensemble_loc_preds.append(per_class_max * weight)
            total_weight += weight
            
        except Exception as e:
            print(f"Error in model {model_dict['name']}: {e}")
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(LABELS)) * 0.1 * weight)
            total_weight += weight
    
    if total_weight > 0:
        final_cls_pred = sum(ensemble_cls_preds) / total_weight
        final_loc_preds = sum(ensemble_loc_preds) / total_weight
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(LABELS)) * 0.1
    
    return final_cls_pred, final_loc_preds

def _predict_inner(series_path):
    """Internal prediction logic with parallel preprocessing and single GPU inference"""
    series_path = Path(series_path)

    dicom_files = collect_series_slices(series_path)
    
    # Parallel DICOM processing
    all_slices: List[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_dicom_file, dcm_path): dcm_path 
                         for dcm_path in dicom_files}
        
        for future in as_completed(future_to_file):
            try:
                slices = future.result()
                all_slices.extend(slices)
            except Exception as e:
                dcm_path = future_to_file[future]
                print(f"Failed processing {dcm_path.name}: {e}")

    # If no valid images were read, return a safe fallback row
    if not all_slices:
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions

    cls_prob, loc_probs = eval_one_series_ensemble(all_slices)

    # Ensure we have the right number of location probabilities
    if len(loc_probs) != len(LABELS):
        loc_probs = np.ones(len(LABELS)) * 0.1

    loc_probs = list(loc_probs)
    values = loc_probs + [cls_prob]

    predictions = pl.DataFrame(
        data=[values],
        schema=LABEL_COLS,
        orient='row'
    )
    return predictions

def predict(series_path: str):
    """
    Top-level prediction function passed to the server.
    """
    try:
        return _predict_inner(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        # Cleanup
        if os.path.exists('/kaggle'):
            shared_dir = '/kaggle/shared'
        else:
            shared_dir = os.path.join(os.getcwd(), 'kaggle_shared')
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # Memory cleanup for single GPU
        torch.cuda.empty_cache()
        gc.collect()



# %% [code] {"execution":{"iopub.status.busy":"2025-08-28T14:35:32.183269Z","iopub.execute_input":"2025-08-28T14:35:32.183587Z","iopub.status.idle":"2025-08-28T14:36:01.359904Z","shell.execute_reply.started":"2025-08-28T14:35:32.183561Z","shell.execute_reply":"2025-08-28T14:36:01.358946Z"}}
# Main execution
if __name__ == "__main__":
    st = time.time()
    # Initialize the inference server with our main `predict` function.
    inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
    
    # Check if the notebook is running in the competition environment or a local session.
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway()
        
        submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
        # Optional: print head instead of display to avoid dependency on notebook environment
        display(submission_df)
    
    print(f"Total time: {time.time() - st}")