# %% [code] {"execution":{"iopub.status.busy":"2025-10-06T17:23:04.032953Z","iopub.execute_input":"2025-10-06T17:23:04.033214Z","iopub.status.idle":"2025-10-06T17:25:23.581752Z","shell.execute_reply.started":"2025-10-06T17:23:04.033193Z","shell.execute_reply":"2025-10-06T17:25:23.580791Z"},"jupyter":{"outputs_hidden":false}}
!tar -xzvf /kaggle/input/offline-install-tensorrt/packages.tar.gz
!pip install --no-index --find-links=./packages tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs onnxruntime-gpu onnxslim

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:25:23.583987Z","iopub.execute_input":"2025-10-06T17:25:23.584467Z","iopub.status.idle":"2025-10-06T17:26:13.995998Z","shell.execute_reply.started":"2025-10-06T17:25:23.584432Z","shell.execute_reply":"2025-10-06T17:26:13.995188Z"}}
import os
import numpy as np
import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import warnings
import gc
import sys
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time
from collections import OrderedDict

warnings.filterwarnings('ignore')

# Data handling
import polars as pl
import pandas as pd

# ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.insert(0, "/kaggle/input/ultralytcs-timm-rsna/ultralytics-timm")

# YOLO
from ultralytics import YOLO

# Competition API
import kaggle_evaluation.rsna_inference_server

import cupy as cp
from cupyx.scipy.ndimage import zoom

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:13.996903Z","iopub.execute_input":"2025-10-06T17:26:13.997351Z","iopub.status.idle":"2025-10-06T17:26:14.003183Z","shell.execute_reply.started":"2025-10-06T17:26:13.997324Z","shell.execute_reply":"2025-10-06T17:26:14.002399Z"}}

# Optimization settings
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ====================================================
# Competition constants
# ====================================================
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# YOLO label mappings
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.004092Z","iopub.execute_input":"2025-10-06T17:26:14.004864Z","iopub.status.idle":"2025-10-06T17:26:14.027078Z","shell.execute_reply.started":"2025-10-06T17:26:14.004832Z","shell.execute_reply":"2025-10-06T17:26:14.026492Z"}}


# ====================================================
# YOLO DICOM Processing
# ====================================================
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
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
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

#def process_dicom_file_yolo(dcm_path: Path) -> List[np.ndarray]:
#    """Process single DICOM file for YOLO - for parallel processing"""
#    try:
#        frames = read_dicom_frames_hu(dcm_path)
#        processed_slices = []
#        for f in frames:
#            img_u8 = min_max_normalize(f)
#            if img_u8.ndim == 2:
#                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
#            processed_slices.append(img_u8)
#        return processed_slices
#    except Exception as e:
#        return []

def process_dicom_file_yolo(dcm_path: Path) -> List[np.ndarray]:
    """Process single DICOM file for YOLO - for parallel processing
    Ensures all outputs are exactly 512x512 to match TensorRT engine expectations
    """
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
        pass
    dcm_paths.sort()
    return dcm_paths

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.028632Z","iopub.execute_input":"2025-10-06T17:26:14.028832Z","iopub.status.idle":"2025-10-06T17:26:14.045321Z","shell.execute_reply.started":"2025-10-06T17:26:14.028816Z","shell.execute_reply":"2025-10-06T17:26:14.044694Z"}}
# %load ../src/models/segmentation_classification.py


# ====================================================
# YOLO Configuration
# ====================================================
IMG_SIZE = 512
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4

YOLO_MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold03/weights/best.engine",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11m_fold0"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold12/weights/best.engine",
        "fold": "1",
        "weight": 1.0,
        "name": "YOLOv11n_fold1"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold2/weights/best.engine",
        "fold": "2",
        "weight": 1.0,
        "name": "YOLOv11n_fold2"
    }, 
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold3/weights/best.engine",
        "fold": "3",
        "weight": 1.0,
        "name": "YOLOv11n_fold3"
    },   
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold4/weights/best.engine",
        "fold": "4",
        "weight": 1.0,
        "name": "YOLOv11n_fold4"
    }
]

# ====================================================
# Model Loading and Inference
# ====================================================
# Global variables
YOLO_MODELS = []
EFFNET_AUX_MODELS = []
EFFNET_AUX_TRANSFORM = None
_MODELS_LOADED = False

CONVNEXT_AUX_LOSS_CKPTS = [

    
]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.045967Z","iopub.execute_input":"2025-10-06T17:26:14.046147Z","iopub.status.idle":"2025-10-06T17:26:14.063924Z","shell.execute_reply.started":"2025-10-06T17:26:14.046133Z","shell.execute_reply":"2025-10-06T17:26:14.063298Z"}}
# !ls /kaggle/input/rsna-iad-modelzoo

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.064502Z","iopub.execute_input":"2025-10-06T17:26:14.064744Z","iopub.status.idle":"2025-10-06T17:26:14.078372Z","shell.execute_reply.started":"2025-10-06T17:26:14.064728Z","shell.execute_reply":"2025-10-06T17:26:14.077878Z"}}

def load_yolo_models():
    """Load YOLO models with proper device assignment to avoid OOM"""
    models = []
    
    num_gpus = torch.cuda.device_count()
    
    for idx, config in enumerate(YOLO_MODEL_CONFIGS):
        device_id = 0
        
        model = YOLO(config["path"], task='detect')
        
        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"],
            "device": device_id  # Store assigned device
        }
        models.append(model_dict)
    
    return models

def load_all_models():
    """Load all models with memory-efficient warmup"""
    global YOLO_MODELS, EFFNET_AUX_MODELS, _MODELS_LOADED

    if _MODELS_LOADED:
        return

    # Load YOLO models
    YOLO_MODELS = load_yolo_models()

    _MODELS_LOADED = True

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.079180Z","iopub.execute_input":"2025-10-06T17:26:14.079727Z","iopub.status.idle":"2025-10-06T17:26:14.100575Z","shell.execute_reply.started":"2025-10-06T17:26:14.079690Z","shell.execute_reply":"2025-10-06T17:26:14.099916Z"}}


###

def aggregate_top3(confs: List[float]) -> float:
    """Aggregate confidences using top3 strategy (mean of top 3)"""
    if not confs:
        return 0.0
    arr = np.array(confs)
    if len(arr) < 1:
        return float(np.mean(arr))
    top_3 = np.partition(arr, -1)[-1:]
    return float(np.mean(top_3))


@torch.no_grad()
def predict_yolo_ensemble(slices: List[np.ndarray]):
    """Run YOLO inference using all models with proper device management"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1
    
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    for model_dict in YOLO_MODELS:
        model = model_dict["model"]
        weight = model_dict["weight"]
        device_id = model_dict.get("device", 0)  # Get assigned device
        
        try:
            all_confs = []
            per_class_confs = [[] for _ in range(len(YOLO_LABELS))]
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                
                with torch.no_grad():
                    results = model.predict(
                        batch_slices, 
                        verbose=False,
                        imgsz=512,
                        batch=len(batch_slices), 
                        device=device_id, 
                        conf=0.01,
                        half=True
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
                            all_confs.append(c)
                            if 0 <= k < len(YOLO_LABELS):
                                per_class_confs[k].append(c)
                    except Exception:
                        try:
                            for c in r.boxes.conf:
                                all_confs.append(float(c.item()))
                        except Exception:
                            pass
            
            # Aggregate using top3 strategy
            agg_conf = aggregate_top3(all_confs) if all_confs else 0.1
            per_class_agg = np.array([aggregate_top3(confs) if confs else 0.0 
                                      for confs in per_class_confs], dtype=np.float32)
            
            print(model_dict["name"], per_class_agg)
            ensemble_cls_preds.append(agg_conf * weight)
            ensemble_loc_preds.append(per_class_agg * weight)
            total_weight += weight
            
            # Clear cache after each model to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with model {model_dict['name']}: {e}")
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            total_weight += weight

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if total_weight > 0:
        final_cls_pred = sum(ensemble_cls_preds) / total_weight
        final_loc_preds = sum(ensemble_loc_preds) / total_weight
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_cls_pred, final_loc_preds

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.101246Z","iopub.execute_input":"2025-10-06T17:26:14.101428Z","iopub.status.idle":"2025-10-06T17:26:14.118359Z","shell.execute_reply.started":"2025-10-06T17:26:14.101414Z","shell.execute_reply":"2025-10-06T17:26:14.117797Z"}}
def process_dicom_for_yolo(series_path: str) -> List[np.ndarray]:
    """Process DICOM for YOLO with parallel processing"""
    series_path = Path(series_path)
    dicom_files = collect_series_slices(series_path)
    
    all_slices: List[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_dicom_file_yolo, dcm_path): dcm_path 
                         for dcm_path in dicom_files}
        
        for future in as_completed(future_to_file):
            try:
                slices = future.result()
                all_slices.extend(slices)
            except Exception as e:
                pass
    
    return all_slices

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.119191Z","iopub.execute_input":"2025-10-06T17:26:14.119392Z","iopub.status.idle":"2025-10-06T17:26:14.135860Z","shell.execute_reply.started":"2025-10-06T17:26:14.119376Z","shell.execute_reply":"2025-10-06T17:26:14.135131Z"}}

def _predict_inner(series_path: str) -> pl.DataFrame:
    """Main ensemble prediction logic"""
    global YOLO_MODELS, EFFNET_AUX_MODELS
    
    # Load models if not already loaded
    if not YOLO_MODELS or not EFFNET_AUX_MODELS:
        load_all_models()
    
    try:
        # print(f"{effnet_volume.shape=}, {effnet_aux_volume.shape=}")
        yolo_slices = process_dicom_for_yolo(series_path)
        
        # Get EfficientNet predictions

        
        # Get YOLO predictions
        yolo_cls_pred, yolo_loc_preds = predict_yolo_ensemble(yolo_slices)

        yolo_full_preds = np.zeros(len(LABEL_COLS))
        for i, label in enumerate(YOLO_LABELS):
            if label in LABEL_COLS:
                label_idx = LABEL_COLS.index(label)
                yolo_full_preds[label_idx] = yolo_loc_preds[i]
        aneurysm_idx = LABEL_COLS.index('Aneurysm Present')
        yolo_full_preds[aneurysm_idx] = yolo_cls_pred
        ensemble_preds = yolo_full_preds
        # Create output dataframe
        predictions_df = pl.DataFrame(
            data=[ensemble_preds.tolist()],
            schema=LABEL_COLS,
            orient='row'
        )
        
        return predictions_df
        
    except Exception as e:
        print(f"Critical error in prediction: {e}")
        # Reset models to force reload on next attempt
        reset_models()
        # Return conservative predictions
        conservative_preds = [0.1] * len(LABEL_COLS)
        predictions_df = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions_df

def predict(series_path: str) -> pl.DataFrame:
    """
    Top-level prediction function passed to the server.
    Combines YOLO and EfficientNet with 50/50 weighting.
    """
    try:
        return _predict_inner(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        conservative_preds = [0.1] * len(LABEL_COLS)
        predictions = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        # Cleanup
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)

        # Aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.136629Z","iopub.execute_input":"2025-10-06T17:26:14.136857Z","iopub.status.idle":"2025-10-06T17:26:14.155081Z","shell.execute_reply.started":"2025-10-06T17:26:14.136838Z","shell.execute_reply":"2025-10-06T17:26:14.154588Z"}}
# _predict_inner("/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:14.155734Z","iopub.execute_input":"2025-10-06T17:26:14.155926Z","iopub.status.idle":"2025-10-06T17:26:49.787248Z","shell.execute_reply.started":"2025-10-06T17:26:14.155906Z","shell.execute_reply":"2025-10-06T17:26:49.786378Z"}}
if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize the inference server
    inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
    
    # Check if running in competition environment
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway()
        
        # Display results if in local mode
        submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
        print(f"Submission shape: {submission_df.shape}")
        display(submission_df)
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-06T17:26:49.788125Z","iopub.execute_input":"2025-10-06T17:26:49.788369Z","iopub.status.idle":"2025-10-06T17:26:49.792430Z","shell.execute_reply.started":"2025-10-06T17:26:49.788340Z","shell.execute_reply":"2025-10-06T17:26:49.791851Z"}}
# !ls /kaggle/input/rsna-iad-modelzoo