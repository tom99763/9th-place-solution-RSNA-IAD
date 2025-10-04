# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.031376Z","iopub.execute_input":"2025-10-01T21:21:49.031922Z","iopub.status.idle":"2025-10-01T21:21:49.038786Z","shell.execute_reply.started":"2025-10-01T21:21:49.031884Z","shell.execute_reply":"2025-10-01T21:21:49.038120Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.039968Z","iopub.execute_input":"2025-10-01T21:21:49.040291Z","iopub.status.idle":"2025-10-01T21:21:49.057025Z","shell.execute_reply.started":"2025-10-01T21:21:49.040267Z","shell.execute_reply":"2025-10-01T21:21:49.056377Z"},"jupyter":{"outputs_hidden":false}}

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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.057630Z","iopub.execute_input":"2025-10-01T21:21:49.057837Z","iopub.status.idle":"2025-10-01T21:21:49.075419Z","shell.execute_reply.started":"2025-10-01T21:21:49.057817Z","shell.execute_reply":"2025-10-01T21:21:49.074940Z"},"jupyter":{"outputs_hidden":false}}


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

def process_dicom_file_yolo(dcm_path: Path) -> List[np.ndarray]:
    """Process single DICOM file for YOLO - for parallel processing"""
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.076473Z","iopub.execute_input":"2025-10-01T21:21:49.076673Z","iopub.status.idle":"2025-10-01T21:21:49.127371Z","shell.execute_reply.started":"2025-10-01T21:21:49.076658Z","shell.execute_reply":"2025-10-01T21:21:49.126820Z"},"jupyter":{"outputs_hidden":false}}
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
        "name": "YOLOv11m_fold1"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold2/weights/best.engine",
        "fold": "3",
        "weight": 1.0,
        "name": "YOLOv11m_fold1"
    }, 
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold3/weights/best.engine",
        "fold": "4",
        "weight": 1.0,
        "name": "mobile_net_more_negatives"
    },   
    #{
    #    "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold2/weights/best.pt",
    #    "fold": "2",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold2"
    #}
]

# ====================================================
# Model Loading and Inference
# ====================================================
# Global variables
YOLO_MODELS = []
EFFNET_AUX_MODELS = []
EFFNET_AUX_TRANSFORM = None

CONVNEXT_AUX_LOSS_CKPTS = [

    
]

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.127992Z","iopub.execute_input":"2025-10-01T21:21:49.128204Z","iopub.status.idle":"2025-10-01T21:21:49.131358Z","shell.execute_reply.started":"2025-10-01T21:21:49.128183Z","shell.execute_reply":"2025-10-01T21:21:49.130657Z"},"jupyter":{"outputs_hidden":false}}
# !ls /kaggle/input/rsna-iad-modelzoo

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.217034Z","iopub.execute_input":"2025-10-01T21:21:49.217308Z","iopub.status.idle":"2025-10-01T21:21:49.223573Z","shell.execute_reply.started":"2025-10-01T21:21:49.217282Z","shell.execute_reply":"2025-10-01T21:21:49.222814Z"},"jupyter":{"outputs_hidden":false}}

def load_yolo_models():
    """Load all YOLO models"""
    models = []
    for config in YOLO_MODEL_CONFIGS:
        # YOLO automatically handles both .pt and .engine files
        model = YOLO(config["path"], task='detect')
        
        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"]
        }
        models.append(model_dict)
    return models

def load_all_models():
    """Load all models (EfficientNet + YOLO)"""
    global YOLO_MODELS, EFFNET_AUX_MODELS, EFFNET_AUX_TRANSFORM
    



    # Load YOLO models
    YOLO_MODELS = load_yolo_models()
    
    # Initialize transforms
    
    # Warm up models
    dummy_yolo_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    
    with torch.no_grad():
        
        for model_dict in YOLO_MODELS:
            model = model_dict["model"]
            _ = model.predict([dummy_yolo_image], verbose=False, device=device)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.224967Z","iopub.execute_input":"2025-10-01T21:21:49.225169Z","iopub.status.idle":"2025-10-01T21:21:49.243813Z","shell.execute_reply.started":"2025-10-01T21:21:49.225155Z","shell.execute_reply":"2025-10-01T21:21:49.243007Z"},"jupyter":{"outputs_hidden":false}}


###

def aggregate_top3(confs: List[float]) -> float:
    """Aggregate confidences using top3 strategy (mean of top 3)"""
    if not confs:
        return 0.0
    arr = np.array(confs)
    if len(arr) < 3:
        return float(np.mean(arr))
    top_3 = np.partition(arr, -3)[-3:]
    return float(np.mean(top_3))


@torch.no_grad()
def predict_yolo_ensemble(slices: List[np.ndarray]):
    """Run YOLO inference using all models"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1
    
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    for model_dict in YOLO_MODELS:
        model = model_dict["model"]
        weight = model_dict["weight"]
        
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
                        batch=len(batch_slices), 
                        device=device, 
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
            
            ensemble_cls_preds.append(agg_conf * weight)
            ensemble_loc_preds.append(per_class_agg * weight)
            total_weight += weight
            
        except Exception as e:
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            total_weight += weight
    
    if total_weight > 0:
        final_cls_pred = sum(ensemble_cls_preds) / total_weight
        final_loc_preds = sum(ensemble_loc_preds) / total_weight
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1
    
    return final_cls_pred, final_loc_preds

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.244621Z","iopub.execute_input":"2025-10-01T21:21:49.244865Z","iopub.status.idle":"2025-10-01T21:21:49.261256Z","shell.execute_reply.started":"2025-10-01T21:21:49.244844Z","shell.execute_reply":"2025-10-01T21:21:49.260611Z"},"jupyter":{"outputs_hidden":false}}



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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-01T21:21:49.261975Z","iopub.execute_input":"2025-10-01T21:21:49.262172Z","iopub.status.idle":"2025-10-01T21:21:49.282583Z","shell.execute_reply.started":"2025-10-01T21:21:49.262157Z","shell.execute_reply":"2025-10-01T21:21:49.281966Z"}}

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

        #generate eff preds match the label cols
        # eff_full_preds = effnet_preds
        # eff_full_preds = np.zeros(len(LABEL_COLS))
        # for i, label in enumerate(EFF_LABELS):
        #     if label in LABEL_COLS:
        #         label_idx = LABEL_COLS.index(label)
        #         eff_full_preds[label_idx] = effnet_preds[i]

        #generate yolo preds match the label cols
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
        print(e)
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
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.284054Z","iopub.execute_input":"2025-10-01T21:21:49.284247Z","iopub.status.idle":"2025-10-01T21:21:49.299796Z","shell.execute_reply.started":"2025-10-01T21:21:49.284232Z","shell.execute_reply":"2025-10-01T21:21:49.299244Z"},"jupyter":{"outputs_hidden":false}}
# _predict_inner("/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636")

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:21:49.300377Z","iopub.execute_input":"2025-10-01T21:21:49.300544Z","iopub.status.idle":"2025-10-01T21:23:06.410319Z","shell.execute_reply.started":"2025-10-01T21:21:49.300529Z","shell.execute_reply":"2025-10-01T21:23:06.409690Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-01T21:23:06.410989Z","iopub.execute_input":"2025-10-01T21:23:06.411231Z","iopub.status.idle":"2025-10-01T21:23:06.414542Z","shell.execute_reply.started":"2025-10-01T21:23:06.411214Z","shell.execute_reply":"2025-10-01T21:23:06.413848Z"},"jupyter":{"outputs_hidden":false}}
# !ls /kaggle/input/rsna-iad-modelzoo