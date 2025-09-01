

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-29T13:01:07.034281Z","iopub.execute_input":"2025-08-29T13:01:07.034574Z","iopub.status.idle":"2025-08-29T13:01:12.873825Z","shell.execute_reply.started":"2025-08-29T13:01:07.034554Z","shell.execute_reply":"2025-08-29T13:01:12.873075Z"}}
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
        # RGB or multi-frame
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

MODEL_CONFIGS_ANEURYSM = [
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
    }
]
MODEL_CONFIGS_LOCATION = [
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
    }
]
def load_models(model_configs):
    """Load all models on single GPU (cuda:0)"""
    models = []
    for config in model_configs:
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
models_aneurysm= load_models(MODEL_CONFIGS_ANEURYSM)
models_loc = load_models(MODEL_CONFIGS_LOCATION)
print(f"Loaded {len(models_aneurysm)} models on single GPU")
print(f"Loaded {len(models_loc)} models on single GPU")

@torch.no_grad()
def eval_one_series_ensemble(slices: List[np.ndarray]):
    """Two-stage inference on single GPU:
    1) Predict all slices with models_aneurysm, get top-3 slice indices by ensemble prob.
    2) Run models_loc on those top-3 slices.
    Final per-class prob = min(aneurysm_prob, localization_prob).
    Aneurysm prob = average of each aneurysm model's max slice confidence.
    Returns (aneurysm_prob, per_class_probs[13]).
    """
    if not slices:
        return 0.1, np.ones(len(LABELS), dtype=np.float32) * 0.1

    num_slices = len(slices)

    # Stage 1: Aneurysm presence per-slice using ensemble of aneurysm models
    per_slice_accum = np.zeros(num_slices, dtype=np.float32)
    cls_total_weight = 0.0
    per_model_maxes: List[float] = []

    for m in models_aneurysm:
        model = m["model"]
        weight = float(m.get("weight", 1.0))

        try:
            # Collect max conf per slice for this model
            per_slice_max = np.zeros(num_slices, dtype=np.float32)

            for i in range(0, num_slices, BATCH_SIZE):
                batch = slices[i:i + BATCH_SIZE]
                results = model.predict(
                    batch,
                    verbose=False,
                    batch=len(batch),
                    device="cuda:0",
                    conf=0.01,
                )

                # Map each result to a slice index (offset by i)
                for j, r in enumerate(results):
                    idx = i + j
                    if r is None or r.boxes is None or getattr(r.boxes, "conf", None) is None or len(r.boxes) == 0:
                        continue
                    try:
                        # Max confidence over all detections = aneurysm presence prob for the slice
                        per_slice_max[idx] = float(r.boxes.conf.max().item())
                    except Exception:
                        # If anything odd, keep zero
                        pass

            # Record per-model max slice confidence
            try:
                per_model_maxes.append(float(per_slice_max.max().item()))
            except Exception:
                per_model_maxes.append(0.1)

            per_slice_accum += per_slice_max * weight
            cls_total_weight += weight
        except Exception as e:
            print(f"Error in aneurysm model {m.get('name','unknown')}: {e}")
            # Add a low-confidence fallback for this model to keep weights aligned
            per_slice_accum += (np.ones(num_slices, dtype=np.float32) * 0.1) * weight
            cls_total_weight += weight
            per_model_maxes.append(0.1)

    if cls_total_weight <= 0:
        # Fallback if no models produced output
        per_slice_probs = np.ones(num_slices, dtype=np.float32) * 0.1
    else:
        per_slice_probs = per_slice_accum / cls_total_weight

    # Final aneurysm probability as average of per-model maximum slice confidences
    if per_model_maxes:
        aneurysm_prob = float(np.mean(per_model_maxes))
    else:
        aneurysm_prob = 0.1

    # Pick top-K slice indices by aneurysm prob
    top_k = min(3, num_slices)
    if top_k == 0:
        return aneurysm_prob if aneurysm_prob > 0 else 0.1, np.ones(len(LABELS), dtype=np.float32) * 0.1

    top_indices = np.argsort(-per_slice_probs)[:top_k]
    top_slices = [slices[idx] for idx in top_indices]

    # Stage 2: Localization per-class over the top slices using ensemble of location models
    loc_total_weight = 0.0
    per_class_accum = np.zeros(len(LABELS), dtype=np.float32)

    for m in models_loc:
        model = m["model"]
        weight = float(m.get("weight", 1.0))
        try:
            # Per-slice per-class maxima for this model
            per_slice_per_class = []  # list of arrays shape [num_classes]

            # Usually only up to 3 images, but keep batching generic
            for i in range(0, len(top_slices), BATCH_SIZE):
                batch = top_slices[i:i + BATCH_SIZE]
                results = model.predict(
                    batch,
                    verbose=False,
                    batch=len(batch),
                    device="cuda:0",
                    conf=0.01,
                )
                for r in results:
                    cls_vec = np.zeros(len(LABELS), dtype=np.float32)
                    if r is not None and r.boxes is not None and getattr(r.boxes, "conf", None) is not None and len(r.boxes) > 0:
                        try:
                            confs = r.boxes.conf
                            clses = r.boxes.cls
                            for j in range(len(confs)):
                                c = float(confs[j].item())
                                k = int(clses[j].item())
                                if 0 <= k < len(LABELS) and c > cls_vec[k]:
                                    cls_vec[k] = c
                        except Exception:
                            try:
                                # If class info missing, fall back to presence only (no per-class boost)
                                max_c = float(r.boxes.conf.max().item())
                                if max_c > 0:
                                    cls_vec[:] = np.maximum(cls_vec, 0)
                            except Exception:
                                pass
                    per_slice_per_class.append(cls_vec)

            if per_slice_per_class:
                # Aggregate over top slices by max per class
                per_slice_stack = np.stack(per_slice_per_class, axis=0)  # [K, C]
                per_class_max_over_slices = per_slice_stack.max(axis=0)  # [C]
            else:
                per_class_max_over_slices = np.zeros(len(LABELS), dtype=np.float32)

            per_class_accum += per_class_max_over_slices * weight
            loc_total_weight += weight

        except Exception as e:
            print(f"Error in location model {m.get('name','unknown')}: {e}")
            per_class_accum += (np.ones(len(LABELS), dtype=np.float32) * 0.1) * weight
            loc_total_weight += weight

    if loc_total_weight <= 0:
        loc_probs = np.ones(len(LABELS), dtype=np.float32) * 0.1
    else:
        loc_probs = per_class_accum / loc_total_weight

    # Final rule: per-class probability = min(aneurysm_prob, localization_prob)
    final_loc_probs = np.minimum(loc_probs, aneurysm_prob).astype(np.float32)

    return float(aneurysm_prob), final_loc_probs

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



# %% [code] {"execution":{"iopub.status.busy":"2025-08-29T13:01:34.980013Z","iopub.execute_input":"2025-08-29T13:01:34.980422Z","iopub.status.idle":"2025-08-29T13:01:35.171471Z","shell.execute_reply.started":"2025-08-29T13:01:34.980402Z","shell.execute_reply":"2025-08-29T13:01:35.170342Z"},"jupyter":{"outputs_hidden":false}}
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
        # Optional: display if available, else print
        try:
            from IPython.display import display as _display  # type: ignore
            _display(submission_df)
        except Exception:
            print(submission_df)
    
    print(f"Total time: {time.time() - st}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
