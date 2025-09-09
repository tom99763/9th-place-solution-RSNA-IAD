# %% [code] {"execution":{"iopub.status.busy":"2025-09-08T16:10:35.692057Z","iopub.execute_input":"2025-09-08T16:10:35.692816Z","iopub.status.idle":"2025-09-08T16:11:01.326376Z","shell.execute_reply.started":"2025-09-08T16:10:35.692790Z","shell.execute_reply":"2025-09-08T16:11:01.325318Z"},"_kg_hide-output":true,"_kg_hide-input":true,"jupyter":{"outputs_hidden":false}}
!tar xfvz /kaggle/input/ultralytics-offlineinstall-yolo12-weights/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Ensemble Pipeline: YOLO + EfficientNet (50/50 weighting)
# 1. **DICOM → 3D Volume**: Normalize to `(32, 384, 384)` for EfficientNet
# 2. **DICOM → 2D Slices**: Process for YOLO detection
# 3. **EfficientNetV2-S**: 32-channel input, 14 binary outputs
# 4. **YOLOv11**: Object detection on 2D slices
# 5. **Ensemble**: Average predictions with 50/50 weighting

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-08T16:11:01.328229Z","iopub.execute_input":"2025-09-08T16:11:01.328482Z","iopub.status.idle":"2025-09-08T16:11:01.466526Z","shell.execute_reply.started":"2025-09-08T16:11:01.328460Z","shell.execute_reply":"2025-09-08T16:11:01.465794Z"}}
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

# YOLO
from ultralytics import YOLO

# Competition API
import kaggle_evaluation.rsna_inference_server

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ====================================================
# DICOM Preprocessor for EfficientNet
# ====================================================
class DICOMPreprocessorKaggle:
    """DICOM preprocessing system for EfficientNet"""
    
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
        
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """Load DICOM series"""
        series_path = Path(series_path)
        series_name = series_path.name
        
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception as e:
                continue
        
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")
        
        return datasets, series_name
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """Extract position information for each slice"""
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    info['z_position'] = float(info['instance_number'])
            except Exception as e:
                info['z_position'] = float(i)
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """Sort slices by z-coordinate"""
        return sorted(slice_info, key=lambda x: x['z_position'])
    
    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[Optional[float], Optional[float]]:
        """Get windowing parameters based on modality"""
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            return "CT", "CT"
        else:
            return None, None
    
    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """Apply windowing or statistical normalization"""
        if center is not None and width is not None:
            p1, p99 = 0, 500
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    return np.zeros_like(img, dtype=np.uint8)
        else:
            p1, p99 = np.percentile(img, [1, 99])
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    return np.zeros_like(img, dtype=np.uint8)
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """Extract 2D pixel array from DICOM"""
        img = ds.pixel_array.astype(np.float32)
        
        if img.ndim == 3:
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
        
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target size"""
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        
        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """Process DICOM series and return as NumPy array"""
        try:
            datasets, series_name = self.load_dicom_series(series_path)
            
            first_ds = datasets[0]
            first_img = first_ds.pixel_array
            
            if len(datasets) == 1 and first_img.ndim == 3:
                return self._process_single_3d_dicom(first_ds, series_name)
            else:
                return self._process_multiple_2d_dicoms(datasets, series_name)
            
        except Exception as e:
            raise
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """Process single 3D DICOM file"""
        volume = ds.pixel_array.astype(np.float32)
        
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            volume = volume * float(slope) + float(intercept)
        
        window_center, window_width = self.get_windowing_params(ds)
        
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)
        
        return final_volume
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """Process multiple 2D DICOM files"""
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        first_img = self.extract_pixel_array(sorted_slices[0]['dataset'])
        window_center, window_width = self.get_windowing_params(sorted_slices[0]['dataset'], first_img)
        processed_slices = []
        
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
            processed_slices.append(resized_img)

        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)
        
        return final_volume

# ====================================================
# YOLO DICOM Processing (aligned with dataset preparation)
# ====================================================
def yolo_get_windowing_params(ds: pydicom.Dataset, img: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[float]]:
    """Match dataset script logic: CT -> fixed 0..500 range, else percentile path."""
    modality = getattr(ds, "Modality", "CT")
    if modality == "CT":
        return 0.0, 500.0
    return None, None

def yolo_apply_windowing_or_normalize(img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
    """Apply CT hard clipping (0..500) else 1st-99th percentile; fallback to min-max.

    Returns uint8 single-channel image (0..255)."""
    if center is not None and width is not None:
        p1, p99 = center, width  # 0,500
        if p99 > p1:
            clipped = np.clip(img, p1, p99)
            norm = (clipped - p1) / (p99 - p1)
            return (norm * 255).astype(np.uint8)
        # fallback
        img_min, img_max = float(img.min()), float(img.max())
        if img_max > img_min:
            norm = (img - img_min) / (img_max - img_min)
            return (norm * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)
    # percentile path
    try:
        p1, p99 = np.percentile(img, [1, 99])
    except Exception:
        p1, p99 = float(img.min()), float(img.max())
    if p99 > p1:
        clipped = np.clip(img, p1, p99)
        norm = (clipped - p1) / (p99 - p1)
        return (norm * 255).astype(np.uint8)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max > img_min:
        norm = (img - img_min) / (img_max - img_min)
        return (norm * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)

def yolo_read_frames_hu(ds: pydicom.Dataset) -> List[np.ndarray]:
    """Return list of frames in HU (float32) replicating dataset behavior."""
    pix = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        frames.append(pix.astype(np.float32) * slope + intercept)
    elif pix.ndim == 3:
        # RGB vs multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            # convert to grayscale (take first channel to stay deterministic like dataset script)
            gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frames.append(pix[i].astype(np.float32) * slope + intercept)
    return frames

def collect_series_slices_yolo(series_dir: Path) -> List[Path]:
    """Collect DICOM paths sorted by spatial position (SliceLocation > ImagePositionPatient[z] > InstanceNumber)."""
    dicom_files: List[Path] = []
    for root, _, files in os.walk(series_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                dicom_files.append(Path(root) / f)
    if not dicom_files:
        return []
    temp = []
    for p in dicom_files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            if hasattr(ds, 'SliceLocation'):
                sort_val = float(ds.SliceLocation)
            elif hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                sort_val = float(ds.ImagePositionPatient[-1])
            else:
                sort_val = float(getattr(ds, 'InstanceNumber', 0))
            temp.append((sort_val, p))
        except Exception:
            temp.append((0.0, p))
    temp.sort(key=lambda x: x[0])
    return [t[1] for t in temp]

def process_dicom_file_yolo(dcm_path: Path) -> List[np.ndarray]:
    """Process one DICOM -> list of preprocessed BGR slices (uint8) using dataset logic."""
    out: List[np.ndarray] = []
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        frames = yolo_read_frames_hu(ds)
        # compute window params once (dataset used per-frame 0..500 for CT) – reuse for all frames
        center, width = yolo_get_windowing_params(ds, frames[0] if frames else None)
        for fr in frames:
            img_u8 = yolo_apply_windowing_or_normalize(fr, center, width)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            if IMG_SIZE > 0 and (img_u8.shape[0] != IMG_SIZE or img_u8.shape[1] != IMG_SIZE):
                img_u8 = cv2.resize(img_u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            out.append(img_u8)
    except Exception:
        return []
    return out

def process_dicom_for_yolo(series_path: str) -> List[np.ndarray]:
    """Full series -> list of preprocessed YOLO-ready slices in spatial order."""
    series_dir = Path(series_path)
    dcm_paths = collect_series_slices_yolo(series_dir)
    if not dcm_paths:
        return []
    slices: List[np.ndarray] = []
    # Parallel frame extraction (I/O bound)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_dicom_file_yolo, p): p for p in dcm_paths}
        for fut in as_completed(futures):
            try:
                slices.extend(fut.result())
            except Exception:
                pass
    return slices

# ====================================================
# EfficientNet Configuration
# ====================================================
class EfficientNetConfig:
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    size = 384
    target_cols = LABEL_COLS
    num_classes = len(target_cols)
    in_chans = 32
    target_shape = (32, 384, 384)
    batch_size = 1
    use_amp = True
    model_dir = '/kaggle/input/rsna2025-effnetv2-32ch'
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]

EFFNET_CFG = EfficientNetConfig()

# ====================================================
# YOLO Configuration
# ====================================================
IMG_SIZE = 512
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4

YOLO_MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11n_069_preprocessing_fold03/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11n_069_preprocessing_fold1/weights/best.pt",
        "fold": "1",
        "weight": 1.0,
        "name": "YOLOv11n_fold1"
    },  
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11n_069_preprocessing_fold22/weights/best.pt",
        "fold": "2",
        "weight": 1.0,
        "name": "YOLOv11n_fold2"
    }
]

# ====================================================
# Model Loading and Inference
# ====================================================
# Global variables
EFFNET_MODELS = {}
YOLO_MODELS = []
EFFNET_TRANSFORM = None

def get_inference_transform():
    """Get inference transformation for EfficientNet"""
    return A.Compose([
        A.Resize(EFFNET_CFG.size, EFFNET_CFG.size),
        A.Normalize(),
        ToTensorV2(),
    ])

def load_effnet_model_fold(fold: int) -> nn.Module:
    """Load a single EfficientNet fold model"""
    model_path = Path(EFFNET_CFG.model_dir) / f'{EFFNET_CFG.model_name}_fold{fold}_best.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"EfficientNet model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = timm.create_model(
        EFFNET_CFG.model_name, 
        num_classes=EFFNET_CFG.num_classes, 
        pretrained=False,
        in_chans=EFFNET_CFG.in_chans
    )
    
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model

def load_yolo_models():
    """Load all YOLO models"""
    models = []
    for config in YOLO_MODEL_CONFIGS:
        model = YOLO(config["path"])
        model.to(device)
        
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
    global EFFNET_MODELS, YOLO_MODELS, EFFNET_TRANSFORM
    
    # Load EfficientNet models
    for fold in EFFNET_CFG.trn_fold:
        try:
            EFFNET_MODELS[fold] = load_effnet_model_fold(fold)
        except Exception as e:
            print(f"Warning: Could not load EfficientNet fold {fold}: {e}")
    
    if not EFFNET_MODELS:
        raise ValueError("No EfficientNet models were loaded successfully")
    
    # Load YOLO models
    YOLO_MODELS = load_yolo_models()
    
    # Initialize transforms
    EFFNET_TRANSFORM = get_inference_transform()
    
    # Warm up models
    dummy_effnet_image = torch.randn(1, EFFNET_CFG.in_chans, EFFNET_CFG.size, EFFNET_CFG.size).to(device)
    dummy_yolo_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    with torch.no_grad():
        for fold, model in EFFNET_MODELS.items():
            _ = model(dummy_effnet_image)
        
        for model_dict in YOLO_MODELS:
            model = model_dict["model"]
            _ = model.predict([dummy_yolo_image], verbose=False, device=device)

def predict_effnet_single_model(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Make prediction with a single EfficientNet model"""
    image = image.transpose(1, 2, 0)  # (D,H,W) -> (H,W,D)
    
    transformed = EFFNET_TRANSFORM(image=image)
    image_tensor = transformed['image']
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        with autocast(enabled=EFFNET_CFG.use_amp):
            output = model(image_tensor)
            return torch.sigmoid(output).cpu().numpy().squeeze()

def predict_effnet_ensemble(image: np.ndarray) -> np.ndarray:
    """Make EfficientNet ensemble prediction across all folds"""
    all_predictions = []
    weights = []
    
    for fold, model in EFFNET_MODELS.items():
        pred = predict_effnet_single_model(model, image)
        all_predictions.append(pred)
        weights.append(1.0)
    
    weights = np.array(weights) / np.sum(weights)
    predictions = np.array(all_predictions)
    return np.average(predictions, weights=weights, axis=0)

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
            max_conf_all = 0.0
            per_class_max = np.zeros(len(YOLO_LABELS), dtype=np.float32)
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                
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
                            if c > max_conf_all:
                                max_conf_all = c
                            if 0 <= k < len(YOLO_LABELS) and c > per_class_max[k]:
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

def process_dicom_for_effnet(series_path: str) -> np.ndarray:
    """Process DICOM for EfficientNet with memory cleanup"""
    try:
        preprocessor = DICOMPreprocessorKaggle(target_shape=EFFNET_CFG.target_shape)
        volume = preprocessor.process_series(series_path)
        return volume
    finally:
        gc.collect()

## (Note) Removed duplicate older process_dicom_for_yolo; unified above.

def _predict_inner(series_path: str) -> pl.DataFrame:
    """Main ensemble prediction logic"""
    global EFFNET_MODELS, YOLO_MODELS
    
    # Load models if not already loaded
    if not EFFNET_MODELS or not YOLO_MODELS:
        load_all_models()
    
    try:
        # Process DICOM for both models
        effnet_volume = process_dicom_for_effnet(series_path)
        yolo_slices = process_dicom_for_yolo(series_path)
        
        # Get EfficientNet predictions
        effnet_preds = predict_effnet_ensemble(effnet_volume)
        
        # Get YOLO predictions
        yolo_cls_pred, yolo_loc_preds = predict_yolo_ensemble(yolo_slices)
        
        # Align YOLO predictions with EfficientNet format
        # Create YOLO prediction array in the same order as LABEL_COLS
        yolo_full_preds = np.zeros(len(LABEL_COLS))
        
        # Map YOLO location predictions to correct positions
        for i, label in enumerate(YOLO_LABELS):
            if label in LABEL_COLS:
                label_idx = LABEL_COLS.index(label)
                yolo_full_preds[label_idx] = yolo_loc_preds[i]
        
        # Set "Aneurysm Present" from YOLO classification prediction
        aneurysm_idx = LABEL_COLS.index('Aneurysm Present')
        yolo_full_preds[aneurysm_idx] = yolo_cls_pred
        
        # Ensemble: 50% EfficientNet + 50% YOLO
        ensemble_preds = 0.5 * effnet_preds + 0.5 * yolo_full_preds
        
        # Create output dataframe
        predictions_df = pl.DataFrame(
            data=[ensemble_preds.tolist()],
            schema=LABEL_COLS,
            orient='row'
        )
        
        return predictions_df
        
    except Exception as e:
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

# %% [code] {"execution":{"iopub.status.busy":"2025-09-08T16:11:01.467526Z","iopub.execute_input":"2025-09-08T16:11:01.467716Z","iopub.status.idle":"2025-09-08T16:11:31.364217Z","shell.execute_reply.started":"2025-09-08T16:11:01.467700Z","shell.execute_reply":"2025-09-08T16:11:31.363557Z"},"jupyter":{"outputs_hidden":false}}
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