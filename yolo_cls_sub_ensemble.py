# %% [code] {"execution":{"iopub.status.busy":"2025-09-21T20:23:55.496858Z","iopub.execute_input":"2025-09-21T20:23:55.497167Z","iopub.status.idle":"2025-09-21T20:24:21.070123Z","shell.execute_reply.started":"2025-09-21T20:23:55.497148Z","shell.execute_reply":"2025-09-21T20:24:21.068968Z"},"_kg_hide-output":true,"_kg_hide-input":true,"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-21T20:24:21.072146Z","iopub.execute_input":"2025-09-21T20:24:21.072387Z","iopub.status.idle":"2025-09-21T20:24:21.171309Z","shell.execute_reply.started":"2025-09-21T20:24:21.072365Z","shell.execute_reply":"2025-09-21T20:24:21.170708Z"}}
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

# YOLO
from ultralytics import YOLO

# CUPY
import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage
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


EFF_LABELS_TO_IDX = {
    'Aneurysm Present': 0,
    'Anterior Communicating Artery': 1,
    'Basilar Tip': 2,
    'Left Anterior Cerebral Artery': 3,
    'Left Infraclinoid Internal Carotid Artery': 4,
    'Left Middle Cerebral Artery': 5,
    'Left Posterior Communicating Artery': 6,
    'Left Supraclinoid Internal Carotid Artery': 7,
    'Other Posterior Circulation': 8,
    'Right Anterior Cerebral Artery': 9,
    'Right Infraclinoid Internal Carotid Artery': 10,
    'Right Middle Cerebral Artery': 11,
    'Right Posterior Communicating Artery': 12,
    'Right Supraclinoid Internal Carotid Artery': 13
}

EFF_LABELS = sorted(list(EFF_LABELS_TO_IDX.keys()))


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

def load_series_as_volume_cupy(series_dir: Path, target_shape: Tuple[int, int, int] = (32, 512, 512)) -> np.ndarray | None:
    """Load entire series as a 3D volume (D, H, W) and resize using CuPy."""
    paths = collect_series_slices(series_dir)
    if not paths:
        return None
    
    slices = []
    for dcm_path in paths:
        try:
            frames = read_dicom_frames_hu(dcm_path)
            for f in frames:
                slices.append(f)
        except Exception:
            continue
    if not slices:
        return None
    
    # Stack slices into volume (D, H, W)
    try:
        volume = np.stack(slices, axis=0)
        
        # Resize volume using CuPy for better performance
        current_shape = volume.shape
        zoom_factors = tuple(target_shape[i] / current_shape[i] for i in range(3))
        
        if cp is not None:
            try:
                # Use CuPy for 3D resizing
                from cupyx.scipy import ndimage as cp_ndimage
                volume_gpu = cp.asarray(volume)
                resized_volume_gpu = cp_ndimage.zoom(volume_gpu, zoom_factors, order=1)
                resized_volume = cp.asnumpy(resized_volume_gpu)
                # Clean up GPU memory
                del volume_gpu, resized_volume_gpu
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"[WARNING] CuPy processing failed: {e}, falling back to CPU")
                resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        else:
            resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Normalize to uint8
        mn, mx = float(resized_volume.min()), float(resized_volume.max())
        if mx - mn < 1e-6:
            normalized_volume = np.zeros_like(resized_volume, dtype=np.uint8)
        else:
            norm = (resized_volume - mn) / (mx - mn)
            normalized_volume = (norm * 255.0).clip(0, 255).astype(np.uint8)
        
        return normalized_volume
    except Exception:
        return None

def process_volume_for_yolo_cls(volume: np.ndarray, rgb_mode: bool = True) -> List[np.ndarray]:
    """Process 3D volume for YOLO classification with RGB mode."""
    if volume is None or volume.shape[0] == 0:
        return []
    
    processed_slices = []
    
    if rgb_mode:
        # Create RGB slices by stacking consecutive slices as channels
        # Skip first and last slice to avoid redundant combinations like [0,0,1] and [30,31,31]
        for i in range(1, volume.shape[0] - 1):
            # Get 3 consecutive slices: [i-1, i, i+1]
            prev_idx = i - 1
            next_idx = i + 1
            
            r = volume[prev_idx]  # Previous slice
            g = volume[i]         # Current slice  
            b = volume[next_idx]  # Next slice
            
            # Stack as RGB image
            rgb_img = np.stack([r, g, b], axis=-1)  # Shape: (H, W, 3)
            processed_slices.append(rgb_img)
    else:
        # Convert grayscale slices to BGR for YOLO
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            if slice_img.ndim == 2:
                slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
            processed_slices.append(slice_img)
    
    return processed_slices

# ====================================================
# YOLO-CLS Configuration
# ====================================================
YOLO_CLS_MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11n_cls_binary_fold0/weights/best.pt",  # Update with actual paths
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11_cls_fold0"
    }
]

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
# YOLO-LOC Configuration
# ====================================================
IMG_SIZE = 512
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4

YOLO_LOC_MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold0/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold1/weights/best.pt",
        "fold": "1",
        "weight": 1.0,
        "name": "YOLOv11n_fold1"
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
EFFNET_MODELS = {}
YOLO_LOC_MODELS = []
YOLO_CLS_MODELS = []
EFFNET_TRANSFORM = None

# EFFNET_CKPTS = [
#     '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/default/1/32-ch-cnn-epoch04-kaggle_score0.6498_fold_id0.ckpt',
#     '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/default/1/32-ch-cnn-epoch04-kaggle_score0.6739_fold_id1.ckpt',
#     '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/default/1/32-ch-cnn-epoch03-kaggle_score0.6735_fold_id2.ckpt',
#     '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/default/1/32-ch-cnn-epoch04-kaggle_score0.6901_fold_id3.ckpt',
#     '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/default/1/32-ch-cnn-epoch04-kaggle_score0.6537_fold_id4.ckpt'
# ]

EFFNET_CKPTS = [
    '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/tf_efficientnetv2_s.in21k_ft_in1k/1/32-ch-cnn-epoch03-kaggle_score0.6584_fold_id0.ckpt',
    '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/tf_efficientnetv2_s.in21k_ft_in1k/1/32-ch-cnn-epoch04-kaggle_score0.6770_fold_id1.ckpt',
    '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/tf_efficientnetv2_s.in21k_ft_in1k/1/32-ch-cnn-epoch02-kaggle_score0.6789_fold_id2.ckpt',
    '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/tf_efficientnetv2_s.in21k_ft_in1k/1/32-ch-cnn-epoch03-kaggle_score0.7006_fold_id3.ckpt',
    '/kaggle/input/rsna-iad-32ch-efficientnet/pytorch/tf_efficientnetv2_s.in21k_ft_in1k/1/32-ch-cnn-epoch01-kaggle_score0.6676_fold_id4.ckpt'
]

def get_inference_transform():
    """Get inference transformation for EfficientNet"""
    return A.Compose([
        A.Resize(EFFNET_CFG.size, EFFNET_CFG.size),
        A.Normalize(),
        ToTensorV2(),
    ])

def load_effnet_model_fold(fold: int) -> nn.Module:
    """Load a single EfficientNet fold model"""
    #model_path = Path(EFFNET_CFG.model_dir) / f'{EFFNET_CFG.model_name}_fold{fold}_best.pth'
    checkpoint = torch.load(EFFNET_CKPTS[fold], map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove "model." prefix if it exists
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v
    
    model = timm.create_model(
        EFFNET_CFG.model_name, 
        num_classes=EFFNET_CFG.num_classes, 
        pretrained=False,
        in_chans=EFFNET_CFG.in_chans
    )
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    return model

def load_YOLO_LOC_MODELS():
    """Load all YOLO localization models"""
    models = []
    for config in YOLO_LOC_MODEL_CONFIGS:
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

def load_YOLO_CLS_MODELS():
    """Load all YOLO classification models"""
    models = []
    for config in YOLO_CLS_MODEL_CONFIGS:
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
    global EFFNET_MODELS, YOLO_LOC_MODELS, YOLO_CLS_MODELS, EFFNET_TRANSFORM
    
    # Load EfficientNet models
    for fold in EFFNET_CFG.trn_fold:
        try:
            EFFNET_MODELS[fold] = load_effnet_model_fold(fold)
        except Exception as e:
            print(f"Warning: Could not load EfficientNet fold {fold}: {e}")
    
    if not EFFNET_MODELS:
        raise ValueError("No EfficientNet models were loaded successfully")
    
    # Load YOLO models
    YOLO_LOC_MODELS = load_YOLO_LOC_MODELS()
    YOLO_CLS_MODELS = load_YOLO_CLS_MODELS()
    
    # Initialize transforms
    EFFNET_TRANSFORM = get_inference_transform()
    
    # Warm up models
    dummy_effnet_image = torch.randn(1, EFFNET_CFG.in_chans, EFFNET_CFG.size, EFFNET_CFG.size).to(device)
    dummy_yolo_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    with torch.no_grad():
        for fold, model in EFFNET_MODELS.items():
            _ = model(dummy_effnet_image)
        
        for model_dict in YOLO_LOC_MODELS:
            model = model_dict["model"]
            _ = model.predict([dummy_yolo_image], verbose=False, device=device)
            
        for model_dict in YOLO_CLS_MODELS:
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
    """Run YOLO localization inference using all models"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1
    
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    for model_dict in YOLO_LOC_MODELS:
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

@torch.no_grad()
def predict_yolo_cls_ensemble(slices: List[np.ndarray]):
    """Run YOLO classification inference using all models - returns only aneurysm probability"""
    if not slices:
        return 0.1
    
    ensemble_preds = []
    total_weight = 0.0
    
    for model_dict in YOLO_CLS_MODELS:
        model = model_dict["model"]
        weight = model_dict["weight"]
        
        try:
            max_aneurysm_prob = 0.0
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                
                results = model.predict(
                    batch_slices, 
                    verbose=False, 
                    batch=len(batch_slices), 
                    device=device
                )
                
                for r in results:
                    if r is None or r.probs is None:
                        continue
                    try:
                        # For binary classification, r.probs.data contains class probabilities
                        probs = r.probs.data.cpu().numpy()
                        # Index 0 is aneurysm present probability
                        aneurysm_prob = float(probs[0])
                        max_aneurysm_prob = max(max_aneurysm_prob, aneurysm_prob)
                        
                    except Exception as e:
                        pass
            
            ensemble_preds.append(max_aneurysm_prob * weight)
            total_weight += weight
            
        except Exception as e:
            # Fallback prediction
            ensemble_preds.append(0.1 * weight)
            total_weight += weight
    
    if total_weight > 0:
        final_pred = sum(ensemble_preds) / total_weight
    else:
        final_pred = 0.1
    
    return final_pred

def process_dicom_for_effnet(series_path: str) -> np.ndarray:
    """Process DICOM for EfficientNet with memory cleanup"""
    try:
        preprocessor = DICOMPreprocessorKaggle(target_shape=EFFNET_CFG.target_shape)
        volume = preprocessor.process_series(series_path)
        return volume
    finally:
        gc.collect()

def process_dicom_for_yolo(series_path: str) -> List[np.ndarray]:
    """Process DICOM for YOLO localization with parallel processing"""
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

def process_dicom_for_yolo_cls(series_path: str) -> List[np.ndarray]:
    """Process DICOM for YOLO classification with CuPy resizing and RGB mode"""
    try:
        # Load and resize volume using CuPy
        volume = load_series_as_volume_cupy(Path(series_path), target_shape=(32, 512, 512))
        if volume is None:
            return []
        
        # Process volume for YOLO classification with RGB mode
        processed_slices = process_volume_for_yolo_cls(volume, rgb_mode=True)
        return processed_slices
    except Exception as e:
        return []

def _predict_inner(series_path: str) -> pl.DataFrame:
    """Main ensemble prediction logic"""
    global EFFNET_MODELS, YOLO_LOC_MODELS, YOLO_CLS_MODELS
    
    # Load models if not already loaded
    if not EFFNET_MODELS or not YOLO_LOC_MODELS or not YOLO_CLS_MODELS:
        load_all_models()
    
    try:
        # Process DICOM for all models
        effnet_volume = process_dicom_for_effnet(series_path)
        yolo_loc_slices = process_dicom_for_yolo(series_path)
        yolo_cls_slices = process_dicom_for_yolo_cls(series_path)
        
        # Get EfficientNet predictions
        effnet_preds = predict_effnet_ensemble(effnet_volume)
        
        # Get YOLO localization predictions
        yolo_loc_cls_pred, yolo_loc_preds = predict_yolo_ensemble(yolo_loc_slices)
        
        # Get YOLO classification predictions (only aneurysm probability)
        yolo_cls_aneurysm_prob = predict_yolo_cls_ensemble(yolo_cls_slices)

        # Generate EfficientNet predictions matching label columns
        eff_full_preds = np.zeros(len(LABEL_COLS))
        for i, label in enumerate(EFF_LABELS):
            if label in LABEL_COLS:
                label_idx = LABEL_COLS.index(label)
                eff_full_preds[label_idx] = effnet_preds[i]

        # Generate YOLO localization predictions matching label columns
        yolo_loc_full_preds = np.zeros(len(LABEL_COLS))
        for i, label in enumerate(YOLO_LABELS):
            if label in LABEL_COLS:
                label_idx = LABEL_COLS.index(label)
                yolo_loc_full_preds[label_idx] = yolo_loc_preds[i]
        # Set aneurysm present from localization model
        aneurysm_idx = LABEL_COLS.index('Aneurysm Present')
        yolo_loc_full_preds[aneurysm_idx] = yolo_loc_cls_pred
        
        # Ensemble predictions with proper weighting
        ensemble_preds = np.zeros(len(LABEL_COLS))
        
        # For anatomical locations (indices 0-12): Only EfficientNet + YOLO Localization
        for i in range(len(LABEL_COLS) - 1):  # Exclude "Aneurysm Present" (last index)
            ensemble_preds[i] = (0.45 * eff_full_preds[i]) + (0.55 * yolo_loc_full_preds[i])
        
        # For "Aneurysm Present" (last index): All three models
        ensemble_preds[aneurysm_idx] = (0.3 * eff_full_preds[aneurysm_idx]) + (0.35 * yolo_loc_full_preds[aneurysm_idx]) + (0.35 * yolo_cls_aneurysm_prob)
        
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
    Combines EfficientNet (30%), YOLO Localization (35%), and YOLO Classification (35%).
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

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"execution":{"iopub.status.busy":"2025-09-21T20:24:21.172106Z","iopub.execute_input":"2025-09-21T20:24:21.172413Z","iopub.status.idle":"2025-09-21T20:25:18.660874Z","shell.execute_reply.started":"2025-09-21T20:24:21.172387Z","shell.execute_reply":"2025-09-21T20:25:18.660142Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
