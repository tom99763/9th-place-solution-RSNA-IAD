# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T07:57:14.88528Z","iopub.status.busy":"2025-10-12T07:57:14.885044Z","iopub.status.idle":"2025-10-12T07:59:22.983348Z","shell.execute_reply":"2025-10-12T07:59:22.982561Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":128.105689,"end_time":"2025-10-12T07:59:22.98504","exception":false,"start_time":"2025-10-12T07:57:14.879351","status":"completed"},"tags":[]}
!tar -xzvf /kaggle/input/offline-install-tensorrt/packages.tar.gz
!pip install --no-index --find-links=./packages tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs onnxruntime-gpu onnxslim
!pip install dicomsdl --no-index --find-links=file:///kaggle/input/read-dicom-set

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T07:59:22.996798Z","iopub.status.busy":"2025-10-12T07:59:22.996548Z","iopub.status.idle":"2025-10-12T08:00:15.937231Z","shell.execute_reply":"2025-10-12T08:00:15.936413Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":52.948089,"end_time":"2025-10-12T08:00:15.938745","exception":false,"start_time":"2025-10-12T07:59:22.990656","status":"completed"},"tags":[]}
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
import joblib

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
import lightgbm as lgb
import xgboost as xgb

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Debug flag to control verbose prints (kept off for speed)
DEBUG = False
# Optional model warmup (first-run latency tradeoff). Keep False for speed.
ENABLE_WARMUP = False

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:15.951791Z","iopub.status.busy":"2025-10-12T08:00:15.951246Z","iopub.status.idle":"2025-10-12T08:00:15.957192Z","shell.execute_reply":"2025-10-12T08:00:15.956669Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.013486,"end_time":"2025-10-12T08:00:15.958135","exception":false,"start_time":"2025-10-12T08:00:15.944649","status":"completed"},"tags":[]}
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

# Fast index lookup for LABEL_COLS
LABEL_INDEX = {label: i for i, label in enumerate(LABEL_COLS)}
YOLO_TO_LABEL_IDX = [LABEL_INDEX.get(label, -1) for label in YOLO_LABELS]
ANEURYSM_IDX = LABEL_INDEX['Aneurysm Present']

VESSEL_LABELS = LABEL_COLS[:-1]
PRESENCE_LABEL = LABEL_COLS[-1]

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:15.970244Z","iopub.status.busy":"2025-10-12T08:00:15.970045Z","iopub.status.idle":"2025-10-12T08:00:15.991632Z","shell.execute_reply":"2025-10-12T08:00:15.991052Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.028384,"end_time":"2025-10-12T08:00:15.992568","exception":false,"start_time":"2025-10-12T08:00:15.964184","status":"completed"},"tags":[]}
class FlayerDICOMPreprocessor:
    """
    DICOM preprocessing system for Kaggle Code Competition
    Converts original DICOMPreprocessor logic to single series processing
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
        
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """
        Load DICOM series
        """
        series_path = Path(series_path)
        series_name = series_path.name
        
        # Search for DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        #print(f"Found {len(dicom_files)} DICOM files in series {series_name}")
        
        # Load DICOM datasets
        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception as e:
                #print(f"Failed to load {filepath}: {e}")
                continue
        
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")
        
        return datasets, series_name
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """
        Extract position information for each slice
        """
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            # Get z-coordinate from ImagePositionPatient
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    # Fallback: use InstanceNumber
                    info['z_position'] = float(info['instance_number'])
                    #print("ImagePositionPatient not found, using InstanceNumber")
            except Exception as e:
                info['z_position'] = float(i)
                #print(f"Failed to extract position info: {e}")
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """
        Sort slices by z-coordinate
        """
        # Sort by z-coordinate
        sorted_slices = sorted(slice_info, key=lambda x: x['z_position'])
        
        #print(f"Sorted {len(sorted_slices)} slices by z-position")
        #print(f"Z-range: {sorted_slices[0]['z_position']:.2f} to {sorted_slices[-1]['z_position']:.2f}")
        
        return sorted_slices
    #original
    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Get windowing parameters based on modality
        """
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            # For CT, apply CTA (angiography) settings
            center, width = (50, 350)
            #print(f"Using CTA windowing for CT: Center={center}, Width={width}")
            # return center, width
            return "CT", "CT"
            
        elif modality == 'MR':
            # For MR, skip windowing (statistical normalization only)
            #print("MR modality detected: skipping windowing, using statistical normalization")
            return None, None
            
        else:
            # Unexpected modality (safety measure)
            #print(f"Unexpected modality '{modality}', using CTA windowing")
            #return (50, 350)
            return None, None

    #YTT 
    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """
        Apply windowing or statistical normalization
        """
        
        # For MR or if windowing is not applied, use statistical normalization
        p1, p99 = np.percentile(img, [1, 99])
        
        if p99 > p1:
            normalized = np.clip(img, p1, p99)
            normalized = (normalized - p1) / (p99 - p1 + 1e-6)
            result = (normalized * 255).astype(np.uint8)
            #print("norm")
            #print(result)
            return result
        else:
            # Fallback: min-max normalization
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                normalized = (img - img_min) / (img_max - img_min + 1e-6)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                return np.zeros_like(img, dtype=np.uint8)
    
    def apply_windowing_or_normalize_vectorized(self, volume: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """
        Vectorized windowing/normalization for entire volume
        """
        # For MR or if windowing is not applied, use statistical normalization
        p1, p99 = np.percentile(volume, [1, 99])
        
        if p99 > p1:
            normalized = np.clip(volume, p1, p99)
            normalized = (normalized - p1) / (p99 - p1 + 1e-6)
            return (normalized * 255).astype(np.uint8)
        else:
            # Fallback: min-max normalization
            vol_min, vol_max = volume.min(), volume.max()
            if vol_max > vol_min:
                normalized = (volume - vol_min) / (vol_max - vol_min + 1e-6)
                return (normalized * 255).astype(np.uint8)
            else:
                return np.zeros_like(volume, dtype=np.uint8)
    
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Extract 2D pixel array from DICOM and apply preprocessing (for 2D DICOM series)
        """
        # Get pixel data
        img = ds.pixel_array.astype(np.float32)
        
        # For 3D volume case (multiple frames) - select middle frame
        if img.ndim == 3:
            #print(f"3D DICOM in 2D processing - using middle frame from shape: {img.shape}")
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
            #print(f"Selected frame {frame_idx} from 3D DICOM")
        
        # Convert color image to grayscale
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            #print("Converted color image to grayscale")
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        #YTT gemini fix1
        #slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
            #print(f"Applied rescaling: slope={slope}, intercept={intercept}")
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize 3D volume to target size
        """
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        #print(f"Resizing volume from {current_shape} to {target_shape}")
        
        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        volume = cp.asarray(volume)
        
        # Resize with linear interpolation
        resized_volume = zoom(volume, zoom_factors, order=1, mode='nearest')
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        resized_volume = cp.asnumpy(resized_volume)
        
        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        #print(resized_volume)
        #print(f"Final volume shape: {resized_volume.shape}")
        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """
        Process DICOM series and return as NumPy array (for Kaggle: no file saving)
        """
        try:
            # 1. Load DICOM files
            datasets, series_name = self.load_dicom_series(series_path)
            
            # Check first DICOM to determine 3D/2D
            first_ds = datasets[0]
            first_img = first_ds.pixel_array
            
            if len(datasets) == 1 and first_img.ndim == 3:
                # Case 1: Single 3D DICOM file
                #print(f"Processing single 3D DICOM with shape: {first_img.shape}")
                return self._process_single_3d_dicom(first_ds, series_name)
            else:
                # Case 2: Multiple 2D DICOM files
                #print(f"Processing {len(datasets)} 2D DICOM files")
                return self._process_multiple_2d_dicoms(datasets, series_name)
            
        except Exception as e:
            #print(f"Failed to process series {series_path}: {e}")
            raise
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """
        Process single 3D DICOM file (for Kaggle: no file saving)
        """
        # Get pixel array
        volume = ds.pixel_array.astype(np.float32)
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        #YTT gemini fix1
        #slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            volume = volume * float(slope) + float(intercept)
            # #print(f"Applied rescaling: slope={slope}, intercept={intercept}")
        
        # Get windowing settings and apply vectorized windowing
        window_center, window_width = self.get_windowing_params(ds)
        volume = self.apply_windowing_or_normalize_vectorized(volume, window_center, window_width)
        ##print(f"3D volume shape after windowing: {volume.shape}")
        
        # 3D resize
        final_volume = self.resize_volume_3d(volume)
        
        ##print(f"Successfully processed 3D DICOM series {series_name}")
        return final_volume
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """
        Process multiple 2D DICOM files (for Kaggle: no file saving)
        """
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
        ##print(f"2D slices stacked to volume shape: {volume.shape}")
        final_volume = self.resize_volume_3d(volume)
        
        ##print(f"Successfully processed 2D DICOM series {series_name}")
        return final_volume

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.00351Z","iopub.status.busy":"2025-10-12T08:00:16.003249Z","iopub.status.idle":"2025-10-12T08:00:16.013771Z","shell.execute_reply":"2025-10-12T08:00:16.013198Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.017305,"end_time":"2025-10-12T08:00:16.014831","exception":false,"start_time":"2025-10-12T08:00:15.997526","status":"completed"},"tags":[]}
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

def collect_series_slices(series_dir: Path) -> List[Path]:
    """Collect all DICOM files in a series directory (recursively) - legacy function."""
    return collect_series_slices_sorted(series_dir)

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.025643Z","iopub.status.busy":"2025-10-12T08:00:16.025381Z","iopub.status.idle":"2025-10-12T08:00:16.033364Z","shell.execute_reply":"2025-10-12T08:00:16.03266Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.014712,"end_time":"2025-10-12T08:00:16.034415","exception":false,"start_time":"2025-10-12T08:00:16.019703","status":"completed"},"tags":[]}
# ====================================================
# Configuration
# ====================================================
class FlayerInferenceConfig:
    # Model settings
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    size = 448#512#448 #384
    target_cols = LABEL_COLS
    num_classes = len(VESSEL_LABELS)
    heatmap_classes = VESSEL_LABELS
    in_chans = 1
    
    # Preprocessing settings
    #target_shape = (32, 384, 384)  # (depth, height, width)
    #target_shape = (32, 448, 448)  
    #target_shape = (32, 512, 512) 
    #target_shape = (32, 576, 576)
    #target_shape = (48, 448, 448)
    target_shape = (64, 448, 448)
    output_stride_depth = 1
    output_stride_height = 16#32
    output_stride_width = 16#32
    base_channels: int = 32
    # Inference settings
    batch_size = 1
    use_amp = True #True
    use_tta = False  # TTA is prohibited due to left/right positional information
    tta_transforms = 0
    
    # Model paths
    #model_dir = '/kaggle/input/rsna2025-effnetv2-32ch'
    #model_dir = '/kaggle/input/rsna-iad-model-atom/outputs_init_6336'
    #model_dir = '/kaggle/input/rsna-iad-model-atom/outputs_flip_swaplabel_6634'
    #model_dir = '/kaggle/input/rsna-iad-model-atom/outputs_contrast_6697'
    #model_dir = '/kaggle/input/rsna-iad-model-atom/outputs_default_6670'

    model_dirs = [
        #"/kaggle/input/rsna-iad-model-atom/outputs_heatmap_3dnorm_feature_layer2_classx2_74" #1fold lb71
        #"/kaggle/input/rsna-iad-model-atom/heatmap_512_flayer2_3dnorm_7275", #lb75+
        #"/kaggle/input/rsna-iad-model-atom/heatmap_512_flayer2_3dnorm_flipud_7176"
        #"/kaggle/input/rsna-iad-model-atom/heatmap_z48_448_flayer2_fold0_7639"
        "/kaggle/input/iad-model/heatmap_z64_448_flayer2_7664"
    ]


    
    n_fold = 5#1#5
    #trn_fold = [0, 1, 2, 3, 4]
    trn_fold = [0,1,2,3,4]
    
    # Ensemble weights (equal weight for all folds)
    ensemble_weights = None  # None means equal weights

FLAYER_CFG = FlayerInferenceConfig()


# ====================================================
# YOLO Configuration
# ====================================================
IMG_SIZE = 512
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4


YOLO_MODEL_CONFIGS = [
    #{
    #    "path": "/kaggle/input/rsna-yolo-models/cv_effnetv2s_v2_drop_path_fold0/weights/best.engine",
    #    "fold": 0,
    #    "weight": 1.0,
    #    "name": "effv2s",
    #    "mode": "2.5D"
    #},
    #{
    #    "path": "/kaggle/input/rsna-yolo-models/cv_effnetv2_s_drop_path_25d_fold1/weights/best.engine",
    #    "fold": 1,
    #    "weight": 1.0,
    #    "name": "effv2s",
    #    "mode": "2.5D"
    #},
    {
        "path": "/kaggle/input/rsna-yolo-models/cv_effnetv2_s_drop_path_25d_fold2/weights/best.engine",
        "fold": 2,
        "weight": 1.0,
        "name": "effv2s",
        "mode": "2.5D"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/cv_effnetv2_s_drop_path_25d_fold3/weights/best.pt",
        "fold": 3,
        "weight": 1.0,
        "name": "effv2s",
        "mode": "2.5D"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/cv_effnetv2_s_drop_path_25d_fold4/weights/best.engine",
        "fold": 4,
        "weight": 1.0,
        "name": "effv2s",
        "mode": "2.5D"
    },
    ####
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo-11m-2.5D_fold0/weights/best.engine",
        "fold": 0,
        "weight": 1.0,
        "name": "yolo11",
        "mode": "2.5D"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo-11m-2.5D_fold1/weights/best.engine",
        "fold": 1,
        "weight": 1.0,
        "name": "yolo11",
        "mode": "2.5D"
    },
    #{
    #    "path": "/kaggle/input/rsna-yolo-models/yolo-11m-2.5D_fold22/weights/best.pt",
    #    "fold": 2,
    #    "weight": 1.0,
    #    "name": "yolo11",
    #    "mode": "2.5D"
    #},
    #{
    #    "path": "/kaggle/input/rsna-yolo-models/yolo-11m-2.5D_fold3/weights/best.engine",
    #    "fold": 3,
    #    "weight": 1.0,
    #    "name": "yolo11",
    #    "mode": "2.5D"
    #},
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo-11m-2.5D_fold4/weights/best.engine",
        "fold": 4,
        "weight": 1.0,
        "name": "yolo11",
        "mode": "2.5D"
    },
]

# ====================================================
# Model Loading and Inference
# ====================================================
# Global variables
YOLO_MODELS = []
FLAYER_MODELS = {}
FLAYER_TRANSFORM = None
FLAYER_TTA_TRANSFORMS = None

# ====================================================
# Transforms
# ====================================================
def get_inference_transform():
    """Get inference transformation"""
    return A.Compose([
        A.Resize(FLAYER_CFG.size, FLAYER_CFG.size),
        A.Normalize(),
        #A.Normalize(mean=(0.0,), std=(1.0,)),  # no-op for 1-channel
        ToTensorV2(),
    ])

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.04553Z","iopub.status.busy":"2025-10-12T08:00:16.045201Z","iopub.status.idle":"2025-10-12T08:00:16.060346Z","shell.execute_reply":"2025-10-12T08:00:16.059677Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.02213,"end_time":"2025-10-12T08:00:16.061539","exception":false,"start_time":"2025-10-12T08:00:16.039409","status":"completed"},"tags":[]}
######################################################################
# 2.5 fate's and atom's model
######################################################################

class CenterNet3DInfer(nn.Module):
    """Inference model mirroring training CenterNet3D architecture."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            FLAYER_CFG.model_name,
            pretrained=False,
            features_only=True,
            #out_indices=(-1,),
            out_indices=(-2,),
        )
        info = self.backbone.feature_info
        self.feature_channels = info.channels()[-1]
        self.encoder_in_channels = getattr(self.backbone, 'in_chans', None)
        if self.encoder_in_channels is None:
            default_input = self.backbone.default_cfg.get('input_size', (3,))
            if isinstance(default_input, (list, tuple)):
                self.encoder_in_channels = default_input[0]
            else:
                self.encoder_in_channels = int(default_input)
        head_channels = FLAYER_CFG.base_channels
        self.temporal_head = nn.Sequential(
            nn.Conv3d(self.feature_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv3d(head_channels, len(FLAYER_CFG.heatmap_classes), kernel_size=1)
        self.offset_head = nn.Conv3d(head_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        if x.shape[1] != self.encoder_in_channels:
            if x.shape[1] == 1 and self.encoder_in_channels == 3:
                x = x.repeat(1, 3, 1, 1)
            else:
                raise ValueError(f"Input has {x.shape[1]} channels but encoder expects {self.encoder_in_channels}")
        feats = self.backbone(x)[0]
        feat_c, feat_h, feat_w = feats.shape[1:]
        feats = feats.view(b, d, feat_c, feat_h, feat_w).permute(0, 2, 1, 3, 4)
        feat3d = self.temporal_head(feats)
        heatmap = self.heatmap_head(feat3d)
        offset = self.offset_head(feat3d)
        return {"heatmap": heatmap, "offset": offset}


@torch.no_grad()
def compute_class_logits_from_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = heatmap.shape
    flat = heatmap.view(b, c, -1)
    class_logits = flat.max(dim=2).values
    presence_logits = class_logits.max(dim=1, keepdim=True).values
    return torch.cat([class_logits, presence_logits], dim=1)


def _resolve_model_dirs() -> list[str]:
    """
    取得要載入的 model 目錄列表。
    - 若 FLAYER_CFG.model_dirs 存在且非空，使用之
    - 否則回退到單一 FLAYER_CFG.model_dir
    """
    if hasattr(FLAYER_CFG, 'model_dirs') and FLAYER_CFG.model_dirs:
        return list(FLAYER_CFG.model_dirs)
    elif hasattr(FLAYER_CFG, 'model_dir') and FLAYER_CFG.model_dir:
        return [FLAYER_CFG.model_dir]
    else:
        raise ValueError("Please specify FLAYER_CFG.model_dirs (list) or FLAYER_CFG.model_dir (str).")

def _dir_label(path_str: str) -> str:
    """
    用於建立 FLAYER_MODELS dict 的 key 前綴，避免不同資料夾的 fold key 衝突。
    會取資料夾名稱當 label。
    """
    return Path(path_str).name

def _get_model_name_for_dir(dir_label: str) -> str:
    """
    若你之後需要不同資料夾有不同 model_name，可在 FLAYER_CFG 內加一個 dict:
      FLAYER_CFG.dir_model_name_map = {"expA": "tf_efficientnetv2_s", "expB": "convnext_base"...}
    若沒有，則回退使用 FLAYER_CFG.model_name。
    """
    if hasattr(FLAYER_CFG, 'dir_model_name_map') and dir_label in FLAYER_CFG.dir_model_name_map:
        return FLAYER_CFG.dir_model_name_map[dir_label]
    return FLAYER_CFG.model_name


def load_model_fold(model_dir: str, fold: int) -> nn.Module:
    """Load a single fold heatmap model from a specific model_dir"""
    dir_label = _dir_label(model_dir)
    model_name = _get_model_name_for_dir(dir_label)

    model_path = Path(model_dir) / f'{model_name}_fold{fold}_best.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if DEBUG: print(f"[{dir_label}] Loading fold {fold} model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = CenterNet3DInfer()
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_flayer_models():
    """Load all fold models from all configured model directories"""
    global FLAYER_MODELS, FLAYER_TRANSFORM, FLAYER_TTA_TRANSFORMS
    if DEBUG: print("Loading all fold models from multiple model_dirs...")

    model_dirs = _resolve_model_dirs()
    total_loaded = 0

    for md in model_dirs:
        dir_label = _dir_label(md)
        for fold in FLAYER_CFG.trn_fold:
            try:
                model = load_model_fold(md, fold)
                # key: (dir_label, fold)
                FLAYER_MODELS[(dir_label, fold)] = model
                total_loaded += 1
            except Exception as e:
                print(f"Warning: Could not load [{dir_label}] fold {fold}: {e}")

    if not FLAYER_MODELS:
        raise ValueError("No models were loaded successfully")

    # Initialize transforms (shared)
    FLAYER_TRANSFORM = get_inference_transform()
    FLAYER_TTA_TRANSFORMS = None  # TTA disabled due to left/right anatomy

    # Optional warmup (disabled by default)
    if ENABLE_WARMUP:
        dummy_volume = torch.randn(1, FLAYER_CFG.in_chans, FLAYER_CFG.target_shape[0], FLAYER_CFG.size, FLAYER_CFG.size).to(device)
        with torch.no_grad():
            for (dir_label, fold), model in FLAYER_MODELS.items():
                _ = model(dummy_volume)

    if DEBUG: print(f"Models ready for inference! Loaded: {total_loaded} models "
          f"from {len(model_dirs)} dirs, folds: {list(FLAYER_CFG.trn_fold)}.")

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.072204Z","iopub.status.busy":"2025-10-12T08:00:16.071998Z","iopub.status.idle":"2025-10-12T08:00:16.076169Z","shell.execute_reply":"2025-10-12T08:00:16.075689Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.010659,"end_time":"2025-10-12T08:00:16.077212","exception":false,"start_time":"2025-10-12T08:00:16.066553","status":"completed"},"tags":[]}
def load_yolo_models():
    models = []
    
    for idx, config in enumerate(YOLO_MODEL_CONFIGS):
        device_id = 0
        
        model = YOLO(config["path"], task='detect')
        
        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"],
            "device": device_id  
        }
        models.append(model_dict)
    
    return models

def load_all_models():
    """Load all models (EfficientNet + YOLO)"""
    global YOLO_MODELS, FLAYER_MODELS
    

    # Load YOLO models
    YOLO_MODELS = load_yolo_models()
    
    if not FLAYER_MODELS:
        load_flayer_models()

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.088146Z","iopub.status.busy":"2025-10-12T08:00:16.087941Z","iopub.status.idle":"2025-10-12T08:00:16.102992Z","shell.execute_reply":"2025-10-12T08:00:16.10249Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.021754,"end_time":"2025-10-12T08:00:16.104006","exception":false,"start_time":"2025-10-12T08:00:16.082252","status":"completed"},"tags":[]}
#ytt avg then sigmoid 
def flayer_predict_single_model(model: nn.Module, tensor_5d: torch.Tensor) -> torch.Tensor:
    """
    Run inference for a single model and return LOGITS (torch.Tensor on device).
    - tensor_5d: (1, 1, D, H, W) 已在 GPU/AMP 準備好的張量
    """
    # 假設 model 已 .eval()，外層用 inference_mode/autocast
    outputs = model(tensor_5d)

    # 可能回 dict/tensor，保守處理
    heatmap = outputs['heatmap'] if isinstance(outputs, dict) else outputs
    logits = compute_class_logits_from_heatmap(heatmap)

    # 確保 logits 在同一裝置、同一 dtype、且為 1D
    logits = logits.to(tensor_5d.device, dtype=torch.float32)
    logits = logits.flatten()  # (num_labels,)
    return logits


def predict_flayer_ensemble(image: np.ndarray) -> np.ndarray:
    """
    先對各模型 logits 做加權平均（在 GPU 上累加），最後一次 sigmoid。
    同時把資料前處理只做一次，避免重工。
    """
    # --- 一次性前處理到 GPU ---
    # 原始 volume 是 (D, H, W)，轉成 (H, W, D) 給 FLAYER_TRANSFORM
    image_hwd = image.transpose(1, 2, 0)  # (H, W, D)
    transformed = FLAYER_TRANSFORM(image=image_hwd)

    tensor = transformed['image']  # 可能是 numpy 或 torch.Tensor（視 FLAYER_TRANSFORM 實作而定）
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    # 期望 (D, H, W)
    if tensor.dim() != 3:
        raise ValueError(f"FLAYER_TRANSFORM['image'] should be 3D (D,H,W), got shape {tuple(tensor.shape)}")

    # 統一 dtype/device，並補上 batch/channel 維度 → (1,1,D,H,W)
    tensor_5d = tensor.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(0).unsqueeze(0)

    def _lookup_weight(dir_label: str, fold: int) -> float:
        if getattr(FLAYER_CFG, 'ensemble_weights', None) is None:
            return 1.0
        ew = FLAYER_CFG.ensemble_weights
        return (
            ew.get((dir_label, fold), None)
            or ew.get(f"{dir_label}/{fold}", None)
            or ew.get(f"{dir_label}_fold{fold}", None)
            or ew.get(f"fold{fold}", None)
            or 1.0
        )

    sum_logits = None
    sum_w = 0.0

    # 比 no_grad() 更快的推論模式
    with torch.inference_mode():
        # 共享一個 autocast，避免在迴圈中重複建立 context
        with autocast(enabled=FLAYER_CFG.use_amp):
            for (dir_label, fold), model in FLAYER_MODELS.items():
                w = float(_lookup_weight(dir_label, fold))
                if w == 0.0:
                    continue

                logits = flayer_predict_single_model(model, tensor_5d)  # torch.Tensor on device, float32, 1D

                # 初始化累加器並做 shape 檢查
                if sum_logits is None:
                    sum_logits = torch.zeros_like(logits)  # 確保 shape/dtype/device 一致
                if sum_logits.shape != logits.shape:
                    raise ValueError(f"Logits shape mismatch: got {tuple(logits.shape)}, "
                                     f"expected {tuple(sum_logits.shape)}")

                # 就地加權累加
                sum_logits.add_(logits, alpha=w)
                sum_w += w

    # 邊界情況：沒有模型或權重總和為 0
    if (sum_logits is None) or (sum_w == 0.0):
        return np.full(len(LABEL_COLS), 0.5, dtype=np.float32)

    avg_logits = sum_logits / float(sum_w)                    # 仍在 GPU
    probs = torch.sigmoid(avg_logits).float().cpu().numpy()   # 只在最後搬回 CPU
    return probs




@torch.no_grad()
def predict_yolo_ensemble(slices: List[np.ndarray]):
    """Run YOLO inference using all models"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1

    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    for fold_id, model_dict in enumerate(YOLO_MODELS):
        model = model_dict["model"]
        weight = model_dict["weight"]
        model_name = model_dict['name']
        mode = model_dict.get('mode', '2.5D')  
        device_id = model_dict.get("device", 0)  # Get assigned device
        
        
        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(YOLO_LABELS), dtype=np.float32)
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                if (len(batch_slices) < 32): #and (model_name == 'effv2s'):
                    batch_slices += [batch_slices[0]] * (32 - len(batch_slices))
                
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
                            if 0 <= k < len(YOLO_LABELS) and c > per_class_max[k]:
                                per_class_max[k] = c
                    except Exception:
                        pass
            
            if DEBUG: print(f"{model_name} ({mode}): {per_class_max}")
            
            # Weighted per-model contributions
            ensemble_loc_preds.append(per_class_max * weight)
            total_weight += weight
            
            
        except Exception as e:
            if DEBUG: print(f"Error with model {model_name} ({mode}): {e}")
            ensemble_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            total_weight += weight


    if total_weight > 0:
        final_loc_preds = sum(ensemble_loc_preds) / total_weight
        if DEBUG: print(f"Final ensemble predictions: {final_loc_preds}")
        final_cls_pred = float(final_loc_preds.max())
        if DEBUG: print(f"Max confidence: {final_cls_pred}")
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1


    return final_cls_pred, final_loc_preds

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.114998Z","iopub.status.busy":"2025-10-12T08:00:16.114816Z","iopub.status.idle":"2025-10-12T08:00:16.123569Z","shell.execute_reply":"2025-10-12T08:00:16.12305Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.015572,"end_time":"2025-10-12T08:00:16.12463","exception":false,"start_time":"2025-10-12T08:00:16.109058","status":"completed"},"tags":[]}
# Safe processing function with memory cleanup
def process_dicom_series_for_flayer(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    """
    Safe DICOM processing with reduced memory cleanup frequency
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        np.ndarray: Processed volume
    """
    preprocessor = FlayerDICOMPreprocessor(target_shape=target_shape)
    return preprocessor.process_series(series_path)



def process_dicom_for_yolo(series_path: str, mode: str = "2.5D") -> List[np.ndarray]:
    """Process DICOM for YOLO with parallel processing and mode support"""
    series_path = Path(series_path)
    dicom_files = collect_series_slices(series_path)

    if mode == "2D":
        # For 2D mode, process each DICOM file individually (convert to RGB)
        all_slices: List[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit in order and maintain order
            futures = [executor.submit(process_dicom_file_yolo, dcm_path, False) 
                      for dcm_path in dicom_files]
            
            # Retrieve results in submission order
            for future in futures:  # ✅ Deterministic order!
                try:
                    slices = future.result()
                    all_slices.extend(slices)
                except Exception as e:
                    pass
        return all_slices

    elif mode == "2.5D":
        # Similar fix for 2.5D mode
        if len(dicom_files) < 3:
            return process_dicom_for_yolo(series_path, "2D")

        all_frames = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit in order and maintain order
            futures = [executor.submit(process_dicom_file_yolo, dcm_path, True) 
                      for dcm_path in dicom_files]
            
            # Retrieve results in submission order
            for future in futures:  # ✅ Deterministic order!
                try:
                    slices = future.result()
                    all_frames.extend(slices)
                except Exception as e:
                    pass

        # Rest of 2.5D logic remains the same...
        if len(all_frames) < 3:
            print(f"Warning: Only {len(all_frames)} frames available, need at least 3 for 2.5D")
            return all_frames if all_frames else []

        rgb_slices = []
        for i in range(1, len(all_frames) - 1):
            try:
                prev_frame = all_frames[i-1]
                curr_frame = all_frames[i]
                next_frame = all_frames[i+1]
                
                if prev_frame is None or curr_frame is None or next_frame is None:
                    continue
      
                if not (prev_frame.shape == curr_frame.shape == next_frame.shape):
                    print(f"Warning: Frame shape mismatch at index {i}")
                    continue
                
                rgb_img = np.stack([prev_frame, curr_frame, next_frame], axis=-1)
                
                if rgb_img.shape[-1] != 3 or rgb_img.ndim != 3:
                    print(f"Warning: Invalid RGB shape {rgb_img.shape} at index {i}")
                    continue
                    
                rgb_slices.append(rgb_img)
                
            except Exception as e:
                print(f"Error creating RGB triplet at index {i}: {e}")
                continue
        
        if DEBUG: print(f"Created {len(rgb_slices)} valid 2.5D slices from {len(all_frames)} frames")
        return rgb_slices if rgb_slices else all_frames

    else:
        raise ValueError(f"Unsupported YOLO mode: {mode}. Use '2D' or '2.5D'")

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.135448Z","iopub.status.busy":"2025-10-12T08:00:16.135215Z","iopub.status.idle":"2025-10-12T08:00:16.142782Z","shell.execute_reply":"2025-10-12T08:00:16.142242Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.014282,"end_time":"2025-10-12T08:00:16.143843","exception":false,"start_time":"2025-10-12T08:00:16.129561","status":"completed"},"tags":[]}
def _predict_inner(series_path: str) -> pl.DataFrame:
    """Main ensemble prediction logic"""
    global YOLO_MODELS, FLAYER_MODELS
    
    # Load models if not already loaded
    if  not YOLO_MODELS or not FLAYER_MODELS:
        load_all_models()
    try:
        
        # Process DICOM for both models
        yolo_slices = process_dicom_for_yolo(series_path, mode="2.5D")
        flayer_volume = process_dicom_series_for_flayer(series_path, FLAYER_CFG.target_shape)
        if DEBUG: print(f"{flayer_volume.shape=}")

        # Get YOLO predictions
        yolo_cls_pred, yolo_loc_preds = predict_yolo_ensemble(yolo_slices)
        flayer_preds = predict_flayer_ensemble(flayer_volume)
        flayer_preds = np.asarray(flayer_preds, dtype=np.float32)
        if flayer_preds.shape[0] != len(LABEL_COLS):
            raise ValueError("Flayer ensemble output length mismatch")

        # yolo_full_preds has preds in LABEL_COLS order now
        yolo_full_preds = np.zeros(len(LABEL_COLS))
        for i, label in enumerate(YOLO_LABELS):
            idx = YOLO_TO_LABEL_IDX[i]
            if idx != -1:
                yolo_full_preds[idx] = yolo_loc_preds[i]
        aneurysm_idx = ANEURYSM_IDX

        ensemble_preds = yolo_full_preds
        ensemble_preds[:aneurysm_idx] = (0.7 * ensemble_preds[:aneurysm_idx]) + (0.3 * flayer_preds[:aneurysm_idx])
        ensemble_preds[aneurysm_idx]  = (0.7 * yolo_cls_pred) + (0.3 * flayer_preds[aneurysm_idx])

        
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
        # Cleanup shared directory only
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # Reduced frequency memory cleanup - only CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.154831Z","iopub.status.busy":"2025-10-12T08:00:16.154633Z","iopub.status.idle":"2025-10-12T08:00:16.18524Z","shell.execute_reply":"2025-10-12T08:00:16.184546Z"},"papermill":{"duration":0.037571,"end_time":"2025-10-12T08:00:16.186348","exception":false,"start_time":"2025-10-12T08:00:16.148777","status":"completed"},"tags":["dicomsdl-opt"],"jupyter":{"outputs_hidden":false}}
# --- dicomsdl-optimized DICOM I/O (overrides) [parity fix] ---
try:
    import dicomsdl as dicom_sdl
    HAS_DICOMSDL = True
except Exception:
    dicom_sdl = None
    HAS_DICOMSDL = False
    print("no dicom sdl")

def _sdl_read_pixels(path: str):
    if not HAS_DICOMSDL:
        return None
    try:
        d = dicom_sdl.open(str(path))
        try:
            pix = d.pixelData()
        except Exception:
            try:
                pix = d.getPixelData()
            except Exception:
                return None
        arr = np.asarray(pix)
        return arr
    except Exception:
        return None

class FlayerDICOMPreprocessor:
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        series_name = os.path.basename(os.path.normpath(series_path))
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
                # Read full dataset (include Pixel Data) for parity
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception:
                continue
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")
        return datasets, series_name

    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
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
            except Exception:
                info['z_position'] = float(i)
            slice_info.append(info)
        return slice_info

    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        return sorted(slice_info, key=lambda x: x['z_position'])

    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[Optional[float], Optional[float]]:
        modality = getattr(ds, 'Modality', 'CT')
        if modality == 'CT':
            return "CT", "CT"
        elif modality == 'MR':
            return None, None
        else:
            return None, None

    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            normalized = np.clip(img, p1, p99)
            normalized = (normalized - p1) / (p99 - p1 + 1e-6)
            return (normalized * 255).astype(np.uint8)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            normalized = (img - img_min) / (img_max - img_min + 1e-6)
            return (normalized * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    def _to_gray_if_rgb(self, arr: np.ndarray) -> Optional[np.ndarray]:
        if arr.ndim == 3 and arr.shape[-1] == 3 and (arr.shape[0] != 3 or arr.shape[1] != 3):
            try:
                return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            except Exception:
                return arr[..., 0].astype(np.float32)
        return None

    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        path = getattr(ds, 'filename', None)
        img = None
        if path is not None:
            arr = _sdl_read_pixels(path)
            if arr is not None:
                gray = self._to_gray_if_rgb(arr)
                if gray is not None:
                    img = gray
                elif arr.ndim == 3:
                    frame_idx = arr.shape[0] // 2
                    img = arr[frame_idx].astype(np.float32)
                else:
                    img = arr.astype(np.float32)
        if img is None:
            arr = ds.pixel_array
            if arr.ndim == 3 and arr.shape[-1] == 3 and (arr.shape[0] != 3 or arr.shape[1] != 3):
                try:
                    img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                except Exception:
                    img = arr[..., 0].astype(np.float32)
            elif arr.ndim == 3:
                frame_idx = arr.shape[0] // 2
                img = arr[frame_idx].astype(np.float32)
            else:
                img = arr.astype(np.float32)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
        return img

    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        if current_shape == target_shape:
            return volume.astype(np.uint8)
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        volume_gpu = cp.asarray(volume)
        resized_volume = zoom(volume_gpu, zoom_factors, order=1, mode='nearest')
        resized_volume = cp.asnumpy(resized_volume)
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
        datasets, series_name = self.load_dicom_series(series_path)
        first_ds = datasets[0]
        nframes = int(getattr(first_ds, 'NumberOfFrames', 0) or 0)
        if nframes <= 1:
            try:
                arr0 = _sdl_read_pixels(getattr(first_ds, 'filename', ''))
                if arr0 is not None and arr0.ndim == 3 and not (arr0.shape[-1] == 3 and (arr0.shape[0] != 3 or arr0.shape[1] != 3)):
                    nframes = arr0.shape[0]
            except Exception:
                pass
        if len(datasets) == 1 and nframes > 1:
            return self._process_single_3d_dicom(first_ds, series_name)
        else:
            return self._process_multiple_2d_dicoms(datasets, series_name)

    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        path = getattr(ds, 'filename', None)
        volume = None
        if path is not None:
            arr = _sdl_read_pixels(path)
            if arr is not None:
                volume = arr.astype(np.float32)
        if volume is None:
            volume = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
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
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        images = []
        for info in sorted_slices:
            img = self.extract_pixel_array(info['dataset'])
            window_center, window_width = self.get_windowing_params(info['dataset'], img)
            img_windowed = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(img_windowed, (self.target_width, self.target_height))
            images.append(resized_img)
        volume = np.stack(images, axis=0)
        final_volume = self.resize_volume_3d(volume)
        return final_volume

def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    ds_meta = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    arr = _sdl_read_pixels(str(path))
    if arr is None:
        ds_pix = pydicom.dcmread(str(path), force=True)
        arr = ds_pix.pixel_array
    slope = float(getattr(ds_meta, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds_meta, 'RescaleIntercept', 0.0))
    frames: List[np.ndarray] = []
    if arr.ndim == 2:
        img = arr.astype(np.float32)
        frames.append(img * slope + intercept)
    elif arr.ndim == 3:
        if arr.shape[-1] == 3 and (arr.shape[0] != 3 or arr.shape[1] != 3):
            try:
                gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = arr[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(arr.shape[0]):
                frm = arr[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    return frames

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.196914Z","iopub.status.busy":"2025-10-12T08:00:16.196715Z","iopub.status.idle":"2025-10-12T08:00:16.201087Z","shell.execute_reply":"2025-10-12T08:00:16.200597Z"},"papermill":{"duration":0.010793,"end_time":"2025-10-12T08:00:16.202116","exception":false,"start_time":"2025-10-12T08:00:16.191323","status":"completed"},"tags":["dicomsdl-patch"],"jupyter":{"outputs_hidden":false}}
# Parity patch: ensure 2D slice resizing matches original
def __patch__process_multiple_2d_dicoms(self, datasets, series_name):
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
FlayerDICOMPreprocessor._process_multiple_2d_dicoms = __patch__process_multiple_2d_dicoms

# %% [code] {"tags":["cache-opt"],"jupyter":{"outputs_hidden":false}}
# --- Per-series shared DICOM cache (output-identical) ---
DICOM_SERIES_CACHE: dict = {}

def reset_series_cache():
    DICOM_SERIES_CACHE.clear()

def _load_arr_cached(path: str):
    if not path:
        return None
    arr = DICOM_SERIES_CACHE.get(path)
    if arr is not None:
        return arr
    # Try dicomsdl first (if available), fallback to pydicom pixel_array
    arr = None
    try:
        if 'dicom_sdl' in globals() and dicom_sdl is not None:
            d = dicom_sdl.open(str(path))
            try:
                pix = d.pixelData()
            except Exception:
                try:
                    pix = d.getPixelData()
                except Exception:
                    pix = None
            if pix is not None:
                arr = np.asarray(pix)
    except Exception:
        arr = None
    if arr is None:
        try:
            ds_pix = pydicom.dcmread(str(path), force=True)
            arr = ds_pix.pixel_array
        except Exception:
            arr = None
    if arr is not None:
        DICOM_SERIES_CACHE[path] = arr
    return arr

# Patch FlayerDICOMPreprocessor.extract_pixel_array to reuse cache
def _extract_pixel_array_cached(self, ds: pydicom.Dataset) -> np.ndarray:
    path = getattr(ds, 'filename', None)
    img = None
    if path:
        arr = _load_arr_cached(path)
        if arr is not None:
            # RGB HxWx3 case: convert to grayscale
            if arr.ndim == 3 and arr.shape[-1] == 3 and (arr.shape[0] != 3 or arr.shape[1] != 3):
                try:
                    img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                except Exception:
                    img = arr[..., 0].astype(np.float32)
            elif arr.ndim == 3:
                # Multi-frame (F,H,W): take middle frame
                frame_idx = arr.shape[0] // 2
                img = arr[frame_idx].astype(np.float32)
            else:
                img = arr.astype(np.float32)
    if img is None:
        # Fallback to original pydicom path
        arr = ds.pixel_array
        if arr.ndim == 3 and arr.shape[-1] == 3 and (arr.shape[0] != 3 or arr.shape[1] != 3):
            try:
                img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            except Exception:
                img = arr[..., 0].astype(np.float32)
        elif arr.ndim == 3:
            frame_idx = arr.shape[0] // 2
            img = arr[frame_idx].astype(np.float32)
        else:
            img = arr.astype(np.float32)
    # Apply slope/intercept from ds (identical to original)
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    if slope != 1 or intercept != 0:
        img = img * float(slope) + float(intercept)
    return img

# Patch YOLO reader to reuse cache
def _read_dicom_frames_hu_cached(path: Path) -> List[np.ndarray]:
    path = str(path)
    ds_meta = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    arr = _load_arr_cached(path)
    if arr is None:
        ds_pix = pydicom.dcmread(path, force=True)
        arr = ds_pix.pixel_array
    slope = float(getattr(ds_meta, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds_meta, 'RescaleIntercept', 0.0))
    frames: List[np.ndarray] = []
    if arr.ndim == 2:
        img = arr.astype(np.float32)
        frames.append(img * slope + intercept)
    elif arr.ndim == 3:
        if arr.shape[-1] == 3 and (arr.shape[0] != 3 or arr.shape[1] != 3):
            try:
                gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = arr[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(arr.shape[0]):
                frm = arr[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    return frames

# Apply monkey patches if symbols exist
try:
    FlayerDICOMPreprocessor.extract_pixel_array = _extract_pixel_array_cached
    read_dicom_frames_hu = _read_dicom_frames_hu_cached
except NameError:
    pass

# Wrap predict to reset cache per series
try:
    _orig_predict = predict
    def predict(series_path: str) -> pl.DataFrame:
        reset_series_cache()
        try:
            return _orig_predict(series_path)
        finally:
            reset_series_cache()
except NameError:
    pass

# %% [code] {"execution":{"iopub.execute_input":"2025-10-12T08:00:16.213556Z","iopub.status.busy":"2025-10-12T08:00:16.21334Z","iopub.status.idle":"2025-10-12T08:01:40.950735Z","shell.execute_reply":"2025-10-12T08:01:40.950033Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":84.750401,"end_time":"2025-10-12T08:01:40.957602","exception":false,"start_time":"2025-10-12T08:00:16.207201","status":"completed"},"tags":[]}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.005575,"end_time":"2025-10-12T08:01:40.969072","exception":false,"start_time":"2025-10-12T08:01:40.963497","status":"completed"},"tags":[]}
