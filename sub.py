# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-10T22:07:04.868963Z","iopub.execute_input":"2025-10-10T22:07:04.869189Z","iopub.status.idle":"2025-10-10T22:09:17.856812Z","shell.execute_reply.started":"2025-10-10T22:07:04.869165Z","shell.execute_reply":"2025-10-10T22:09:17.856077Z"}}
!tar -xzvf /kaggle/input/offline-install-tensorrt/packages.tar.gz
!pip install --no-index --find-links=./packages tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs onnxruntime-gpu onnxslim

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:09:17.857856Z","iopub.execute_input":"2025-10-10T22:09:17.858120Z","iopub.status.idle":"2025-10-10T22:10:09.789744Z","shell.execute_reply.started":"2025-10-10T22:09:17.858086Z","shell.execute_reply":"2025-10-10T22:10:09.788953Z"},"papermill":{"duration":58.742328,"end_time":"2025-10-03T05:10:24.555508","exception":false,"start_time":"2025-10-03T05:09:25.81318","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
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
import pickle

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.insert(0, "/kaggle/input/ultralytcs-timm-rsna/ultralytics-timm")

# YOLO
from ultralytics import YOLO

# Competition API
import kaggle_evaluation.rsna_inference_server
from tqdm import tqdm

import cupy as cp
from cupyx.scipy.ndimage import zoom
import lightgbm as lgb
import xgboost as xgb

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xgb.set_config(verbosity=0)

with open('/kaggle/input/2x-yolo-flayer-meta-training/label_encoder_sex.pkl', 'rb') as f:
    le = pickle.load(f)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:09.791708Z","iopub.execute_input":"2025-10-10T22:10:09.792277Z","iopub.status.idle":"2025-10-10T22:10:09.798160Z","shell.execute_reply.started":"2025-10-10T22:10:09.792255Z","shell.execute_reply":"2025-10-10T22:10:09.797480Z"},"papermill":{"duration":0.011534,"end_time":"2025-10-03T05:10:24.942479","exception":false,"start_time":"2025-10-03T05:10:24.930945","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}

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

VESSEL_LABELS = LABEL_COLS[:-1]
PRESENCE_LABEL = LABEL_COLS[-1]

#ensemble_w = [0.44894133 0.50395086 0.01779959 0.0278795  0.00142871] #nelder-meand optimized

#lgb, xgb, cat, flayer, yolo11m, effv2s
ensemble_w = [0.16666, 0.16666,  0.16666,  0.16666,  0.16666, 0.16666]
_MODELS_LOADED = False

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:09.798936Z","iopub.execute_input":"2025-10-10T22:10:09.799096Z","iopub.status.idle":"2025-10-10T22:10:09.823690Z","shell.execute_reply.started":"2025-10-10T22:10:09.799082Z","shell.execute_reply":"2025-10-10T22:10:09.823100Z"},"jupyter":{"outputs_hidden":false}}
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
        
        # Get windowing settings
        window_center, window_width = self.get_windowing_params(ds)
        
        # Apply windowing to each slice
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
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

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## meta classifier

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-10-10T22:10:09.824324Z","iopub.execute_input":"2025-10-10T22:10:09.824484Z","iopub.status.idle":"2025-10-10T22:10:11.910353Z","shell.execute_reply.started":"2025-10-10T22:10:09.824471Z","shell.execute_reply":"2025-10-10T22:10:11.909643Z"}}
meta_cls_path = '/kaggle/input/2x-yolo-flayer-meta-training'
model_prefix="meta_classifier"
n_folds = 5


lgb_models = {label: [] for label in LABEL_COLS}
xgb_models = {label: [] for label in LABEL_COLS}
cat_models = {label: [] for label in LABEL_COLS}
meta_models = {'lgb': lgb_models, 'xgb': xgb_models, 'cat': cat_models}


for label in tqdm(LABEL_COLS):
    for model_file in ['lgb', 'xgb', 'cat']:
        for fold in range(n_folds):
            model_path = f"{meta_cls_path}/{model_file}/{model_prefix}_{label}_fold_fold{fold}.pkl"
            model = joblib.load(model_path)
            meta_models[model_file][label].append(model)

def predict_prob_lgb(X, fold_id):
    preds_fold = []
    for k, model in lgb_models.items():
        preds_fold.append(model[fold_id].predict_proba(X)[:, 1])
    preds_fold = np.array(preds_fold)
    return preds_fold

def predict_prob_xgb(X, fold_id):
    preds_fold = []
    for k, model in xgb_models.items():
        preds_fold.append(model[fold_id].predict_proba(X)[:, 1])
    preds_fold = np.array(preds_fold)
    return preds_fold

def predict_prob_cat(X, fold_id):
    preds_fold = []
    for k, model in cat_models.items():
        preds_fold.append(model[fold_id].predict_proba(X)[:, 1])
    preds_fold = np.array(preds_fold)
    return preds_fold

sex_map = {name[0]: name for name in le.classes_}
def parse_meta_data(ds):
    # ---- Patient Age ----
    try:
        age_str = getattr(ds, 'PatientAge', '050Y')
        age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
        age = min(age, 100)
    except Exception:
        age = 50

    # ---- Patient Sex ----
    try:
        sex = sex_map[getattr(ds, 'PatientSex', 'M')]
        sex = le.transform(np.array([sex]))[0]
    except Exception as e:
        sex = 0
    metadata = np.array([age, sex])
    return metadata

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:11.911243Z","iopub.execute_input":"2025-10-10T22:10:11.911745Z","iopub.status.idle":"2025-10-10T22:10:11.924782Z","shell.execute_reply.started":"2025-10-10T22:10:11.911724Z","shell.execute_reply":"2025-10-10T22:10:11.924256Z"},"papermill":{"duration":0.013845,"end_time":"2025-10-03T05:10:24.993772","exception":false,"start_time":"2025-10-03T05:10:24.979927","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


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
        # Direct conversion for 2D images
        frames.append(pix.astype(np.float32) * slope + intercept)
    elif pix.ndim == 3:
        # If RGB (H,W,3), take first channel; else assume multi-frame (N,H,W)
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            frames.append(pix[..., 0].astype(np.float32) * slope + intercept)
        else:
            # Pre-allocate for multi-frame images
            num_frames = pix.shape[0]
            frame_shape = pix.shape[1:]
            combined_slope = slope + intercept / 255.0 if intercept != 0 else slope

            for i in range(num_frames):
                frames.append(pix[i].astype(np.float32) * combined_slope + intercept)
    return frames

def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255"""
    mn = float(img.min())
    mx = float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)

    # Use in-place operations for better memory efficiency
    img = img - mn
    img = img / (mx - mn)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def process_dicom_file_yolo_single(dcm_path: Path) -> np.ndarray:
    """Process single DICOM file for YOLO (non-RGB mode)."""
    try:
        frames = read_dicom_frames_hu(dcm_path)
        if not frames:
            return None

        # For single slice mode, use first frame
        img = min_max_normalize(frames[0])
        return img
    except Exception as e:
        return None



def collect_series_slices(series_dir: Path) -> List[Path]:
    """Collect all DICOM files in a series directory and sort by spatial position."""
    # Use recursive glob for better performance than os.walk
    dicom_files = []
    for dcm_file in series_dir.rglob("*.dcm"):
        dicom_files.append(dcm_file)

    if not dicom_files:
        return []

    # Sort DICOM files by spatial position (same as training)
    temp_slices = []
    for filepath in dicom_files:
        try:
            # Use stop_before_pixels for faster metadata reading
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:11.925697Z","iopub.execute_input":"2025-10-10T22:10:11.926423Z","iopub.status.idle":"2025-10-10T22:10:11.946282Z","shell.execute_reply.started":"2025-10-10T22:10:11.926399Z","shell.execute_reply":"2025-10-10T22:10:11.945538Z"},"papermill":{"duration":0.014746,"end_time":"2025-10-03T05:10:25.039561","exception":false,"start_time":"2025-10-03T05:10:25.024815","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
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
    {
        "path": "/kaggle/input/rsna-yolo-models/cv_effnetv2s_v2_drop_path_fold0/weights/best.engine",
        "fold": 0,
        "weight": 1.0,
        "name": "effv2s",
        "mode": "2D"
    },
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
        "path": "/kaggle/input/rsna-yolo-models/cv_effnetv2_s_drop_path_25d_fold3/weights/best.engine",
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
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold03/weights/best.engine",
        "fold": 0,
        "weight": 1.0,
        "name": "yolo11",
        "mode": "2.5D"
    },
    #{
    #    "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold12/weights/best.engine",
    #    "fold": 1,
    #    "weight": 1.0,
    #    "name": "yolo11",
    #    "mode": "2D"
    #},
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold2/weights/best.engine",
        "fold": 2,
        "weight": 1.0,
        "name": "yolo11",
        "mode": "2.5D"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold3/weights/best.engine",
        "fold": 3,
        "weight": 1.0,
        "name": "yolo11",
        "mode": "2.5D"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/yolo_11_m_fold4/weights/best.engine",
        "fold": 4,
        "weight": 1.0,
        "name": "yolo11",
        "mode": "2D"
    }
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:11.947084Z","iopub.execute_input":"2025-10-10T22:10:11.947357Z","iopub.status.idle":"2025-10-10T22:10:11.967809Z","shell.execute_reply.started":"2025-10-10T22:10:11.947333Z","shell.execute_reply":"2025-10-10T22:10:11.967130Z"},"jupyter":{"outputs_hidden":false}}
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

    print(f"[{dir_label}] Loading fold {fold} model from {model_path}...")
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
    print("Loading all fold models from multiple model_dirs...")

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

    # Warm up models
    dummy_volume = torch.randn(1, FLAYER_CFG.in_chans, FLAYER_CFG.target_shape[0], FLAYER_CFG.size, FLAYER_CFG.size).to(device)
    with torch.no_grad():
        for (dir_label, fold), model in FLAYER_MODELS.items():
            _ = model(dummy_volume)

    print(f"Models ready for inference! Loaded: {total_loaded} models "
          f"from {len(model_dirs)} dirs, folds: {list(FLAYER_CFG.trn_fold)}.")

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:11.969559Z","iopub.execute_input":"2025-10-10T22:10:11.969759Z","iopub.status.idle":"2025-10-10T22:10:11.987123Z","shell.execute_reply.started":"2025-10-10T22:10:11.969743Z","shell.execute_reply":"2025-10-10T22:10:11.986373Z"},"papermill":{"duration":0.023125,"end_time":"2025-10-03T05:10:25.099131","exception":false,"start_time":"2025-10-03T05:10:25.076006","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}

def load_yolo_models():
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:11.987925Z","iopub.execute_input":"2025-10-10T22:10:11.988208Z","iopub.status.idle":"2025-10-10T22:10:12.012419Z","shell.execute_reply.started":"2025-10-10T22:10:11.988185Z","shell.execute_reply":"2025-10-10T22:10:12.011834Z"},"papermill":{"duration":0.018904,"end_time":"2025-10-03T05:10:25.122436","exception":false,"start_time":"2025-10-03T05:10:25.103532","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
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
    flayer_preds = []
    # Initialize flayer_fold_preds as a dictionary to maintain fold order
    flayer_fold_preds_dict = {}

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

                # Store predictions by fold for proper ordering
                flayer_fold_preds_dict[fold] = logits.sigmoid().float().cpu().numpy()

    # 邊界情況：沒有模型或權重總和為 0
    if (sum_logits is None) or (sum_w == 0.0):
        return np.full(len(LABEL_COLS), 0.5, dtype=np.float32), [np.full(len(LABEL_COLS), 0.5, dtype=np.float32) for _ in range(5)]

    avg_logits = sum_logits / float(sum_w)                    # 仍在 GPU
    probs = torch.sigmoid(avg_logits).float().cpu().numpy()   # 只在最後搬回 CPU

    # Create properly ordered flayer_fold_preds
    flayer_fold_preds = []
    for fold in range(5):  # FLAYER_CFG.trn_fold = [0,1,2,3,4]
        if fold in flayer_fold_preds_dict:
            flayer_fold_preds.append(flayer_fold_preds_dict[fold])
        else:
            flayer_fold_preds.append(np.full(len(LABEL_COLS), 0.5, dtype=np.float32))

    return probs, flayer_fold_preds


@torch.no_grad()
def predict_yolo_ensemble_with_meta(slices, metadata, flayer_fold_preds):
    """Run YOLO inference with meta classifier integration"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1, np.ones(len(YOLO_LABELS)) * 0.1, np.ones(len(YOLO_LABELS)) * 0.1

    yolo11m_cls_preds = []
    yolo11m_loc_preds = []
    yolo_effv2s_cls_preds = []
    yolo_effv2s_loc_preds = []

    total_weight = 0.0

    for fold_id, model_dict in enumerate(YOLO_MODELS):
        model = model_dict["model"]
        model_name = model_dict['name']
        weight = model_dict["weight"]
        mode = model_dict.get('mode', '2.5D')  # Default to 2.5D for backward compatibility
        device_id = model_dict.get("device", 0)

        try:
            # Process input based on model mode (keeping 2.5D logic)
            if mode == "2D":
                yolo_slices = process_dicom_for_yolo_2d_from_loaded(slices)
            elif mode == "2.5D":
                yolo_slices = process_dicom_for_yolo_25d_from_loaded(slices)
            else:
                print(f"Warning: Unknown mode '{mode}' for model {model_name}, using 2.5D")
                yolo_slices = process_dicom_for_yolo_25d_from_loaded(slices)

            if not yolo_slices:
                print(f"Warning: No slices generated for model {model_name} in mode {mode}")
                if model_name == 'yolo11':
                    yolo11m_cls_preds.append(0.1 * weight)
                    yolo11m_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
                elif model_name == 'effv2s':
                    yolo_effv2s_cls_preds.append(0.1 * weight)
                    yolo_effv2s_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
                total_weight += weight
                continue

            max_conf_all = 0.0
            per_class_max = np.zeros(len(YOLO_LABELS), dtype=np.float32)

            # Process in batches
            for i in range(0, len(yolo_slices), BATCH_SIZE):
                batch_slices = yolo_slices[i:i+BATCH_SIZE]

                if (len(batch_slices) < 32): #and (model_name == 'effv2s'):
                    batch_slices += [batch_slices[0]] * (32 - len(batch_slices))
                results = model.predict(
                    batch_slices,
                    verbose=False,
                    batch=len(batch_slices),
                    device=device_id,
                    conf=0.01,
                    half=True,
                    imgsz=512
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

            try:
                if model_name == 'yolo11':
                    yolo11m_cls_preds.append(max_conf_all * weight)
                    yolo11m_loc_preds.append(per_class_max * weight)
                elif model_name == 'effv2s':
                    yolo_effv2s_cls_preds.append(max_conf_all * weight)
                    yolo_effv2s_loc_preds.append(per_class_max * weight)
                total_weight += weight

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(e)

        except Exception as e:
            print(e)
            if model_name == 'yolo11':
                    yolo11m_cls_preds.append(0.1 * weight)
                    yolo11m_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            elif model_name == 'effv2s':
                yolo_effv2s_cls_preds.append(0.1 * weight)
                yolo_effv2s_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            total_weight += weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if total_weight > 0:
        yolo11m_final_cls_pred = sum(yolo11m_cls_preds) / 4
        yolo11m_final_loc_preds = sum(yolo11m_loc_preds) / 4
        yolo_effv2s_final_cls_pred = sum(yolo_effv2s_cls_preds) / 4
        yolo_effv2s_final_loc_preds = sum(yolo_effv2s_loc_preds) / 4
    else:
        yolo11m_final_cls_pred = 0.1
        yolo_effv2s_final_cls_pred = 0.1
        yolo11m_final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1
        yolo_effv2s_final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1

    #meta stage
    meta_lgb_preds = []
    meta_xgb_preds = []
    meta_cat_preds = []

    # Get the actual fold IDs that were used (excluding fold 1)
    used_folds = []
    for model_dict in YOLO_MODELS:
        fold_id = model_dict["fold"]
        if fold_id != 1 and fold_id not in used_folds:  # Avoid duplicates
            used_folds.append(fold_id)

    for fold_id in used_folds:
        # Find the model index for this fold_id
        model_idx = None
        for idx, model_dict in enumerate(YOLO_MODELS):
            if model_dict["fold"] == fold_id:
                model_idx = idx
                break

        if model_idx is None:
            print(f"Warning: No model found for fold {fold_id}")
            continue

        try:
            yolo11m_cls_pred = yolo11m_cls_preds[model_idx]
            yolo11m_loc_pred = yolo11m_loc_preds[model_idx]
            yolo_effv2s_cls_pred = yolo_effv2s_cls_preds[model_idx]
            yolo_effv2s_loc_pred = yolo_effv2s_loc_preds[model_idx]
            X = np.concatenate([np.array([yolo11m_cls_pred]), yolo11m_loc_pred, np.array([yolo_effv2s_cls_pred]),
                                yolo_effv2s_loc_pred, flayer_fold_preds[fold_id], metadata], axis=0)[None, :]
            lgb_pred = predict_prob_lgb(X, fold_id)
            xgb_pred = predict_prob_xgb(X, fold_id)
            cat_pred = predict_prob_cat(X, fold_id)
            meta_lgb_preds.append(lgb_pred)
            meta_xgb_preds.append(xgb_pred)
            meta_cat_preds.append(cat_pred)
        except Exception as e:
            print(e)
            meta_lgb_preds.append(np.array([[0]]))
            meta_xgb_preds.append(np.array([[0]]))
            meta_cat_preds.append(np.array([[0]]))

    meta_lgb_preds = np.mean(meta_lgb_preds, axis=0)[:, 0]
    meta_xgb_preds = np.mean(meta_xgb_preds, axis=0)[:, 0]
    meta_cat_preds = np.mean(meta_cat_preds, axis=0)[:, 0]
    return yolo11m_final_cls_pred, yolo11m_final_loc_preds, yolo_effv2s_final_cls_pred, yolo_effv2s_final_loc_preds, meta_lgb_preds, meta_xgb_preds, meta_cat_preds

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:12.013242Z","iopub.execute_input":"2025-10-10T22:10:12.013451Z","iopub.status.idle":"2025-10-10T22:10:12.033815Z","shell.execute_reply.started":"2025-10-10T22:10:12.013436Z","shell.execute_reply":"2025-10-10T22:10:12.033079Z"},"papermill":{"duration":0.010055,"end_time":"2025-10-03T05:10:25.135982","exception":false,"start_time":"2025-10-03T05:10:25.125927","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}


def load_dicom_series_once(series_path: str) -> Tuple[List[np.ndarray], List[pydicom.Dataset], str, pydicom.Dataset]:
    """
    Load DICOM series once and return processed pixel arrays and metadata.
    This eliminates duplicate DICOM loading for both YOLO and Flayer models.

    KEY OPTIMIZATION: Previously, DICOM files were loaded twice - once for YOLO processing
    and once for Flayer processing. This function loads DICOM files only once, then both
    models can create their required input formats from the same loaded data.

    Returns:
        pixel_arrays: List of HU-converted and processed 2D arrays
        datasets: List of DICOM datasets (for metadata if needed)
        series_name: Name of the series
        first_ds: First DICOM dataset for metadata parsing
    """
    series_path = Path(series_path)
    series_name = series_path.name

    # Collect and sort DICOM files once
    dicom_files = collect_series_slices(series_path)

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_path}")

    # Load all DICOM datasets and extract pixel arrays in one pass
    datasets = []
    pixel_arrays = []

    for dcm_path in dicom_files:
        try:
            ds = pydicom.dcmread(str(dcm_path), force=True)
            datasets.append(ds)

            # Extract pixel array and apply HU conversion
            pix = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

            if pix.ndim == 2:
                hu_array = pix * slope + intercept
            elif pix.ndim == 3:
                if pix.shape[-1] == 3 and pix.shape[0] != 3:
                    # RGB image, take first channel
                    hu_array = pix[..., 0] * slope + intercept
                else:
                    # Multi-frame, take middle frame
                    frame_idx = pix.shape[0] // 2
                    hu_array = pix[frame_idx] * slope + intercept
            else:
                raise ValueError(f"Unexpected pixel array dimensions: {pix.shape}")

            pixel_arrays.append(hu_array)

        except Exception as e:
            print(f"Warning: Failed to load {dcm_path}: {e}")
            continue

    if not datasets or not pixel_arrays:
        raise ValueError(f"No valid DICOM data in {series_path}")

    return pixel_arrays, datasets, series_name, datasets[0]


def process_dicom_for_yolo_2d_from_loaded(pixel_arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Create YOLO 2D grayscale images from pre-loaded pixel arrays"""
    if not pixel_arrays:
        return []
    
    # Normalize all frames and convert to grayscale
    grayscale_slices = []
    for frame in pixel_arrays:
        normalized_frame = min_max_normalize(frame)
        # Convert to 3-channel grayscale for YOLO (RGB with same values)
        grayscale_img = np.stack([normalized_frame] * 3, axis=-1)
        grayscale_slices.append(grayscale_img)
    
    return grayscale_slices

def process_dicom_for_yolo_25d_from_loaded(pixel_arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Create YOLO RGB triplets from pre-loaded pixel arrays (2.5D mode)"""
    if len(pixel_arrays) < 3:
        # Fallback for insufficient slices - use 2D mode
        return process_dicom_for_yolo_2d_from_loaded(pixel_arrays)

    # Normalize all frames
    normalized_frames = [min_max_normalize(frame) for frame in pixel_arrays]

    # Create RGB triplets more efficiently
    rgb_slices = []
    for i in range(1, len(normalized_frames) - 1):
        rgb_img = np.empty((normalized_frames[0].shape[0], normalized_frames[0].shape[1], 3), dtype=np.uint8)
        rgb_img[..., 0] = normalized_frames[i-1]  # Previous slice
        rgb_img[..., 1] = normalized_frames[i]     # Current slice
        rgb_img[..., 2] = normalized_frames[i+1]   # Next slice
        rgb_slices.append(rgb_img)

    return rgb_slices

def process_dicom_for_yolo_from_loaded(pixel_arrays: List[np.ndarray], mode: str = "2.5D") -> List[np.ndarray]:
    """Create YOLO input from pre-loaded pixel arrays based on mode"""
    if mode == "2D":
        return process_dicom_for_yolo_2d_from_loaded(pixel_arrays)
    elif mode == "2.5D":
        return process_dicom_for_yolo_25d_from_loaded(pixel_arrays)
    else:
        raise ValueError(f"Unsupported YOLO mode: {mode}. Use '2D' or '2.5D'")


def process_dicom_for_flayer_from_loaded(pixel_arrays: List[np.ndarray], target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Create Flayer 3D volume from pre-loaded pixel arrays"""
    if not pixel_arrays:
        raise ValueError("No pixel arrays provided")

    # Normalize all slices for Flayer
    normalized_slices = [min_max_normalize(frame) for frame in pixel_arrays]

    # Stack into 3D volume
    volume = np.stack(normalized_slices, axis=0)

    # Resize to target shape if needed
    if volume.shape != target_shape:
        # Use the existing resize logic from FlayerDICOMPreprocessor
        current_shape = volume.shape
        target_depth, target_height, target_width = target_shape

        if current_shape != (target_depth, target_height, target_width):
            zoom_factors = [
                target_depth / current_shape[0],
                target_height / current_shape[1],
                target_width / current_shape[2]
            ]

            volume = cp.asarray(volume)
            resized_volume = zoom(volume, zoom_factors, order=1, mode='nearest')
            resized_volume = resized_volume[:target_depth, :target_height, :target_width]
            volume = cp.asnumpy(resized_volume).astype(np.uint8)

            # Padding if necessary
            pad_width = [
                (0, max(0, target_depth - resized_volume.shape[0])),
                (0, max(0, target_height - resized_volume.shape[1])),
                (0, max(0, target_width - resized_volume.shape[2]))
            ]

            if any(pw[1] > 0 for pw in pad_width):
                volume = np.pad(volume, pad_width, mode='edge')

    return volume


def process_dicom_for_yolo(series_path: str, mode: str = "2.5D") -> List[np.ndarray]:
    """Process DICOM for YOLO based on mode (2D or 2.5D)."""
    pixel_arrays, _, _, _ = load_dicom_series_once(series_path)
    return process_dicom_for_yolo_from_loaded(pixel_arrays, mode)

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:12.034529Z","iopub.execute_input":"2025-10-10T22:10:12.034716Z","iopub.status.idle":"2025-10-10T22:10:12.055337Z","shell.execute_reply.started":"2025-10-10T22:10:12.034701Z","shell.execute_reply":"2025-10-10T22:10:12.054784Z"},"jupyter":{"outputs_hidden":false},"papermill":{"duration":0.012524,"end_time":"2025-10-03T05:10:25.151964","exception":false,"start_time":"2025-10-03T05:10:25.13944","status":"completed"},"tags":[]}
def _predict_inner(series_path: str) -> pl.DataFrame:
    """Main ensemble prediction logic - optimized to load DICOMs only once"""
    global YOLO_MODELS, FLAYER_MODELS

    # Load models if not already loaded
    if not YOLO_MODELS or not FLAYER_MODELS:
        load_all_models()

    # Pre-allocate arrays for better memory efficiency
    conservative_preds = np.full(len(LABEL_COLS), 0.1, dtype=np.float32)
    yolo_full_preds = np.zeros(len(LABEL_COLS), dtype=np.float32)

    try:
        # Load DICOM series ONCE and create both YOLO and Flayer inputs
        pixel_arrays, datasets, series_name, first_ds = load_dicom_series_once(series_path)

        # Create Flayer 3D volume from loaded data
        flayer_volume = process_dicom_for_flayer_from_loaded(pixel_arrays, FLAYER_CFG.target_shape)
        print(f"{flayer_volume.shape=}")

        # Parse metadata for meta classifiers
        metadata = parse_meta_data(first_ds)

        # Get predictions from all models
        flayer_preds, flayer_fold_preds = predict_flayer_ensemble(flayer_volume)
        yolo11m_final_cls_pred, yolo11m_final_loc_preds, yolo_effv2s_final_cls_pred, yolo_effv2s_final_loc_preds, meta_lgb_preds, meta_xgb_preds, meta_cat_preds = predict_yolo_ensemble_with_meta(pixel_arrays, metadata, flayer_fold_preds)

        # Convert to numpy array once
        flayer_preds = np.asarray(flayer_preds, dtype=np.float32)

        if flayer_preds.shape[0] != len(LABEL_COLS):
            raise ValueError("Flayer ensemble output length mismatch")

        # Ensemble predictions - aneurysm index calculation
        aneurysm_idx = LABEL_COLS.index('Aneurysm Present')

        # yolo_full_preds has preds in LABEL_COLS order now
        yolo11m_preds = np.zeros(len(LABEL_COLS))
        yolo_effv2s_preds = np.zeros(len(LABEL_COLS))
        for i, label in enumerate(YOLO_LABELS):
            if label in LABEL_COLS:
                label_idx = LABEL_COLS.index(label)
                yolo11m_preds[label_idx] = yolo11m_final_loc_preds[i]
                yolo_effv2s_preds[label_idx] = yolo_effv2s_final_loc_preds[i]
        yolo11m_preds[aneurysm_idx] = yolo11m_final_cls_pred
        yolo_effv2s_preds[aneurysm_idx] = yolo_effv2s_final_cls_pred

        ensemble_preds =  ensemble_w[0] * meta_lgb_preds + ensemble_w[1] * meta_xgb_preds + ensemble_w[2] * meta_cat_preds + ensemble_w[3] * flayer_preds +  ensemble_w[4] * yolo11m_preds + ensemble_w[5] * yolo_effv2s_preds

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
        predictions_df = pl.DataFrame(
            data=[conservative_preds.tolist()],
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

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T22:10:12.056005Z","iopub.execute_input":"2025-10-10T22:10:12.056395Z","iopub.status.idle":"2025-10-10T22:11:50.436324Z","shell.execute_reply.started":"2025-10-10T22:10:12.056377Z","shell.execute_reply":"2025-10-10T22:11:50.435739Z"},"papermill":{"duration":114.451079,"end_time":"2025-10-03T05:12:19.61779","exception":false,"start_time":"2025-10-03T05:10:25.166711","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
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