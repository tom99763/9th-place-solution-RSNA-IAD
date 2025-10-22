
import os
import numpy as np
import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import gc
import sys
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# Data handling
import polars as pl
import joblib

# ML/DL
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
import pickle

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.insert(0, "./ultralytics-timm")

# YOLO
from ultralytics import YOLO

from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import zoom
import xgboost as xgb
import argparse


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
        
        # Load DICOM datasets
        datasets = []
        for filepath in dicom_files:
            ds = pydicom.dcmread(filepath, force=True)
            datasets.append(ds)
        
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
        """
        Sort slices by z-coordinate
        """

        sorted_slices = sorted(slice_info, key=lambda x: x['z_position'])
        
        return sorted_slices
    #original
    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Get windowing parameters based on modality
        """
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            return "CT", "CT"
            
        elif modality == 'MR':
            return None, None
            
        else:
            return None, None

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
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
        
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)

        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize 3D volume to target size
        """
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        volume = cp.asarray(volume)
        
        resized_volume = zoom(volume, zoom_factors, order=1, mode='nearest')
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        resized_volume = cp.asnumpy(resized_volume)
        
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')

        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """
        Process DICOM series and return as NumPy array (for Kaggle: no file saving)
        """
   
        datasets, series_name = self.load_dicom_series(series_path)
        
        first_ds = datasets[0]
        first_img = first_ds.pixel_array
        
        if len(datasets) == 1 and first_img.ndim == 3:
            return self._process_single_3d_dicom(first_ds, series_name)
        else:
            return self._process_multiple_2d_dicoms(datasets, series_name)
            
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """
        Process single 3D DICOM file (for Kaggle: no file saving)
        """

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

# ====================================================
# YOLO DICOM Processing
# ====================================================
def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    """Read DICOM file and return HU frames (with slope/intercept conversion)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:   
        for i in range(pix.shape[0]):
            frm = pix[i].astype(np.float32)
            frames.append(frm * slope + intercept)
    else:
        # Unsupported layout
        pass
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
    frames = read_dicom_frames_hu(dcm_path)
    processed_slices = []
    for f in frames:
        img_u8 = min_max_normalize(f)
        processed_slices.append(img_u8)
    return processed_slices

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

            if hasattr(ds, "SliceLocation"):
                sort_val = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                sort_val = float(ds.ImagePositionPatient[-1])
            else:
                sort_val = float(getattr(ds, "InstanceNumber", 0))

            temp_slices.append((sort_val, filepath))

        except Exception as e:
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


# ====================================================
# Transforms
# ====================================================
def get_inference_transform():
    """Get inference transformation"""
    return A.Compose([
        A.Resize(FLAYER_CFG.size, FLAYER_CFG.size),
        A.Normalize(),
        ToTensorV2(),
    ])

######################################################################
# 2.5 Flayer
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
    if hasattr(FLAYER_CFG, 'model_dirs') and FLAYER_CFG.model_dirs:
        return list(FLAYER_CFG.model_dirs)
    elif hasattr(FLAYER_CFG, 'model_dir') and FLAYER_CFG.model_dir:
        return [FLAYER_CFG.model_dir]
    else:
        raise ValueError("Please specify FLAYER_CFG.model_dirs (list) or FLAYER_CFG.model_dir (str).")

def _dir_label(path_str: str) -> str:
    return Path(path_str).name

def _get_model_name_for_dir(dir_label: str) -> str:
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


def load_yolo_models():
    """Load all YOLO models"""
    models = []
    for config in YOLO_MODEL_CONFIGS:
        model = YOLO(config["path"], task='detect')
        if not str(config["path"]).endswith('.engine'):
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
    global YOLO_MODELS, FLAYER_MODELS
    
    YOLO_MODELS = load_yolo_models()
    
    if not FLAYER_MODELS:
        load_flayer_models()

def flayer_predict_single_model(model: nn.Module, tensor_5d: torch.Tensor) -> torch.Tensor:
    """
    Run inference for a single model and return LOGITS (torch.Tensor on device).
    - tensor_5d: (1, 1, D, H, W) 已在 GPU/AMP 準備好的張量
    """
   
    outputs = model(tensor_5d)

    heatmap = outputs['heatmap'] if isinstance(outputs, dict) else outputs
    logits = compute_class_logits_from_heatmap(heatmap)

    logits = logits.to(tensor_5d.device, dtype=torch.float32)
    logits = logits.flatten()  # (num_labels,)
    return logits


def predict_flayer_ensemble(image: np.ndarray) -> np.ndarray:
    image_hwd = image.transpose(1, 2, 0)  # (H, W, D)
    transformed = FLAYER_TRANSFORM(image=image_hwd)

    tensor = transformed['image']
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if tensor.dim() != 3:
        raise ValueError(f"FLAYER_TRANSFORM['image'] should be 3D (D,H,W), got shape {tuple(tensor.shape)}")

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

    
    with torch.inference_mode():
       
        with autocast(enabled=FLAYER_CFG.use_amp):
            for (dir_label, fold), model in FLAYER_MODELS.items():
                w = float(_lookup_weight(dir_label, fold))
                if w == 0.0:
                    continue

                logits = flayer_predict_single_model(model, tensor_5d)

                
                if sum_logits is None:
                    sum_logits = torch.zeros_like(logits)
                if sum_logits.shape != logits.shape:
                    raise ValueError(f"Logits shape mismatch: got {tuple(logits.shape)}, "
                                     f"expected {tuple(sum_logits.shape)}")

                
                sum_logits.add_(logits, alpha=w)
                sum_w += w

   
    if (sum_logits is None) or (sum_w == 0.0):
        return np.full(len(LABEL_COLS), 0.5, dtype=np.float32)

    avg_logits = sum_logits / float(sum_w)                    # 仍在 GPU
    probs = torch.sigmoid(avg_logits).float().cpu().numpy()   # 只在最後搬回 CPU
    return probs




def predict_flayer_ensemble(image: np.ndarray) -> np.ndarray:
    image_hwd = image.transpose(1, 2, 0)  # (H, W, D)
    transformed = FLAYER_TRANSFORM(image=image_hwd)

    tensor = transformed['image']
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

   
    if tensor.dim() != 3:
        raise ValueError(f"FLAYER_TRANSFORM['image'] should be 3D (D,H,W), got shape {tuple(tensor.shape)}")

    
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

    
    with torch.inference_mode():
        
        with autocast(enabled=FLAYER_CFG.use_amp):
            for (dir_label, fold), model in FLAYER_MODELS.items():
                w = float(_lookup_weight(dir_label, fold))
                if w == 0.0:
                    continue

                logits = flayer_predict_single_model(model, tensor_5d)

                
                if sum_logits is None:
                    sum_logits = torch.zeros_like(logits)
                if sum_logits.shape != logits.shape:
                    raise ValueError(f"Logits shape mismatch: got {tuple(logits.shape)}, "
                                     f"expected {tuple(sum_logits.shape)}")

                # 就地加權累加
                sum_logits.add_(logits, alpha=w)
                sum_w += w
                flayer_preds.append(logits.sigmoid().float().cpu().numpy())

    if (sum_logits is None) or (sum_w == 0.0):
        return np.full(len(LABEL_COLS), 0.5, dtype=np.float32)

    avg_logits = sum_logits / float(sum_w)
    probs = torch.sigmoid(avg_logits).float().cpu().numpy()
    return probs, flayer_preds


@torch.no_grad()
def predict_yolo_ensemble(slices, metadata, flayer_fold_preds):
    """Run YOLO inference using all models"""
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1

    yolo11m_cls_preds = []
    yolo11m_loc_preds = []
    effv2s_cls_preds = []
    effv2s_loc_preds = []
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    for fold_id, model_dict in enumerate(YOLO_MODELS):
        model = model_dict["model"]
        model_name = model_dict["name"]
        weight = model_dict["weight"]
        
        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(YOLO_LABELS), dtype=np.float32)
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                if (len(batch_slices) < 32): #and (model_name == 'effv2s'):
                    batch_slices += [batch_slices[0]] * (32 - len(batch_slices))
                
                results = model.predict(
                    batch_slices, 
                    verbose=False, 
                    batch=len(batch_slices), 
                    device=device, 
                    conf=0.01,
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

            if model_name == 'YOLOv11m':
                yolo11m_cls_preds.append(max_conf_all * weight)
                yolo11m_loc_preds.append(per_class_max * weight)
            elif model_name == 'effv2s':
                effv2s_cls_preds.append(max_conf_all * weight)
                effv2s_loc_preds.append(per_class_max * weight)
            
            ensemble_cls_preds.append(max_conf_all * weight)
            ensemble_loc_preds.append(per_class_max * weight)
            total_weight += weight
            
        except Exception as e:
            if model_name == 'YOLOv11m':
                yolo11m_cls_preds.append(0.1 * weight)
                yolo11m_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            elif model_name == 'effv2s':
                effv2s_cls_preds.append(0.1 * weight)
                effv2s_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
                
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            total_weight += weight
    
    if total_weight > 0:
        final_cls_pred = sum(ensemble_cls_preds) / (total_weight)
        final_loc_preds = sum(ensemble_loc_preds) / (total_weight)
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1

    meta_lgb_preds = []
    meta_xgb_preds = []
    meta_cat_preds = []

    for fold_id in range(len(YOLO_MODELS)//2):
        try:
            X = np.concatenate([np.array([yolo11m_cls_preds[fold_id]]), yolo11m_loc_preds[fold_id],
                                np.array([effv2s_cls_preds[fold_id]]), effv2s_loc_preds[fold_id],
                                flayer_fold_preds[fold_id], metadata], axis=0)[None, :]
            lgb_pred = predict_prob_lgb(X, fold_id)
            xgb_pred = predict_prob_xgb(X, fold_id)
            cat_pred = predict_prob_cat(X, fold_id)
            meta_lgb_preds.append(lgb_pred)
            meta_xgb_preds.append(xgb_pred)
            meta_cat_preds.append(cat_pred)
        except Exception as e:
            meta_lgb_preds.append(np.ones(len(YOLO_LABELS) + 1) * 0.1)
            meta_xgb_preds.append(np.ones(len(YOLO_LABELS) + 1) * 0.1)
            meta_cat_preds.append(np.ones(len(YOLO_LABELS) + 1) * 0.1)

    meta_lgb_preds = np.mean(meta_lgb_preds, axis=0)[:, 0]
    meta_xgb_preds = np.mean(meta_xgb_preds, axis=0)[:, 0]
    meta_cat_preds = np.mean(meta_cat_preds, axis=0)[:, 0]
    return final_cls_pred, final_loc_preds, meta_lgb_preds, meta_xgb_preds, meta_cat_preds

# Safe processing function with memory cleanup
def process_dicom_series_for_flayer(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    """
    Safe DICOM processing with memory cleanup
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        np.ndarray: Processed volume
    """
    try:
        preprocessor = FlayerDICOMPreprocessor(target_shape=target_shape)
        return preprocessor.process_series(series_path)
    finally:
        gc.collect()



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


def process_dicom_for_yolo(series_path: str, mode: str = "2.5D") -> List[np.ndarray]:
    """Process DICOM for YOLO with parallel processing and mode support"""
    series_path = Path(series_path)
    dicom_files = collect_series_slices(series_path)
    ds = pydicom.dcmread(dicom_files[0], force=True)
    metadata = parse_meta_data(ds)

    if mode == "2D":
        all_slices: List[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_dicom_file_yolo, dcm_path, False) 
                      for dcm_path in dicom_files]
            for future in futures:
                slices = future.result()
                all_slices.extend(slices)
        return all_slices, metadata

    elif mode == "2.5D":
       
        if len(dicom_files) < 3:
            return process_dicom_for_yolo(series_path, "2D")

        all_frames = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            futures = [executor.submit(process_dicom_file_yolo, dcm_path, True) 
                      for dcm_path in dicom_files]
            
            for future in futures:
                slices = future.result()
                all_frames.extend(slices)

        if len(all_frames) < 3:
            print(f"Warning: Only {len(all_frames)} frames available, need at least 3 for 2.5D")
            return (all_frames, metadata) if all_frames else ([], metadata)

        rgb_slices = []
        for i in range(1, len(all_frames) - 1):
            prev_frame = all_frames[i-1]
            curr_frame = all_frames[i]
            next_frame = all_frames[i+1]
            
            if prev_frame is None or curr_frame is None or next_frame is None:
                continue
  
            if not (prev_frame.shape == curr_frame.shape == next_frame.shape):
                print(f"Warning: Frame shape mismatch at index {i}")
                continue
            
            rgb_img = np.stack([prev_frame, curr_frame, next_frame], axis=-1)

            if IMG_SIZE > 0 and (rgb_img.shape[0] != IMG_SIZE or rgb_img.shape[1] != IMG_SIZE):
                rgb_img = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            
            if rgb_img.shape[-1] != 3 or rgb_img.ndim != 3:
                print(f"Warning: Invalid RGB shape {rgb_img.shape} at index {i}")
                continue
                
            rgb_slices.append(rgb_img)
        
        print(f"Created {len(rgb_slices)} valid 2.5D slices from {len(all_frames)} frames")
        return (rgb_slices, metadata) if rgb_slices else (all_frames, metadata)

    else:
        raise ValueError(f"Unsupported YOLO mode: {mode}. Use '2D' or '2.5D'")

def predict(series_path: str) -> pl.DataFrame:
    """Main ensemble prediction logic"""
    global YOLO_MODELS, FLAYER_MODELS
    
    # Load models if not already loaded
    if  not YOLO_MODELS or not FLAYER_MODELS:
        load_all_models()

    aneurysm_idx = LABEL_COLS.index('Aneurysm Present')
    
    yolo_slices, metadata = process_dicom_for_yolo(series_path)
    flayer_volume = process_dicom_series_for_flayer(series_path, FLAYER_CFG.target_shape)
    print(f"{flayer_volume.shape=}")
    
    flayer_preds, flayer_fold_preds = predict_flayer_ensemble(flayer_volume)
    flayer_preds = np.asarray(flayer_preds, dtype=np.float32)
    if flayer_preds.shape[0] != len(LABEL_COLS):
        raise ValueError("Flayer ensemble output length mismatch")

    yolo_cls_pred, yolo_loc_preds, meta_lgb_preds, meta_xgb_preds, meta_cat_preds = predict_yolo_ensemble(yolo_slices, metadata, flayer_fold_preds)

    yolo_full_preds = np.zeros(len(LABEL_COLS))
    for i, label in enumerate(YOLO_LABELS):
        if label in LABEL_COLS:
            label_idx = LABEL_COLS.index(label)
            yolo_full_preds[label_idx] = yolo_loc_preds[i]
    yolo_full_preds[aneurysm_idx] = yolo_cls_pred

    ensemble_preds =  ensemble_w[0] * meta_lgb_preds + ensemble_w[1] * meta_xgb_preds + ensemble_w[2] * meta_cat_preds + ensemble_w[3] * flayer_preds +  ensemble_w[4] * yolo_full_preds
   
    # Create output dataframe
    predictions_df = pl.DataFrame(
        data=[ensemble_preds.tolist()],
        schema=LABEL_COLS,
        orient='row'
    )
    
    return predictions_df
        

def parse_args():
    ap = argparse.ArgumentParser(description='Train and validate meta classifier pipeline')
    ap.add_argument('--data_path', type=str, default='./', help='path where all the series are present')
    ap.add_argument('--meta_cls_weight_path', type=str, default='./meta_classifiers')
    ap.add_argument('--yolo_weight_path', type=str, default='./yolo25d/yolo_aneurysm_locations')
    ap.add_argument('--flayer_weight_path', type=str, default='./flayer/flayer_weights')
    return ap.parse_args()

def make_yolo_config(model_dir):
    global YOLO_MODEL_CONFIGS

    model_names = [
        "yolo_11m_fold0",
        "yolo_11m_fold1",
        "yolo_11m_fold2",
        "yolo_effnetv2_fold0",
        "yolo_effnetv2_fold1",
        "yolo_effnetv2_fold2",
    ]

    for model in model_names:
        model = model_dir / model 
        name = "effv2s" if "eff" in model.stem else "YOLOv11m"
        YOLO_MODEL_CONFIGS.append(
            {
                "path": model / "weights/best.pt",
                "fold": int(model.stem[-1]),
                "weight": 1.0,
                "name": name
            }
        )

if __name__ == "__main__":

    args = parse_args()
    print(args.yolo_weight_path)


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xgb.set_config(verbosity=0)

    META_CLS_PATH = Path(args.meta_cls_weight_path)

    with open(META_CLS_PATH/'label_encoder_sex.pkl', 'rb') as f:
        le = pickle.load(f)
    sex_map = {name[0]: name for name in le.classes_}

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

    ensemble_w = [0.2793572,  0.58535173, 0.00420708, 0.01056496, 0.06308367, 0.05743536]


    model_prefix="meta_classifier"
    n_folds = 5


    lgb_models = {label: [] for label in LABEL_COLS}
    xgb_models = {label: [] for label in LABEL_COLS}
    cat_models = {label: [] for label in LABEL_COLS}
    meta_models = {'lgb': lgb_models, 'xgb': xgb_models, 'cat': cat_models}


    for label in tqdm(LABEL_COLS):
        for model_file in ['lgb', 'xgb', 'cat']:
            for fold in range(n_folds):
                model_path = f"{META_CLS_PATH}/{model_file}/{model_prefix}_{label}_fold_fold{fold}.pkl"
                model = joblib.load(model_path)
                meta_models[model_file][label].append(model)

    # ====================================================
    # Configuration
    # ====================================================
    class FlayerInferenceConfig:
        # Model settings
        model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
        size = 448
        target_cols = LABEL_COLS
        num_classes = len(VESSEL_LABELS)
        heatmap_classes = VESSEL_LABELS
        in_chans = 1
        
        target_shape = (64, 448, 448)
        output_stride_depth = 1
        output_stride_height = 16
        output_stride_width = 16
        base_channels: int = 32

        batch_size = 1
        use_amp = True
        use_tta = False 
        tta_transforms = 0
        
        model_dirs = None
        n_fold = 5
        trn_fold = [0,1,2,3,4]
        ensemble_weights = None  # None means equal weights

    FLAYER_CFG = FlayerInferenceConfig()
    FLAYER_CFG.model_dirs = [ Path(args.flayer_weight_path) ]


    # ====================================================
    # YOLO Configuration
    # ====================================================
    IMG_SIZE = 512
    BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
    MAX_WORKERS = 4

    YOLO_MODEL_CONFIGS = []

    make_yolo_config(Path(args.yolo_weight_path))

    # ====================================================
    # Model Loading and Inference
    # ====================================================

    YOLO_MODELS = []
    FLAYER_MODELS = {}
    FLAYER_TRANSFORM = None
    FLAYER_TTA_TRANSFORMS = None

    all_preds = []
    for series_path in os.listdir(args.data_path):
        print(f"Evaluating series: {series_path}")
        res = predict(os.path.join(args.data_path, series_path))
        all_preds.append(res)

    all_preds = pl.concat(all_preds)
    all_preds.write_csv("preds.csv")
    print(all_preds)
