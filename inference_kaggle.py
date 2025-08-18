# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:14.863114Z","iopub.execute_input":"2025-08-07T18:49:14.863339Z","iopub.status.idle":"2025-08-07T18:49:14.8681Z","shell.execute_reply.started":"2025-08-07T18:49:14.86332Z","shell.execute_reply":"2025-08-07T18:49:14.867372Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:17.908678Z","iopub.execute_input":"2025-08-07T18:49:17.908983Z","iopub.status.idle":"2025-08-07T18:49:17.992562Z","shell.execute_reply.started":"2025-08-07T18:49:17.908961Z","shell.execute_reply":"2025-08-07T18:49:17.991743Z"},"jupyter":{"outputs_hidden":false}}
import kaggle_evaluation.rsna_inference_server
import shutil
import gc

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:19.600774Z","iopub.execute_input":"2025-08-07T18:49:19.601432Z","iopub.status.idle":"2025-08-07T18:49:19.605431Z","shell.execute_reply.started":"2025-08-07T18:49:19.601405Z","shell.execute_reply":"2025-08-07T18:49:19.604707Z"},"jupyter":{"outputs_hidden":false}}
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:20.385899Z","iopub.execute_input":"2025-08-07T18:49:20.386191Z","iopub.status.idle":"2025-08-07T18:49:20.391622Z","shell.execute_reply.started":"2025-08-07T18:49:20.386169Z","shell.execute_reply":"2025-08-07T18:49:20.390663Z"},"jupyter":{"outputs_hidden":false}}
IMG_SIZE = 512
FACTOR = 1

# HU clipping bounds & window definitions matching training pipeline
RAW_MIN_HU = -1200.0
RAW_MAX_HU = 4000.0
BASE_WINDOWS = [
    (40.0, 80.0),    # brain narrow
    (50.0, 150.0),   # soft tissue / brain standard
    (60.0, 300.0),   # wider soft
    (300.0, 700.0),  # bone/high contrast
]


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

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:21.22941Z","iopub.execute_input":"2025-08-07T18:49:21.230139Z","iopub.status.idle":"2025-08-07T18:49:21.238649Z","shell.execute_reply.started":"2025-08-07T18:49:21.230113Z","shell.execute_reply":"2025-08-07T18:49:21.237889Z"},"jupyter":{"outputs_hidden":false}}
def _convert_to_hu(pixel_array: np.ndarray, dcm) -> np.ndarray:
    """Convert raw DICOM pixel data to HU if possible (float64)."""
    img = pixel_array.astype(np.float64)
    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        try:
            img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        except Exception:
            pass
    return img

def _fixed_hu_window(img: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(img, low, high)
    return (windowed - low) / max(width, 1e-6)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:22.542526Z","iopub.execute_input":"2025-08-07T18:49:22.542834Z","iopub.status.idle":"2025-08-07T18:49:22.552064Z","shell.execute_reply.started":"2025-08-07T18:49:22.542811Z","shell.execute_reply":"2025-08-07T18:49:22.551244Z"},"jupyter":{"outputs_hidden":false}}
def process_dicom_series(series_path: str) -> np.ndarray:
    """Load series, convert all slices to HU (float32), return 3D array (N,H,W) without resizing.
    Resizing is deferred until after MIP (to mimic training data creation)."""
    series_path = Path(series_path)
    filepaths = []
    for root, _, files in os.walk(series_path):
        for f in files:
            if f.endswith('.dcm'):
                filepaths.append(os.path.join(root, f))
    if not filepaths:
        return np.zeros((0, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    filepaths.sort()
    slices = []
    ordering = []
    for fp in filepaths:
        try:
            ds = pydicom.dcmread(fp, force=True)
            arr = ds.pixel_array
            if arr.ndim == 3:
                # RGB or multi-frame
                if arr.shape[-1] == 3:  # RGB -> grayscale
                    arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    arrs = [arr]
                else:
                    arrs = [arr[i] for i in range(arr.shape[0])]
            else:
                arrs = [arr]
            for a in arrs:
                hu = _convert_to_hu(a, ds)
                z_key = ds.ImagePositionPatient[-1] if hasattr(ds, 'ImagePositionPatient') else (getattr(ds, 'InstanceNumber', 0))
                slices.append((z_key, hu))
        except Exception:
            continue
    if not slices:
        return np.zeros((0, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    slices.sort(key=lambda x: x[0])
    vol = np.stack([s[1] for s in slices], axis=0).astype(np.float32)
    return vol

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:23.89839Z","iopub.execute_input":"2025-08-07T18:49:23.89872Z","iopub.status.idle":"2025-08-07T18:49:26.967686Z","shell.execute_reply.started":"2025-08-07T18:49:23.898697Z","shell.execute_reply":"2025-08-07T18:49:26.966772Z"},"jupyter":{"outputs_hidden":false}}
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones (matches training model behavior)."""
    def __init__(self, model_name, in_chans, img_size, num_classes=13, pretrained=True,
                 drop_rate=0.3, drop_path_rate=0.2, global_pool_override: str | None = None):
        super().__init__()
        
        self.model_name = model_name
        
        # Build kwargs similar to training code
        backbone_kwargs = dict(
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            img_size=img_size,
        )
        if global_pool_override is not None:
            backbone_kwargs["global_pool"] = global_pool_override

        try:
            self.backbone = timm.create_model(model_name, **backbone_kwargs)
        except TypeError:
            backbone_kwargs.pop("img_size", None)
            self.backbone = timm.create_model(model_name, **backbone_kwargs)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_chans, img_size, img_size)
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                # Conv features (batch, channels, height, width)
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                # Transformer features (batch, sequence, features)
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                # Already flat features (batch, features)
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        
        print(f"Model {model_name}: detected {num_features} features, output shape: {features.shape}")
        
        # Add global pooling for models that output spatial features
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Combined classifier with batch norm for stability
        self.loc_classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        self.aneurysm_classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 1)
        )
        
    def forward(self, image):
        # Extract image features
        img_features = self.backbone(image)
        
        # Apply appropriate pooling based on model type
        if hasattr(self, 'needs_pool') and self.needs_pool:
            # Conv features - apply global pooling
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            # Transformer features - average across sequence dimension
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            # Fallback for any 4D output
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            # Fallback for any 3D output
            img_features = img_features.mean(dim=1)
        
        # Classification
        loc_output = self.loc_classifier(img_features)
        cls_logit  = self.aneurysm_classifier(img_features)
        
        return cls_logit, loc_output

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:26.968919Z","iopub.execute_input":"2025-08-07T18:49:26.96922Z","iopub.status.idle":"2025-08-07T18:49:26.973026Z","shell.execute_reply.started":"2025-08-07T18:49:26.969187Z","shell.execute_reply":"2025-08-07T18:49:26.972447Z"},"jupyter":{"outputs_hidden":false}}
# !ls /kaggle/input/rsna-iad-modelzoo/

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:27.93013Z","iopub.execute_input":"2025-08-07T18:49:27.930452Z","iopub.status.idle":"2025-08-07T18:49:28.90629Z","shell.execute_reply.started":"2025-08-07T18:49:27.930429Z","shell.execute_reply":"2025-08-07T18:49:28.905424Z"},"jupyter":{"outputs_hidden":false}}
model=MultiBackboneModel( model_name="resnet18.a1_in1k",
                          in_chans=5,
                          img_size=IMG_SIZE,
                          num_classes=13,
                          drop_path_rate=0.0,
                          drop_rate=0.0,
                          pretrained=False,
                          global_pool_override=None )

def _adapt_first_conv(state_dict, expected_in: int) -> dict:
    """If checkpoint first conv has 3 channels and we need 5, expand weights by mean replication."""
    new_sd = state_dict.copy()
    # Common timm first conv keys
    conv_keys = [k for k in state_dict.keys() if (k.endswith('conv1.weight') or k.endswith('patch_embed.proj.weight')) and state_dict[k].ndim == 4]
    if not conv_keys:
        return state_dict
    for ck in conv_keys:
        w = state_dict[ck]
        if w.shape[1] == expected_in:
            continue
        if w.shape[1] == 3 and expected_in > 3:
            extra = expected_in - 3
            mean_channel = w.mean(dim=1, keepdim=True)  # (out,1,k,k)
            expand = mean_channel.repeat(1, extra, 1, 1)
            new_w = torch.cat([w, expand], dim=1)
            new_sd[ck] = new_w
    return new_sd

# Load checkpoint (update path as needed for 5-channel model). Provide fallback adaptation.
checkpoint_path = "/kaggle/input/rsna-sergio-models/mip_resnet18_a1_in1k_e150_no_flip_fold0-epoch79-val_kaggle_score0.6803fold_id0.ckpt"
checkpoint = torch.load(checkpoint_path, weights_only=False)
if 'state_dict' in checkpoint:
    raw_sd = { (k[6:] if k.startswith('model.') else k): v for k,v in checkpoint['state_dict'].items() }
else:
    raw_sd = checkpoint
raw_sd = _adapt_first_conv(raw_sd, expected_in=5)
missing, unexpected = model.load_state_dict(raw_sd, strict=False)
if missing:
    print(f"Warning: missing keys: {missing[:5]} ... total {len(missing)}")
if unexpected:
    print(f"Warning: unexpected keys: {unexpected[:5]} ... total {len(unexpected)}")
model.cuda()
model.eval()
print("Done")

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:29.042674Z","iopub.execute_input":"2025-08-07T18:49:29.043647Z","iopub.status.idle":"2025-08-07T18:49:29.052343Z","shell.execute_reply.started":"2025-08-07T18:49:29.043605Z","shell.execute_reply":"2025-08-07T18:49:29.051482Z"},"jupyter":{"outputs_hidden":false}}
def create_multiwindow_mip(volume_hu: np.ndarray) -> np.ndarray:
    """Compute axial MIP in HU then build 5-channel tensor (raw_norm + 4 fixed windows) in [0,1]."""
    if volume_hu.shape[0] == 0:
        return np.zeros((5, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    mip = np.max(volume_hu, axis=0).astype(np.float32)  # raw HU
    mip = np.clip(mip, RAW_MIN_HU, RAW_MAX_HU)
    # resize AFTER MIP to mimic training script
    if mip.shape[0] != IMG_SIZE or mip.shape[1] != IMG_SIZE:
        mip = cv2.resize(mip, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    raw_norm = np.clip((mip - RAW_MIN_HU) / (RAW_MAX_HU - RAW_MIN_HU), 0.0, 1.0)
    window_channels = [_fixed_hu_window(mip, c, w) for (c, w) in BASE_WINDOWS]
    img = np.stack([raw_norm] + window_channels, axis=0).astype(np.float32)  # (5,H,W)
    return img

@torch.no_grad()
def eval_one_series(volume_hu: np.ndarray, model):
    """Compute multi-window MIP and run inference."""
    img = create_multiwindow_mip(volume_hu)  # (5,H,W)
    tensor = torch.from_numpy(img).unsqueeze(0).cuda()  # 1,5,H,W
    with torch.cuda.amp.autocast():
        pred_cls, pred_locs = model(tensor)
    pred_cls = torch.sigmoid(pred_cls.squeeze()).item()
    pred_locs = torch.sigmoid(pred_locs.squeeze(0)).detach().cpu().numpy()
    return pred_cls, pred_locs

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:30.794699Z","iopub.execute_input":"2025-08-07T18:49:30.794994Z","iopub.status.idle":"2025-08-07T18:49:30.799976Z","shell.execute_reply.started":"2025-08-07T18:49:30.794973Z","shell.execute_reply":"2025-08-07T18:49:30.799124Z"},"jupyter":{"outputs_hidden":false}}
def _predict_inner(series_path):
    volume = process_dicom_series(series_path)  # HU volume float32
    cls_prob, loc_probs = eval_one_series(volume, model)

    loc_probs = list(loc_probs)

    values = loc_probs + [cls_prob]


    predictions = pl.DataFrame(
            data=[values],
            schema=LABEL_COLS,
            orient='row'
    )
    return predictions

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:54.886633Z","iopub.execute_input":"2025-08-07T18:49:54.887578Z","iopub.status.idle":"2025-08-07T18:49:54.893856Z","shell.execute_reply.started":"2025-08-07T18:49:54.887524Z","shell.execute_reply":"2025-08-07T18:49:54.892979Z"},"jupyter":{"outputs_hidden":false}}


def predict(series_path: str):
    """
    Top-level prediction function passed to the server.
    It calls the core logic and guarantees cleanup in a `finally` block.
    """
    try:
        # Call the internal prediction logic
        return _predict_inner(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        # Return a fallback dataframe with the correct schema
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        # This code is required to prevent "out of disk space" and "directory not empty" errors.
        # It deletes the shared folder and then immediately recreates it, ensuring it's
        # empty and ready for the next prediction.
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # Also perform memory cleanup here
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:57.43293Z","iopub.execute_input":"2025-08-07T18:49:57.433574Z","iopub.status.idle":"2025-08-07T18:49:57.43719Z","shell.execute_reply.started":"2025-08-07T18:49:57.433544Z","shell.execute_reply":"2025-08-07T18:49:57.436441Z"},"jupyter":{"outputs_hidden":false}}
import time

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:58.426728Z","iopub.execute_input":"2025-08-07T18:49:58.427228Z","iopub.status.idle":"2025-08-07T18:50:10.953514Z","shell.execute_reply.started":"2025-08-07T18:49:58.427204Z","shell.execute_reply":"2025-08-07T18:50:10.952804Z"},"jupyter":{"outputs_hidden":false}}
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
    print(submission_df.head())

print(time.time() - st)

# %% [code] {"jupyter":{"outputs_hidden":false}}

