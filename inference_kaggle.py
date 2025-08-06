# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.659208Z","iopub.execute_input":"2025-08-05T03:01:06.659530Z","iopub.status.idle":"2025-08-05T03:01:06.665109Z","shell.execute_reply.started":"2025-08-05T03:01:06.659509Z","shell.execute_reply":"2025-08-05T03:01:06.664045Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.666883Z","iopub.execute_input":"2025-08-05T03:01:06.667218Z","iopub.status.idle":"2025-08-05T03:01:06.680978Z","shell.execute_reply.started":"2025-08-05T03:01:06.667199Z","shell.execute_reply":"2025-08-05T03:01:06.680326Z"},"jupyter":{"outputs_hidden":false}}
import kaggle_evaluation.rsna_inference_server
import shutil
import gc

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.681954Z","iopub.execute_input":"2025-08-05T03:01:06.682244Z","iopub.status.idle":"2025-08-05T03:01:06.695529Z","shell.execute_reply.started":"2025-08-05T03:01:06.682226Z","shell.execute_reply":"2025-08-05T03:01:06.694539Z"},"jupyter":{"outputs_hidden":false}}
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.697860Z","iopub.execute_input":"2025-08-05T03:01:06.698235Z","iopub.status.idle":"2025-08-05T03:01:06.708709Z","shell.execute_reply.started":"2025-08-05T03:01:06.698207Z","shell.execute_reply":"2025-08-05T03:01:06.708061Z"},"jupyter":{"outputs_hidden":false}}
IMG_SIZE = 640
FACTOR = 3


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

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.709566Z","iopub.execute_input":"2025-08-05T03:01:06.709873Z","iopub.status.idle":"2025-08-05T03:01:06.721952Z","shell.execute_reply.started":"2025-08-05T03:01:06.709835Z","shell.execute_reply":"2025-08-05T03:01:06.721163Z"},"jupyter":{"outputs_hidden":false}}
def process_slice(img,ds):
    modality = getattr(ds, 'Modality', 'CT')
    
    # Apply rescale if available
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    
    img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-7)
    img = (img_normalized * 255).astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.722859Z","iopub.execute_input":"2025-08-05T03:01:06.723087Z","iopub.status.idle":"2025-08-05T03:01:06.739428Z","shell.execute_reply.started":"2025-08-05T03:01:06.723069Z","shell.execute_reply":"2025-08-05T03:01:06.738754Z"},"jupyter":{"outputs_hidden":false}}
def process_dicom_series(series_path):
    """Process a DICOM series and extract metadata"""
    series_path = Path(series_path)
    
    all_filepaths = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(series_path)
        for file in files if file.endswith('.dcm')
    ])
    
    if not all_filepaths:
        print(f"No DCM files found for {series_path}")
        return np.array([])
        
    temp_slices = []
    instance_numbers = []
    
    for filepath in all_filepaths:
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)

            if img.ndim == 3 and img.shape[-1] == 3:
                imgs = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)]
            elif img.ndim == 3:
                imgs = list(img)
            else:
                imgs = [img]

            for img_idx, img in enumerate(imgs):
                if hasattr(ds, "InstanceNumber"):
                    instance_numbers.append(ds.InstanceNumber)
                
                z_val = getattr(ds, "ImagePositionPatient", [0])[-1] if hasattr(ds, "ImagePositionPatient") else int(getattr(ds, "InstanceNumber", 0))
                processed_img = process_slice(img, ds)
                temp_slices.append((z_val, processed_img, ds.InstanceNumber if hasattr(ds, "InstanceNumber") else 0))
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    if not temp_slices:
        return np.array([])

    temp_slices = sorted(temp_slices, key=lambda x: x[0])
    
    total_slices = len(temp_slices)
    selected_idxs = list(range(0, total_slices, FACTOR))
    
    # Extract selected slices
    volume = []
    for idx in selected_idxs:
        if idx < len(temp_slices):
            _, processed_img, _ = temp_slices[idx]
            volume.append(processed_img)
    
    return np.array(volume)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.740205Z","iopub.execute_input":"2025-08-05T03:01:06.740408Z","iopub.status.idle":"2025-08-05T03:01:06.757378Z","shell.execute_reply.started":"2025-08-05T03:01:06.740392Z","shell.execute_reply":"2025-08-05T03:01:06.756675Z"},"jupyter":{"outputs_hidden":false}}
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones"""

    def __init__(self, model_name, in_chans, img_size, num_classes=13, pretrained=True,
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()

        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=0,  # Remove classifier head
            global_pool=''  # Remove global pooling
        )

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
        cls_logit = self.aneurysm_classifier(img_features)

        return cls_logit, loc_output


# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:06.758332Z","iopub.execute_input":"2025-08-05T03:01:06.758612Z","iopub.status.idle":"2025-08-05T03:01:07.477599Z","shell.execute_reply.started":"2025-08-05T03:01:06.758583Z","shell.execute_reply":"2025-08-05T03:01:07.476830Z"},"jupyter":{"outputs_hidden":false}}
model=MultiBackboneModel( model_name="tf_efficientnetv2_b0.in1k"
                                            , in_chans=3
                                            , img_size=640  
                                            , num_classes=13
                                            , drop_path_rate=0.0
                                            , drop_rate=0.0
                                            , pretrained=False
                                            )

# Load PyTorch Lightning checkpoint and extract model weights
checkpoint = torch.load("/kaggle/input/rsna-sergio-models/slice_based_efficientnet_v2_baseline_2d-epoch119-val_kaggle_score0.7476fold_id0.ckpt", weights_only=False)

# Extract the actual model state_dict from Lightning checkpoint
if 'state_dict' in checkpoint:
    # Remove 'model.' prefix from Lightning checkpoint keys
    model_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            # Remove 'model.' prefix
            new_key = key[6:]  # Remove 'model.' (6 characters)
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    
    model.load_state_dict(model_state_dict)
else:
    # Fallback: try loading directly if it's already a pure state dict
    model.load_state_dict(checkpoint)
model.cuda()
model.eval()
print("Done")

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:07.478397Z","iopub.execute_input":"2025-08-05T03:01:07.478631Z","iopub.status.idle":"2025-08-05T03:01:07.484854Z","shell.execute_reply.started":"2025-08-05T03:01:07.478612Z","shell.execute_reply":"2025-08-05T03:01:07.483917Z"},"jupyter":{"outputs_hidden":false}}
@torch.no_grad()
def eval_one_series(volume, model):
    volume = torch.from_numpy(volume).cuda()
    volume = torch.stack([volume, volume, volume], dim=1)

    pred_cls = []
    pred_locs = []

    # Process slices in batches for efficiency
    with amp.autocast():  # Enable mixed-precision inference
        batch_size = 64  # Process up to 64 slices at once
        for batch_idx in range(0, volume.shape[0], batch_size):
            batch_slices = volume[batch_idx:batch_idx + batch_size]
            pc, pl = model(batch_slices)
            pred_cls.append(pc)
            pred_locs.append(pl)

    pred_cls = torch.vstack(pred_cls)
    pred_locs = torch.vstack(pred_locs)

    pred_cls = pred_cls.squeeze()

    return pred_cls.max().sigmoid().item(), pred_locs.max(dim=0).values.sigmoid().cpu().numpy()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:07.487306Z","iopub.execute_input":"2025-08-05T03:01:07.487530Z","iopub.status.idle":"2025-08-05T03:01:07.499831Z","shell.execute_reply.started":"2025-08-05T03:01:07.487514Z","shell.execute_reply":"2025-08-05T03:01:07.498914Z"},"jupyter":{"outputs_hidden":false}}
def _predict_inner(series_path):
    volume = process_dicom_series(series_path)
    cls_prob, loc_probs = eval_one_series(volume.astype(np.float32),model)

    loc_probs = list(loc_probs)

    values = loc_probs + [cls_prob]


    predictions = pl.DataFrame(
            data=[values],
            schema=LABEL_COLS,
            orient='row'
    )
    return predictions

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:07.500710Z","iopub.execute_input":"2025-08-05T03:01:07.500984Z","iopub.status.idle":"2025-08-05T03:01:07.512217Z","shell.execute_reply.started":"2025-08-05T03:01:07.500956Z","shell.execute_reply":"2025-08-05T03:01:07.511514Z"},"jupyter":{"outputs_hidden":false}}


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

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:07.513084Z","iopub.execute_input":"2025-08-05T03:01:07.513372Z","iopub.status.idle":"2025-08-05T03:01:07.525499Z","shell.execute_reply.started":"2025-08-05T03:01:07.513348Z","shell.execute_reply":"2025-08-05T03:01:07.524865Z"},"jupyter":{"outputs_hidden":false}}
import time

# %% [code] {"execution":{"iopub.status.busy":"2025-08-05T03:01:07.526320Z","iopub.execute_input":"2025-08-05T03:01:07.526564Z","iopub.status.idle":"2025-08-05T03:01:40.930844Z","shell.execute_reply.started":"2025-08-05T03:01:07.526546Z","shell.execute_reply":"2025-08-05T03:01:40.929939Z"},"jupyter":{"outputs_hidden":false}}
st = time.time()
# Initialize the inference server with our main `predict` function.
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Check if the notebook is running in the competition environment or a local session.
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    
    submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
    display(submission_df)

print(time.time() - st)
