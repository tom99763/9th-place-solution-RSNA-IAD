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
def preprocess_dcm_slice(image, dcm, output_size=(IMG_SIZE, IMG_SIZE), window_level=150, window_width=350):
    """
    Reads and preprocesses a single DICOM slice from a CTA or MRA scan.

    For CTA scans, it applies a specific vascular window to highlight arteries.
    For other modalities like MRA, it performs standard min-max normalization.
    The final image is resized and returned as an 8-bit grayscale numpy array.

    Args:
        dcm_path (str): The full path to the .dcm file.
        output_size (tuple): The target dimensions for the output image (width, height).
        window_level (int): The window level (center) for CTA windowing in HU.
        window_width (int): The window width for CTA windowing in HU.

    Returns:
        numpy.ndarray: The preprocessed 8-bit grayscale image, or None if an error occurs.
    """
    try:

        # Get the pixel data from the DICOM file
        image = image.astype(np.float64)

        # 2. Check if the modality is 'CT' to decide on the processing method
        # The DICOM tag (0008,0060) specifies the modality
        is_ct_scan = 'CT' in dcm.get('Modality', '').upper()

        if is_ct_scan:
            # For CT scans, convert pixel data to Hounsfield Units (HU)
            # using the Rescale Slope and Intercept values from DICOM metadata
            if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
                image = image * dcm.RescaleSlope + dcm.RescaleIntercept

            # Apply the vascular windowing
            lower_bound = window_level - (window_width / 2)
            upper_bound = window_level + (window_width / 2)
            
            # Clip the image to the window range
            image = np.clip(image, lower_bound, upper_bound)
            
            # Normalize the windowed image to a 0-255 scale
            image = ((image - lower_bound) / window_width) * 255.0

        else:
            # 3. For non-CT scans (like MRA), perform standard min-max normalization
            if np.max(image) != np.min(image):
                image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0

        # Convert the final processed image to an 8-bit unsigned integer format
        image = image.astype(np.uint8)

        # 4. Resize the image to the desired output size
        processed_image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)

        return processed_image

    except Exception as e:
        print(f"Error processing DICOM slice: {e}")
        return None

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:22.542526Z","iopub.execute_input":"2025-08-07T18:49:22.542834Z","iopub.status.idle":"2025-08-07T18:49:22.552064Z","shell.execute_reply.started":"2025-08-07T18:49:22.542811Z","shell.execute_reply":"2025-08-07T18:49:22.551244Z"},"jupyter":{"outputs_hidden":false}}
def process_dicom_series(series_path):
    """Process a DICOM series and extract metadata"""
    series_path = Path(series_path)
    
    # Find all DICOM files
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    if len(all_filepaths) == 0:
        print(f"No DCM files found in {series_path}")
        return np.array([])
        
    # Process DICOM files
    slices = []
    instance_numbers = []
    
    for _, filepath in enumerate(all_filepaths):
        ds = pydicom.dcmread(filepath, force=True)
        
        # print(ds.InstanceNumber)
        img = ds.pixel_array.astype(np.float32)
        if img.ndim == 3:
            if img.shape[-1] == 3:
                imgs = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)]
            else:
                imgs = []
                for i in range(img.shape[0]):
                    imgs.append(img[i, :, :])

        else:
            imgs = [img]
            
        for img in imgs:
            if hasattr(ds, "InstanceNumber"):
                instance_numbers.append(ds.InstanceNumber)
            
            if hasattr(ds, "ImagePositionPatient"):
                slices.append((ds.ImagePositionPatient[-1],preprocess_dcm_slice(img,ds)))
            elif hasattr(ds, "InstanceNumber"):
                slices.append((int(ds.InstanceNumber),preprocess_dcm_slice(img,ds)))
            else:
                slices.append((0,preprocess_dcm_slice(img,ds)))


    # sometimes it's the case that the starting instance number is greater than 1. So we want to get the start_instance_number and then substract it from the z axis.
    instance_numbers = sorted(instance_numbers)
    start_instance_number = instance_numbers[0] - 1


    # we sort all the slices by ImagePositionPatient or InstanceNumber
    slices = sorted(slices, key = lambda x: x[0])
    
    volume = np.array([slice[-1] for slice in slices])

    # # We get nth slice of the volume
    selected_idxs = [*range(0,volume.shape[0],FACTOR)]
    return volume[selected_idxs]

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
"""
Instantiate two separate models:
- model_cls: EfficientNetV2 for aneurysm-present classification (Kaggle checkpoint)
- model_loc: DeiT v2 for location probabilities (local checkpoint)
"""

# Aneurysm-present classifier (EfficientNetV2)
model_cls = MultiBackboneModel(
    model_name="tf_efficientnetv2_b0.in1k",
    in_chans=3,
    img_size=IMG_SIZE,
    num_classes=13,
    drop_path_rate=0.0,
    drop_rate=0.0,
    pretrained=False,
    global_pool_override=None,
)

checkpoint_cls = torch.load(
    "/kaggle/input/rsna-sergio-models/mip_efficientnet_v2_baseline-epoch49-val_kaggle_score0.7158fold_id0.ckpt",
    weights_only=False,
)

if 'state_dict' in checkpoint_cls:
    state_dict_cls = {}
    for key, value in checkpoint_cls['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[6:]
            state_dict_cls[new_key] = value
        else:
            state_dict_cls[key] = value
    model_cls.load_state_dict(state_dict_cls)
else:
    model_cls.load_state_dict(checkpoint_cls)
model_cls.cuda()
model_cls.eval()

# Location classifier (DeiT v2)
model_loc = MultiBackboneModel(
    model_name="deit_small_patch16_224.fb_in1k",
    in_chans=3,
    img_size=IMG_SIZE,
    num_classes=13,
    drop_path_rate=0.0,
    drop_rate=0.0,
    pretrained=False,
    global_pool_override=None,
)

checkpoint_loc = torch.load(
    "/home/sersasj/RSNA-IAD-Codebase/models/mip_deit_v2_baseline-epoch=39-val_kaggle_score=0.6665fold_id=0.ckpt",
    weights_only=False,
)

if 'state_dict' in checkpoint_loc:
    state_dict_loc = {}
    for key, value in checkpoint_loc['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[6:]
            state_dict_loc[new_key] = value
        else:
            state_dict_loc[key] = value
    model_loc.load_state_dict(state_dict_loc)
else:
    model_loc.load_state_dict(checkpoint_loc)
model_loc.cuda()
model_loc.eval()
print("Models loaded: model_cls (EfficientNetV2) and model_loc (DeiT v2)")

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:29.042674Z","iopub.execute_input":"2025-08-07T18:49:29.043647Z","iopub.status.idle":"2025-08-07T18:49:29.052343Z","shell.execute_reply.started":"2025-08-07T18:49:29.043605Z","shell.execute_reply":"2025-08-07T18:49:29.051482Z"},"jupyter":{"outputs_hidden":false}}
def create_rgb_slices(volume):
    """Create a 3-channel MIP image scaled to [0,1] to match training MIP pipeline."""
    mip = np.max(volume, axis=0).astype(np.float32)  # H, W in [0,255]
    image = np.stack([mip, mip, mip], axis=0)  # 3, H, W
    image = image #/ 255.0
    return image.astype(np.float32)

@torch.no_grad()
def eval_one_series(volume, model_cls, model_loc):
    """Evaluate a single series: use EfficientNetV2 for cls and DeiT v2 for loc."""
    img = create_rgb_slices(volume)
    tensor = torch.from_numpy(img).unsqueeze(0).cuda()

    with torch.cuda.amp.autocast():
        cls_logit, _ = model_cls(tensor)
        _, loc_logits = model_loc(tensor)

    pred_cls = torch.sigmoid(cls_logit.squeeze()).item()
    pred_locs = torch.sigmoid(loc_logits.squeeze(0)).detach().cpu().numpy()
    return pred_cls, pred_locs

# %% [code] {"execution":{"iopub.status.busy":"2025-08-07T18:49:30.794699Z","iopub.execute_input":"2025-08-07T18:49:30.794994Z","iopub.status.idle":"2025-08-07T18:49:30.799976Z","shell.execute_reply.started":"2025-08-07T18:49:30.794973Z","shell.execute_reply":"2025-08-07T18:49:30.799124Z"},"jupyter":{"outputs_hidden":false}}
def _predict_inner(series_path):
    volume = process_dicom_series(series_path)
    cls_prob, loc_probs = eval_one_series(volume.astype(np.float32), model_cls, model_loc)

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
    try:
        from IPython.display import display as ipy_display  # type: ignore
        ipy_display(submission_df)
    except Exception:
        print(submission_df)

print(time.time() - st)

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
