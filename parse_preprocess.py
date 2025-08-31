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
import shutil
import gc
import time
from torch_geometric.nn import LayerNorm
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.data import Data
from torch_cluster import knn_graph
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import AddRandomWalkPE
from scipy.spatial import Delaunay

class MODEL_CONIFG_1:
    num_layers = 8
    hidden_channels = 256
    pos_weight = 8
    jk = 'lstm'
    dropout = 0.3
    conf = 0.04
    graph_augment = False
    use_pe = True
    walk_length = 8

class MODEL_CONIFG_2:
    num_layers = 8
    hidden_channels = 256
    pos_weight = 8
    jk = 'lstm'
    dropout = 0.3
    conf = 0.05
    graph_augment = False
    use_pe = True
    walk_length = 8

# Model configurations - Add your models here
MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold02/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    # {
    #     "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold12/weights/best.pt",
    #     "fold": "1",
    #     "weight": 1.0,
    #     "name": "YOLOv11n_fold1"
    # },
    # {
    #     "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold22/weights/best.pt",
    #     "fold": "2",
    #     "weight": 1.0,
    #     "name": "YOLOv11n_fold2"
    # }
    # {
    #    "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold32/weights/best.pt",
    #    "fold": "3",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold3"
    # },
    # {
    #    "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold42/weights/best.pt",
    #    "fold": "4",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold4"
    # }
]

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

IDX_TO_LABEL = {v: k for k, v in LABELS_TO_IDX.items()}

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


def get_feature_map(model):
    features = {}
    def make_hook(name):
        def hook(module, input, output):
            features[name] = output
        return hook
    model.model.model[16].register_forward_hook(make_hook("C3K2"))
    return features


def delaunay_graph(x):
    points = x.cpu().numpy()  # assuming x is a tensor of shape [num_points, dims]
    tri = Delaunay(points)
    edges = set()

    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.add(tuple(sorted((simplex[i], simplex[j]))))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t()
    return edge_index


def assign_feat(x, feat, tomo_id, vol_size):
    d, h, w = vol_size
    fd, fh, fw = feat.shape[0], feat.shape[2], feat.shape[3]
    z = x[:, 0].astype('int32')
    y = ((x[:, 1]/h) * fh).astype('int32')
    x = ((x[:, 2]/w) * fw).astype('int32')
    extract_feat = feat[z, :, y, x]
    return extract_feat


def sample_uniform_3d_ball(points, vol_size, radius=30, num_samples=10):
    Z, Y, X = vol_size
    N = points.shape[0]

    def uniform_ball(n):
        vec = np.random.randn(n, 3)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        r = np.random.rand(n) ** (1/3)
        return vec * (r[:, None] * radius)

    result = []
    np.random.seed(42)
    for center in points:
        attempts = 0
        accepted = []

        # Accept valid samples until we have enough or hit retry limit
        while len(accepted) < num_samples and attempts < num_samples * 10:
            samples = uniform_ball(num_samples)
            candidates = samples + center  # shifted samples

            # Keep only those within volume bounds
            mask = (
                (candidates[:, 0] >= 0) & (candidates[:, 0] < Z) &
                (candidates[:, 1] >= 0) & (candidates[:, 1] < Y) &
                (candidates[:, 2] >= 0) & (candidates[:, 2] < X)
            )
            accepted.extend(candidates[mask])
            attempts += 1

        # If not enough valid, pad with center point
        if len(accepted) < num_samples:
            accepted.extend([center] * (num_samples - len(accepted)))

        result.append(np.array(accepted[:num_samples]))

    result = np.concatenate(result, axis=0)
    return result


def assign_feat(x, feat, vol_size):
    d, h, w = vol_size
    fd, fh, fw = feat.shape[0], feat.shape[2], feat.shape[3]
    z = x[:, 0].astype('int32')
    y = ((x[:, 1]/h) * fh).astype('int32')
    x = ((x[:, 2]/w) * fw).astype('int32')
    extract_feat = feat[z, :, y, x]
    return extract_feat


def delaunay_graph(x):
    points = x.cpu().numpy()  # assuming x is a tensor of shape [num_points, dims]
    tri = Delaunay(points)
    edges = set()

    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.add(tuple(sorted((simplex[i], simplex[j]))))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t()
    return edge_index

transform_1 = AddRandomWalkPE(walk_length=MODEL_CONIFG_1.walk_length, attr_name=None)
transform_2 = AddRandomWalkPE(walk_length=MODEL_CONIFG_2.walk_length, attr_name=None)

def extract_tomo(all_locations, all_features, vol_size, k_neibs = 15, radius = 30,  num_samples = 10):
    points = sample_uniform_3d_ball(all_locations, vol_size, radius, num_samples)
    extract_feat = assign_feat(points, all_features, vol_size)
    #convert to torch
    points = torch.from_numpy(points)
    extract_feat = torch.from_numpy(extract_feat)
    batch = torch.zeros(points.shape[0], dtype=torch.int64)

    #multi-graph
    edge_index_1 = knn_graph(points, k=k_neibs, loop=False)
    edge_index_2 = delaunay_graph(points)
    data1= Data(points = points, x = extract_feat, edge_index = edge_index_1, batch = batch)
    data2= Data(points = points, x = extract_feat, edge_index = edge_index_2, batch = batch)

    if MODEL_CONIFG_1.use_pe:
        data1 = transform_1(data1)

    if MODEL_CONIFG_2.use_pe:
        data2 = transform_2(data2)
    return data1, data2