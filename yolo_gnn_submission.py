# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-29T01:15:04.307339Z","iopub.execute_input":"2025-08-29T01:15:04.307594Z","iopub.status.idle":"2025-08-29T01:15:04.317292Z","shell.execute_reply.started":"2025-08-29T01:15:04.307567Z","shell.execute_reply":"2025-08-29T01:15:04.316575Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-29T01:15:04.317964Z","iopub.execute_input":"2025-08-29T01:15:04.318526Z","iopub.status.idle":"2025-08-29T01:15:05.307265Z","shell.execute_reply.started":"2025-08-29T01:15:04.318504Z","shell.execute_reply":"2025-08-29T01:15:05.306543Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-29T01:15:05.308209Z","iopub.execute_input":"2025-08-29T01:15:05.308464Z","iopub.status.idle":"2025-08-29T01:15:05.345326Z","shell.execute_reply.started":"2025-08-29T01:15:05.308447Z","shell.execute_reply":"2025-08-29T01:15:05.344589Z"}}
#GNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


class SimpleGNNBinary(nn.Module):
    """
    Simple GNN for binary aneurysm detection (present/absent).

    Node features:
        [confidence, x_center, y_center, z_norm, z_mm_norm, class_one_hot]
    (When z is unavailable, set z_norm and z_mm_norm to 0.)
    Edge features: spatial/physical distance-based weights
    """

    def __init__(self, num_classes: int = 13, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Node feature dimension: conf(1) + coords(4: x,y,z,z_mm) + class_onehot(13) = 18
        self.node_feat_dim = 1 + 4 + num_classes

        # Node embedding
        self.node_embed = nn.Linear(self.node_feat_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Binary classification head: aneurysm present/absent
        self.binary_head = nn.Linear(hidden_dim, 1)  # Series-level binary classification
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: [num_nodes, node_feat_dim] 
            edge_index: [2, num_edges] adjacency
            edge_weights: [num_edges] optional edge weights
            
        Returns:
            dict with binary_logit
        """
        # Embed nodes
        x = F.relu(self.node_embed(node_features))
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_weights)

        # Global pooling to get series-level representation
        if x.size(0) > 0:
            # Max pooling across all nodes to get series representation
            series_repr = torch.max(x, dim=0)[0]  # [hidden_dim]
        else:
            # Handle empty graphs
            series_repr = torch.zeros(self.hidden_dim, device=x.device)
        
        # Series-level binary prediction
        binary_logit = self.binary_head(series_repr)  # [1]
        
        return {
            'binary_logit': binary_logit.squeeze(-1)  # scalar
        }


class GNNLayer(nn.Module):
    """Single GNN layer with attention-weighted message passing."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for confidence-aware message passing
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention-weighted message passing layer."""
        row, col = edge_index
        
        # Compute messages
        messages = self.message_mlp(torch.cat([x[row], x[col]], dim=-1))
        
        # Compute attention weights based on node features
        attention_input = torch.cat([x[row], x[col]], dim=-1)
        attention_weights = torch.sigmoid(self.attention(attention_input)).squeeze(-1)
        
        # Combine edge weights with attention weights
        if edge_weights is not None:
            final_weights = edge_weights * attention_weights
        else:
            final_weights = attention_weights
        
        # Apply combined weights to messages
        messages = messages * final_weights.unsqueeze(-1)
        
        # Aggregate messages for each node with normalization
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        norm_weights = torch.zeros(num_nodes, device=x.device)
        
        aggregated.index_add_(0, col, messages)
        norm_weights.index_add_(0, col, final_weights)
        
        # Normalize by sum of weights (avoid division by zero)
        norm_weights = torch.clamp(norm_weights, min=1e-6)
        aggregated = aggregated / norm_weights.unsqueeze(-1)
        
        # Update node features
        x_new = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return F.relu(x_new + x)  # Residual connection


def knn_edges_weighted(centers: np.ndarray, k: int, radius: Optional[float], sigma: float = 0.1) -> List[Tuple[int, int, float]]:
    """Build kNN edges with Gaussian distance-based weights."""
    n = centers.shape[0]
    if n <= 1:
        return []
    
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        order = np.argsort(d)
        cnt = 0
        for j in order[1:]:  # skip self
            if radius is not None and d[j] > radius:
                break
            
            # Gaussian kernel weight
            weight = float(np.exp(-d[j] / sigma))
            edge_key = (min(i, j), max(i, j))
            
            # Keep maximum weight if edge already exists
            if edge_key in edges:
                edges[edge_key] = max(edges[edge_key], weight)
            else:
                edges[edge_key] = weight
            
            cnt += 1
            if cnt >= k:
                break
    
    return [(u, v, w) for (u, v), w in sorted(edges.items())]


class YOLOGNNBinaryProcessor:
    """Processes YOLO detections into graphs and applies binary GNN."""
    
    def __init__(self, gnn_model: SimpleGNNBinary, distance_threshold: float = 0.25, 
                 knn_intra: int = 4, knn_inter: int = 4, sigma: float = 0.1):
        self.gnn_model = gnn_model
        self.distance_threshold = distance_threshold
        self.knn_intra = knn_intra
        self.knn_inter = knn_inter
        self.sigma = sigma
        
    def yolo_to_graph_with_slices(self, all_detections: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert YOLO detections from multiple slices to graph format with proper z-info and kNN edges.
        
        Args:
            all_detections: List of detection dicts with keys:
                - 'bbox': [x_center, y_center, width, height] (normalized)
                - 'conf': confidence score
                - 'class': class index
                - 'slice_idx': slice index
                - 'z_norm': normalized z position [0,1]
                - 'z_mm': actual z position in mm (optional)
                
        Returns:
            node_features, edge_index, edge_weights
        """
        # Get device from the GNN model
        device = next(self.gnn_model.parameters()).device
        
        if not all_detections:
            # Empty graph
            return (torch.empty(0, self.gnn_model.node_feat_dim, device=device), 
                    torch.empty(2, 0, dtype=torch.long, device=device), 
                    None)
            
        num_nodes = len(all_detections)
        node_features = []
        
        # Build node features
        for det in all_detections:
            bbox = det['bbox']  # [x_center, y_center, width, height]
            conf = det['conf']
            cls_idx = int(det['class'])
            z_norm = det.get('z_norm', 0.0)
            z_mm = det.get('z_mm', 0.0)
            
            # One-hot encode class
            class_onehot = torch.zeros(self.gnn_model.num_classes, device=device)
            if 0 <= cls_idx < self.gnn_model.num_classes:
                class_onehot[cls_idx] = 1.0
            
            # Concatenate features: [conf, x, y, z_norm, z_mm, class_onehot...]
            x, y, w, h = bbox
            feat = torch.cat([
                torch.tensor([conf], dtype=torch.float32, device=device),
                torch.tensor([x, y, z_norm, z_mm], dtype=torch.float32, device=device),
                class_onehot.to(torch.float32)
            ])
            node_features.append(feat)
            
        node_features = torch.stack(node_features)
        
        # Normalize z_mm column to match training (divide by max value across nodes)
        if node_features.shape[0] > 0:
            z_mm_col = 4  # [conf, x, y, z_norm, z_mm, ...]
            z_mm_vals = node_features[:, z_mm_col]
            z_max = torch.max(z_mm_vals)
            if z_max > 0:
                node_features[:, z_mm_col] = z_mm_vals / z_max
        
        # Build edges using kNN approach similar to extract_yolo_graphs.py
        edges: Dict[Tuple[int, int], float] = {}
        radius = self.distance_threshold if self.distance_threshold > 0 else None
        
        # Group detections by slice
        slice_to_node_idxs: Dict[int, List[int]] = {}
        for i, det in enumerate(all_detections):
            slice_idx = det.get('slice_idx', 0)
            if slice_idx not in slice_to_node_idxs:
                slice_to_node_idxs[slice_idx] = []
            slice_to_node_idxs[slice_idx].append(i)
        
        # Intra-slice edges with kNN and Gaussian weights
        for slice_idx, node_idxs in slice_to_node_idxs.items():
            if len(node_idxs) <= 1:
                continue
            centers = np.array([[all_detections[i]['bbox'][0], all_detections[i]['bbox'][1]] for i in node_idxs], dtype=np.float32)
            e_local = knn_edges_weighted(centers, k=max(0, self.knn_intra), radius=radius, sigma=self.sigma)
            for u, v, w in e_local:
                a, b = node_idxs[u], node_idxs[v]
                if a != b:
                    edge_key = (min(a, b), max(a, b))
                    if edge_key in edges:
                        edges[edge_key] = max(edges[edge_key], w)
                    else:
                        edges[edge_key] = w
        
        # Inter-slice edges (multiple adjacent slices with distance-based weights)
        slice_indices = sorted(slice_to_node_idxs.keys())
        for i, slice_idx in enumerate(slice_indices):
            idxs_src = slice_to_node_idxs[slice_idx]
            if not idxs_src:
                continue
            
            # Consider adjacent slices
            for offset in [-2, -1, 1, 2]:
                target_slice_idx = slice_idx + offset
                if target_slice_idx not in slice_to_node_idxs:
                    continue
                
                idxs_dst = slice_to_node_idxs[target_slice_idx]
                if not idxs_dst:
                    continue
                
                # Weight by slice distance
                slice_weight = 1.0 / (1.0 + abs(offset))
                
                # Use 3D coordinates (x, y, z) for distance calculation
                src_cent = np.array([
                    [all_detections[i]['bbox'][0], all_detections[i]['bbox'][1], all_detections[i].get('z_norm', 0.0)] 
                    for i in idxs_src
                ], dtype=np.float32)
                dst_cent = np.array([
                    [all_detections[i]['bbox'][0], all_detections[i]['bbox'][1], all_detections[i].get('z_norm', 0.0)] 
                    for i in idxs_dst
                ], dtype=np.float32)
                
                # Pairwise distances
                dists = np.linalg.norm(src_cent[:, None, :] - dst_cent[None, :, :], axis=2)
                
                for ii in range(dists.shape[0]):
                    order = np.argsort(dists[ii])
                    added = 0
                    for jj in order:
                        if radius is not None and dists[ii, jj] > radius:
                            break
                        
                        # Combined weight: Gaussian distance * slice distance weight
                        dist_weight = float(np.exp(-dists[ii, jj] / self.sigma))
                        combined_weight = dist_weight * slice_weight
                        
                        a, b = idxs_src[ii], idxs_dst[jj]
                        if a != b:
                            edge_key = (min(a, b), max(a, b))
                            if edge_key in edges:
                                edges[edge_key] = max(edges[edge_key], combined_weight)
                            else:
                                edges[edge_key] = combined_weight
                            added += 1
                        if added >= self.knn_inter:
                            break
        
        # Convert edges dict to tensors
        if edges:
            edge_list = list(edges.keys())
            edge_weights_list = list(edges.values())
            
            # Create bidirectional edges
            edge_list_bidir = []
            edge_weights_bidir = []
            for (u, v), w in zip(edge_list, edge_weights_list):
                edge_list_bidir.extend([[u, v], [v, u]])
                edge_weights_bidir.extend([w, w])
            
            edge_index = torch.tensor(edge_list_bidir, device=device).t().contiguous()
            edge_weights = torch.tensor(edge_weights_bidir, device=device)
        else:
            # No edges - create self-loops
            edge_index = torch.stack([torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)])
            edge_weights = torch.ones(num_nodes, device=device)
        return node_features, edge_index, edge_weights
    
    def process_series(self, yolo_results: List, slice_thickness: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Process YOLO results for an entire series through binary GNN.
        
        Args:
            yolo_results: List of YOLO result objects (from model.predict())
            slice_thickness: Slice thickness in mm (if available)
            
        Returns:
            Dictionary with binary prediction
        """
        all_detections = []
        num_slices = len(yolo_results)
        
        # Use provided slice thickness or fallback to 1.0mm
        thickness_norm = slice_thickness if slice_thickness is not None else 1.0
        
        # Extract detections from all slices with proper z information
        for slice_idx, result in enumerate(yolo_results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Calculate normalized z position
                    z_norm = float(slice_idx / max(1, num_slices - 1)) if num_slices > 1 else 0.0
                    # Calculate actual z position in mm
                    z_mm = float(slice_idx * thickness_norm)
                    
                    det = {
                        'bbox': boxes.xywhn[i].cpu().numpy().tolist(),  # Normalized xywh
                        'conf': float(boxes.conf[i].cpu()),
                        'class': int(boxes.cls[i].cpu()),
                        'slice_idx': slice_idx,
                        'z_norm': z_norm,
                        'z_mm': z_mm
                    }
                    all_detections.append(det)
        
        if not all_detections:
            # No detections - return zero prediction
            device = next(self.gnn_model.parameters()).device
            return {
                'aneurysm_prob': torch.tensor(0.0, device=device),
                'num_detections': 0
            }
        
        # Convert to graph and process with proper z-info and kNN edges
        node_features, edge_index, edge_weights = self.yolo_to_graph_with_slices(all_detections)
        
        with torch.no_grad():
            gnn_output = self.gnn_model(node_features, edge_index, edge_weights)
            
            # Get binary probability
            aneurysm_prob = torch.sigmoid(gnn_output['binary_logit'])  # scalar
            
        return {
            'aneurysm_prob': aneurysm_prob,
            'num_detections': len(all_detections)
        }


def create_binary_gnn_model(num_classes: int = 13, hidden_dim: int = 64) -> SimpleGNNBinary:
    """Factory function to create binary GNN model."""
    return SimpleGNNBinary(num_classes=num_classes, hidden_dim=hidden_dim)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-29T01:15:05.346164Z","iopub.execute_input":"2025-08-29T01:15:05.346431Z","iopub.status.idle":"2025-08-29T01:15:05.455616Z","shell.execute_reply.started":"2025-08-29T01:15:05.346414Z","shell.execute_reply":"2025-08-29T01:15:05.455011Z"}}
def read_dicom_frames_hu(path: Path) -> Tuple[List[np.ndarray], Optional[float]]:
    """Read DICOM file and return raw frames plus SliceThickness if available."""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    
    # Get SliceThickness if available
    slice_thickness = None
    if hasattr(ds, "SliceThickness"):
        try:
            slice_thickness = float(ds.SliceThickness)
        except:
            pass
    
    frames: List[np.ndarray] = []
    
    if pix.ndim == 2:
        frames.append(pix.astype(np.float32))
    elif pix.ndim == 3:
        # RGB or multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append(gray)
        else:
            for i in range(pix.shape[0]):
                frames.append(pix[i].astype(np.float32))
    return frames, slice_thickness

def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)

def process_dicom_file(dcm_path: Path) -> Tuple[List[np.ndarray], Optional[float]]:
    """Process single DICOM file - for parallel processing"""
    try:
        frames, slice_thickness = read_dicom_frames_hu(dcm_path)
        processed_slices = []
        for f in frames:
            img_u8 = min_max_normalize(f)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            processed_slices.append(img_u8)
        return processed_slices, slice_thickness
    except Exception as e:
        print(f"Failed processing {dcm_path.name}: {e}")
        return [], None

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

# Model configurations - Add your models here
MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_yolo_11n_mix_up_no_flip/cv_y11n_with_mix_up_mosaic_no_flip_fold02/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    }
    
]

# GNN Model configurations - Add your trained GNN models here
GNN_MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/gnn_simple_v2.pth",
        "fold": "0",
        "weight": 1.0,
        "name": "GNN_Binary_fold0"
    }
]

def load_models():
    """Load all YOLO models on single GPU (cuda:0)"""
    models = []
    for config in MODEL_CONFIGS:
        print(f"Loading YOLO model: {config['name']} on cuda:0")
        
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

def load_gnn_models():
    """Load all GNN models"""
    gnn_models = []
    for config in GNN_MODEL_CONFIGS:
        try:
            print(f"Loading GNN model: {config['name']} on cuda:0")
            
            # Create GNN model
            gnn_model = create_binary_gnn_model(num_classes=13, hidden_dim=64)
            
            # Load weights if file exists
            if os.path.exists(config["path"]):
                checkpoint = torch.load(config["path"], map_location="cuda:0", weights_only=False)
                
                # Handle different checkpoint formats
                if 'model_state' in checkpoint:
                    # Checkpoint contains metadata, extract model state
                    state_dict = checkpoint['model_state']
                    print(f"Loaded checkpoint with metadata: epoch={checkpoint.get('epoch', 'unknown')}, "
                          f"val_auc={checkpoint.get('best_val_auc', 'unknown'):.4f}")
                elif isinstance(checkpoint, dict) and any(key.startswith(('node_embed', 'gnn_layers', 'binary_head')) for key in checkpoint.keys()):
                    # Direct state dict
                    state_dict = checkpoint
                else:
                    # Try to use checkpoint as-is
                    state_dict = checkpoint
                
                gnn_model.load_state_dict(state_dict)
            else:
                print(f"GNN model file not found: {config['path']}, using randomly initialized weights")
            
            gnn_model.to("cuda:0")
            gnn_model.eval()
            
            # Create processor with parameters matching extract_yolo_graphs.py
            processor = YOLOGNNBinaryProcessor(
                gnn_model, 
                distance_threshold=0.25,  # radius parameter from extract_yolo_graphs.py
                knn_intra=4,              # knn parameter
                knn_inter=4,              # inter parameter  
                sigma=0.1                 # sigma parameter
            )
            
            model_dict = {
                "model": gnn_model,
                "processor": processor,
                "weight": config["weight"],
                "name": config["name"],
                "fold": config["fold"]
            }
            gnn_models.append(model_dict)
        except Exception as e:
            print(f"Failed to load GNN model {config['name']}: {e}")
            continue
    
    return gnn_models

# Load all models
models = load_models()
gnn_models = load_gnn_models()
print(f"Loaded {len(models)} YOLO models and {len(gnn_models)} GNN models on single GPU")

@torch.no_grad()
def eval_one_series_ensemble_with_gnn(slices: List[np.ndarray], slice_thickness: Optional[float] = None):
    """Run inference using all models on single GPU with GNN post-processing"""
    if not slices:
        return 0.1, np.ones(len(LABELS)) * 0.1
    
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    
    # Store all YOLO results for GNN processing
    all_yolo_results = []
    
    # First, run YOLO models to get detections and location predictions
    for model_dict in models:
        model = model_dict["model"]
        weight = model_dict["weight"]
        
        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(LABELS), dtype=np.float32)
            
            # Store results for this model
            model_results = []
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                
                results = model.predict(
                    batch_slices, 
                    verbose=False, 
                    batch=len(batch_slices), 
                    device="cuda:0", 
                    conf=0.01
                )
                
                model_results.extend(results)
                
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
                            if 0 <= k < len(LABELS) and c > per_class_max[k]:
                                per_class_max[k] = c
                    except Exception:
                        try:
                            batch_max = float(r.boxes.conf.max().item())
                            if batch_max > max_conf_all:
                                max_conf_all = batch_max
                        except Exception:
                            pass
            
            all_yolo_results.append((model_results, weight, model_dict["name"]))
            ensemble_cls_preds.append(max_conf_all * weight)
            ensemble_loc_preds.append(per_class_max * weight)
            total_weight += weight
            
        except Exception as e:
            print(f"Error in YOLO model {model_dict['name']}: {e}")
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(LABELS)) * 0.1 * weight)
            total_weight += weight
    
    # Get YOLO location predictions (always use these)
    if total_weight > 0:
        yolo_cls_pred = sum(ensemble_cls_preds) / total_weight  # Keep as fallback
        yolo_loc_preds = sum(ensemble_loc_preds) / total_weight
    else:
        yolo_cls_pred = 0.1  # Fallback
        yolo_loc_preds = np.ones(len(LABELS)) * 0.1
    
    # Always try GNN first for classification
    final_cls_pred = None
    gnn_success = False
    
    if gnn_models and all_yolo_results:
        gnn_predictions = []
        gnn_total_weight = 0.0
        
        for gnn_dict in gnn_models:
            processor = gnn_dict["processor"]
            gnn_weight = gnn_dict["weight"]
            
            try:
                # Use the first YOLO model's results for GNN (you could ensemble here too)
                yolo_results, _, _ = all_yolo_results[0]
                
                # Process through GNN with slice thickness
                gnn_output = processor.process_series(yolo_results, slice_thickness)
                aneurysm_prob = float(gnn_output['aneurysm_prob'])
                
                gnn_predictions.append(aneurysm_prob * gnn_weight)
                gnn_total_weight += gnn_weight
                gnn_success = True
                
            except Exception as e:
                print(f"Error in GNN model {gnn_dict['name']}: {e}")
                # Don't add fallback predictions here - we'll use YOLO as fallback
                continue
        
        if gnn_success and gnn_total_weight > 0:
            final_cls_pred = sum(gnn_predictions) / gnn_total_weight
            print(f"Using GNN prediction: {final_cls_pred:.4f}")
        else:
            print("GNN processing failed, falling back to YOLO prediction")
    
    # Fallback to YOLO if GNN failed or is not available
    if final_cls_pred is None:
        final_cls_pred = yolo_cls_pred
        print(f"Using YOLO fallback prediction: {final_cls_pred:.4f}")
    
    return final_cls_pred, yolo_loc_preds

def _predict_inner(series_path):
    """Internal prediction logic with parallel preprocessing and single GPU inference"""
    series_path = Path(series_path)

    dicom_files = collect_series_slices(series_path)
    
    # Parallel DICOM processing
    all_slices: List[np.ndarray] = []
    slice_thicknesses: List[float] = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_dicom_file, dcm_path): dcm_path 
                         for dcm_path in dicom_files}
        
        for future in as_completed(future_to_file):
            try:
                slices, thickness = future.result()
                all_slices.extend(slices)
                if thickness is not None:
                    slice_thicknesses.append(thickness)
            except Exception as e:
                dcm_path = future_to_file[future]
                print(f"Failed processing {dcm_path.name}: {e}")

    # Calculate average slice thickness
    avg_thickness = None
    if slice_thicknesses:
        avg_thickness = float(np.median(slice_thicknesses))

    # If no valid images were read, return a safe fallback row
    if not all_slices:
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions

    cls_prob, loc_probs = eval_one_series_ensemble_with_gnn(all_slices, avg_thickness)

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

# %% [code] {"execution":{"iopub.status.busy":"2025-08-29T01:15:05.456183Z","iopub.execute_input":"2025-08-29T01:15:05.456411Z","iopub.status.idle":"2025-08-29T01:15:19.522195Z","shell.execute_reply.started":"2025-08-29T01:15:05.456395Z","shell.execute_reply":"2025-08-29T01:15:19.521496Z"},"jupyter":{"outputs_hidden":false}}
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
        # Optional: print head instead of display to avoid dependency on notebook environment
        display(submission_df)
    
    print(f"Total time: {time.time() - st}")