"""
Binary aneurysm detection with PyTorch Lightning.
Uses full tomograms with shape (128, 384, 384) for aneurysm presence classification.

Weighted Metrics Implementation:
- Training: Volume-level weights based on class imbalance (pos_weight from dataset statistics)
- Validation: Tomogram-level weights for whole volume predictions
- Metrics: Standard + weighted versions of accuracy, F1, precision, recall, AP
- Balanced accuracy: Uses sklearn's built-in class balancing
"""

import argparse
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import timm_3d
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the specific warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', message='A single label was found in')
warnings.filterwarnings('ignore', message='.*confusion matrix.*', category=UserWarning)

torch.set_float32_matmul_precision('medium')

class FocalLoss(nn.Module):
    """Focal Loss implementation for binary classification."""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities
        probs = torch.sigmoid(logits.squeeze(-1))
        
        # Calculate binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, 
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction='none'
        )
        
        # Calculate focal weight: (1 - p_t)^gamma
        # For positive samples: p_t = p, for negative samples: p_t = 1 - p
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine focal and alpha weights
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


class VolumeDataset(Dataset):
    def __init__(self, data_dir: str, fold: int, mode: str = 'train', debug: bool = False, crop_size: tuple = (128, 128, 312)):
        """
        Load volume data from fold directories and crop random patches.
        Args:
            data_dir: Root directory containing fold_0, fold_1, etc.
            fold: Fold number to use as validation (0-4)
            mode: 'train' or 'val'
            debug: If True, use only 1 percent of the data for faster testing
            crop_size: Size of random crop (H, W, D)
        """
        self.items = []
        self.mode = mode
        self.debug = debug
        self.crop_size = crop_size  # (H, W, D) = (128, 128, 312)
        
        if mode == 'train':
            # Use all folds except the validation fold
            train_folds = [f for f in range(5) if f != fold]
        else:
            # Use only the validation fold
            train_folds = [fold]
        
        for f in train_folds:
            fold_dir = Path(data_dir) / f"fold_{f}"
            if fold_dir.exists():
                for npz_file in fold_dir.glob("*.npz"):
                    self.items.append(str(npz_file))
        
        # Apply debug sampling if enabled
        if self.debug and len(self.items) > 0:
            original_count = len(self.items)
            # Sample 10 percent of the data, but ensure at least 1 sample
            sample_size = max(1, int(len(self.items) * 0.1))
            if sample_size < len(self.items):
                # Use deterministic sampling based on fold and mode for reproducibility
                # This ensures same samples are selected even if setup() is called multiple times
                seed_offset = hash(f"{fold}_{mode}") % 1000
                rng = np.random.RandomState(42 + seed_offset)
                sampled_indices = rng.choice(len(self.items), sample_size, replace=False)
                self.items = [self.items[i] for i in sampled_indices]
                print(f"DEBUG MODE: Sampled {len(self.items)} out of {original_count} samples (1 percent)")
        
        print(f"Found {len(self.items)} samples for {mode} (folds: {train_folds})")
        print(f"Data format: float16 [0,1] -> float32 [0,1] for training. Volume: (128,384,384) -> Crop: {self.crop_size} -> Model: (1,{self.crop_size[0]},{self.crop_size[1]},{self.crop_size[2]})")
        if self.debug:
            print(f"DEBUG MODE: Using reduced dataset for faster testing")
    
    def __len__(self):
        # Return number of volumes for both train and validation
        return len(self.items)
    
    def __getitem__(self, idx):
        path = self.items[idx]
        
        with np.load(path) as npz:
            volume = npz['volume']  # (128, 384, 384) full volume - float16 [0,1] or uint8 [0,255]
            mask = npz.get('mask', None)  # Mask if available
        
        # Handle different data formats: uint8 [0,255] or float16 [0,1]
        if volume.dtype == np.uint8:
            # Convert uint8 [0,255] to float32 [0,1] for training
            volume_normalized = (volume.astype(np.float32) / 255.0)
        else:
            # Volume is already normalized to [0,1] as float16, convert to float32 for training
            volume_normalized = volume.astype(np.float32)
        
        # Crop random patch of size (crop_H, crop_W, crop_D) from volume (D, H, W)
        d, h, w = volume_normalized.shape  # (128, 384, 384)
        crop_h, crop_w, crop_d = self.crop_size  # (128, 128, 312)
        
        # Calculate valid crop ranges
        max_h_start = max(0, h - crop_h)
        max_w_start = max(0, w - crop_w) 
        max_d_start = max(0, d - crop_d)
        
        if self.mode == 'train':
            # Random crop for training
            h_start = random.randint(0, max_h_start) if max_h_start > 0 else 0
            w_start = random.randint(0, max_w_start) if max_w_start > 0 else 0
            d_start = random.randint(0, max_d_start) if max_d_start > 0 else 0
        else:
            # Center crop for validation
            h_start = max_h_start // 2
            w_start = max_w_start // 2
            d_start = max_d_start // 2
        
        # Extract crop from volume
        volume_crop = volume_normalized[
            d_start:d_start + crop_d,
            h_start:h_start + crop_h,
            w_start:w_start + crop_w
        ]
        
        # Handle case where volume is smaller than crop size
        if volume_crop.shape != (crop_d, crop_h, crop_w):
            # Pad with zeros if needed
            padded_volume = np.zeros((crop_d, crop_h, crop_w), dtype=volume_normalized.dtype)
            actual_d, actual_h, actual_w = volume_crop.shape
            padded_volume[:actual_d, :actual_h, :actual_w] = volume_crop
            volume_crop = padded_volume
        
        # Determine label from mask in the cropped region
        if mask is not None:
            # Crop the mask using the same coordinates
            mask_crop = mask[
                d_start:d_start + crop_d,
                h_start:h_start + crop_h,
                w_start:w_start + crop_w
            ]
            
            # Handle case where mask is smaller than crop size
            if mask_crop.shape != (crop_d, crop_h, crop_w):
                padded_mask = np.zeros((crop_d, crop_h, crop_w), dtype=mask.dtype)
                actual_d, actual_h, actual_w = mask_crop.shape
                padded_mask[:actual_d, :actual_h, :actual_w] = mask_crop
                mask_crop = padded_mask
            
            # Handle different mask formats: uint8 [0,255] or float16 [0,1]
            if mask_crop.dtype == np.uint8:
                threshold = 2  # For uint8, use threshold of 2/255
                label = 1 if np.any(mask_crop > threshold) else 0
            else:
                threshold = 0.01  # For float16, use threshold of 0.01
                label = 1 if np.any(mask_crop > threshold) else 0
        else:
            # Infer from filename if mask not available
            label = 1 if 'pos_' in path else 0
        
        # Apply correct transpose for timm_3d: (D, H, W) -> (C, H, W, D)
        # volume_crop shape: (crop_d, crop_h, crop_w) = (312, 128, 128)
        #volume_transposed = volume_crop.transpose(1, 2, 0)  # (128, 128, 312) = (H, W, D)
        x = torch.from_numpy(volume_crop).unsqueeze(0)  # (1, 128, 128, 312) = (C, H, W, D)
  
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


class SlidingWindowVolumeDataset(Dataset):
    """Dataset that applies sliding window validation on full volumes."""
    def __init__(self, data_dir: str, fold: int, crop_size: tuple = (128, 128, 312), stride_ratio: float = 0.5):
        self.items = []
        self.crop_size = crop_size
        self.stride_ratio = stride_ratio
        
        # Use only validation fold
        fold_dir = Path(data_dir) / f"fold_{fold}"
        if fold_dir.exists():
            for npz_file in fold_dir.glob("*.npz"):
                self.items.append(str(npz_file))
        
        print(f"Found {len(self.items)} volumes for sliding window validation")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path = self.items[idx]
        
        with np.load(path) as npz:
            volume = npz['volume']  # (128, 384, 384) full volume
            mask = npz.get('mask', None)
        
        # Normalize volume
        if volume.dtype == np.uint8:
            volume_normalized = (volume.astype(np.float32) / 255.0)
        else:
            volume_normalized = volume.astype(np.float32)
        
        # Get global label from mask
        if mask is not None:
            if mask.dtype == np.uint8:
                threshold = 2
                global_label = 1 if np.any(mask > threshold) else 0
            else:
                threshold = 0.01
                global_label = 1 if np.any(mask > threshold) else 0
        else:
            global_label = 1 if 'pos_' in path else 0
        
        # Generate sliding window crops
        crops, crop_labels = self.get_sliding_window_crops(volume_normalized, mask)
        
        # Convert to tensors
        crop_tensors = []
        for crop in crops:
            x = torch.from_numpy(crop).unsqueeze(0)  # (1, H, W, D)
            crop_tensors.append(x)
        
        crop_tensors = torch.stack(crop_tensors)  # (num_crops, 1, H, W, D)
        
        return {
            'crops': crop_tensors,
            'global_label': torch.tensor(global_label, dtype=torch.float32),
            'path': path
        }
    
    def get_sliding_window_crops(self, volume_normalized, mask=None):
        """Generate sliding window crops."""
        d, h, w = volume_normalized.shape  # (128, 384, 384)
        crop_h, crop_w, crop_d = self.crop_size  # (128, 128, 312)
        
        # Calculate stride
        stride_h = max(1, int(crop_h * self.stride_ratio))
        stride_w = max(1, int(crop_w * self.stride_ratio))
        stride_d = max(1, int(crop_d * self.stride_ratio))
        
        crops = []
        labels = []
        
        # Generate all possible crop positions
        for d_start in range(0, max(1, d - crop_d + 1), stride_d):
            for h_start in range(0, max(1, h - crop_h + 1), stride_h):
                for w_start in range(0, max(1, w - crop_w + 1), stride_w):
                    # Ensure we don't go out of bounds
                    d_end = min(d_start + crop_d, d)
                    h_end = min(h_start + crop_h, h)
                    w_end = min(w_start + crop_w, w)
                    
                    # Extract crop
                    volume_crop = volume_normalized[d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # Pad if necessary
                    if volume_crop.shape != (crop_d, crop_h, crop_w):
                        padded_volume = np.zeros((crop_d, crop_h, crop_w), dtype=volume_normalized.dtype)
                        actual_d, actual_h, actual_w = volume_crop.shape
                        padded_volume[:actual_d, :actual_h, :actual_w] = volume_crop
                        volume_crop = padded_volume
                    
                    # Determine label for this crop
                    if mask is not None:
                        mask_crop = mask[d_start:d_end, h_start:h_end, w_start:w_end]
                        if mask_crop.shape != (crop_d, crop_h, crop_w):
                            padded_mask = np.zeros((crop_d, crop_h, crop_w), dtype=mask.dtype)
                            actual_d, actual_h, actual_w = mask_crop.shape
                            padded_mask[:actual_d, :actual_h, :actual_w] = mask_crop
                            mask_crop = padded_mask
                        
                        if mask_crop.dtype == np.uint8:
                            threshold = 2
                            label = 1 if np.any(mask_crop > threshold) else 0
                        else:
                            threshold = 0.01
                            label = 1 if np.any(mask_crop > threshold) else 0
                    else:
                        # Use global label if no local mask available
                        label = None  # Will use global label
                    
                    # Transpose for model input
                    volume_transposed = volume_crop.transpose(1, 2, 0)  # (H, W, D)
                    crops.append(volume_transposed)
                    labels.append(label)
        
        return crops, labels



class LightningModel(pl.LightningModule):
    def __init__(self, model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k", lr: float = 1e-3, pos_weight: float = 1.0, 
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.save_hyperparameters()
        set_seed()

        self.model = timm_3d.create_model(
            model_name, 
            pretrained=True,
            num_classes=1,  # Single binary output
            global_pool="avg",
            in_chans=1,
            drop_path_rate=0.2, 
            drop_rate=0.2,      
        )
        
        # Disable gradient checkpointing for speed (trades memory for speed)
        if hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(False)
        
        # Initialize focal loss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        
        # For metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        
        # Check for numerical issues before clamping
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: Found NaN/Inf in logits: nan={torch.isnan(logits).sum()}, inf={torch.isinf(logits).sum()}")
        
        # Ensure we have valid targets
        targets_clean = targets.to(device).float()
        
        # Use focal loss instead of BCE
        loss = self.focal_loss(logits, targets_clean)
        
        # Debug very small losses
        if loss.item() < 1e-6:
            print(f"Warning: Very small loss detected: {loss.item():.8f}")
            print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"Targets: unique values = {torch.unique(targets_clean)}")
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        
        logits = self(x)  # Shape: (batch_size, 1)
        loss = self.compute_loss(logits, y)
        
        # Store outputs for epoch-end metrics calculation
        self.training_step_outputs.append({
            'logits': logits.detach().cpu(),
            'targets': y.detach().cpu(),
            'loss': loss.detach().cpu(),
        })
        
        # Only log loss per step for monitoring training progress
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
 
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Standard validation with full volumes
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
        
        # Store outputs for epoch-end metrics calculation
        self.validation_step_outputs.append({
            'logits': logits.detach().cpu(),
            'targets': y.detach().cpu(),
            'loss': loss.detach().cpu(),
        })
        
        return loss
        
    def on_train_epoch_end(self):
        # Calculate training metrics at epoch end using all collected outputs
        if not self.training_step_outputs:
            return
        
            
        all_logits = torch.cat([x['logits'] for x in self.training_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.training_step_outputs])
        
        # Convert to numpy for binary classification
        probs = torch.sigmoid(all_logits.squeeze(-1)).numpy()
        targets_np = all_targets.numpy()
        
        try:
            # Binary classification metrics at threshold 0.5
            y_pred = (probs > 0.5).astype(int)
            
            # Calculate per-class metrics and take their average
            if len(np.unique(targets_np)) > 1 and len(np.unique(y_pred)) > 1:
                # F1 score for each class, then average
                f1_per_class = f1_score(targets_np, y_pred, average=None, zero_division=0)
                f1_macro = np.mean(f1_per_class)  # Average of no_aneurysm and aneurysm F1
                
                # Precision and recall for each class, then average
                precision_per_class = precision_score(targets_np, y_pred, average=None, zero_division=0)
                precision_macro = np.mean(precision_per_class)
                
                recall_per_class = recall_score(targets_np, y_pred, average=None, zero_division=0)
                recall_macro = np.mean(recall_per_class)
            else:
                f1_macro = 0.0
                precision_macro = 0.0
                recall_macro = 0.0
            
            # Overall accuracy (unchanged)
            acc = accuracy_score(targets_np, y_pred)
            
            # Log macro-averaged metrics (average of both classes)
            self.log("train_f1_macro", f1_macro, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_precision_macro", precision_macro, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_recall_macro", recall_macro, on_epoch=True, prog_bar=True, sync_dist=True)
            # AUC calculation if both classes present
            if len(np.unique(targets_np)) > 1:
                auc = roc_auc_score(targets_np, probs)
                self.log("train_auc", auc, on_epoch=True, prog_bar=True, sync_dist=True)
            
            if np.sum(targets_np) > 0:  # Positive samples exist
                ap = average_precision_score(targets_np, probs)
                self.log("train_ap", ap, on_epoch=True, prog_bar=True, sync_dist=True)
            
                
        except Exception as e:
            print(f"Warning: Could not calculate training metrics: {e}")
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # Calculate additional metrics at epoch end
        if not self.validation_step_outputs:
            return
            
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Convert to numpy for binary classification
        probs = torch.sigmoid(all_logits.squeeze(-1)).numpy()
        targets_np = all_targets.numpy()
        
        try:
            # Binary classification metrics at threshold 0.5
            y_pred = (probs > 0.5).astype(int)
            
            # Calculate per-class metrics and take their average
            if len(np.unique(targets_np)) > 1 and len(np.unique(y_pred)) > 1:
                # F1 score for each class, then average
                f1_per_class = f1_score(targets_np, y_pred, average=None, zero_division=0)
                f1_macro = np.mean(f1_per_class)  # Average of no_aneurysm and aneurysm F1
                
                # Precision and recall for each class, then average
                precision_per_class = precision_score(targets_np, y_pred, average=None, zero_division=0)
                precision_macro = np.mean(precision_per_class)
                
                recall_per_class = recall_score(targets_np, y_pred, average=None, zero_division=0)
                recall_macro = np.mean(recall_per_class)
            else:
                f1_macro = 0.0
                precision_macro = 0.0
                recall_macro = 0.0
            
            # Overall accuracy and AUC
            acc = accuracy_score(targets_np, y_pred)
            
            if len(np.unique(targets_np)) > 1:
                auc = roc_auc_score(targets_np, probs)
            else:
                auc = 0.5
                
            # Log macro-averaged metrics (average of both classes)
            self.log("val_f1_macro", f1_macro, prog_bar=True, sync_dist=True)
            self.log("val_precision_macro", precision_macro, prog_bar=True, sync_dist=True)
            self.log("val_recall_macro", recall_macro, prog_bar=True, sync_dist=True)
            self.log("val_auc", auc, prog_bar=True, sync_dist=True)
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)
                
        except Exception as e:
            print(f"Warning: Could not calculate advanced metrics: {e}")
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)




class DataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, fold: int, batch_size: int = 4, debug: bool = False, crop_size: tuple = (128, 128, 312)):
        super().__init__()
        self.data_root = data_root
        self.fold = fold
        self.batch_size = batch_size
        self.pos_weight = 1.0
        self.debug = debug
        self.crop_size = crop_size
        
    def setup(self, stage=None):
        # Prevent multiple setup calls from re-creating datasets
        if hasattr(self, 'train_ds') and hasattr(self, 'val_ds'):
            print(f"Setup already called - Training samples: {len(self.train_ds)}, Validation samples: {len(self.val_ds)}")
            return
            
        self.train_ds = VolumeDataset(
            self.data_root, 
            self.fold, 
            mode='train',
            debug=self.debug,
            crop_size=self.crop_size
        )
        self.val_ds = VolumeDataset(
            self.data_root, 
            self.fold, 
            mode='val',
            debug=self.debug,
            crop_size=self.crop_size
        )
        
        print(f"Training samples: {len(self.train_ds)}")
        print(f"Validation samples: {len(self.val_ds)}")
        
        # Ensure we have some data to train on
        if len(self.train_ds) == 0:
            raise ValueError(f"No training samples found in data_root={self.data_root} for fold={self.fold}")
        if len(self.val_ds) == 0:
            raise ValueError(f"No validation samples found in data_root={self.data_root} for fold={self.fold}")

        # Calculate pos_weight based on volume-level label distribution
        pos_weight = 1.0
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, 
                         num_workers=4, persistent_workers=True, pin_memory=True, 
                         prefetch_factor=2)
    
    def val_dataloader(self):
        val_batch_size = self.batch_size     
       
        return DataLoader(self.val_ds, batch_size=val_batch_size, num_workers=4, 
                         persistent_workers=True, pin_memory=True, prefetch_factor=2)


def main():
    # Set memory allocation strategy for better GPU memory management
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="aneurysm_cubes_v2", help="Root directory with fold_0, fold_1, etc.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold (0-4)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21k_ft_in1k")
    # Focal loss parameters
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha parameter")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    # Crop size parameters
    parser.add_argument("--crop-h", type=int, default=128, help="Crop height")
    parser.add_argument("--crop-w", type=int, default=128, help="Crop width") 
    parser.add_argument("--crop-d", type=int, default=32, help="Crop depth")
    parser.add_argument("--sliding-window", action="store_true", help="Use sliding window validation")
    parser.add_argument("--stride-ratio", type=float, default=0.5, help="Stride ratio for sliding window (0.5 means 50 percent overlap)")
    # Weights & Biases logging options
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="rsna-iad-binary")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--debug", action="store_true", help="Debug mode: use only 1 percent of the data for faster testing")
    args = parser.parse_args()
    
    crop_size = (args.crop_h, args.crop_w, args.crop_d)
    dm = DataModule(
        args.data_root, 
        args.fold, 
        args.batch_size,
        debug=args.debug,
        crop_size=crop_size
    )
    # Setup early to compute class weights
    dm.setup()
    model = LightningModel(
        model_name=args.model_name, 
        lr=args.lr, 
        pos_weight=dm.pos_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    
    # Configure Weights & Biases logger if enabled
    logger = None
    if args.wandb != "disabled":
        if args.wandb == "offline":
            os.environ["WANDB_MODE"] = "offline"
        try:
            run_name = args.run_name or f"{args.model_name}_fold{args.fold}"
            logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                log_model=True,
            )
            # Log CLI and model hyperparameters
            logger.log_hyperparams({
                "data_root": args.data_root,
                "fold": args.fold,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "model_name": args.model_name,
                "pos_weight": dm.pos_weight,
                "task_type": "binary_aneurysm_detection_full_volumes",
                "input_volume_shape_d_h_w": "(128, 384, 384)",
                "crop_size_h_w_d": f"({args.crop_h}, {args.crop_w}, {args.crop_d})",
                "model_input_shape_c_h_w_d": f"(1, {args.crop_h}, {args.crop_w}, {args.crop_d})",
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "augmentations_disabled": True,
                "debug_mode": args.debug,
                "balanced_metrics": True,  # Indicate that balanced metrics are calculated
                "volume_level_training": True,  # Training uses full volumes
                "auc_weighting": False,  # AUC is class-invariant, no weighting applied
            })
            # Log gradients/parameters periodically
            #logger.watch(model, log="gradients", log_freq=200)
        except Exception as e:
            print(f"WandbLogger initialization failed: {e}. Continuing without W&B.")
            logger = None


    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc", 
        mode="max",
        save_top_k=2,
        filename=f'best-fold{args.fold}-{{epoch}}-{{val_auc:.3f}}'
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=logger,
        precision="16-mixed",  # Use mixed precision for memory efficiency
        gradient_clip_val=1.0,  # Gradient clipping for stability
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        log_every_n_steps=10,  # Reduce logging frequency
        enable_model_summary=True,
        accumulate_grad_batches=1,  # No gradient accumulation for now
        deterministic=True,  # Allow non-deterministic operations for speed
    )
    
    trainer.fit(model, dm)
    
    # Print final results
    if checkpoint_callback.best_model_path:
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best AUC: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()


# Usage examples:
# python3 train_timm3d_binary_crop.py --data-root volume_data --fold 0  # Default settings with focal loss and random cropping
# python3 train_timm3d_binary_crop.py --data-root volume_data --fold 0 --batch-size 1  # Even smaller batch for limited GPU memory
# python3 train_timm3d_binary_crop.py --data-root volume_data --fold 0 --debug  # Debug mode with 1 percent of data
# python3 train_timm3d_binary_crop.py --data-root volume_data --fold 0 --focal-alpha 0.5 --focal-gamma 1.5  # Custom focal loss parameters
# python3 train_timm3d_binary_crop.py --data-root volume_data --fold 0 --crop-h 96 --crop-w 96 --crop-d 256  # Custom crop size

