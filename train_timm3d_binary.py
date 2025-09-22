"""
Binary aneurysm detection with PyTorch Lightning.
Uses cube patches with shape (32, 128, 128) for aneurysm presence classification.

Weighted Metrics Implementation:
- Training: Crop-level weights based on class imbalance (pos_weight from dataset statistics)
- Validation: Tomogram-level weights for whole volume predictions
- Metrics: Standard + weighted versions of accuracy, F1, precision, recall, AP
- Balanced accuracy: Uses sklearn's built-in class balancing
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# TorchIO for 3D medical image augmentation
try:
    import torchio as tio
    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False
    print("Warning: TorchIO not available. Install with 'pip install torchio' for advanced 3D augmentations.")

torch.set_float32_matmul_precision('medium')

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_torchio_transforms(mode: str = 'train', intensity_augmentation: bool = True, spatial_augmentation: bool = True):
    """
    Create TorchIO transforms for CT image preprocessing and augmentation.
    
    Args:
        mode: 'train' or 'val' - controls which augmentations are applied
        intensity_augmentation: Whether to apply intensity-based augmentations
        spatial_augmentation: Whether to apply spatial augmentations
    
    Returns:
        torchio.Compose transform or None if TorchIO not available
    """
    if not TORCHIO_AVAILABLE:
        return None
    
    transforms_list = []
    
    if mode == 'train':
        # Spatial augmentations for training
        if spatial_augmentation:
            transforms_list.extend([
                tio.RandomFlip(axes=('LR',), p=0.5),  # Left-Right flip
            ])
        
        # Intensity augmentations for training
        if intensity_augmentation:
            transforms_list.extend([
                tio.RandomNoise(std=(0, 0.05), p=0.3),
                tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3),
                tio.RandomBiasField(coefficients=0.3, p=0.2),
            ])
    
    # Common preprocessing for both train and validation
    # Note: Skip normalization here since we handle it in the dataset
    
    if transforms_list:
        return tio.Compose(transforms_list)
    else:
        return None


class VolumeDataset(Dataset):
    def __init__(self, data_dir: str, fold: int, mode: str = 'train', 
                 use_torchio: bool = True, intensity_augmentation: bool = True, 
                 spatial_augmentation: bool = True, crop_size: Tuple[int, int, int] = (32, 128, 128),
                 crops_per_volume: int = 4, overlap: float = 0.25, debug: bool = False,
                 positive_crop_ratio: float = 0.8):  # New parameter
        """
        Load full volume data from fold directories and do random/sliding window cropping.
        Args:
            data_dir: Root directory containing fold_0, fold_1, etc.
            fold: Fold number to use as validation (0-4)
            mode: 'train' or 'val'
            use_torchio: Whether to apply TorchIO transforms
            intensity_augmentation: Whether to apply intensity-based augmentations
            spatial_augmentation: Whether to apply spatial augmentations
            crop_size: Size of crops to extract (D, H, W)
            crops_per_volume: Number of random crops per volume during training
            overlap: Overlap for sliding window during validation (0.0-1.0)
            debug: If True, use only 1% of the data for faster testing
            positive_crop_ratio: Ratio of positive crops to sample during training (0.0-1.0)
        """
        self.items = []
        self.mode = mode
        self.use_torchio = use_torchio and TORCHIO_AVAILABLE
        self.crop_size = crop_size
        self.crops_per_volume = crops_per_volume
        self.overlap = overlap
        self.debug = debug
        self.positive_crop_ratio = positive_crop_ratio if mode == 'train' else 0.5  # Only apply to training
        
        # Initialize TorchIO transforms
        if self.use_torchio:
            self.transforms = get_torchio_transforms(
                mode=mode, 
                intensity_augmentation=intensity_augmentation,
                spatial_augmentation=spatial_augmentation
            )
        else:
            self.transforms = None
        
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
            # Sample 1% of the data, but ensure at least 1 sample
            sample_size = max(1, int(len(self.items) * 0.01))
            if sample_size < len(self.items):
                # Use random sampling with fixed seed for reproducibility
                np.random.seed(42)
                sampled_indices = np.random.choice(len(self.items), sample_size, replace=False)
                self.items = [self.items[i] for i in sampled_indices]
                print(f"DEBUG MODE: Sampled {len(self.items)} out of {original_count} samples (1%)")
        
        print(f"Found {len(self.items)} samples for {mode} (folds: {train_folds})")
        print(f"Crop size: {crop_size}, Mode: {mode}")
        if mode == 'train':
            print(f"Random crops per volume: {crops_per_volume}")
            print(f"Positive crop ratio: {positive_crop_ratio:.1%}")
        else:
            print(f"Sliding window overlap: {overlap}")
        if self.use_torchio and self.transforms is not None:
            print(f"TorchIO transforms enabled for {mode}")
        else:
            print(f"TorchIO transforms disabled for {mode}")
        if self.debug:
            print(f"DEBUG MODE: Using reduced dataset for faster testing")
    
    def __len__(self):
        # For training, we take multiple crops per volume
        if self.mode == 'train':
            return len(self.items) * self.crops_per_volume
        # For validation, length equals number of volumes
        return len(self.items)
    
    def _find_positive_crop_locations(self, mask: np.ndarray, max_attempts: int = 50) -> List[Tuple[int, int, int]]:
        """
        Find potential crop locations that contain positive regions.
        
        Args:
            mask: Binary mask array
            max_attempts: Maximum number of attempts to find positive locations
            
        Returns:
            List of (start_d, start_h, start_w) coordinates for positive crops
        """
        d, h, w = mask.shape
        crop_d, crop_h, crop_w = self.crop_size
        
        # Find all positive voxel coordinates
        positive_coords = np.where(mask > 0.01)
        if len(positive_coords[0]) == 0:
            return []  # No positive regions found
        
        positive_locations = []
        attempts = 0
        
        # Try to find crop locations that include positive voxels
        while len(positive_locations) < max_attempts and attempts < max_attempts * 2:
            attempts += 1
            
            # Randomly select a positive voxel as anchor
            idx = np.random.randint(len(positive_coords[0]))
            anchor_d, anchor_h, anchor_w = positive_coords[0][idx], positive_coords[1][idx], positive_coords[2][idx]
            
            # Generate crop coordinates that include this positive voxel
            # Allow some randomness in crop placement while ensuring the positive voxel is included
            max_start_d = min(anchor_d, d - crop_d)
            max_start_h = min(anchor_h, h - crop_h)
            max_start_w = min(anchor_w, w - crop_w)
            
            min_start_d = max(0, anchor_d - crop_d + 1)
            min_start_h = max(0, anchor_h - crop_h + 1)
            min_start_w = max(0, anchor_w - crop_w + 1)
            
            if max_start_d >= min_start_d and max_start_h >= min_start_h and max_start_w >= min_start_w:
                start_d = np.random.randint(min_start_d, max_start_d + 1)
                start_h = np.random.randint(min_start_h, max_start_h + 1)
                start_w = np.random.randint(min_start_w, max_start_w + 1)
                
                # Verify this crop would actually be positive
                crop_mask = mask[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
                if np.any(crop_mask > 0.01):
                    coord = (start_d, start_h, start_w)
                    if coord not in positive_locations:  # Avoid duplicates
                        positive_locations.append(coord)
        
        return positive_locations
    
    def _extract_targeted_crop(self, volume: np.ndarray, mask: np.ndarray, target_positive: bool = True) -> Tuple[np.ndarray, int]:
        """
        Extract a crop targeting either positive or negative regions.
        
        Args:
            volume: Volume array
            mask: Mask array
            target_positive: If True, try to get a positive crop; if False, try to get a negative crop
            
        Returns:
            Tuple of (crop_volume, label)
        """
        d, h, w = volume.shape
        crop_d, crop_h, crop_w = self.crop_size
        
        if target_positive:
            # Try to find a positive crop
            positive_locations = self._find_positive_crop_locations(mask)
            
            if positive_locations:
                # Randomly select one of the positive locations
                start_d, start_h, start_w = positive_locations[np.random.randint(len(positive_locations))]
            else:
                # Fallback to random crop if no positive locations found
                max_d = max(0, d - crop_d)
                max_h = max(0, h - crop_h)
                max_w = max(0, w - crop_w)
                
                start_d = np.random.randint(0, max_d + 1) if max_d > 0 else 0
                start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
                start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
        else:
            # Try to find a negative crop (avoid positive regions)
            attempts = 0
            max_attempts = 20
            
            while attempts < max_attempts:
                max_d = max(0, d - crop_d)
                max_h = max(0, h - crop_h)
                max_w = max(0, w - crop_w)
                
                start_d = np.random.randint(0, max_d + 1) if max_d > 0 else 0
                start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
                start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
                
                # Check if this crop is negative
                crop_mask = mask[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
                if not np.any(crop_mask > 0.01):
                    break  # Found a negative crop
                
                attempts += 1
            
            # If we couldn't find a negative crop after max_attempts, just use the last attempt
        
        # Extract the crop
        crop_volume = volume[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        crop_mask = mask[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Pad if necessary (edge case)
        if crop_volume.shape != self.crop_size:
            pad_d = self.crop_size[0] - crop_volume.shape[0]
            pad_h = self.crop_size[1] - crop_volume.shape[1]
            pad_w = self.crop_size[2] - crop_volume.shape[2]
            crop_volume = np.pad(crop_volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            crop_mask = np.pad(crop_mask, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        # Determine actual label
        label = 1 if np.any(crop_mask > 0.01) else 0
        
        return crop_volume, label
    
    def _extract_random_crop(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Extract a random crop from volume and determine label based on mask."""
        d, h, w = volume.shape
        crop_d, crop_h, crop_w = self.crop_size
        
        # Random crop coordinates
        max_d = max(0, d - crop_d)
        max_h = max(0, h - crop_h)
        max_w = max(0, w - crop_w)
        
        start_d = np.random.randint(0, max_d + 1) if max_d > 0 else 0
        start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
        start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
        
        # Extract crop
        crop_volume = volume[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        crop_mask = mask[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Pad if necessary (edge case)
        if crop_volume.shape != self.crop_size:
            pad_d = self.crop_size[0] - crop_volume.shape[0]
            pad_h = self.crop_size[1] - crop_volume.shape[1]
            pad_w = self.crop_size[2] - crop_volume.shape[2]
            crop_volume = np.pad(crop_volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            crop_mask = np.pad(crop_mask, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        # Determine label: positive if any mask value > 0.01
        label = 1 if np.any(crop_mask > 0.01) else 0
        
        return crop_volume, label
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            # For training: use targeted crop sampling based on positive_crop_ratio
            volume_idx = idx // self.crops_per_volume
            crop_idx = idx % self.crops_per_volume
            path = self.items[volume_idx]
            
            with np.load(path) as npz:
                volume = npz['volume']  # (128, 384, 384) full volume
                mask = npz.get('mask', np.zeros_like(volume))  # Mask if available
                
            # Normalize to [0, 1] 
            volume_normalized = volume.astype('float32') / 255.0
            mask_normalized = mask.astype('float32')
            
            # Decide whether to target positive or negative crop based on ratio
            target_positive = np.random.random() < self.positive_crop_ratio
            
            # Extract targeted crop
            crop_volume, label = self._extract_targeted_crop(
                volume_normalized, mask_normalized, target_positive=target_positive
            )
            
            # Apply TorchIO transforms if enabled
            if self.use_torchio and self.transforms is not None:
                try:
                    # Create TorchIO subject
                    subject = tio.Subject(
                        image=tio.ScalarImage(tensor=crop_volume[None, ...])  # Add channel dim for TorchIO
                    )
                    
                    # Apply transforms
                    transformed = self.transforms(subject)
                    
                    # Extract the transformed volume and remove the extra dimension
                    x = transformed['image'].data.squeeze(0)  # Remove the channel dim added for TorchIO
                    
                    # Add channel dimension back for the model
                    x = x.unsqueeze(0)
                    
                except Exception as e:
                    print(f"Warning: TorchIO transform failed for {path}: {e}")
                    # Fallback to original volume if transforms fail
                    x = torch.from_numpy(crop_volume).unsqueeze(0)
            else:
                # No transforms, just convert to tensor
                x = torch.from_numpy(crop_volume).unsqueeze(0)  # Add channel dim
            
            y = torch.tensor(label, dtype=torch.float32)
            return x, y
            
        else:
            # For validation: return volume info for sliding window (unchanged)
            path = self.items[idx]
            
            with np.load(path) as npz:
                volume = npz['volume']  # (128, 384, 384) full volume
                mask = npz.get('mask', np.zeros_like(volume))  # Mask if available
                
            # Normalize to [0, 1] 
            volume_normalized = volume.astype('float32') / 255.0
            mask_normalized = mask.astype('float32')
            
            # Infer label from mask only: positive if any mask value > 0.01
            volume_label = 1 if np.any(mask_normalized > 0.01) else 0
            
            # Return volume info for sliding window processing
            return {
                'volume': volume_normalized,
                'mask': mask_normalized,
                'label': volume_label,
                'path': path
            }
    
    def _get_sliding_window_crops(self, volume: np.ndarray, mask: np.ndarray) -> List[Tuple[np.ndarray, int, Tuple[int, int, int]]]:
        """Get all sliding window crops for validation."""
        d, h, w = volume.shape
        crop_d, crop_h, crop_w = self.crop_size
        
        # Calculate step sizes based on overlap
        step_d = max(1, int(crop_d * (1 - self.overlap)))
        step_h = max(1, int(crop_h * (1 - self.overlap)))
        step_w = max(1, int(crop_w * (1 - self.overlap)))
        
        crops = []
        
        # Generate all possible crop positions
        for start_d in range(0, d - crop_d + 1, step_d):
            for start_h in range(0, h - crop_h + 1, step_h):
                for start_w in range(0, w - crop_w + 1, step_w):
                    # Extract crop
                    crop_volume = volume[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    crop_mask = mask[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    
                    # Determine label
                    label = 1 if np.any(crop_mask > 0.1) else 0
                    
                    crops.append((crop_volume, label, (start_d, start_h, start_w)))
        
        # If no crops were generated (volume too small), just take the whole volume padded
        if not crops:
            pad_d = max(0, crop_d - d)
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            
            padded_volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            padded_mask = np.pad(mask, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            crop_volume = padded_volume[:crop_d, :crop_h, :crop_w]
            crop_mask = padded_mask[:crop_d, :crop_h, :crop_w]
            label = 1 if np.any(crop_mask > 0.1) else 0
            
            crops.append((crop_volume, label, (0, 0, 0)))
        
        return crops
    



class LightningModel(pl.LightningModule):
    def __init__(self, model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k", lr: float = 1e-3, pos_weight: float = 1.0):
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
        
        # Store pos_weight for binary classification
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        
        # For metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        # Guard against NaNs/Infs in logits
        safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).float()
        return F.binary_cross_entropy_with_logits(
            safe_logits.squeeze(-1), targets.to(device).float(), pos_weight=self.pos_weight.to(device).float(), reduction="mean"
        )
    
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
        # Initialize variables that will be used regardless of batch type
        batch_logits = []
        batch_targets = []
        batch_volume_logits = []  # For tomogram-level metrics
        batch_volume_targets = []  # For tomogram-level metrics
        batch_patch_losses = []  # For storing patch losses per volume
        
        # Handle sliding window validation
        if isinstance(batch, dict):
            # Sliding window validation - process one volume at a time to save memory
            volume = batch['volume']  # (B, D, H, W)
            mask = batch['mask']      # (B, D, H, W) 
            labels = batch['label']   # (B,)
            paths = batch['path']
            
            # Process volumes one by one to reduce memory usage
            for i in range(volume.size(0)):
                vol = volume[i].detach().cpu().numpy()  # (D, H, W)
                msk = mask[i].detach().cpu().numpy()    # (D, H, W)
                # Always derive label from mask only
                label = 1 if np.any(msk > 0.01) else 0
                path = paths[i] if isinstance(paths, (list, tuple)) else str(paths)
                
                # Get sliding window crops using dataset method
                crops = self.trainer.datamodule.val_ds._get_sliding_window_crops(vol, msk)
                
                if not crops:
                    continue
                
                crop_logits = []
                crop_labels = []
                crop_losses = []
                crop_probs = []
                crop_positions = []
                
                # Process crops in batches for efficiency
                crop_batch_size = 8  # Adjust based on GPU memory
                num_crops = len(crops)
                
                for batch_start in range(0, num_crops, crop_batch_size):
                    batch_end = min(batch_start + crop_batch_size, num_crops)
                    batch_crops = crops[batch_start:batch_end]
                    
                    # Prepare batch tensors
                    batch_volumes = []
                    batch_labels = []
                    batch_positions = []
                    
                    for crop_vol, crop_label, crop_pos in batch_crops:
                        batch_volumes.append(torch.from_numpy(crop_vol))
                        batch_labels.append(crop_label)
                        batch_positions.append(crop_pos)
                    
                    # Stack into batch tensor
                    batch_tensor = torch.stack(batch_volumes).unsqueeze(1)  # (batch, 1, D, H, W)
                    batch_tensor = batch_tensor.to(self.device)
                    batch_targets_tensor = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                    
                    # Get predictions for batch
                    with torch.no_grad():
                        batch_logits_tensor = self(batch_tensor)  # (batch, 1)
                        batch_logits_squeezed = batch_logits_tensor.squeeze(-1)  # (batch,)
                        
                        # Calculate loss for each patch in batch
                        batch_losses = F.binary_cross_entropy_with_logits(
                            batch_logits_squeezed, batch_targets_tensor, 
                            pos_weight=self.pos_weight.to(self.device).float(), 
                            reduction='none'  # Get loss for each sample
                        )
                        
                        # Convert to probabilities
                        batch_probs = torch.sigmoid(batch_logits_squeezed)
                    
                    # Store results
                    for j, (logit, label, loss, prob, pos) in enumerate(zip(
                        batch_logits_squeezed.cpu(), batch_labels, batch_losses.cpu(), 
                        batch_probs.cpu(), batch_positions
                    )):
                        crop_logits.append(logit)
                        crop_labels.append(label)
                        crop_losses.append(loss)
                        crop_probs.append(prob.item())
                        crop_positions.append(pos)
                    
                    # Free memory
                    del batch_tensor, batch_targets_tensor, batch_logits_tensor, batch_losses, batch_probs
                
                if not crop_logits:
                    continue
                
                # Print individual patch probabilities
                print(f"\nTomogram: {path}")
                print(f"True label: {label}")
                print("Individual patch probabilities:")
                for j, (prob, pos, label) in enumerate(zip(crop_probs, crop_positions, crop_labels)):
                    print(f"  Patch {j:2d}: prob={prob:.4f}, position={pos}, is_positive={crop_labels[j]}")
                
                # Store individual patch losses (already calculated in batch processing above)
                
                # Use max over logits across crops for tomogram-level probability (for metrics only)
                crop_logits_tensor = torch.stack(crop_logits)  # (num_crops,)
                crop_logits_tensor = torch.nan_to_num(crop_logits_tensor, 
                                                    nan=-float('inf'), 
                                                    posinf=float('inf'), 
                                                    neginf=-float('inf'))
                max_logit = torch.max(crop_logits_tensor)
                max_logit_idx = torch.argmax(crop_logits_tensor).item()
                
                # Tomogram-level probability (for metrics only)
                tomogram_prob = torch.sigmoid(max_logit).item()
                
                print(f"Tomogram probability: {tomogram_prob:.4f} (from patch {max_logit_idx})")
                print(f"Max patch prob directly: {max(crop_probs):.4f}")
                avg_patch_loss = torch.stack(crop_losses).mean()
                print(f"Average patch loss for this volume: {avg_patch_loss:.4f}")
                
                
                # Store patch-level losses and tomogram-level predictions for metrics
                batch_logits.extend(crop_logits)  # All patch logits for loss calculation
                batch_targets.extend([torch.tensor(cl, dtype=torch.float32) for cl in crop_labels])  # All patch labels
                batch_patch_losses.append(crop_losses)  # Store patch losses for this volume
                
                # Also store volume-level info for metrics (we'll separate this later)
                batch_volume_logits.append(max_logit.unsqueeze(0))
                batch_volume_targets.append(torch.tensor(label, dtype=torch.float32))
            
            if not batch_logits:
                return None
                
            # Combine all patch predictions and losses
            all_logits = torch.stack(batch_logits)  # (total_patches,)
            all_targets = torch.stack(batch_targets)  # (total_patches,)
            
            # Use the pre-calculated patch losses and aggregate them
            all_patch_losses = []
            for vol_losses in batch_patch_losses:
                all_patch_losses.extend(vol_losses)
            loss = torch.stack(all_patch_losses).mean()  # Average loss across all patches
            
            # For metrics, we'll use tomogram-level predictions
            all_volume_logits = torch.stack(batch_volume_logits)  # (num_volumes, 1)
            all_volume_targets = torch.stack(batch_volume_targets)  # (num_volumes,)
            
        else:
            # Standard validation (for backward compatibility)
            x, y = batch
            logits = self(x)
            loss = self.compute_loss(logits, y)
            all_volume_logits = logits
            all_volume_targets = y
        
        # Store outputs for epoch-end metrics calculation (use tomogram-level for metrics)
        self.validation_step_outputs.append({
            'logits': all_volume_logits.detach().cpu(),
            'targets': all_volume_targets.detach().cpu(),
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
            
            # Handle edge cases for metrics calculation
            if len(np.unique(targets_np)) > 1 and len(np.unique(y_pred)) > 1:
                f1 = f1_score(targets_np, y_pred, average='binary', zero_division=0)
            else:
                f1 = 0.0
            acc = accuracy_score(targets_np, y_pred)
            
            self.log("train_f1", f1, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)

            # Balanced accuracy (sklearn handles class imbalance automatically)
            balanced_acc = balanced_accuracy_score(targets_np, y_pred)
            
            # Log balanced metrics
            self.log("train_balanced_acc", balanced_acc, on_epoch=True, prog_bar=True, sync_dist=True)

            # AUC calculation if both classes present
            if len(np.unique(targets_np)) > 1:
                auc = roc_auc_score(targets_np, probs)
                self.log("train_auc", auc, on_epoch=True, prog_bar=True, sync_dist=True)
            
            if np.sum(targets_np) > 0:  # Positive samples exist
                ap = average_precision_score(targets_np, probs)
                self.log("train_ap", ap, on_epoch=True, prog_bar=True, sync_dist=True)
            
            # Ensure metrics appear in W&B even if Lightning optimization skips some logs
            try:
                metrics = {
                    "train_f1": float(f1),
                    "train_acc": float(acc),
                    "train_balanced_acc": float(balanced_acc),
                    "epoch": int(self.current_epoch),
                    "step": int(self.global_step),
                }
                if 'auc' in locals():
                    metrics["train_auc"] = float(auc)
                if 'ap' in locals():
                    metrics["train_ap"] = float(ap)

                wandb_experiment = None
                # Try trainer.loggers (PL >= 2.x) first
                if hasattr(self.trainer, "loggers") and self.trainer.loggers is not None:
                    for lg in self.trainer.loggers:
                        if lg.__class__.__name__ == "WandbLogger" and hasattr(lg, "experiment"):
                            wandb_experiment = lg.experiment
                            break
                # Fallback to single logger
                if wandb_experiment is None and hasattr(self.trainer, "logger") and self.trainer.logger is not None:
                    lg = self.trainer.logger
                    if lg.__class__.__name__ == "WandbLogger" and hasattr(lg, "experiment"):
                        wandb_experiment = lg.experiment

                if wandb_experiment is not None:
                    wandb_experiment.log(metrics, step=self.global_step)
            except Exception as wandb_log_err:
                print(f"Warning: direct W&B logging failed: {wandb_log_err}")
                
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
            
            # Handle edge cases for metrics calculation
            if len(np.unique(targets_np)) > 1 and len(np.unique(y_pred)) > 1:
                f1 = f1_score(targets_np, y_pred, average='binary', zero_division=0)
            else:
                f1 = 0.0
            acc = accuracy_score(targets_np, y_pred)
            
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)

            # Balanced accuracy (sklearn handles class imbalance automatically)
            balanced_acc = balanced_accuracy_score(targets_np, y_pred)
            
            # Log balanced metrics
            self.log("val_balanced_acc", balanced_acc, prog_bar=True, sync_dist=True, logger=True)

            # AUC and AP
            if len(np.unique(targets_np)) > 1:
                auc = roc_auc_score(targets_np, probs)
            else:
                # Fallback when only one class present so checkpoint monitor key always exists
                auc = 0.5
            self.log("val_auc", auc, prog_bar=True, sync_dist=True)
            
            # AUC is inherently class-invariant, so weighted AUC = regular AUC
            
            if np.sum(targets_np) > 0:  # Positive samples exist
                ap = average_precision_score(targets_np, probs)
                self.log("val_ap", ap, prog_bar=True, sync_dist=True)
                
        except Exception as e:
            print(f"Warning: Could not calculate advanced metrics: {e}")
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)


def validation_collate_fn(batch):
    """Custom collate function for validation to handle dictionary returns."""
    if isinstance(batch[0], dict):
        # Dictionary batch for sliding window validation
        volumes = torch.stack([torch.from_numpy(item['volume']) for item in batch])
        masks = torch.stack([torch.from_numpy(item['mask']) for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        paths = [item['path'] for item in batch]
        
        return {
            'volume': volumes,
            'mask': masks,
            'label': labels,
            'path': paths
        }
    else:
        # Standard batch (tensors)
        return torch.utils.data.dataloader.default_collate(batch)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, fold: int, batch_size: int = 4,
                 use_torchio: bool = True, intensity_augmentation: bool = True,
                 spatial_augmentation: bool = True, crop_size: Tuple[int, int, int] = (32, 128, 128),
                 crops_per_volume: int = 4, overlap: float = 0.25, debug: bool = False,
                 positive_crop_ratio: float = 0.8):  # New parameter
        super().__init__()
        self.data_root = data_root
        self.fold = fold
        self.batch_size = batch_size
        self.pos_weight = 1.0
        self.use_torchio = use_torchio
        self.intensity_augmentation = intensity_augmentation
        self.spatial_augmentation = spatial_augmentation
        self.crop_size = crop_size
        self.crops_per_volume = crops_per_volume
        self.overlap = overlap
        self.debug = debug
        self.positive_crop_ratio = positive_crop_ratio
        
    def setup(self, stage=None):
        self.train_ds = VolumeDataset(
            self.data_root, 
            self.fold, 
            mode='train',
            use_torchio=self.use_torchio,
            intensity_augmentation=self.intensity_augmentation,
            spatial_augmentation=self.spatial_augmentation,
            crop_size=self.crop_size,
            crops_per_volume=self.crops_per_volume,
            overlap=self.overlap,
            debug=self.debug,
            positive_crop_ratio=self.positive_crop_ratio  # Pass the parameter
        )
        self.val_ds = VolumeDataset(
            self.data_root, 
            self.fold, 
            mode='val',
            use_torchio=False,
            intensity_augmentation=False,  # No augmentation for validation
            spatial_augmentation=False,    # No spatial transforms for validation
            crop_size=self.crop_size,
            crops_per_volume=1,  # Not used in validation
            overlap=self.overlap,
            debug=self.debug,
            positive_crop_ratio=0.5  # Not used in validation
        )
        
        print(f"Training samples: {len(self.train_ds)}")
        print(f"Validation samples: {len(self.val_ds)}")

        # Note: You might want to adjust pos_weight calculation since the 
        # training distribution will now be artificially balanced
        # The pos_weight should reflect the true class imbalance, not the sampling ratio
        if len(self.train_ds) > 0:
            # For pos_weight calculation, we should use the natural distribution
            # not the artificially balanced one from positive crop sampling
            pos_count = 0
            total_count = 0
            
            # Sample using random crops to get natural distribution
            sample_size = min(1000, len(self.train_ds.items))  # Sample from volumes, not crops
            indices = np.random.choice(len(self.train_ds.items), sample_size, replace=False)
            
            for idx in indices:
                path = self.train_ds.items[idx]
                with np.load(path) as npz:
                    mask = npz.get('mask', np.zeros_like(npz['volume']))
                # Check if volume contains any positive regions
                if np.any(mask > 0.01):
                    pos_count += 1
                total_count += 1
            
            neg_count = total_count - pos_count
            if pos_count > 0:
                self.pos_weight = float(neg_count) / float(pos_count)
            else:
                self.pos_weight = 1.0
                
            print(f"\nVolume-level label distribution (from sample of {total_count}):")
            print(f"  Positive volumes: {pos_count}")
            print(f"  Negative volumes: {neg_count}")
            print(f"  Computed pos_weight (volume-level): {self.pos_weight:.4f}")
            print(f"  Training will use {self.positive_crop_ratio:.1%} positive crops")
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, 
                         num_workers=4, pin_memory=True)
    
    def val_dataloader(self):
        val_batch_size = self.batch_size  
        return DataLoader(self.val_ds, batch_size=val_batch_size, num_workers=2, 
                         pin_memory=False, collate_fn=validation_collate_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="aneurysm_cubes_v2", help="Root directory with fold_0, fold_1, etc.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold (0-4)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21k_ft_in1k")
    # Volume cropping options
    parser.add_argument("--crop-size", nargs=3, type=int, default=[32, 128, 128], help="Crop size (D H W)")
    parser.add_argument("--crops-per-volume", type=int, default=1, help="Random crops per volume during training")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap for sliding window validation (0.0-1.0)")
    # TorchIO augmentation options
    parser.add_argument("--no-torchio", action="store_true", help="Disable TorchIO transforms")
    parser.add_argument("--no-intensity-aug", action="store_true", help="Disable intensity augmentations")
    parser.add_argument("--no-spatial-aug", action="store_true", help="Disable spatial augmentations")
    # Weights & Biases logging options
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="rsna-iad-binary")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--debug", action="store_true", help="Debug mode: use only 1% of the data for faster testing")
    args = parser.parse_args()
    
    dm = DataModule(
        args.data_root, 
        args.fold, 
        args.batch_size,
        use_torchio=not args.no_torchio,
        intensity_augmentation=not args.no_intensity_aug,
        spatial_augmentation=not args.no_spatial_aug,
        crop_size=tuple(args.crop_size),
        crops_per_volume=args.crops_per_volume,
        overlap=args.overlap,
        debug=args.debug
    )
    # Setup early to compute class weights
    dm.setup()
    model = LightningModel(model_name=args.model_name, lr=args.lr, pos_weight=dm.pos_weight)
    
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
                "task_type": "binary_aneurysm_detection_volume_crops",
                "crop_size": args.crop_size,
                "crops_per_volume": args.crops_per_volume,
                "overlap": args.overlap,
                "use_torchio": not args.no_torchio,
                "intensity_augmentation": not args.no_intensity_aug,
                "spatial_augmentation": not args.no_spatial_aug,
                "debug_mode": args.debug,
                "balanced_metrics": True,  # Indicate that balanced metrics are calculated
                "crop_level_balancing": True,  # Training uses balanced accuracy
                "tomogram_level_balancing": True,  # Validation uses balanced accuracy
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
        accumulate_grad_batches=1,  # Gradient accumulation to effectively double batch size
        check_val_every_n_epoch=5, 
    )
    
    trainer.fit(model, dm)
    
    # Print final results
    if checkpoint_callback.best_model_path:
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best AUC: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()


# Usage examples:
# python3 train_timm3d_binary.py --data-root volume_data --fold 0  # Default settings with weighted metrics
# python3 train_timm3d_binary.py --data-root volume_data --fold 0 --debug  # Debug mode with 1% of data
# python3 train_timm3d_binary.py --data-root volume_data --fold 0 --crop-size 32 128 128 --crops-per-volume 6 --overlap 0.5
# python3 train_timm3d_binary.py --data-root volume_data --fold 0 --no-torchio  # Disable TorchIO
# python3 train_timm3d_binary.py --data-root volume_data --fold 0 --no-intensity-aug  # Only spatial augmentation
# python3 train_timm3d_binary.py --data-root volume_data --fold 0 --no-spatial-aug  # Only intensity augmentation
