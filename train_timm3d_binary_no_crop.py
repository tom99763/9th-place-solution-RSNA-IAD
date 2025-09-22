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

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)




class VolumeDataset(Dataset):
    def __init__(self, data_dir: str, fold: int, mode: str = 'train', debug: bool = False):
        """
        Load full volume data from fold directories.
        Args:
            data_dir: Root directory containing fold_0, fold_1, etc.
            fold: Fold number to use as validation (0-4)
            mode: 'train' or 'val'
            debug: If True, use only 1% of the data for faster testing
        """
        self.items = []
        self.mode = mode
        self.debug = debug
        
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
                # Use deterministic sampling based on fold and mode for reproducibility
                # This ensures same samples are selected even if setup() is called multiple times
                seed_offset = hash(f"{fold}_{mode}") % 1000
                rng = np.random.RandomState(42 + seed_offset)
                sampled_indices = rng.choice(len(self.items), sample_size, replace=False)
                self.items = [self.items[i] for i in sampled_indices]
                print(f"DEBUG MODE: Sampled {len(self.items)} out of {original_count} samples (1%)")
        
        print(f"Found {len(self.items)} samples for {mode} (folds: {train_folds})")
        print(f"Full volume size: (128, 384, 384), Mode: {mode}")
        if self.debug:
            print(f"DEBUG MODE: Using reduced dataset for faster testing")
    
    def __len__(self):
        # Return number of volumes for both train and validation
        return len(self.items)
    
    def __getitem__(self, idx):
        path = self.items[idx]
        
        with np.load(path) as npz:
            volume = npz['volume']  # (128, 384, 384) full volume
            mask = npz.get('mask', None)  # Mask if available
            
        # Normalize to [0, 1] more efficiently
        volume_normalized = (volume / 255.0).astype(np.float32)
        
        # Determine label from mask: positive if any mask value > 0.01
        if mask is not None:
            label = 1 if np.any(mask > 0.01) else 0
        else:
            # Infer from filename if mask not available
            label = 1 if 'pos_' in path else 0
        
        # Convert to tensor (no augmentations for speed)
        x = torch.from_numpy(volume_normalized).unsqueeze(0)  # Add channel dim
        
        y = torch.tensor(label, dtype=torch.float32)
        return x, y
    



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
            drop_path_rate=0.05,  # Reduced for speed
            drop_rate=0.05,       # Reduced for speed
        )
        
        # Disable gradient checkpointing for speed (trades memory for speed)
        if hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(False)
        
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
        # Standard validation with full volumes
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
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
            
            # Essential metrics only for speed
            if len(np.unique(targets_np)) > 1:
                auc = roc_auc_score(targets_np, probs)
            else:
                auc = 0.5
            self.log("val_auc", auc, prog_bar=True, sync_dist=True)
            
            acc = accuracy_score(targets_np, y_pred)
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)
                
        except Exception as e:
            print(f"Warning: Could not calculate advanced metrics: {e}")
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)




class DataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, fold: int, batch_size: int = 4, debug: bool = False):
        super().__init__()
        self.data_root = data_root
        self.fold = fold
        self.batch_size = batch_size
        self.pos_weight = 1.0
        self.debug = debug
        
    def setup(self, stage=None):
        # Prevent multiple setup calls from re-creating datasets
        if hasattr(self, 'train_ds') and hasattr(self, 'val_ds'):
            print(f"Setup already called - Training samples: {len(self.train_ds)}, Validation samples: {len(self.val_ds)}")
            return
            
        self.train_ds = VolumeDataset(
            self.data_root, 
            self.fold, 
            mode='train',
            debug=self.debug
        )
        self.val_ds = VolumeDataset(
            self.data_root, 
            self.fold, 
            mode='val',
            debug=self.debug
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
                         num_workers=8, persistent_workers=True, pin_memory=True, 
                         prefetch_factor=2)
    
    def val_dataloader(self):
        val_batch_size = self.batch_size     
       
        return DataLoader(self.val_ds, batch_size=val_batch_size, num_workers=8, 
                         persistent_workers=True, pin_memory=True, prefetch_factor=2)


def main():
    # Set memory allocation strategy for better GPU memory management
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="aneurysm_cubes_v2", help="Root directory with fold_0, fold_1, etc.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold (0-4)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21k_ft_in1k")
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
                "task_type": "binary_aneurysm_detection_full_volumes",
                "volume_size_d": 128,
                "volume_size_h": 384,
                "volume_size_w": 384,
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
        check_val_every_n_epoch=5,
        enable_progress_bar=True,
        log_every_n_steps=10,  # Reduce logging frequency
        enable_model_summary=True,
        accumulate_grad_batches=1,  # No gradient accumulation for now
        deterministic=False,  # Allow non-deterministic operations for speed
    )
    
    trainer.fit(model, dm)
    
    # Print final results
    if checkpoint_callback.best_model_path:
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best AUC: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()


# Usage examples:
# python3 train_timm3d_binary_no_crop.py --data-root volume_data --fold 0  # Default settings with optimized memory usage
# python3 train_timm3d_binary_no_crop.py --data-root volume_data --fold 0 --batch-size 1  # Even smaller batch for limited GPU memory
# python3 train_timm3d_binary_no_crop.py --data-root volume_data --fold 0 --debug  # Debug mode with 1% of data
# python3 train_timm3d_binary_no_crop.py --data-root volume_data --fold 0 --no-torchio  # Disable TorchIO
# python3 train_timm3d_binary_no_crop.py --data-root volume_data --fold 0 --no-intensity-aug  # Only spatial augmentation
# python3 train_timm3d_binary_no_crop.py --data-root volume_data --fold 0 --no-spatial-aug  # Only intensity augmentation

