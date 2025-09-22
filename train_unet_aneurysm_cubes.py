import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from monai.networks.nets import UNet
from monai.networks.nets import DynUNet

from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    EnsureTyped,
    Activations,
    AsDiscrete,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropSamplesd
)
from monai.data import CacheDataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
import logging
import warnings
from pathlib import Path
import argparse
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class AneurysmCubesDataset(Dataset):
    def __init__(self, npz_files, transform=None, label_threshold=1e-6):
        self.npz_files = npz_files
        self.transform = transform
        self.label_threshold = label_threshold

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        data = np.load(self.npz_files[idx])
        
        # Convert uint8 volume back to float [0, 1] range
        volume = data['volume'].astype(np.float32) / 255.0
        mask = data['mask'].astype(np.float32)
        
        sample = {'image': volume, 'label': mask}

        if self.transform:
            sample = self.transform(sample)

        # Keep soft Gaussian labels; ensure float32 dtype
        if isinstance(sample, list):
            for s in sample:
                s['label'] = s['label'].astype(np.float32)
        else:
            sample['label'] = sample['label'].astype(np.float32)
        
        return sample

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dice_weight=1.0, alpha=1, gamma=2):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

class UNetModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, pos_weight: float = 3.0):
        super().__init__()
        self.save_hyperparameters()
        #self.model = UNet(
        #    spatial_dims=3,
        #    in_channels=1,
        #    out_channels=1,
        #    channels= (32, 64, 128, 256),
        #    strides=(2, 2, 1),
        #    num_res_units=2,
        #    dropout=0.1,
        #)

        self.model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2],
            filters=[64, 128, 256, 512], 
            dropout=0.1,
        )
        # BCE with logits for soft Gaussian labels; class imbalance via pos_weight
        self.register_buffer("bce_pos_weight", torch.tensor(float(pos_weight)))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
        #self.loss_fn = CombinedLoss(focal_weight=1.0, dice_weight=1.0, gamma=2)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        # Use 0.1 threshold for tiny lesion sensitivity
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.1)])
        self.post_label = Compose([AsDiscrete(threshold=0.1)]) 
        # For classification accuracy based on max confidence
        self.val_preds = []
        self.val_targets = []
        self.val_pred_probs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch["image"], batch["label"]
        y_hat = sliding_window_inference(x, (32, 128, 128), 4, self.model, overlap=0.25)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Compute Dice on binarized predictions/labels
        y_pred_list = [self.post_pred(b) for b in decollate_batch(y_hat)]
        y_list = [self.post_label(b) for b in decollate_batch(y)]
        self.dice_metric(y_pred=y_pred_list, y=y_list)

        # Compute classification accuracy based on max confidence
        max_prob = torch.sigmoid(y_hat).max().item()  # Use sigmoid for probability
        pred_prob = max_prob
        is_pred_pos = max_prob > 0.5
        is_gt_pos = (y > 0.001).any().item()
        self.val_preds.append(is_pred_pos)
        self.val_targets.append(is_gt_pos)
        self.val_pred_probs.append(pred_prob)

    def on_validation_epoch_end(self):
        dice_score = self.dice_metric.aggregate()
        self.log("val_dice", dice_score, prog_bar=True)
        self.dice_metric.reset()

        # Compute classification metrics
        if self.val_preds:
            tp = sum(p and t for p, t in zip(self.val_preds, self.val_targets))
            tn = sum(not p and not t for p, t in zip(self.val_preds, self.val_targets))
            fp = sum(p and not t for p, t in zip(self.val_preds, self.val_targets))
            fn = sum(not p and t for p, t in zip(self.val_preds, self.val_targets))
            
            overall_accuracy = (tp + tn) / len(self.val_preds)
            pos_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            neg_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            self.log("val_acc", overall_accuracy, prog_bar=True)
            self.log("val_pos_acc", pos_accuracy, prog_bar=True)
            self.log("val_neg_acc", neg_accuracy, prog_bar=True)
            
            # Compute AUC-ROC if there are both positive and negative samples
            if len(set(self.val_targets)) > 1:
                auc = roc_auc_score(self.val_targets, self.val_pred_probs)
                self.log("val_auc", auc, prog_bar=True)
            else:
                self.log("val_auc", 0.0, prog_bar=True)
        else:
            self.log("val_acc", 0.0, prog_bar=True)
            self.log("val_pos_acc", 0.0, prog_bar=True)
            self.log("val_neg_acc", 0.0, prog_bar=True)
            self.log("val_auc", 0.0, prog_bar=True)
        # Reset lists
        self.val_preds = []
        self.val_targets = []
        self.val_pred_probs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

def load_fold_files(cubes_dir: Path, val_fold: int = 0):
    """Load NPZ files from aneurysm cubes, using specified fold as validation."""
    cubes_dir = Path(cubes_dir)
    
    # Get all fold directories
    fold_dirs = [d for d in cubes_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')]
    
    train_files = []
    val_files = []
    
    for fold_dir in fold_dirs:
        fold_num = int(fold_dir.name.split('_')[1])
        npz_files = list(fold_dir.glob('*.npz'))
        
        if fold_num == val_fold:
            val_files.extend(npz_files)
            print(f"Validation fold {fold_num}: {len(npz_files)} files")
        else:
            train_files.extend(npz_files)
            print(f"Training fold {fold_num}: {len(npz_files)} files")
    
    return train_files, val_files

def main(args):
    set_determinism(seed=42)
    # Silence MONAI sampler warnings like "Num foregrounds 0..."
    logging.getLogger("monai").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r"Num foregrounds .* unable to generate class balanced samples")
    # Enable Tensor Core friendly matmul for speed on RTX 30xx
    try:
        torch.set_float16_matmul_precision('medium')
    except Exception:
        pass

    # Get label threshold from args (default 0.1 for aneurysm cubes)
    label_threshold = getattr(args, 'label_threshold', 0.1)

    # Load file paths using fold-based splitting
    train_files, val_files = load_fold_files(args.cubes_dir, args.val_fold)

    # Use small dataset if specified
    if getattr(args, 'small_dataset', False):
        train_files = train_files[:len(train_files)//10]
        val_files = val_files[:len(val_files)//10]
    
    # Identify positive volumes and optionally oversample them
    def has_foreground(npz_path: Path, thr: float) -> bool:
        try:
            d = np.load(npz_path)
            m = d['mask']
            return bool((m > thr).any())
        except Exception:
            return False

    pos_thr = float(getattr(args, 'pos_thr', 0.1))
    pos_files = [p for p in train_files if has_foreground(p, pos_thr)]
    neg_files = [p for p in train_files if p not in set(pos_files)]

    # Also compute positives for validation set
    val_pos_files = [p for p in val_files if has_foreground(p, pos_thr)]
    val_neg_files = [p for p in val_files if p not in set(val_pos_files)]

    print(f"Train vols: {len(train_files)} (pos={len(pos_files)}, neg={len(neg_files)}), Val vols: {len(val_files)} (pos={len(val_pos_files)}, neg={len(val_neg_files)}), pos_thr={pos_thr}")

    # Optionally keep only positive volumes for train/val
    if getattr(args, 'only_positive', False):
        print(f"--only-positive: keeping {len(pos_files)} positive training volumes (out of {len(train_files)})")
        train_files = pos_files
        neg_files = []

    if getattr(args, 'only_positive_val', False):
        print(f"--only-positive-val: keeping {len(val_pos_files)} positive validation volumes (out of {len(val_files)})")
        val_files = val_pos_files
        val_neg_files = []

    # Transforms - adapted for aneurysm cubes (32x128x128)
    train_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # rotate
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1,2)),
        # flip
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # Random spatial cropping - simpler and more reliable
        RandSpatialCropSamplesd(
            keys=['image', 'label'],
            roi_size=(32, 128, 128),  # Smaller crops to fit in 32x128x128 volumes
            num_samples=8
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"]),
    ])

    # Datasets
    train_ds = AneurysmCubesDataset(train_files, transform=train_transforms, label_threshold=label_threshold)
    val_ds = AneurysmCubesDataset(val_files, transform=val_transforms, label_threshold=label_threshold)

    # Optional: visualize transformed training samples
    if getattr(args, "viz_samples", 0) and args.viz_samples > 0:
        out_dir = Path(getattr(args, "viz_dir", "outputs/aneurysm_cubes_viz"))
        out_dir.mkdir(parents=True, exist_ok=True)
        viz_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=list_data_collate,
        )
        sample_count = 0
        for bidx, batch in enumerate(viz_loader):
            x = batch["image"]
            y = batch["label"]
            # expected shapes: 3D -> (B,C,D,H,W), 2D -> (B,C,H,W)
            B = x.shape[0]
            for i in range(B):
                if sample_count >= args.viz_samples:
                    break
                xi = x[i].detach().cpu().numpy()
                yi = y[i].detach().cpu().numpy()
                if xi.ndim == 4:  # C,D,H,W
                    img_vol = xi[0]
                    lbl_vol = yi[0] if yi.ndim == 4 else yi
                    for z in range(img_vol.shape[0]):
                        img2d = img_vol[z]
                        lbl2d = lbl_vol[z]
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax.imshow(img2d, cmap="gray")
                        thr = 0.5  # Use 0.5 since labels are binarized to 0/1
                        mask_vis = np.ma.masked_where(lbl2d <= thr, lbl2d)
                        ax.imshow(mask_vis, cmap="turbo", alpha=0.65)
                        # add a bright contour for extra visibility
                        bin_lbl = (lbl2d > thr).astype(np.uint8)
                        if bin_lbl.any():
                            ax.contour(bin_lbl, levels=[0.5], colors="lime", linewidths=0.8)
                            # print location stats for positives on this slice
                            pos_idx = np.argwhere(lbl2d > thr)
                            if pos_idx.size > 0:
                                y_max, x_max = np.unravel_index(lbl2d.argmax(), lbl2d.shape)
                                print(f"[viz] sample {sample_count} z={z}, max@(y={y_max}, x={x_max})")
                        else:
                            print(f"[viz] sample {sample_count} z={z}: no label > {thr}")
                        ax.set_title(f"sample {sample_count} / z={z}")
                        ax.axis("off")
                        fig.savefig(out_dir / f"train_transform_{sample_count:03d}_z{z:03d}.png", bbox_inches="tight", dpi=150)
                        plt.close(fig)
                sample_count += 1
            if sample_count >= args.viz_samples:
                break
        
        print(f"Visualization complete! Saved {sample_count} samples to {out_dir}")
        return  # Exit early after visualization

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=2,  # Smaller batch size for 3D data
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=False,  
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=list_data_collate,
        pin_memory=False,  
    )

    # Model
    model = UNetModel(lr=args.lr, pos_weight=args.pos_weight)

    # Setup W&B logger if enabled
    logger = None
    if getattr(args, 'wandb', False) and WANDB_AVAILABLE:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.wandb_name if args.wandb_name else None,
            group=args.wandb_group if args.wandb_group else None,
            save_dir='lightning_logs'
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10,
        precision="16-mixed",
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cubes-dir', type=str, default='aneurysm_cubes_v2', help='Aneurysm cubes directory')
    parser.add_argument('--val-fold', type=int, default=0, help='Fold to use for validation (0-4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--viz-samples', type=int, default=0, help='Save this many transformed training samples and exit after saving')
    parser.add_argument('--viz-dir', type=str, default='outputs/aneurysm_cubes_viz', help='Directory to save transform visualizations')
    parser.add_argument('--pos-weight', type=float, default=1024.0, help='Positive weight for BCE loss (higher for foreground)')
    parser.add_argument('--small-dataset', action='store_true', help='Use 10% of the dataset for quick testing')
    parser.add_argument('--wandb', default=True, action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='unet_aneurysm_cubes', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default='', help='W&B entity (team/user)')
    parser.add_argument('--wandb-name', type=str, default='', help='W&B run name')
    parser.add_argument('--wandb-group', type=str, default='', help='W&B group')
    parser.add_argument('--label-threshold', type=float, default=0.1, help='Threshold for binarizing labels to hard (0 or 1)')
    parser.add_argument('--pos-thr', type=float, default=0.1, help='Threshold for identifying positive volumes')
    parser.add_argument('--only-positive', default=False, action='store_true', help='Keep only positive training volumes (mask > pos-thr)')
    parser.add_argument('--only-positive-val', default=False, action='store_true', help='Keep only positive validation volumes (mask > pos-thr)')
    args = parser.parse_args()
    main(args)

# Example usage:
# python3 train_unet_aneurysm_cubes.py --cubes-dir aneurysm_cubes --val-fold 0 --lr 1e-4 --epochs 500 --small-dataset
