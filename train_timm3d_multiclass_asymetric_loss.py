"""
Multilabel aneurysm detection with PyTorch Lightning.
Uses native tomogram size (32, 384, 384) without interpolation.
Predicts 14 anatomical locations independently with separated metrics.
Now includes Asymmetric Loss for Multilabel (ASL) for better imbalanced multilabel handling.
"""

import argparse
import os
import csv
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import timm_3d
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

torch.set_float32_matmul_precision('medium')

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Define the 14 target columns
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

# Separate indices for localization vs classification
LOCALIZATION_INDICES = list(range(13))  # First 13 columns
CLASSIFICATION_INDEX = 13  # Last column (Aneurysm Present)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    
    Paper: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    https://arxiv.org/abs/2009.14119
    https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    This loss addresses both class imbalance and multilabel nature by:
    1. Using different focusing parameters for positive/negative samples
    2. Probability shifting to reduce easy negatives
    3. Hard negative mining through focusing
    """
    
    def __init__(self, 
                 gamma_neg: float = 4, # 4 original
                 gamma_pos: float = 1, # 1 original
                 clip: float = 0.05, 
                 eps: float = 1e-6, # modified because of NaN
                 disable_torch_grad_focal_loss: bool = True):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples (higher = focus more on hard negatives)
            gamma_pos: Focusing parameter for positive samples 
            clip: Probability margin to shift easy negatives (reduces their contribution)
            eps: Small constant for numerical stability
            disable_torch_grad_focal_loss: Disable gradients for focal loss computation (memory efficient)
        """
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

def build_uid_to_multilabel(csv_path: str) -> Dict[str, List[int]]:
    """Map SeriesInstanceUID -> 14-element binary label vector."""
    uid2labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row['SeriesInstanceUID']
            labels = []
            for col in LABEL_COLS:
                labels.append(int(row[col]))
            uid2labels[uid] = labels
    return uid2labels


class VolumeDataset(Dataset):
    def __init__(self, split_dir: str, uid2labels: Dict[str, List[int]]):
        items = []
        for fn in os.listdir(split_dir):
            if fn.endswith('.npz'):
                uid = fn[:-4]
                if uid in uid2labels:
                    items.append((os.path.join(split_dir, fn), uid2labels[uid]))
        self.items = sorted(items)
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, labels = self.items[idx]
        with np.load(path) as npz:
            volume = npz['volume']  # (32, 384, 384)

        x = torch.from_numpy(volume.astype('float32') / 255.0).unsqueeze(0) 
        y = torch.tensor(labels, dtype=torch.float32)  # 14-element vector
        return x, y


class LightningModel(pl.LightningModule):
    def __init__(self, 
                 model_name: str = "resnet18.a1_in1k", 
                 lr: float = 1e-3, 
                 loss_type: str = "bce",  # "bce" or "asl"
                 pos_weight: torch.Tensor | None = None,
                 # ASL parameters
                 gamma_neg: float = 4,
                 gamma_pos: float = 1,
                 clip: float = 0.05):
        super().__init__()
        self.save_hyperparameters()
        set_seed()

        self.model = timm_3d.create_model(
            model_name, 
            pretrained=True,
            num_classes=14,  # 14 outputs for multilabel
            global_pool="avg",
            in_chans=1,
            drop_path_rate=0.2,
            drop_rate=0.2,
        )
        
        # Loss function setup
        self.loss_type = loss_type
        if loss_type == "asl":
            self.criterion = AsymmetricLoss(
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip=clip
            )
        else:  # BCE with logits
            # Register pos_weight as buffer so it moves with devices and is saved in checkpoints
            if pos_weight is None:
                pos_weight = torch.ones(len(LABEL_COLS), dtype=torch.float32)
            pos_weight = pos_weight.to(dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
            self.loss_reduction = "mean"
        
        # For metrics tracking
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "asl":
            return self.criterion(logits, targets)
        else:
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction=self.loss_reduction
            )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Macro-averaged F1 scores (threshold @ 0.5)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            y_true = y.detach().cpu().numpy().astype(int)
            macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
            macro_f1_loc = f1_score(
                y_true[:, LOCALIZATION_INDICES], preds[:, LOCALIZATION_INDICES],
                average='macro', zero_division=0
            )
            cls_f1 = f1_score(
                y_true[:, CLASSIFICATION_INDEX], preds[:, CLASSIFICATION_INDEX],
                average='binary', zero_division=0
            )
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_macro_f1", macro_f1, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_loc_macro_f1", macro_f1_loc, on_step=False, on_epoch=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_cls_f1", cls_f1, on_step=False, on_epoch=True, 
                batch_size=x.size(0), sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Store outputs for epoch-end metrics calculation
        self.validation_step_outputs.append({
            'logits': logits.detach().cpu(),
            'targets': y.detach().cpu(),
            'loss': loss.detach().cpu(),
        })
        
        # Log basic metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
    
    def on_validation_epoch_end(self):
        # Calculate additional metrics at epoch end
        if not self.validation_step_outputs:
            return
            
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Convert to numpy
        probs = torch.sigmoid(all_logits).numpy()
        targets_np = all_targets.numpy()
        
        # Calculate separated metrics
        try:
            # Macro-averaged F1s at threshold 0.5
            y_pred = (probs > 0.5).astype(int)
            macro_f1 = f1_score(targets_np, y_pred, average='macro', zero_division=0)
            macro_f1_loc = f1_score(targets_np[:, LOCALIZATION_INDICES], y_pred[:, LOCALIZATION_INDICES],
                                    average='macro', zero_division=0)
            cls_f1 = f1_score(targets_np[:, CLASSIFICATION_INDEX], y_pred[:, CLASSIFICATION_INDEX],
                               average='binary', zero_division=0)
            self.log("val_macro_f1", macro_f1, prog_bar=True, sync_dist=True)
            self.log("val_loc_macro_f1", macro_f1_loc, prog_bar=True, sync_dist=True)
            self.log("val_cls_f1", cls_f1, prog_bar=True, sync_dist=True)

            # LOCALIZATION METRICS (first 13 columns)
            aucs_loc = []
            aps_loc = []
            for i in LOCALIZATION_INDICES:
                if len(np.unique(targets_np[:, i])) > 1:  # Both classes present
                    auc = roc_auc_score(targets_np[:, i], probs[:, i])
                    aucs_loc.append(auc)
                
                if np.sum(targets_np[:, i]) > 0:  # Positive samples exist
                    ap = average_precision_score(targets_np[:, i], probs[:, i])
                    aps_loc.append(ap)
            
            if aucs_loc:
                macro_auc_loc = np.mean(aucs_loc)
                self.log("val_loc_auc", macro_auc_loc, prog_bar=True, sync_dist=True)
                
            if aps_loc:
                macro_ap_loc = np.mean(aps_loc)
                self.log("val_loc_ap", macro_ap_loc, prog_bar=True, sync_dist=True)
            
            # CLASSIFICATION METRICS (aneurysm present - column 13)
            cls_auc = None
            if len(np.unique(targets_np[:, CLASSIFICATION_INDEX])) > 1:
                cls_auc = roc_auc_score(targets_np[:, CLASSIFICATION_INDEX], probs[:, CLASSIFICATION_INDEX])
                self.log("val_cls_auc", cls_auc, prog_bar=True, sync_dist=True)
            
            if np.sum(targets_np[:, CLASSIFICATION_INDEX]) > 0:
                cls_ap = average_precision_score(targets_np[:, CLASSIFICATION_INDEX], probs[:, CLASSIFICATION_INDEX])
                self.log("val_cls_ap", cls_ap, prog_bar=True, sync_dist=True)

            # Kaggle metric only if both components are available
            if aucs_loc and cls_auc is not None:
                kaggle_metric = (np.mean(aucs_loc) + cls_auc) / 2
                self.log("val_kaggle_metric", kaggle_metric, prog_bar=True, sync_dist=True)
        except Exception as e:
            print(f"Warning: Could not calculate advanced metrics: {e}")
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, csv_path: str, batch_size: int = 2):
        super().__init__()
        self.data_root = data_root
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.pos_weight: torch.Tensor | None = None
        
    def setup(self, stage=None):
        uid2labels = build_uid_to_multilabel(self.csv_path)
        
        self.train_ds = VolumeDataset(os.path.join(self.data_root, 'train'), uid2labels)
        self.val_ds = VolumeDataset(os.path.join(self.data_root, 'val'), uid2labels)
        
        print(f"Training samples: {len(self.train_ds)}")
        print(f"Validation samples: {len(self.val_ds)}")
        
        # Compute pos_weight from the full training label distribution
        if hasattr(self, 'train_ds') and len(self.train_ds) > 0:
            all_labels = np.array([labels for _, labels in self.train_ds.items], dtype=np.int64)

            pos_counts = np.sum(all_labels, axis=0)
            neg_counts = len(all_labels) - pos_counts
            
            # Calculate proper pos_weights for BCE (neg/pos ratio)
            #pos_weights = [float(neg) / float(max(pos, 1)) for pos, neg in zip(pos_counts, neg_counts)]
            pos_weights = [1 for pos, neg in zip(pos_counts, neg_counts)]

            self.pos_weight = torch.tensor(pos_weights, dtype=torch.float32)

            # Print label distribution
            print("\nLabel distribution (positive samples):")
            print("LOCALIZATION LABELS:")
            for i in LOCALIZATION_INDICES:
                print(f"  {LABEL_COLS[i]}: {int(pos_counts[i])}")
            print("CLASSIFICATION LABEL:")
            print(f"  {LABEL_COLS[CLASSIFICATION_INDEX]}: {int(pos_counts[CLASSIFICATION_INDEX])}")
            print("\nComputed pos_weight for BCE:")
            for name, w in zip(LABEL_COLS, self.pos_weight.tolist()):
                print(f"  {name}: {w:.4f}")
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, 
                         num_workers=2, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2, 
                         pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/sersasj/RSNA-IAD-Codebase/data/unet_dataset_v2")
    parser.add_argument("--csv-path", default="/home/sersasj/RSNA-IAD-Codebase/data/train_df.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21k_ft_in1k")
    
    # Loss function selection
    parser.add_argument("--loss-type", choices=["bce", "asl"], default="asl",
                       help="Loss function: bce (binary cross entropy) or asl (asymmetric loss)")
    
    # ASL hyperparameters
    parser.add_argument("--gamma-neg", type=float, default=4,
                       help="ASL negative focusing parameter (higher = focus more on hard negatives)")
    parser.add_argument("--gamma-pos", type=float, default=1,
                       help="ASL positive focusing parameter")  
    parser.add_argument("--clip", type=float, default=0.05,
                       help="ASL probability margin for shifting easy negatives")
    
    # Weights & Biases logging options
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="rsna-iad-multilabel")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    
    dm = DataModule(args.data_root, args.csv_path, args.batch_size)
    # Setup early to compute class weights
    dm.setup()
    
    model = LightningModel(
        model_name=args.model_name, 
        lr=args.lr, 
        loss_type=args.loss_type,
        pos_weight=dm.pos_weight if args.loss_type == "bce" else None,
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.clip
    )
    
    print(f"\nUsing {args.loss_type.upper()} loss function")
    if args.loss_type == "asl":
        print(f"ASL Parameters: gamma_neg={args.gamma_neg}, gamma_pos={args.gamma_pos}, clip={args.clip}")
    
    # Configure Weights & Biases logger if enabled
    logger = None
    if args.wandb != "disabled":
        if args.wandb == "offline":
            os.environ["WANDB_MODE"] = "offline"
        try:
            logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                log_model=True,
            )
            # Log CLI and model hyperparameters
            logger.log_hyperparams({
                "data_root": args.data_root,
                "csv_path": args.csv_path,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "model_name": args.model_name,
                "loss_type": args.loss_type,
                "gamma_neg": args.gamma_neg if args.loss_type == "asl" else None,
                "gamma_pos": args.gamma_pos if args.loss_type == "asl" else None,
                "clip": args.clip if args.loss_type == "asl" else None,
                "num_labels": len(LABEL_COLS),
                "num_loc_labels": len(LOCALIZATION_INDICES),
                "task_type": "multilabel_binary_separated",
            })
            # Log gradients/parameters periodically
            logger.watch(model, log="gradients", log_freq=200)
        except Exception as e:
            print(f"WandbLogger initialization failed: {e}. Continuing without W&B.")
            logger = None

    # Use kaggle metric for model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_kaggle_metric", 
        mode="max",
        save_top_k=2,
        filename='best-{epoch}-{val_kaggle_metric:.3f}'
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=logger,
        precision="32",
        gradient_clip_val=1.0,
    )
    
    trainer.fit(model, dm)
    
    # Print final results
    if checkpoint_callback.best_model_path:
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best Kaggle metric: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()