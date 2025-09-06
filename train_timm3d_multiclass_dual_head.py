"""
Multilabel aneurysm detection with PyTorch Lightning.
Uses native tomogram size (32, 384, 384) without interpolation.
Predicts 14 anatomical locations independently with separated metrics.
Fixed to properly use 2 separate MLP heads for localization vs classification.
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
    def __init__(self, model_name: str = "resnet18.a1_in1k", lr: float = 1e-3, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.save_hyperparameters()
        set_seed()

        # Feature extractor backbone
        self.backbone = timm_3d.create_model(
            model_name, 
            pretrained=True,
            in_chans=1,
            drop_path_rate=0.2,
            drop_rate=0.2,
            features_only=True
        )
        
        # Get the number of output channels from the backbone
        feature_dim = self.backbone.feature_info[-1]['num_chs']

        # Separate MLP heads for localization and classification
        self.localization_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(LOCALIZATION_INDICES))  # 13 outputs for anatomical locations
        )
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # 1 output for aneurysm presence
        )
         
        # Store pos_weight for loss computation
        self.register_buffer('pos_weight_loc', pos_weight[:13] if pos_weight is not None else None)
        self.register_buffer('pos_weight_cls', pos_weight[13:14] if pos_weight is not None else None)
        
        # For metrics tracking
        self.validation_step_outputs = []
        
    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)[-1]  # Use the last feature map
        
        # Pass features through both heads
        localization_logits = self.localization_head(features)  # Shape: (batch_size, 13)
        classification_logits = self.classification_head(features)  # Shape: (batch_size, 1)
        
        return localization_logits, classification_logits

    def compute_loss(self, localization_logits, classification_logits, targets) -> torch.Tensor:
        # Separate losses for localization and classification
        loss_localization = F.binary_cross_entropy_with_logits(
            localization_logits, 
            targets[:, LOCALIZATION_INDICES], 
            pos_weight=self.pos_weight_loc,
            reduction="mean"
        )
        
        loss_classification = F.binary_cross_entropy_with_logits(
            classification_logits.squeeze(-1),  # Remove last dimension to match target
            targets[:, CLASSIFICATION_INDEX], 
            pos_weight=self.pos_weight_cls,
            reduction="mean"
        )
        
        # You can weight these losses differently if needed
        total_loss = loss_localization + loss_classification
        return total_loss, loss_localization, loss_classification

    def training_step(self, batch, batch_idx):
        x, y = batch
        localization_logits, classification_logits = self(x)
        total_loss, loc_loss, cls_loss = self.compute_loss(localization_logits, classification_logits, y)

        # Combine logits for metric computation
        combined_logits = torch.cat([localization_logits, classification_logits], dim=1)  # (batch_size, 14)

        # Macro-averaged F1 scores (threshold @ 0.5)
        with torch.no_grad():
            preds = (torch.sigmoid(combined_logits) > 0.5).cpu().numpy().astype(int)
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
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_loc_loss", loc_loss, on_step=False, on_epoch=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_cls_loss", cls_loss, on_step=False, on_epoch=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_macro_f1", macro_f1, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_loc_macro_f1", macro_f1_loc, on_step=False, on_epoch=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("train_cls_f1", cls_f1, on_step=False, on_epoch=True, 
                batch_size=x.size(0), sync_dist=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        localization_logits, classification_logits = self(x)
        total_loss, loc_loss, cls_loss = self.compute_loss(localization_logits, classification_logits, y)
        
        # Combine logits for storage
        combined_logits = torch.cat([localization_logits, classification_logits], dim=1)
        
        # Store outputs for epoch-end metrics calculation
        self.validation_step_outputs.append({
            'logits': combined_logits.detach().cpu(),
            'targets': y.detach().cpu(),
            'loss': total_loss.detach().cpu(),
            'loc_loss': loc_loss.detach().cpu(),
            'cls_loss': cls_loss.detach().cpu(),
        })
        
        # Log basic metrics
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("val_loc_loss", loc_loss, on_step=False, on_epoch=True, 
                batch_size=x.size(0), sync_dist=True)
        self.log("val_cls_loss", cls_loss, on_step=False, on_epoch=True, 
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
        
        # Compute pos_weight from the full training label distribution without loading volumes
        if hasattr(self, 'train_ds') and len(self.train_ds) > 0:
            # Extract labels directly from train_ds.items (path, labels)
            all_labels = np.array([labels for _, labels in self.train_ds.items], dtype=np.int64)

            # Positive samples per class
            pos_counts = np.sum(all_labels, axis=0)
            neg_counts = len(all_labels) - pos_counts
            # Avoid division by zero by using max(pos_count, 1)
            #pos_weights = [float(neg) / float(max(pos, 1)) for pos, neg in zip(pos_counts, neg_counts)]
            # Uncomment line below if you want to use equal weighting instead:
            pos_weights = [1 for pos, neg in zip(pos_counts, neg_counts)]
            self.pos_weight = torch.tensor(pos_weights, dtype=torch.float32)

            # Print label distribution
            print("\nLabel distribution (positive samples):")
            print("LOCALIZATION LABELS:")
            for i in LOCALIZATION_INDICES:
                print(f"  {LABEL_COLS[i]}: {int(pos_counts[i])}")
            print("CLASSIFICATION LABEL:")
            print(f"  {LABEL_COLS[CLASSIFICATION_INDEX]}: {int(pos_counts[CLASSIFICATION_INDEX])}")
            print("\nComputed pos_weight:")
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
    parser.add_argument("--data-root", default="/home/sersasj/RSNA-IAD-Codebase/data/unet_dataset")
    parser.add_argument("--csv-path", default="/home/sersasj/RSNA-IAD-Codebase/data/train_df.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21k_ft_in1k")
    # Weights & Biases logging options
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="rsna-iad-multilabel")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    
    dm = DataModule(args.data_root, args.csv_path, args.batch_size)
    # Setup early to compute class weights
    dm.setup()
    model = LightningModel(model_name=args.model_name, lr=args.lr, pos_weight=dm.pos_weight)
    
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
                "num_labels": len(LABEL_COLS),
                "num_loc_labels": len(LOCALIZATION_INDICES),
                "task_type": "multilabel_binary_separated",
                "architecture": "dual_mlp_heads",
            })
            # Log gradients/parameters periodically
            logger.watch(model, log="gradients", log_freq=200)
        except Exception as e:
            print(f"WandbLogger initialization failed: {e}. Continuing without W&B.")
            logger = None

    # Use classification AUC for model checkpointing (most important metric)
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
        precision="16-mixed",  # Use mixed precision for memory efficiency
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )
    
    trainer.fit(model, dm)
    
    # Print final results
    if checkpoint_callback.best_model_path:
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best kaggle metric: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()