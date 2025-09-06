"""
Multilabel aneurysm detection with PyTorch Lightning.
Uses native tomogram size (32, 384, 384) without interpolation.
Predicts 14 anatomical locations independently with separated metrics.
"""

import argparse
import os
import csv
import sys
from pathlib import Path
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
    def __init__(self, model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k", lr: float = 1e-3, pos_weight: torch.Tensor | None = None,
                 medicalnet_root: str | None = None, pretrained_path: str | None = None):
        super().__init__()
        self.save_hyperparameters()
        set_seed()

        self.model = self._build_backbone(model_name, medicalnet_root, pretrained_path)
        # Register pos_weight as buffer so it moves with devices and is saved in checkpoints
        if pos_weight is None:
            pos_weight = torch.ones(len(LABEL_COLS), dtype=torch.float32)
        # Ensure correct dtype
        pos_weight = pos_weight.to(dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)
        
        # We'll compute loss with functional API to ensure correct device for pos_weight
        self.loss_reduction = "mean"
        
        # For metrics tracking
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)

    def _build_backbone(self, model_name: str, medicalnet_root: str | None, pretrained_path: str | None) -> nn.Module:
        """Create the backbone. Supports timm_3d and MedicalNet's 3D ResNet-18.

        model_name:
          - Any timm_3d model name (default path)
          - 'medicalnet_resnet18' to use MedicalNet 3D ResNet-18 with optional pretrained weights
        """
        num_classes = len(LABEL_COLS)
        if model_name.lower() != "medicalnet_resnet18":
            # Default: use timm_3d
            return timm_3d.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
                global_pool="avg",
                in_chans=1,
                drop_path_rate=0.2,
                drop_rate=0.2,
            )

        # MedicalNet path
        if medicalnet_root is None:
            # Assume folder exists in repo root by default
            medicalnet_root = str(Path(__file__).resolve().parent / "MedicalNet-master")
        medicalnet_root_path = Path(medicalnet_root)
        assert medicalnet_root_path.exists(), f"MedicalNet root not found: {medicalnet_root_path}"

        # Find the actual package root that contains models/resnet.py
        def find_medicalnet_pkg_root(root: Path) -> Path:
            candidates = [root, root / "MedicalNet-master", root.parent / "MedicalNet-master"]
            for c in candidates:
                if (c / "models" / "resnet.py").exists():
                    return c
            # Fallback: recursive search
            for p in root.rglob("resnet.py"):
                if p.parent.name == "models":
                    return p.parent.parent
            return root

        pkg_root = find_medicalnet_pkg_root(medicalnet_root_path)
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))

        try:
            # Import MedicalNet resnet definitions
            from models import resnet as medicalnet_resnet  # type: ignore
        except Exception as e:
            raise ImportError(f"Failed to import MedicalNet models from {pkg_root}: {e}")

        # Build MedicalNet ResNet-18 backbone (expects sample input sizes)
        sample_D, sample_H, sample_W = 32, 384, 384
        backbone = medicalnet_resnet.resnet18(
            sample_input_D=sample_D,
            sample_input_H=sample_H,
            sample_input_W=sample_W,
            num_seg_classes=max(1, num_classes),
            shortcut_type='B',
        )

        # Optionally load pretrained weights
        if pretrained_path is None:
            # default to repo pretrain folder
            default_path = Path(__file__).resolve().parent / "pretrain" / "resnet_18_23dataset.pth"
            pretrained_path = str(default_path)
        pretrain_path = Path(pretrained_path)
        if pretrain_path.exists():
            print(pretrain_path)
            try:
                state = torch.load(pretrain_path, map_location="cpu")
                # Some checkpoints wrap state dict
                state_dict = state.get("state_dict", state)
                # Remove possible 'module.' prefixes
                new_state = {}
                for k, v in state_dict.items():
                    nk = k
                    if nk.startswith("module."):
                        nk = nk[len("module."):]
                    # In some MedicalNet checkpoints, final classifier may be named 'fc' or 'last_linear'
                    new_state[nk] = v
                # Load into backbone; ignore segmentation/classifier head mismatches
                missing, unexpected = backbone.load_state_dict(new_state, strict=False)
                if unexpected:
                    print(f"Warning: unexpected keys in pretrained weights: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
                if missing:
                    print(f"Info: missing keys when loading pretrained weights: {missing[:5]}{'...' if len(missing)>5 else ''}")
                print(f"Loaded MedicalNet ResNet-18 weights from {pretrain_path}")
            except Exception as e:
                print(f"Warning: failed to load pretrained weights from {pretrain_path}: {e}")
        else:
            print(f"Warning: pretrained weights not found at {pretrain_path}; training from scratch")

        # Wrap backbone with a classification head (global pooling + linear)
        class MedicalNetResNetClassifier(nn.Module):
            def __init__(self, base: nn.Module, num_classes: int):
                super().__init__()
                # Reuse backbone modules up to layer4
                self.conv1 = base.conv1
                self.bn1 = base.bn1
                self.relu = base.relu
                self.maxpool = base.maxpool
                self.layer1 = base.layer1
                self.layer2 = base.layer2
                self.layer3 = base.layer3
                self.layer4 = base.layer4
                self.avgpool = nn.AdaptiveAvgPool3d(1)
                # Infer feature dim from layer4 last block batchnorm features
                try:
                    feat_dim = self.layer4[-1].bn2.num_features
                except Exception:
                    feat_dim = 512  # fallback for ResNet-18
                self.fc = nn.Linear(feat_dim, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x

        model = MedicalNetResNetClassifier(backbone, num_classes)
        return model
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        # F1 metrics are logged at epoch end using all accumulated batches
    
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
           # pos_weights = [float(neg) / float(max(pos, 1)) for pos, neg in zip(pos_counts, neg_counts)]
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
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21k_ft_in1k",
                        help="Backbone name. Use 'medicalnet_resnet18' to fine-tune MedicalNet 3D ResNet-18.")
    parser.add_argument("--medicalnet-root", default=None,
                        help="Path to MedicalNet root folder (containing models/resnet.py). Defaults to ./MedicalNet-master")
    parser.add_argument("--pretrained-path", default=None,
                        help="Path to MedicalNet pretrained weights .pth. Defaults to ./pretrain/resnet_18_23dataset.pth")
    # Weights & Biases logging options
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="rsna-iad-multilabel")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    
    dm = DataModule(args.data_root, args.csv_path, args.batch_size)
    # Setup early to compute class weights
    dm.setup()
    model = LightningModel(model_name=args.model_name, lr=args.lr, pos_weight=dm.pos_weight,
                           medicalnet_root=args.medicalnet_root, pretrained_path=args.pretrained_path)
    
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
        precision="32",  # Use mixed precision for memory efficiency
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )
    
    trainer.fit(model, dm)
    
    # Print final results
    if checkpoint_callback.best_model_path:
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best classification AUC: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()