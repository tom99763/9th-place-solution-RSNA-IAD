import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torchmetrics
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from configs.data_config import *

from hydra.utils import instantiate
torch.set_float32_matmul_precision('medium')

class NpzVolumeSliceDataset(Dataset):
    """
    Dataset to load .npz image volumes and serve random 2D slices.
    """
    def __init__(self, uids, cfg, transform=None, mode="train"):

        self.uids = uids
        self.cfg = cfg

        data_path = Path(self.cfg.data_dir)

        self.train_df = pd.read_csv(data_path / "train_df.csv")
        self.label_df = pd.read_csv(data_path / "label_df.csv")

        self.num_classes = 13
        self.transform = transform

        self.mode = mode

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]
        labeldf = self.label_df[self.label_df["SeriesInstanceUID"] == uid]
        with np.load(f"./src/data/processed/slices/{uid}.npz") as data:
            volume = data['vol'].astype(np.float32)


        if self.mode == "train":

            loc_labels = np.zeros(self.num_classes)
            label = 0
            # Load the volume from the .npz file

            if int(rowdf["Aneurysm Present"].iloc[0]) == 1:
                slice_idx = random.choice(labeldf["z"].tolist())
                class_idxs = [LABELS_TO_IDX[c] for c in  labeldf[labeldf["z"] == slice_idx]["location"]]
                slice = volume[slice_idx]

                loc_labels[class_idxs] = 1
                label = 1

            else:
                slice_idx = np.random.randint(0, volume.shape[0])
                slice = volume[slice_idx, :, :]

          
            img = np.stack([slice] * 3, axis=-1)
             
         
            # Apply transforms if any (e.g., normalization, resizing)
            if self.transform:
                img = self.transform(image=img)["image"]
                
            return img, label, loc_labels

        else:
            loc_labels = np.zeros((volume.shape[0],self.num_classes))
            label = np.zeros(volume.shape[0])

            volume = np.stack([volume,volume,volume],axis=1)

            if self.transform:
                volume = self.transform(vol=volume)["vol"]
            
            if int(rowdf["Aneurysm Present"].iloc[0]) == 1:
                for slice_idx in labeldf["z"]:
                    class_idxs = [LABELS_TO_IDX[c] for c in  labeldf[labeldf["z"] == slice_idx]["location"]]
                    loc_labels[slice_idx,class_idxs] = 1
                    label[slice_idx] = 1

            return volume, label, loc_labels
            

class NpzDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
       
        self.cfg = cfg

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.7
            ),
            A.ElasticTransform(p=0.3, alpha=10, sigma=120 * 0.05, alpha_affine=120 * 0.03),

            # Intensity
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),

            # Conversion to Tensor (if using PyTorch)
            ToTensorV2() # This should be the last step if needed
        ])

    def setup(self, stage: str = None):

        data_path = Path(self.cfg.data_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = NpzVolumeSliceDataset(uids=list(train_uids), cfg=self.cfg,transform=self.train_transforms)
        self.val_dataset = NpzVolumeSliceDataset(uids=list(val_uids), cfg=self.cfg, transform=None, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True)


class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters() # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=13)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=13)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, cls_labels, loc_labels = batch

        pred_cls, pred_locs =self(x)
        pred_cls = pred_cls.squeeze()

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels.float())

        loss = 2*cls_loss + loc_loss

        self.train_loc_auroc.update(pred_locs, loc_labels.long())
        self.train_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        # Log the metric object. Lightning computes and logs it at epoch end.
        self.log('train_loc_auroc', self.train_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, sample, batch_idx):
        x,cls_labels, loc_labels = sample
        x.squeeze_()
        cls_labels.squeeze_()
        loc_labels.squeeze_()

        pred_cls = []
        pred_locs = []

        for batch_idx in range(0,x.shape[0], 64):
            pc,pl = self(x[batch_idx:batch_idx+64])
            pred_cls.append(pc)
            pred_locs.append(pl)



        pred_cls = torch.vstack(pred_cls)
        pred_locs = torch.vstack(pred_locs)

        pred_cls = pred_cls.squeeze()

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels)

        loss = 2*cls_loss + loc_loss

        self.val_loc_auroc.update(pred_locs, loc_labels.long())
        self.val_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=False)
        # Log the metric object. Lightning computes and logs it at epoch end.
        self.log('val_loc_auroc', self.val_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cls_auroc', self.val_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Your main training function.
    The 'cfg' object contains all your configuration.
    """
    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    datamodule = NpzDataModule(cfg)

    model = instantiate(cfg.model)
    pl_model = LitTimmClassifier(model, cfg)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
                          monitor="val_loss"
                        , mode="min"
                        , dirpath="./models"
                        , filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_loss:.4f}'+f"fold_id={cfg.fold_id}"
                        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=pl.loggers.TensorBoardLogger("logs/", name=cfg.experiment),
        callbacks=[lr_monitor, ckpt_callback]
    )
    trainer.fit(pl_model, datamodule=datamodule)



if __name__ == "__main__":
    train()
    
