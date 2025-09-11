import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from configs.data_config import *

torch.set_float32_matmul_precision('medium')

class NpzVolumeSliceDataset(Dataset):
    """
    Dataset to load .npz image volumes and serve random 2D slices.
    """
    def __init__(self, uids, cfg, transform=None, mode="train"):

        self.uids = uids
        self.cfg = cfg

        self.data_path = Path(self.cfg.params.data_dir)

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")
        self.label_df = pd.read_csv(self.data_path / "label_df.csv")

        self.num_classes = cfg.params.num_classes
        self.transform = transform

        self.mode = mode
        self.depth_mr = self.cfg.params.depth.mr
        self.depth_ct = self.cfg.params.depth.ct

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]
        labeldf = self.label_df[self.label_df["SeriesInstanceUID"] == uid]
        modality = rowdf["Modality"].iloc[0]

        with np.load(self.data_path / "series" / f"{uid}.npz") as data:
            volume = data['vol'].astype(np.float32)
            volume /= 255.0

        labels = np.zeros(self.num_classes)

        if int(rowdf["Aneurysm Present"].iloc[0]) == 1:
            labels[0] = 1
            class_idxs = [LABELS_TO_IDX[loc] + 1 for loc in labeldf["location"].tolist()]
            labels[class_idxs] = 1

        if "CT" not in modality:
            volume = volume[self.depth_mr[0]:self.depth_mr[1]]
        else:
            volume = volume[self.depth_ct[0]:self.depth_ct[1]]

        if self.cfg.params.use_mip:
            volume = volume.max(axis=0, keepdims=True)

        volume = volume.transpose(1,2,0)

        if self.transform:
            # BxDxHxW
            volume = self.transform(image=volume)["image"]

        return uid, volume, labels
           

class NpzDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
       
        self.cfg = cfg

        self.train_transforms = A.Compose([
            A.ShiftScaleRotate(
                    shift_limit=0.05,  # up to 5% shift
                    scale_limit=0.1,   # zoom in/out 10%
                    rotate_limit=10,   # rotate ±10 degrees
                    border_mode=0,     # constant fill
                    p=0.25
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # ±10%
                contrast_limit=0.1,
                p=0.25
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ])
        self.val_transforms = A.Compose([
            ToTensorV2()
        ])

    def setup(self, stage: str = None):

        data_path = Path(self.cfg.params.data_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = NpzVolumeSliceDataset(uids=list(train_uids), cfg=self.cfg,transform=self.train_transforms)
        self.val_dataset = NpzVolumeSliceDataset(uids=list(val_uids), cfg=self.cfg, transform=self.val_transforms, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.params.batch_size, shuffle=True, num_workers=self.cfg.params.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset
                          , batch_size=1
                          , num_workers=1
                          , pin_memory=True)
