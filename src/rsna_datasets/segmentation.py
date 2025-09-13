import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from pathlib import Path
from configs.data_config import *
from pathlib import Path
import cv2

from monai import transforms as T

torch.set_float32_matmul_precision('medium')

class SafeRandCropd(T.MapTransform):
    """
    Wrapper around RandCropByPosNegLabeld:
    - If mask has positives -> use RandCropByPosNegLabeld
    - If no positives -> fall back to RandSpatialCropd (random crop)
    """
    def __init__(self, keys, label_key, spatial_size, num_samples=1, pos=1, neg=1, image_key="volume"):
        super().__init__(keys)
        self.label_key = label_key
        self.num_samples = num_samples
        self.pos_crop = T.RandCropByPosNegLabeld(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            num_samples=num_samples,
            pos=pos,
            neg=neg,
            image_key=image_key,
            image_threshold=0,
        )
        self.rand_crop = T.RandSpatialCropd(
            keys=keys,
            roi_size=spatial_size,
            random_center=True,
            random_size=False
        )

    def __call__(self, data):
        mask = data[self.label_key]
        if not np.all(mask == 0):
            return self.pos_crop(data)
        else:                  
            return [self.rand_crop(dict(data)) for _ in range(self.num_samples)]


class NpzData(Dataset):
    def __init__(self, uids, cfg, transform=None, mode="train"):
        self.uids = uids
        self.cfg = cfg
        self.data_path = Path(cfg.params.data_path)
        self.traindf = pd.read_csv(self.data_path / "train.csv")
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        if self.mode == "train":
            return self.get_train_sample(idx)
        else:
            return self.get_val_sample(idx)

    def get_val_sample(self, idx):
        uid = self.uids[idx]
        sample = np.load(self.data_path / f"{uid}.npz")
        vol, mask = sample["vol"].astype(np.float32), sample["mask"]

        if self.transform:
            result = self.transform({"volume": vol, "mask": mask})
            return result


    def get_train_sample(self, idx):
        uid = self.uids[idx]
        sample = np.load(self.data_path / f"{uid}.npz")
        vol, mask = sample["vol"].astype(np.float32), sample["mask"]

        if self.transform:
            result = self.transform({"volume": vol, "mask": mask})
            return result

class NpzDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
       
        self.cfg = cfg

        # Define augmentations with normalization
        self.train_transforms = T.Compose ([
            T.EnsureChannelFirstd(keys=["volume", "mask"], channel_dim="no_channel"),
            T.ScaleIntensityd(keys="volume"),  # normalize volume intensity
            SafeRandCropd(
                keys=["volume", "mask"],
                label_key="mask",
                spatial_size=(64, 128, 128),
                num_samples=2,
                pos=1,
                neg=1,
                image_key="volume",
            ),

             # --- Spatial Augmentations ---
            T.RandFlipd(keys=["volume", "mask"], prob=0.5, spatial_axis=0),  # flip along depth
            T.RandFlipd(keys=["volume", "mask"], prob=0.5, spatial_axis=1),  # flip along height
            T.RandFlipd(keys=["volume", "mask"], prob=0.5, spatial_axis=2),  # flip along width
    
    #         T.RandAffined(
    #             keys=["volume", "mask"],
    #             prob=0.3,
    #             rotate_range=(0.1, 0.1, 0.1),   # small rotations (rad)
    #             scale_range=(0.1, 0.1, 0.1),    # 10% scaling
    #             mode=("bilinear", "nearest"),   # volume=bilinear, mask=nearest
    #         ),
    # 
    #         T.Rand3DElasticd(
    #             keys=["volume", "mask"],
    #             prob=0.2,
    #             sigma_range=(5, 7),
    #             magnitude_range=(50, 100),
    #             mode=("bilinear", "nearest"),
    #         ),
    #
    #         # --- Intensity Augmentations ---
    #         T.RandGaussianNoised(keys="volume", prob=0.2, mean=0.0, std=0.01),
    #         T.RandAdjustContrastd(keys="volume", prob=0.2, gamma=(0.7, 1.5)),
    #         T.RandShiftIntensityd(keys="volume", offsets=0.1, prob=0.3),


        ])
        self.val_transforms = T.Compose ([
            T.EnsureChannelFirstd(keys=["volume", "mask"], channel_dim="no_channel"),
            T.ScaleIntensityd(keys="volume"),  # normalize volume intensity
        ])


    def setup(self, stage: str = None):

        data_path = Path(self.cfg.params.data_path)
        df = pd.read_csv(data_path / "train.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        print(f"train_uids: {len(train_uids)}, val_uids: {len(val_uids)}")

        self.train_dataset = NpzData(uids=list(train_uids), cfg=self.cfg, transform=self.train_transforms)
        self.val_dataset = NpzData(uids=list(val_uids), cfg=self.cfg, transform=self.val_transforms, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.params.batch_size, shuffle=True, num_workers=self.cfg.params.num_workers, pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.params.batch_size, num_workers=4, pin_memory=False)
