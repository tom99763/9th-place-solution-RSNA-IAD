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
from typing import List, Tuple, Dict, Optional
import os

torch.set_float32_matmul_precision('medium')


class NpzPatchDataset(Dataset):
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

        with np.load(self.data_path / "patch_data" / f"fold{self.cfg.fold_id}"/ f"{uid}/patch_0.npz") as data0:
            mip_0 = data0['cartesian'].astype(np.float32)[:, 1] #(3, 128, 128)
            mip_logpolar_0 = data0['logpolar'].astype(np.float32)[:, 1]
            axial_vol_0 = data0['axial'].astype(np.float32)
            sagittal_vol_0 = data0['sagittal'].astype(np.float32)
            coronal_vol_0 = data0['coronal'].astype(np.float32)

        with np.load(self.data_path / "patch_data" / f"fold{self.cfg.fold_id}"/ f"{uid}/patch_1.npz") as data1:
            mip_1 = data1['cartesian'].astype(np.float32)[:, 1]
            mip_logpolar_1 = data1['logpolar'].astype(np.float32)[:, 1]
            axial_vol_1 = data1['axial'].astype(np.float32)
            sagittal_vol_1 = data1['sagittal'].astype(np.float32)
            coronal_vol_1 = data1['coronal'].astype(np.float32)

        with np.load(self.data_path / "patch_data" / f"fold{self.cfg.fold_id}"/ f"{uid}/patch_2.npz") as data2:
            mip_2 = data2['cartesian'].astype(np.float32)[:, 1]
            mip_logpolar_2 = data2['logpolar'].astype(np.float32)[:, 1]
            axial_vol_2 = data2['axial'].astype(np.float32)
            sagittal_vol_2 = data2['sagittal'].astype(np.float32)
            coronal_vol_2 = data2['coronal'].astype(np.float32)

        #3 patches
        data = {
            "axial_mip": torch.stack([mip_0[0], mip_1[0], mip_2[0]], dim=0).unsqueeze(1), #(3, 1, 128, 128)
            "sagittal_mip": torch.stack([mip_0[1], mip_1[1], mip_2[1]], dim=0).unsqueeze(1),
            "coronal_mip": torch.stack([mip_0[2], mip_1[2], mip_2[2]], dim=0).unsqueeze(1),
            "axial_lp": torch.stack([mip_logpolar_0[0] , mip_logpolar_1[0], mip_logpolar_2[0]], dim=0).unsqueeze(1),
            "sagittal_lp": torch.stack([mip_logpolar_0[1] , mip_logpolar_1[1], mip_logpolar_2[1]], dim=0).unsqueeze(1),
            "coronal_lp": torch.stack([mip_logpolar_0[2] , mip_logpolar_1[2], mip_logpolar_2[2]], dim=0).unsqueeze(1),
            "axial_vol": torch.stack([axial_vol_0, axial_vol_1, axial_vol_2], dim=0), #(3, 31, 128, 128)
            "sagittal_vol": torch.stack([sagittal_vol_0, sagittal_vol_1, sagittal_vol_2], dim=0),
            "coronal_vol": torch.stack([coronal_vol_0, coronal_vol_1, coronal_vol_2], dim=0)
        }
        labels = int(rowdf["Aneurysm Present"].iloc[0])
        return data, labels


class NpzPatchDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        img_size = 128
        self.train_transforms = A.Compose(
            [A.Resize(img_size, img_size), ToTensorV2()],
        )
        self.val_transforms = A.Compose(
            [A.Resize(img_size, img_size), ToTensorV2()],
        )

    def setup(self, stage: str = None):
        data_path = Path(self.cfg.params.data_dir)
        valid_uids = os.listdir(data_path/f'patch_data/{self.cfg.fold_id}')
        df = pd.read_csv(data_path / "train_df.csv")
        df = df[df.SeriesInstanceUID.isin(valid_uids)]
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = NpzPatchDataset(uids=list(train_uids), cfg=self.cfg, transform=self.train_transforms)
        self.val_dataset = NpzPatchDataset(uids=list(val_uids), cfg=self.cfg, transform=self.val_transforms,
                                                 mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.params.batch_size, shuffle=True,
                          num_workers=self.cfg.params.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset
                          , batch_size=1
                          , num_workers=1
                          , pin_memory=True)
