import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from configs.data_config import *
from typing import Dict
import os
import torch.nn.functional as F
import pywt

torch.set_float32_matmul_precision('medium')

class NpzPatchDataset(Dataset):
    """
    Dataset to load .npz image volumes and serve random 2D slices.
    """

    def __init__(self, uids, cfg, transform=None, mode="train", vol_size=(31, 128, 128)):
        self.uids = uids
        self.cfg = cfg
        self.data_path = Path(self.cfg.params.data_dir)

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")
        self.label_df = pd.read_csv(self.data_path / "label_df.csv")

        self.num_classes = cfg.params.num_classes
        self.transform = transform
        self.mode = mode
        self.vol_size = vol_size  # (C, H, W) target volume size

    def __len__(self):
        return len(self.uids)

    def resize_vol2d(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Resize a volume treating depth as channels.
        Input:  (num_patch, C, H, W)
        Output: (num_patch, C, H_out, W_out)
        """
        target_h, target_w = self.vol_size[1], self.vol_size[2]
        return F.interpolate(vol, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]

        # Load 3 patches
        patch_data = []
        for i in range(3):
            with np.load(self.data_path / "patch_data" / f"fold{self.cfg.fold_id_yolo}" / f"{uid}" / f"patch_{i}.npz") as data:
                patch_data.append({
                    "mip": data["cartesian"].astype(np.float32)[:, 1],    # (3,128,128)
                    "lp": data["logpolar"].astype(np.float32)[:, 1],
                    "axial": data["axial"].astype(np.float32),           # (31,128,128)
                    "sagittal": data["sagittal"].astype(np.float32),
                    "coronal": data["coronal"].astype(np.float32),
                })

        # Stack across patches
        data = {
            "axial_mip": torch.stack([torch.from_numpy(p["mip"][0]) for p in patch_data], dim=0).unsqueeze(1),
            "sagittal_mip": torch.stack([torch.from_numpy(p["mip"][1]) for p in patch_data], dim=0).unsqueeze(1),
            "coronal_mip": torch.stack([torch.from_numpy(p["mip"][2]) for p in patch_data], dim=0).unsqueeze(1),
            "axial_lp": torch.stack([torch.from_numpy(p["lp"][0]) for p in patch_data], dim=0).unsqueeze(1),
            "sagittal_lp": torch.stack([torch.from_numpy(p["lp"][1]) for p in patch_data], dim=0).unsqueeze(1),
            "coronal_lp": torch.stack([torch.from_numpy(p["lp"][2]) for p in patch_data], dim=0).unsqueeze(1),
            "axial_vol": torch.stack([torch.from_numpy(p["axial"]) for p in patch_data], dim=0),
            "sagittal_vol": torch.stack([torch.from_numpy(p["sagittal"]) for p in patch_data], dim=0),
            "coronal_vol": torch.stack([torch.from_numpy(p["coronal"]) for p in patch_data], dim=0),
        }

        # Apply transforms only on 2D slices (MIPs & LPs)
        if self.transform is not None:
            for k in ["axial_mip", "sagittal_mip", "coronal_mip",
                      "axial_lp", "sagittal_lp", "coronal_lp"]:
                transformed = []
                for img in data[k]:
                    img_np = img.squeeze(0).numpy()
                    t = self.transform(image=img_np)
                    transformed.append(t["image"].unsqueeze(0))
                data[k] = torch.cat(transformed, dim=0)

        # Resize volumes spatially (H,W), keeping depth as channels
        for k in ["axial_vol", "sagittal_vol", "coronal_vol"]:
            data[k] = self.resize_vol2d(data[k])  # (3,31,H_out,W_out)

        labels = int(rowdf["Aneurysm Present"].iloc[0])
        return data, labels


class NpzPatchWaveletDataset(Dataset):
    """
    Dataset to load .npz image volumes and serve random 2D slices.
    """

    def __init__(self, uids, cfg, transform=None, mode="train", vol_size=(13, 64, 64)):
        self.uids = uids
        self.cfg = cfg
        self.data_path = Path(self.cfg.params.data_dir)

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")
        self.label_df = pd.read_csv(self.data_path / "label_df.csv")

        self.num_classes = cfg.params.num_classes
        self.transform = transform
        self.mode = mode
        self.vol_size = vol_size  # (C, H, W) target volume size

    def __len__(self):
        return len(self.uids)

    def resize_vol3d(self, vol: torch.Tensor) -> torch.Tensor:
        target_d, target_h, target_w = self.vol_size[0] * 8, self.vol_size[1], self.vol_size[2]
        return F.interpolate(
                vol,
                size=(target_d, target_h, target_w),
                mode="trilinear",
                align_corners=False
            )

    def apply_3d_dwt(self, x):
        # (15,64,64) -> (8, 13, 32, 32) -> (64, 32, 32)
        coeffs = pywt.dwtn(x, wavelet='bior3.5', axes=(0,1,2))
        normalized_coeffs = {}
        for band, band_data in coeffs.items():
            normalized_coeffs[band] = (band_data - np.mean(band_data)) / (np.std(band_data) + 1e-10)
        bands = np.stack([normalized_coeffs[k] for k in normalized_coeffs.keys()], axis=0)
        bands = bands.reshape(-1, 37, 37) #(13 * 8, 37, 37)
        return  bands

    def __getitem__(self, idx):
        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]

        # Load 3 patches
        patch_data = []
        for i in range(2):
            with np.load(self.data_path / "patch_data" / f"fold{self.cfg.fold_id_yolo}" / f"{uid}" / f"patch_{i}.npz") as data:
                patch_data.append({
                    "axial": self.apply_3d_dwt(data["axial"].astype(np.float32)),
                    "sagittal": self.apply_3d_dwt(data["sagittal"].astype(np.float32)),
                    "coronal": self.apply_3d_dwt(data["coronal"].astype(np.float32)),
                })

        # Stack across patches
        data = {
            "axial_vol": torch.stack([torch.from_numpy(p["axial"]) for p in patch_data], dim=0), #(2, 13 * 8, 32, 32)->(#patch, #bands, h, w)
            "sagittal_vol": torch.stack([torch.from_numpy(p["sagittal"]) for p in patch_data], dim=0),
            "coronal_vol": torch.stack([torch.from_numpy(p["coronal"]) for p in patch_data], dim=0),
        }

        # Resize volumes spatially (H,W), keeping depth as channels
        for k in ["axial_vol", "sagittal_vol", "coronal_vol"]:
            data[k] = self.resize_vol3d(data[k])  #(2, 13 * 8, 64, 64)->(#patch, #bands, h, w)

        labels = int(rowdf["Aneurysm Present"].iloc[0])
        return data, labels

def patch_collate_fn(batch):
    """
    Collate function for NpzPatchDataset.
    Each batch item is (dict, label).
    Output is (dict of batched tensors, labels).
    """
    batch_dict = {}
    labels = []

    # batch is a list of (data_dict, label)
    for data, label in batch:
        for k, v in data.items():
            if k not in batch_dict:
                batch_dict[k] = []
            batch_dict[k].append(v)
        labels.append(label)

    # stack tensors
    for k in batch_dict:
        batch_dict[k] = torch.stack(batch_dict[k], dim=0)  # (B, 2, 64, 64, 64)

    labels = torch.tensor(labels, dtype=torch.float32)
    return batch_dict, labels


class NpzPatchDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        img_size = 64
        self.train_transforms = A.Compose(
            [A.Resize(img_size, img_size), ToTensorV2()],
        )
        self.val_transforms = A.Compose(
            [A.Resize(img_size, img_size), ToTensorV2()],
        )

    def setup(self, stage: str = None):
        data_path = Path(self.cfg.params.data_dir)

        # List all available uids (directories inside fold dir)
        fold_path = data_path / "patch_data" / f"fold{self.cfg.fold_id}"
        valid_uids = [d for d in os.listdir(fold_path) if (fold_path / d).is_dir()]

        df = pd.read_csv(data_path / "train_df.csv")
        df = df[df.SeriesInstanceUID.isin(valid_uids)]

        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"].tolist()
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"].tolist()

        print(len(train_uids))
        print(len(val_uids))

        self.train_dataset = NpzPatchWaveletDataset(uids=train_uids, cfg=self.cfg, transform=self.train_transforms)
        self.val_dataset = NpzPatchWaveletDataset(uids=val_uids, cfg=self.cfg, transform=self.val_transforms, mode="val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.params.batch_size,
            shuffle=True,
            num_workers=self.cfg.params.num_workers,
            pin_memory=True,
            collate_fn=patch_collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.params.batch_size * 2,
            num_workers=self.cfg.params.num_workers,
            pin_memory=True,
            collate_fn=patch_collate_fn,
            persistent_workers=True
        )