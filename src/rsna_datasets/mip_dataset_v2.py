"""MipDatasetV2

Uses precomputed uint8 min-max normalized MIP images saved by prepare_data_mip_slice_v2.py.
Each .npz is expected to contain:
  - mip_uint8: (H,W) uint8 in [0,255]
  - mip_raw (optional): float32 HU-clipped (retained but unused here)
  - meta: [[raw_min, raw_max]] HU range (optional)

Differences from original MipDataset:
  * Loads the uint8 channel directly (no HU windowing / multi-window stack)
  * Optionally duplicates the single channel to 3 channels for models pretrained on RGB
  * Normalizes to [0,1] float before transforms that convert to tensor (unless transform handles uint8)

Returned sample: (image_tensor, binary_label, location_multihot, series_uid)

Config expectations (attributes accessed):
  - data_dir
  - fold_id
  - model.img_size
  - model.num_classes
  - model.num_loc_classes (optional, default 13)
  - batch_size, val_batch_size (optional), num_workers
  - duplicate_mip_to_rgb (optional bool, default True)
  - use_balanced_sampling / pos_ratio (optional for sampler)
  - use_small_dataset (optional quick subset)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from configs.data_config import LABELS_TO_IDX, data_path as GLOBAL_DATA_PATH

import sys
sys.path.append('./src')

# Balanced sampler (optional reuse)
try:
    from .balanced_sampler import SimpleImbalanceSampler  # type: ignore
except Exception:  # pragma: no cover
    SimpleImbalanceSampler = None  # type: ignore

class MipDatasetV2(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, transform=None, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.mode = mode
        self.num_classes = cfg.model.num_classes
        self.data_dir = Path(self.cfg.data_dir)
        self.duplicate_to_rgb = getattr(cfg, 'duplicate_mip_to_rgb', True)

        self.series_to_loc_vec: Dict[str, np.ndarray] = self._build_series_location_map()

    def _build_series_location_map(self) -> Dict[str, np.ndarray]:
        processed_label_path = self.data_dir / "label_df_slices.csv"
        if processed_label_path.exists():
            label_df = pd.read_csv(processed_label_path)
        else:
            raw_labels_path = Path(GLOBAL_DATA_PATH) / "train_localizers.csv"
            label_df = pd.read_csv(raw_labels_path) if raw_labels_path.exists() else pd.DataFrame(columns=["SeriesInstanceUID","location"])

        num_loc_classes = getattr(self.cfg.model, 'num_loc_classes', 13)
        mapping: Dict[str, np.ndarray] = {}
        if not label_df.empty and 'location' in label_df.columns:
            loc_groups = (label_df.dropna(subset=['location'])
                                   .groupby('SeriesInstanceUID')['location']
                                   .apply(list))
        else:
            loc_groups = pd.Series(dtype=object)

        for uid in self.df['series_uid'].unique():
            vec = np.zeros(num_loc_classes, dtype=np.float32)
            if uid in loc_groups.index:
                for loc in loc_groups.loc[uid]:
                    if loc in LABELS_TO_IDX:
                        vec[LABELS_TO_IDX[loc]] = 1.0
            mapping[uid] = vec
        return mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_uid = row['series_uid']
        npz_path = self.data_dir / "mip_images" / row['mip_filename']
        with np.load(npz_path) as data:
            if 'mip_uint8' in data.files:
                mip_u8 = data['mip_uint8']  # uint8
            elif 'mip' in data.files:  # backward compatibility
                mip_u8 = (data['mip'] - data['mip'].min()) / (data['mip'].ptp() + 1e-6)
                mip_u8 = (mip_u8 * 255.0).clip(0,255).astype(np.uint8)
            else:  # fallback to mip_raw
                mip_raw = data['mip_raw'] if 'mip_raw' in data.files else data[list(data.files)[0]]
                norm = (mip_raw - mip_raw.min()) / (mip_raw.ptp() + 1e-6)
                mip_u8 = (norm * 255.0).clip(0,255).astype(np.uint8)

        # Shape (H,W); convert to H,W,C
        img = mip_u8[..., None]  # keep single channel
        if self.duplicate_to_rgb:
            img = np.repeat(img, 3, axis=-1)  # (H,W,3)

        # Albumentations expects uint8 or float; we can pass uint8 directly.
        if self.transform:
            img = self.transform(image=img)["image"]  # may come back as uint8 tensor
            # Ensure float32 and scale to 0-1
            if img.dtype != torch.float32:
                img = img.float()
            if img.max() > 1.0:
                img = img / 255.0
        else:
            # Convert to float tensor [0,1]
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0


        binary_label = int(row['series_has_aneurysm'])
        num_loc_classes = getattr(self.cfg.model, 'num_loc_classes', 13)
        loc_labels = self.series_to_loc_vec.get(series_uid, np.zeros(num_loc_classes, dtype=np.float32))
        loc_labels = torch.from_numpy(loc_labels)

        return img, binary_label, loc_labels, series_uid

class MipDataModuleV2(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Transforms: keep light; normalization left to model or add here if needed.
        train_list = [
            #A.RandomResizedCrop(
            #    size=(self.cfg.model.img_size, self.cfg.model.img_size),
            #    scale=(0.8, 1),
            #    ratio=(0.9, 1.1),
            #    p=0.3,
            #),
            #A.HorizontalFlip(p=0.5),
            #A.RandomRotate90(p=0.2),
            #A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.3, border_mode=0),
            #A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
            #A.GaussianBlur(blur_limit=(3, 5), p=0.05),
            #A.RandomGamma(gamma_limit=(90, 110), p=0.1),
            #A.Normalize(mean=train_mean, std=train_std),
            ToTensorV2(),
        ]
        self.train_transforms = A.Compose(train_list)
        self.val_transforms = A.Compose([
            ToTensorV2(),
        ])

        self.use_balanced_sampling = getattr(cfg, 'use_balanced_sampling', False)
        self.pos_ratio = getattr(cfg, 'pos_ratio', 0.5)

    def setup(self, stage: Optional[str] = None):
        data_path = Path(self.cfg.data_dir)
        mip_df = pd.read_csv(data_path / 'mip_df.csv')
        train_df = mip_df[mip_df['fold_id'] != self.cfg.fold_id]
        val_df = mip_df[mip_df['fold_id'] == self.cfg.fold_id]

        if getattr(self.cfg, 'use_small_dataset', False):
            train_series = train_df['series_uid'].unique()
            val_series = val_df['series_uid'].unique()
            sampled_train = np.random.choice(train_series, size=max(1,int(len(train_series)*0.1)), replace=False)
            sampled_val = np.random.choice(val_series, size=max(1,int(len(val_series)*0.1)), replace=False)
            train_df = train_df[train_df['series_uid'].isin(sampled_train)]
            val_df = val_df[val_df['series_uid'].isin(sampled_val)]

        self.train_dataset = MipDatasetV2(train_df, cfg=self.cfg, transform=self.train_transforms, mode='train')
        self.val_dataset = MipDatasetV2(val_df, cfg=self.cfg, transform=self.val_transforms, mode='val')

    def train_dataloader(self):
        if self.use_balanced_sampling and SimpleImbalanceSampler is not None:
            sampler = SimpleImbalanceSampler(
                dataset=self.train_dataset,
                pos_ratio=self.pos_ratio,
                samples_per_epoch=self.cfg.batch_size * 100,
            )
            return DataLoader(
                self.train_dataset,
                sampler=sampler,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        val_batch_size = getattr(self.cfg, 'val_batch_size', self.cfg.batch_size)
        return DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
