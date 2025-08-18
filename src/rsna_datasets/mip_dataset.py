import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from configs.data_config import LABELS_TO_IDX, data_path

import sys
sys.path.append('./src')

# Balanced sampler (optional)
try:
    from .balanced_sampler import SimpleImbalanceSampler
except Exception:
    SimpleImbalanceSampler = None

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def random_hu_window(img: np.ndarray, base_center: float, base_width: float, jitter: Tuple[float, float]) -> np.ndarray:
    """Apply a random HU window (center, width) around a base with jitter fractions.
    Returns image scaled to [0,1]."""
    c_jit = base_center + np.random.uniform(-jitter[0], jitter[0]) * base_width
    w_jit = base_width * np.random.uniform(1 - jitter[1], 1 + jitter[1])
    low = c_jit - w_jit / 2.0
    high = c_jit + w_jit / 2.0
    windowed = np.clip(img, low, high)
    return (windowed - low) / max(w_jit, 1e-6)

def fixed_hu_window(img: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(img, low, high)
    return (windowed - low) / max(width, 1e-6)

class MipDataset(Dataset):
    """
    Dataset for per-series axial MIP images stored as .npz files.
    Expects a `mip_df.csv` with columns: series_uid, mip_filename, fold_id, series_has_aneurysm
    """
    def __init__(self, df: pd.DataFrame, cfg, transform=None, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.mode = mode
        self.num_classes = cfg.model.num_classes

        self.data_dir = Path(self.cfg.data_dir)

        # Build series -> location labels (multi-hot) mapping
        self.series_to_loc_vec: Dict[str, np.ndarray] = self._build_series_location_map()

    def _build_series_location_map(self) -> Dict[str, np.ndarray]:
        """Aggregate series-level location labels into multi-hot vectors."""
        mapping: Dict[str, np.ndarray] = {}

        # Prefer processed label_df if available
        processed_label_path = self.data_dir / "label_df_slices.csv"
        if processed_label_path.exists():
            label_df = pd.read_csv(processed_label_path)
            src_uid_col = 'SeriesInstanceUID'
            locations_series = (
                label_df.dropna(subset=['location'])
                        .groupby(src_uid_col)['location']
                        .apply(list)
            )
        else:
            # Fallback to raw labels in root data directory
            raw_labels_path = Path(data_path) / "train_localizers.csv"
            if raw_labels_path.exists():
                label_df = pd.read_csv(raw_labels_path)
                src_uid_col = 'SeriesInstanceUID'
                locations_series = (
                    label_df.dropna(subset=['location'])
                            .groupby(src_uid_col)['location']
                            .apply(list)
                )
            else:
                locations_series = pd.Series(dtype=object)

        num_loc_classes = getattr(self.cfg.model, 'num_loc_classes', 13)
        for uid in self.df['series_uid'].unique():
            vec = np.zeros(num_loc_classes, dtype=np.float32)
            if uid in locations_series.index:
                for loc in locations_series.loc[uid]:
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
            mip = data['mip'].astype(np.float32)  # raw clipped HU
        # Multi-window stack creation
        raw_min = getattr(self.cfg, 'raw_min_hu', -1200.0)
        raw_max = getattr(self.cfg, 'raw_max_hu', 4000.0)
        raw_norm = np.clip((mip - raw_min) / (raw_max - raw_min), 0.0, 1.0)

        # Define canonical windows (center,width)
        base_windows = [
            (40.0, 80.0),    # brain narrow
            (50.0, 150.0),   # soft tissue / brain standard
            (60.0, 300.0),   # wider soft
            (300.0, 700.0),  # bone/high contrast
        ]
        jitter = getattr(self.cfg, 'window_jitter_hu', 10.0)
        use_jitter = (self.mode == 'train') and getattr(self.cfg, 'use_window_aug', True)
        window_channels = []
        for (c, w) in base_windows:
            if use_jitter:
                c = c + np.random.uniform(-jitter, jitter)
                w = max(1.0, w + np.random.uniform(-jitter, jitter))
            window_channels.append(fixed_hu_window(mip, c, w))

        img = np.stack([raw_norm] + window_channels, axis=-1).astype(np.float32)  # (H,W,5)

        # Apply transforms (Albumentations expects HWC)
        if self.transform:
            # If Normalize in transform expects 3 channels it will fail; ensure config updated for 5 channels.
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))  # CHW

        # Labels
        binary_label = int(row['series_has_aneurysm'])
        num_loc_classes = getattr(self.cfg.model, 'num_loc_classes', 13)
        loc_labels = self.series_to_loc_vec.get(series_uid, np.zeros(num_loc_classes, dtype=np.float32))


        return img, binary_label, loc_labels, series_uid


class MipDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_transforms = A.Compose([
            A.RandomResizedCrop(
                size=(self.cfg.model.img_size, self.cfg.model.img_size),
                scale=(0.8, 1),
                ratio=(0.9, 1.1),
                p=0.3,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.3, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.05),
            A.RandomGamma(gamma_limit=(90, 110), p=0.1),
            #A.Normalize(mean=train_mean, std=train_std),
            ToTensorV2(),
        ])

        self.val_transforms = A.Compose([
            #A.Normalize(mean=train_mean, std=train_std),
            ToTensorV2(),
        ])

        # Balanced sampling (optional)
        self.use_balanced_sampling = getattr(cfg, 'use_balanced_sampling', False)
        self.pos_ratio = getattr(cfg, 'pos_ratio', 0.5)

    def setup(self, stage: Optional[str] = None):
        data_path = Path(self.cfg.data_dir)
        mip_df = pd.read_csv(data_path / "mip_df.csv")

        train_df = mip_df[mip_df['fold_id'] != self.cfg.fold_id]
        val_df = mip_df[mip_df['fold_id'] == self.cfg.fold_id]

        if getattr(self.cfg, 'use_small_dataset', False):
            # Sample a subset of series for quick experiments
            train_series = train_df['series_uid'].unique()
            val_series = val_df['series_uid'].unique()
            import numpy as np
            sampled_train = np.random.choice(train_series, size=max(1, int(len(train_series) * 0.1)), replace=False)
            sampled_val = np.random.choice(val_series, size=max(1, int(len(val_series) * 0.1)), replace=False)
            train_df = train_df[train_df['series_uid'].isin(sampled_train)]
            val_df = val_df[val_df['series_uid'].isin(sampled_val)]


        self.train_dataset = MipDataset(train_df, cfg=self.cfg, transform=self.train_transforms, mode="train")
        self.val_dataset = MipDataset(val_df, cfg=self.cfg, transform=self.val_transforms, mode="val")


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
        else:
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

