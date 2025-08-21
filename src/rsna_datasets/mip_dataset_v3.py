"""MipDatasetV3 / MipDataModuleV3 (no in-memory caching)

This dataset expects precomputed multi-fraction cumulative axial MIP npz files
produced by `prepare_data_mip_slice_v3.py`:

    data_dir/processed/mip_fraction_images/<SeriesUID>_mip_fracs.npz containing
        mip_frac_uint8: (H,W,C) uint8 (C = #fractions)

And a dataframe (e.g. `mip_df_v3.csv`) with columns:
    series_uid, mip_filename, fold_id, series_has_aneurysm

__getitem__ now simply loads the npz (or computes on-the-fly if allowed) and
applies geometric transforms. No results are cached between samples; this keeps
RAM usage bounded regardless of dataset size or number of workers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pydicom
import cv2
from collections import OrderedDict

from configs.data_config import LABELS_TO_IDX, data_path as GLOBAL_DATA_PATH

sys.path.append('./src')

try:
    from .balanced_sampler import SimpleImbalanceSampler  # type: ignore
except Exception:  # pragma: no cover
    SimpleImbalanceSampler = None  # type: ignore


class MipDatasetV3(Dataset):
    """Dataset for multi-fraction cumulative axial MIPs (precomputed preferred).

    Parameters
    ---------
    df : pd.DataFrame
        Must have columns: series_uid, series_has_aneurysm, (optionally mip_filename).
    cfg : object
        Config with required attributes (see module docstring).
    transform : albumentations.Compose | None
        Geometric / safe transforms. Expected to include ToTensorV2 at end.
    mode : str
        'train' or 'val'. (Only affects potential future behavior.)
    """

    def __init__(self, df: pd.DataFrame, cfg, transform=None, mode: str = 'train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.mode = mode
        self.data_dir = Path(cfg.data_dir)
        self.num_classes = cfg.model.num_classes
        # Ranges used only if on-the-fly compute fallback triggered
        self.raw_min = getattr(cfg, 'raw_min_hu', -1200.0)
        self.raw_max = getattr(cfg, 'raw_max_hu', 4000.0)
        self.default_fractions: List[float] = sorted(getattr(cfg, 'mip_v3_fractions', [0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0]))
    # No caching retained (intentionally removed to avoid RAM growth)
        self.allow_on_the_fly = bool(getattr(cfg, 'allow_on_the_fly', False))
        self.series_to_loc_vec: Dict[str, np.ndarray] = self._build_series_location_map()

    # ---------------- DICOM / preprocessing helpers -----------------
    def _read_series_slices_hu(self, series_uid: str) -> List[np.ndarray]:
        """Fallback DICOM loading (only if allow_on_the_fly=True & npz missing)."""
        series_dir = self.data_dir / 'series' / series_uid
        dicoms = sorted(series_dir.glob('*.dcm'))
        slices: List[np.ndarray] = []
        for dcm_path in dicoms:
            try:
                ds = pydicom.dcmread(str(dcm_path), force=True)
                px = ds.pixel_array
                if px.ndim == 3:  # multi-frame or RGB
                    if px.shape[-1] == 3:  # RGB -> grayscale
                        px = cv2.cvtColor(px.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        px = px.astype(np.float32)
                        slope = float(getattr(ds,'RescaleSlope',1.0))
                        inter = float(getattr(ds,'RescaleIntercept',0.0))
                        px = np.clip(px * slope + inter, self.raw_min, self.raw_max)
                        slices.append(px)
                    else:  # frames,H,W
                        slope = float(getattr(ds,'RescaleSlope',1.0))
                        inter = float(getattr(ds,'RescaleIntercept',0.0))
                        for i in range(px.shape[0]):
                            frame = px[i].astype(np.float32)
                            frame = np.clip(frame * slope + inter, self.raw_min, self.raw_max)
                            slices.append(frame)
                    continue
                img = px.astype(np.float32)
                slope = float(getattr(ds,'RescaleSlope',1.0))
                inter = float(getattr(ds,'RescaleIntercept',0.0))
                img = np.clip(img * slope + inter, self.raw_min, self.raw_max)
                slices.append(img)
            except Exception:
                continue
        return slices

    def _compute_fraction_mips(self, slices: List[np.ndarray], fractions: List[float]) -> np.ndarray:
        if not slices:
            target = int(getattr(self.cfg.model, 'img_size', 512))
            return np.zeros((target, target, len(fractions)), dtype=np.float32)
        h0, w0 = slices[0].shape
        proc = []
        for s in slices:
            if s.shape != (h0, w0):
                s = cv2.resize(s, (w0, h0), interpolation=cv2.INTER_LINEAR)
            proc.append(s)
        stack = np.stack(proc, axis=0)
        total = stack.shape[0]
        mips = []
        for f in fractions:
            n = max(1, int(round(f * total)))
            mip = stack[:n].max(axis=0)
            norm = (mip - self.raw_min) / (self.raw_max - self.raw_min)
            mips.append(np.clip(norm, 0.0, 1.0))
        mips_arr = np.stack(mips, axis=-1).astype(np.float32)  # (H,W,C)
        target = int(getattr(self.cfg.model, 'img_size', 512))
        if mips_arr.shape[0] != target or mips_arr.shape[1] != target:
            mips_arr = cv2.resize(mips_arr, (target, target), interpolation=cv2.INTER_LINEAR)
        return mips_arr

    # ---------------- Label mapping -----------------
    def _build_series_location_map(self) -> Dict[str, np.ndarray]:
        processed_label_path = self.data_dir / 'label_df_slices.csv'
        if processed_label_path.exists():
            label_df = pd.read_csv(processed_label_path)
        else:
            raw_labels_path = Path(GLOBAL_DATA_PATH) / 'train_localizers.csv'
            label_df = pd.read_csv(raw_labels_path) if raw_labels_path.exists() else pd.DataFrame(columns=['SeriesInstanceUID','location'])
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
                    from configs.data_config import LABELS_TO_IDX  # local import to avoid circular
                    if loc in LABELS_TO_IDX:
                        vec[LABELS_TO_IDX[loc]] = 1.0
            mapping[uid] = vec
        return mapping

    # ---------------- PyTorch Dataset API -----------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row['series_uid']
        arr = None
        mip_fname = row.get('mip_filename', None)
        if mip_fname is not None:
            npz_path = self.data_dir / 'mip_fraction_images' / mip_fname
            if npz_path.exists():
                with np.load(npz_path) as data:
                    if 'mip_frac_uint8' in data.files:
                        u8 = data['mip_frac_uint8']
                    else:
                        u8 = data[list(data.files)[0]]
                    arr = (u8.astype(np.float32) / 255.0)
        if arr is None:
            if not self.allow_on_the_fly:
                raise FileNotFoundError(f"Missing precomputed fraction MIP for {uid} and on-the-fly disabled")
            slices = self._read_series_slices_hu(uid)
            arr = self._compute_fraction_mips(slices, self.default_fractions)

        # Ensure target square size (rare if precomputed already sized)
        target_sz = int(getattr(self.cfg.model, 'img_size', 512))
        if arr.shape[0] != target_sz or arr.shape[1] != target_sz:
            arr = cv2.resize(arr, (target_sz, target_sz), interpolation=cv2.INTER_LINEAR)

        # Albumentations expects HWC float; apply transforms
        if self.transform:
            arr = self.transform(image=arr)['image']  # Should be torch.FloatTensor CHW after ToTensorV2
        else:
            arr = torch.from_numpy(arr.transpose(2,0,1)).float()

        # Optional ImageNet norm only if 3 channels (rare here unless user reduces fractions)
        if getattr(self.cfg, 'imagenet_norm', True) and arr.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=arr.device)[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=arr.device)[:, None, None]
            arr = (arr - mean) / std

        binary_label = int(row['series_has_aneurysm'])
        num_loc_classes = getattr(self.cfg.model, 'num_loc_classes', 13)
        loc_vec = self.series_to_loc_vec.get(uid, np.zeros(num_loc_classes, dtype=np.float32))
        loc_vec_t = torch.from_numpy(loc_vec)
        return arr, binary_label, loc_vec_t, uid


class MipDataModuleV3(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_balanced_sampling = getattr(cfg, 'use_balanced_sampling', False)
        self.pos_ratio = getattr(cfg, 'pos_ratio', 0.5)
        # For multi-channel (>3) images many color/intensity transforms in
        # albumentations expect 1 or 3 channels. Keep only geometric + light blur/noise.
        train_list = [
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
        ]
        self.train_transforms = A.Compose(train_list)
        self.val_transforms = A.Compose([
            ToTensorV2(),
        ])

    def setup(self, stage: Optional[str] = None):
        data_path = Path(self.cfg.data_dir)
        # Prefer v3 fraction dataframe if available, else fall back to legacy mip_df.csv
        v3_csv = data_path / 'mip_df_v3.csv'
        mip_df = pd.read_csv(v3_csv)

        train_df = mip_df[mip_df['fold_id'] != self.cfg.fold_id]
        val_df = mip_df[mip_df['fold_id'] == self.cfg.fold_id]
        if getattr(self.cfg, 'use_small_dataset', False):
            train_series = train_df['series_uid'].unique()
            val_series = val_df['series_uid'].unique()
            sampled_train = np.random.choice(train_series, size=max(1,int(len(train_series)*0.1)), replace=False)
            sampled_val = np.random.choice(val_series, size=max(1,int(len(val_series)*0.1)), replace=False)
            train_df = train_df[train_df['series_uid'].isin(sampled_train)]
            val_df = val_df[val_df['series_uid'].isin(sampled_val)]
        self.train_dataset = MipDatasetV3(train_df, cfg=self.cfg, transform=self.train_transforms, mode='train')
        self.val_dataset = MipDatasetV3(val_df, cfg=self.cfg, transform=self.val_transforms, mode='val')

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
            persistent_workers=False,
            drop_last=True,  # avoid BatchNorm issues with last small batch
        )

    def val_dataloader(self):
        val_bs = getattr(self.cfg, 'val_batch_size', self.cfg.batch_size)
        return DataLoader(
            self.val_dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
