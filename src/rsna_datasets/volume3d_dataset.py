import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pytorch_lightning as pl
from configs.data_config import LABELS_TO_IDX
import random

class Volume3DDataset(Dataset):
    """Load preprocessed 3D volumes (D,H,W) from NPZ and return as (C,H,W) where C=depth.
    """

    def __init__(self, df: pd.DataFrame, cfg, mode: str = 'train', transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.transform = transform
        self.data_dir = Path(self.cfg.data_dir)
        self.num_loc_classes = getattr(self.cfg.model, 'num_classes', 13)
        # Build location multi-hot (series-level) from label_df_slices.csv if present else raw
        self.series_to_loc_vec = self._build_series_location_map()

    def _build_series_location_map(self):
        mapping = {}
        processed_label_path = self.data_dir / 'label_df_slices.csv'
        if processed_label_path.exists():
            label_df = pd.read_csv(processed_label_path)
        else:
            raw_path = Path(self.cfg.data_dir).parent / 'train_localizers.csv'
            label_df = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=['SeriesInstanceUID','location'])
        grouped = (
            label_df.dropna(subset=['location'])
                     .groupby('SeriesInstanceUID')['location']
                     .apply(list)
        ) if 'location' in label_df.columns else {}
        for uid in self.df['series_uid'].unique():
            vec = np.zeros(self.num_loc_classes, dtype=np.float32)
            if isinstance(grouped, pd.Series) and uid in grouped.index:
                for loc in grouped.loc[uid]:
                    if loc in LABELS_TO_IDX:
                        vec[LABELS_TO_IDX[loc]] = 1.0
            mapping[uid] = vec
        return mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_uid = row['series_uid']
        vol_path = self.data_dir / 'volumes_3d' / row['volume_filename']
        with np.load(vol_path) as data:
            vol = data['vol'].astype(np.float32)  # (D,H,W) raw HU (un-normalized)

        raw_min = float(getattr(self.cfg, 'raw_min_hu', -1200.0))
        raw_max = float(getattr(self.cfg, 'raw_max_hu', 4000.0))
        # Clip first, then scale to [0,1]
        vol = np.clip(vol, raw_min, raw_max)
        img = (vol - raw_min) / max(raw_max - raw_min, 1e-6)
        # Shape remains (D,H,W); depth acts as channels after torch.from_numpy

        tensor = torch.from_numpy(img)  # (C,H,W)
        loc_vec = self.series_to_loc_vec.get(series_uid, np.zeros(self.num_loc_classes, dtype=np.float32))
        binary_label = int(row['series_has_aneurysm'])
        return tensor, binary_label, torch.from_numpy(loc_vec), series_uid

class Volume3DDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_df = None
        self.val_df = None

    def setup(self, stage=None):
        data_path = Path(self.cfg.data_dir)
        vol_df = pd.read_csv(data_path / 'volume_df.csv')
        self.train_df = vol_df[vol_df['fold_id'] != self.cfg.fold_id]
        self.val_df = vol_df[vol_df['fold_id'] == self.cfg.fold_id]
        self.train_dataset = Volume3DDataset(self.train_df, cfg=self.cfg, mode='train')
        self.val_dataset = Volume3DDataset(self.val_df, cfg=self.cfg, mode='val')

    def train_dataloader(self):
        # drop_last avoids BatchNorm errors on final small batch (size=1) during training
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
