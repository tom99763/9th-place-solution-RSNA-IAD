import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from configs.data_config import *
import sys
sys.path.append('./src')

# Import balanced sampler - handle both relative and absolute imports
try:
    from .balanced_sampler import SimpleImbalanceSampler
except ImportError:
    # Fallback for when running as script
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from balanced_sampler import SimpleImbalanceSampler
torch.set_float32_matmul_precision('medium')

class SliceDataModule(pl.LightningDataModule):
    """
    DataModule for loading individual slice files instead of volumes.
    Much more memory efficient for training.
    
    Supports both 2D and 2.5D modes:
    - 2D: Single slice replicated 3 times as RGB channels
    - 2.5D: Configurable number of adjacent slices as channels
      * num_adjacent_slices=1: 3 channels (prev, current, next)
      * num_adjacent_slices=2: 5 channels (current±2, current±1, current)
      * reduce_channels_to_rgb: Optionally average down to 3 channels for model compatibility
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Balanced sampling configuration
        self.use_balanced_sampling = getattr(cfg, 'use_balanced_sampling', False)
        self.pos_ratio = getattr(cfg, 'pos_ratio', 0.8)  # 0.8 = 4 positive + 1 negative in batch of 5
        self.image_mode = getattr(cfg, 'image_mode', "2D")
        
        # 2.5D configuration
        self.num_adjacent_slices = getattr(cfg, 'num_adjacent_slices', 1)  # 1 = prev + current + next (3 total)
        self.reduce_channels_to_rgb = getattr(cfg, 'reduce_channels_to_rgb', True)  # True = average down to 3 channels for model compatibility

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),            
            #A.ShiftScaleRotate(
            #    shift_limit=0.06,
            #    scale_limit=0.1,
            #    rotate_limit=15,
            #    p=0.7
            #),
            #A.ElasticTransform(p=0.3, alpha=10, sigma=120 * 0.05, alpha_affine=120 * 0.03),

            # Intensity
            #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),

            # Conversion to Tensor
            ToTensorV2()
        ])

    def setup(self, stage: str = None):
        data_path = Path(self.cfg.data_dir)
        slice_df = pd.read_csv(data_path / "slice_df.csv")
        

        train_slices = slice_df[slice_df["fold_id"] != self.cfg.fold_id]
        print(f"Training on all folds except {self.cfg.fold_id} ({len(train_slices)} slices)")
        
        val_slices = slice_df[slice_df["fold_id"] == self.cfg.fold_id]
        print(f"Validation on fold {self.cfg.fold_id} ({len(val_slices)} slices)")
        

        if(hasattr(self.cfg, 'use_small_dataset') and self.cfg.use_small_dataset):
            # Sample at series level to preserve complete volumes
            train_series = train_slices['series_uid'].unique()
            val_series = val_slices['series_uid'].unique()
            
            # Sample 10% of series, then include all slices from those series (for debugging)
            sampled_train_series = np.random.choice(train_series, size=int(len(train_series) * 0.1), replace=False)
            sampled_val_series = np.random.choice(val_series, size=int(len(val_series) * 0.1), replace=False)
            
            train_slices = train_slices[train_slices['series_uid'].isin(sampled_train_series)]
            val_slices = val_slices[val_slices['series_uid'].isin(sampled_val_series)]
            
            print(f"Small dataset: Using {len(sampled_train_series)} train series ({len(train_slices)} slices)")
            print(f"Small dataset: Using {len(sampled_val_series)} val series ({len(val_slices)} slices)")

        self.train_dataset = IndividualSliceDataset(
            slice_df=train_slices, 
            cfg=self.cfg, 
            transform=self.train_transforms,
            mode="train",
            num_adjacent_slices=self.num_adjacent_slices,
            reduce_channels_to_rgb=self.reduce_channels_to_rgb
        )
        self.val_dataset = IndividualSliceDataset(
            slice_df=val_slices, 
            cfg=self.cfg, 
            transform=None,
            mode="val",
            num_adjacent_slices=self.num_adjacent_slices,
            reduce_channels_to_rgb=self.reduce_channels_to_rgb
        )

    def train_dataloader(self):
        if self.use_balanced_sampling:
            sampler = SimpleImbalanceSampler(
                dataset=self.train_dataset,
                pos_ratio=self.pos_ratio,
                samples_per_epoch=self.cfg.batch_size * 100
            )
            return DataLoader(
                self.train_dataset,
                sampler=sampler,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            # Standard random sampling
            return DataLoader(
                self.train_dataset, 
                batch_size=self.cfg.batch_size, 
                shuffle=True, 
                num_workers=self.cfg.num_workers,
                pin_memory=True, 
                persistent_workers=True
            )

    def val_dataloader(self):
        val_batch_size = getattr(self.cfg, 'val_batch_size', self.cfg.batch_size)
        return DataLoader(
            self.val_dataset, 
            batch_size=val_batch_size,  
            num_workers=self.cfg.num_workers, 
            pin_memory=True,
            persistent_workers=True,
            shuffle=False
        )


class IndividualSliceDataset(Dataset):
    """
    Dataset to load individual slice .npz files.
    Much more memory efficient than loading full volumes.
    """
    def __init__(self, slice_df, cfg, transform=None, mode="train", num_adjacent_slices=1, reduce_channels_to_rgb=True):
        self.slice_df = slice_df.reset_index(drop=True)
        self.cfg = cfg
        self.num_classes = 13
        self.transform = transform
        self.mode = mode
        
        # 2.5D configuration
        self.num_adjacent_slices = num_adjacent_slices
        self.reduce_channels_to_rgb = reduce_channels_to_rgb
        
        # Pre-compute data path
        self.data_path = Path(self.cfg.data_dir)
        
        # OPTIMIZATION: Pre-build lookup dictionaries for fast slice access
        if self.cfg.image_mode == "2.5D":
            print("Building fast lookup indices for 2.5D mode...")
            self._build_slice_lookup_cache()
            print("✅ Fast lookup cache built!")
        else:
            self.slice_lookup = None  # No lookup needed for 2D mode
    
    def _build_slice_lookup_cache(self):
        """Build fast lookup cache for 2.5D slice access."""
        # Create nested dictionary: {series_uid: {slice_idx: filename}}
        self.slice_lookup = {}
        
        for _, row in self.slice_df.iterrows():
            series_uid = row['series_uid']
            slice_idx = row['slice_idx_in_series']
            filename = row['slice_filename']
            
            if series_uid not in self.slice_lookup:
                self.slice_lookup[series_uid] = {}
            
            self.slice_lookup[series_uid][slice_idx] = filename

    def __len__(self):
        return len(self.slice_df)

    def __getitem__(self, idx):
        row = self.slice_df.iloc[idx]
        if(self.cfg.image_mode == "2D"):
        # Load individual slice file
            slice_path = self.data_path / "individual_slices" / row['slice_filename']
            
            with np.load(slice_path) as data:
                slice_img = data['slice'].astype(np.float32)

            # Create 3-channel image (required for most models)
            img = np.stack([slice_img] * 3, axis=-1)
        elif(self.cfg.image_mode == "2.5D"):
            # Load current slice
            slice_path = self.data_path / "individual_slices" / row['slice_filename']
            with np.load(slice_path) as data:
                current_slice = data['slice'].astype(np.float32)
            
            # Get configurable number of adjacent slices for 2.5D
            series_uid = row['series_uid']
            current_idx = row['slice_idx_in_series']
            
            # Calculate total window size
            total_slices = 2 * self.num_adjacent_slices + 1
            
            # Always use centered approach to preserve anatomical ordering
            # This is critical for medical tomography data
            slice_indices = list(range(current_idx - self.num_adjacent_slices, 
                                     current_idx + self.num_adjacent_slices + 1))
            
            all_slices = []
            for slice_idx in slice_indices:
                # Fast dictionary lookup instead of DataFrame filtering
                if (series_uid in self.slice_lookup and 
                    slice_idx in self.slice_lookup[series_uid]):
                    
                    filename = self.slice_lookup[series_uid][slice_idx]
                    target_slice_path = self.data_path / "individual_slices" / filename
                    
                    with np.load(target_slice_path) as data:
                        target_slice = data['slice'].astype(np.float32)
                else:
                    # Use current slice if target slice doesn't exist (edge cases)
                    target_slice = current_slice
                
                all_slices.append(target_slice)
            
            # Stack all slices as channels
            img = np.stack(all_slices, axis=-1)
            
            # Optionally reduce channels to 3 for RGB model compatibility
            if self.reduce_channels_to_rgb and img.shape[-1] > 3:
                # Average groups of channels down to 3 channels
                num_channels = img.shape[-1]
                if num_channels == 5:
                    # For 5 channels, group as: [0,1] -> R, [2] -> G, [3,4] -> B
                    r_channel = np.mean(img[:, :, :2], axis=-1)
                    g_channel = img[:, :, 2]
                    b_channel = np.mean(img[:, :, 3:], axis=-1)
                    img = np.stack([r_channel, g_channel, b_channel], axis=-1)
                elif num_channels == 7:
                    # For 7 channels, group as: [0,1,2] -> R, [3] -> G, [4,5,6] -> B
                    r_channel = np.mean(img[:, :, :3], axis=-1)
                    g_channel = img[:, :, 3]
                    b_channel = np.mean(img[:, :, 4:], axis=-1)
                    img = np.stack([r_channel, g_channel, b_channel], axis=-1)
                else:
                    # For other numbers, divide into 3 equal groups
                    third = num_channels // 3
                    r_channel = np.mean(img[:, :, :third], axis=-1)
                    g_channel = np.mean(img[:, :, third:2*third], axis=-1)
                    b_channel = np.mean(img[:, :, 2*third:], axis=-1)
                    img = np.stack([r_channel, g_channel, b_channel], axis=-1)
        else:
            raise ValueError(f"Invalid image mode: {self.cfg.image_mode}")
        # Prepare labels
        binary_label = 1 if row['has_aneurysm'] else 0
        
        # Location labels (multi-class)
        loc_labels = np.zeros(self.num_classes, dtype=np.float32)
        if row['has_aneurysm'] and not pd.isna(row['aneurysm_locations']):
            # Handle the case where aneurysm_locations might be stored as string
            if isinstance(row['aneurysm_locations'], str):
                import ast
                locations = ast.literal_eval(row['aneurysm_locations'])
            else:
                locations = row['aneurysm_locations']
            
            if isinstance(locations, list):
                for location in locations:
                    if location in LABELS_TO_IDX:
                        loc_labels[LABELS_TO_IDX[location]] = 1

        # Apply transforms
        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            # Convert to tensor if no transforms
            img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC to CHW

        # Include series UID for validation aggregation
        series_uid = row['series_uid']
        
        return img, binary_label, loc_labels, series_uid


class VolumeSliceDataset(Dataset):
    """
    Dataset that loads individual slices but groups them by volume for validation.
    Useful when you need volume-level predictions.
    """
    def __init__(self, slice_df, cfg, transform=None, num_adjacent_slices=1, reduce_channels_to_rgb=True):
        self.slice_df = slice_df
        self.cfg = cfg
        self.num_classes = 13
        self.transform = transform
        self.data_path = Path(self.cfg.data_dir)
        
        # 2.5D configuration
        self.num_adjacent_slices = num_adjacent_slices
        self.reduce_channels_to_rgb = reduce_channels_to_rgb
        
        # Group slices by series UID
        self.series_groups = self.slice_df.groupby('series_uid')
        self.series_uids = list(self.series_groups.groups.keys())

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, idx):
        series_uid = self.series_uids[idx]
        series_slices = self.series_groups.get_group(series_uid).sort_values('slice_idx_in_series')
        
        # Load all slices for this series
        volume_slices = []
        volume_labels = []
        volume_loc_labels = []
        
        for _, row in series_slices.iterrows():
            slice_path = self.data_path / "individual_slices" / row['slice_filename']
            
            with np.load(slice_path) as data:
                slice_img = data['slice'].astype(np.float32)
            
            # Create 3-channel image
            img = np.stack([slice_img] * 3, axis=-1)
            
            # Prepare labels
            binary_label = 1 if row['has_aneurysm'] else 0
            
            # Location labels
            loc_labels = np.zeros(self.num_classes, dtype=np.float32)
            if row['has_aneurysm'] and not pd.isna(row['aneurysm_locations']):
                if isinstance(row['aneurysm_locations'], str):
                    import ast
                    locations = ast.literal_eval(row['aneurysm_locations'])
                else:
                    locations = row['aneurysm_locations']
                
                if isinstance(locations, list):
                    for location in locations:
                        if location in LABELS_TO_IDX:
                            loc_labels[LABELS_TO_IDX[location]] = 1
            
            # Apply transforms (optional for validation)
            if self.transform:
                img = self.transform(image=img)["image"]
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1))
            
            volume_slices.append(img)
            volume_labels.append(binary_label)
            volume_loc_labels.append(loc_labels)
        
        # Stack into volume tensors
        volume = torch.stack(volume_slices)  # Shape: [num_slices, 3, H, W]
        labels = torch.tensor(volume_labels, dtype=torch.long)
        loc_labels = torch.stack([torch.from_numpy(ll) for ll in volume_loc_labels])
        
        return volume, labels, loc_labels, series_uid