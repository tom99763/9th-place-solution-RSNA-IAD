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
torch.set_float32_matmul_precision('medium')

class NpzVolumeSliceDataset(Dataset):
    """
    Dataset to load .npz image volumes and serve random 2D slices.
    """
    def __init__(self, uids, cfg, transform=None, mode="train"):

        self.uids = uids
        self.cfg = cfg

        self.data_path = Path(self.cfg.data_dir)

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")
        self.label_df = pd.read_csv(self.data_path / "label_df.csv")

        self.num_classes = cfg.model.num_classes
        self.transform = transform

        self.mode = mode

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]
        labeldf = self.label_df[self.label_df["SeriesInstanceUID"] == uid]
        with np.load(self.data_path / "slices" / f"{uid}.npz") as data:
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
                
                # items = list(range(volume.shape[0]))
                # mu = volume.shape[0] / 2
                # sigma = volume.shape[0] * 0.25
                #
                #
                # weights = np.exp(-((items - mu) ** 2) / (2 * sigma ** 2))
                # weights /= weights.sum()
                #
                # # Sample one item
                # sample = np.random.choice(items, p=weights)
                # slice = volume[sample, :, :]

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
                volume = self.transform(image=volume)["image"]
            
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
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate( shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7),
            A.ElasticTransform(p=0.3, alpha=10, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            ToTensorV2(), # This should be the last step if needed
        ])
        self.val_transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def setup(self, stage: str = None):

        data_path = Path(self.cfg.data_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = NpzVolumeSliceDataset(uids=list(train_uids), cfg=self.cfg,transform=self.train_transforms)
        self.val_dataset = NpzVolumeSliceDataset(uids=list(val_uids), cfg=self.cfg, transform=self.val_transforms, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True)
