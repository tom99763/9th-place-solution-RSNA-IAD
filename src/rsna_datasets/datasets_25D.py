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
        self.depth = 32

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]
        labeldf = self.label_df[self.label_df["SeriesInstanceUID"] == uid]

        with np.load(self.data_path / "slices" / f"{uid}.npz") as data:
            volume = data['vol'].astype(np.float32)
            volume /= 255.0


        loc_labels = np.zeros(self.num_classes)
        label = 0

        if int(rowdf["Aneurysm Present"].iloc[0]) == 1:
            label = 1
            class_idxs = [LABELS_TO_IDX[loc] for loc in labeldf["location"].tolist()]
            loc_labels[class_idxs] = 1
           

        if self.mode == "train":

            # Load the volume from the .npz file

            if int(rowdf["Aneurysm Present"].iloc[0]) == 1:
                slice_idx = random.choice(labeldf["z"].tolist())
            else:
                items = np.arange(volume.shape[0])
                slice_idx = np.random.choice(items)

            start_idx = max(0, slice_idx - self.depth // 2)
            end_idx = min(volume.shape[0], slice_idx + self.depth // 2)

            # DxHxW
            img = volume[start_idx:end_idx]

            if (img.shape[0] != self.depth):
                diff = self.depth - img.shape[0]
                img = np.vstack([img, np.zeros((diff, 512, 512))])

            assert img.shape[0] == self.depth

            if self.transform:
                # DxHxW
                img = self.transform(image=img)["image"]

            
            return img, label, loc_labels

        else:

            imgs = []
            if volume.shape[0] > 0: # Ensure the volume is not empty
                for i in range(0, volume.shape[0], self.depth):
                    img_chunk = volume[i:i + self.depth]

                    # If the chunk is smaller than self.depth, pad it with zeros
                    if img_chunk.shape[0] != self.depth:
                        diff = self.depth - img_chunk.shape[0]
                        # Assumes image dimensions are (512, 512)
                        padding = np.zeros((diff, 512, 512), dtype=np.float32)
                        img_chunk = np.vstack([img_chunk, padding])

                    imgs.append(img_chunk)

            # Handle cases where the original volume was smaller than self.depth
            if len(imgs) == 0 and volume.shape[0] > 0:
                diff = self.depth - volume.shape[0]
                img_chunk = np.vstack([volume, np.zeros((diff, 512, 512), dtype=np.float32)])
                imgs.append(img_chunk)


            # BxDxHxW
            imgs = np.stack(imgs)
            
            if self.transform:
                # BxDxHxW
                imgs = self.transform(image=imgs)["image"]
            
            return imgs, label, loc_labels

class NpzDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
       
        self.cfg = cfg

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25)
        ])
        self.val_transforms = A.Compose([
        ])

    def setup(self, stage: str = None):

        data_path = Path(self.cfg.data_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = NpzVolumeSliceDataset(uids=list(train_uids), cfg=self.cfg,transform=self.train_transforms)
        self.val_dataset = NpzVolumeSliceDataset(uids=list(val_uids), cfg=self.cfg, transform=None, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=1, pin_memory=True)
