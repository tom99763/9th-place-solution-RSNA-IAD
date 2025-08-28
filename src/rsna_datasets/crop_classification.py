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

from monai.transforms import MapTransform, RandCropByLabelClassesd
from albumentations.pytorch.transforms import ToTensorV2

torch.set_float32_matmul_precision('medium')

class DynamicRandCropByLabelClassesd(MapTransform):
    def __init__(self, keys, label_key="label", num_classes=14, num_samples=1):
        super().__init__(keys)
        self.label_key = label_key
        self.num_classes = num_classes
        self.num_samples = num_samples

    def __call__(self, data):
        d = dict(data)
        spatial_size = d.pop("spatial_size")

        # get unique labels from this mask
        labels = d[self.label_key]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        present_classes = np.unique(labels)

        # build ratios vector dynamically
        ratios = [1 if c in present_classes else 0 for c in range(self.num_classes)]

        # fallback if only background (all zeros)
        if sum(ratios) == 0:
            ratios[0] = 1

        # apply cropping
        cropper = RandCropByLabelClassesd(
            keys=self.keys,
            label_key=self.label_key,
            spatial_size=spatial_size,
            num_classes=self.num_classes,
            num_samples=self.num_samples,
            ratios=ratios,
        )
        cropped = cropper(d)
        return cropped[0], cropper.cropper.centers[0]



class WindowedMIP(Dataset):
    def __init__(self, uids, cfg, transform=None, mode="train"):
        self.uids = uids
        self.cfg = cfg
        self.data_path = Path(cfg.params.data_path)
        self.traindf = pd.read_csv(self.data_path / "train.csv")
        self.extract_patch = DynamicRandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    num_classes=self.cfg.params.num_classes,   # required since label is one-hot
                    num_samples=1,
        )
        self.transform = transform
        self.mode = mode
        self.num_classes = self.cfg.params.num_classes

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.get_train_sample(idx)
        else:
            return self.get_val_sample(idx)

    def get_val_sample(self, idx):


        uid = self.uids[idx]
        series_info = self.traindf[self.traindf["SeriesInstanceUID"] == uid]
        modality = series_info["Modality"].iloc[0]

        sample = np.load(self.data_path / f"{uid}.npz")

        vol,_,loc_labels = sample["vol"],sample["coords"], sample["location_labels"]

        vol_depth,_,_ = vol.shape

        if modality == "CTA":
            z_stride = self.cfg.params.sliding_stride.cta
            spatial_size = np.array(self.cfg.params.spatial_size.cta)
        else:
            z_stride = self.cfg.params.sliding_stride.mr
            spatial_size = np.array(self.cfg.params.spatial_size.mr)


        images = []

        (patch_z,_,_) = spatial_size

        for center_z in range(0, vol_depth, z_stride):
            slice     = vol[center_z]
            sidx = max(0, center_z-patch_z//2)
            eidx = center_z+patch_z//2+1
            slice_mip = vol[sidx:eidx].max(axis=0)

            images.append(np.stack([slice_mip,slice], axis=-1))


        for i in range(len(images)):
            image = images[i]

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            images[i] = image

        images = torch.stack(images, dim=0)

        labels = torch.zeros(self.num_classes)
        labels[loc_labels] = 1
        return uid, images, labels

    def get_train_sample(self, idx):


        uid = self.uids[idx]
        series_info = self.traindf[self.traindf["SeriesInstanceUID"] == uid]
        modality = series_info["Modality"].iloc[0]

        sample = np.load(self.data_path / f"{uid}.npz")

        vol,coords,loc_labels = sample["vol"],sample["coords"], sample["location_labels"]
        mask = np.zeros_like(vol, dtype=np.uint8)

        for coord,label in zip(coords,loc_labels):
            mask = self.create_hard_ball_mask(mask,coord,label)

        if modality == "CTA":

            # depth, height, width
            spatial_size = np.array(self.cfg.params.spatial_size.cta)
        else:
            spatial_size = np.array(self.cfg.params.spatial_size.mr)

        (patch_z,_,_) = spatial_size
        patch, (center_z,center_y,center_x) = self.extract_patch({"image": np.expand_dims(vol, 0), "label": np.expand_dims(mask, 0), "spatial_size": spatial_size})
        center_label = mask[center_z,center_y, center_x]


        img_width,img_height = (self.cfg.params.img_width, self.cfg.params.img_height)

        # ZxYxX
        patch_image = patch["image"][0].numpy()
        patch_slice = cv2.resize(patch_image[patch_z//2], (img_width, img_height))
        patch_mip   = cv2.resize(patch_image.max(axis=0), (img_width, img_height))

        slice     = cv2.resize(vol[center_z], (img_width, img_height))
        slice_mip = cv2.resize(vol[center_z-patch_z//2:center_z+patch_z//2+1].max(axis=0), (img_width, img_height))

        image = np.stack([patch_mip, patch_slice, slice_mip, slice],axis=-1)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        ## Classification related stuff
        return image, center_label

    def create_hard_ball_mask(self, mask, center, fill_value):
        volume_shape = mask.shape

        # Create a 3D coordinate grid. `ogrid` is memory-efficient.
        z, y, x = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]

        # Unpack the center coordinates
        cz, cy, cx = center
        
        # Calculate the squared Euclidean distance from the center for every point
        dist_sq = (z - cz)**2 + (y - cy)**2 + (x - cx)**2
        mask[dist_sq <= self.cfg.params.radius ** 2] = fill_value
        return mask

class WindowedMIPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
       
        self.cfg = cfg

        # Define augmentations with normalization
        self.train_transforms = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0,0,0,0), std=(1,1,1,1)),  # normalize only image
            ToTensorV2()
        ])
        self.val_transforms = A.Compose([
            A.Normalize(mean=(0,0,0,0), std=(1,1,1,1)),  # normalize only image
            ToTensorV2()
        ])

    def setup(self, stage: str = None):

        data_path = Path(self.cfg.params.data_path)
        df = pd.read_csv(data_path / "train.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        print(f"train_uids: {len(train_uids)}, val_uids: {len(val_uids)}")

        self.train_dataset = WindowedMIP(uids=list(train_uids), cfg=self.cfg, transform=self.train_transforms)
        self.val_dataset = WindowedMIP(uids=list(train_uids[:400]), cfg=self.cfg, transform=self.val_transforms, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.params.batch_size, shuffle=True, num_workers=self.cfg.params.num_workers, pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=1, pin_memory=False)
