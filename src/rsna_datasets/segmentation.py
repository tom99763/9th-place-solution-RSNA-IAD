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

        vol,coords,loc_labels = sample["vol"],sample["coords"], sample["location_labels"]
        mask = np.zeros_like(vol, dtype=np.uint8)

        vol_depth,_,_ = vol.shape

        for coord,label in zip(coords,loc_labels):
            mask = self.create_hard_ball_mask(mask,coord,label)

        if modality == "CTA":
            z_stride = self.cfg.params.sliding_stride.cta
            spatial_size = np.array(self.cfg.params.spatial_size.cta)
        else:
            z_stride = self.cfg.params.sliding_stride.mr
            spatial_size = np.array(self.cfg.params.spatial_size.mr)


        images = []
        masks = []

        (patch_z,_,_) = spatial_size

        for center_z in range(0, vol_depth, z_stride):
            slice     = vol[center_z]
            sidx = max(0, center_z-patch_z//2)
            eidx = center_z+patch_z//2+1
            slice_mip = vol[sidx:eidx].max(axis=0)

            images.append(np.stack([slice_mip,slice], axis=0))
            masks.append(mask[sidx:eidx].max(axis=0))

        images = np.stack(images, axis=0)
        masks  = np.stack(masks, axis=0)

        if self.transform:
            transformed = self.transform(image=images, label=masks)
            images = transformed["image"]
            masks = transformed["label"]

        return images, masks

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
        patch, (center_z,_,_) = self.extract_patch({"image": np.expand_dims(vol, 0), "label": np.expand_dims(mask, 0), "spatial_size": spatial_size})

        img_width,img_height = (self.cfg.params.img_width, self.cfg.params.img_height)

        # ZxYxX
        patch_image = patch["image"][0].numpy()
        print(f"{patch_image.shape=}")
        patch_slice = cv2.resize(patch_image[patch_z//2], (img_width, img_height))
        patch_mip   = cv2.resize(patch_image.max(axis=0), (img_width, img_height))

        slice     = cv2.resize(vol[center_z], (img_width, img_height))
        slice_mip = cv2.resize(vol[center_z-patch_z//2:center_z+patch_z//2+1].max(axis=0), (img_width, img_height))

        patch_mask  = cv2.resize(patch["label"][0].numpy().T, (img_width, img_height)).T
        mask = patch_mask.max(axis=0)

        image = np.stack([patch_mip,patch_slice, slice_mip, slice])

        if self.transform:
            transformed = self.transform(image=image, label=mask)
            image = transformed["image"]
            mask = transformed["label"]

        return image, mask

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
