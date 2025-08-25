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

from monai.transforms import MapTransform, RandCropByLabelClassesd

torch.set_float32_matmul_precision('medium')

class DynamicRandCropByLabelClassesd(MapTransform):
    def __init__(self, keys, label_key="label", spatial_size=(4,128,128), num_classes=14, num_samples=1):
        super().__init__(keys)
        self.label_key = label_key
        self.num_classes = num_classes
        self.spatial_size = spatial_size
        self.num_samples = num_samples

    def __call__(self, data):
        d = dict(data)

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
            spatial_size=self.spatial_size,
            num_classes=self.num_classes,
            num_samples=self.num_samples,
            ratios=ratios,
        )
        return cropper(d)



class WindowedMIP(Dataset):
    def __init__(self, uids, cfg, transform=None, mode="train"):
        self.uids = uids
        self.cfg = cfg
        self.data_path = Path(cfg.param.data_path)
        self.traindf = pd.read_csv(self.data_path / "train.csv")
        self.extract_patch = DynamicRandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    num_classes=self.cfg.params.num_classes,   # required since label is one-hot
                    num_samples=1,
        )

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        series_info = self.traindf[self.traindf["SeriesInstanceUID"] == uid]
        modality = series_info["Modality"].iloc[0]

        sample = np.load(self.data_path / f"{uid}.npz")

        vol,coords,loc_labels = sample["vol"],sample["coords"], sample["location_labels"]
        mask = np.zeros_like(vol, dtype=np.uint8)

        for coord,label in zip(coords,loc_labels):
            mask = self.create_hard_ball_mask(mask,coord,label + 1)

        if modality == "CTA":

            # depth, height, width
            spatial_size = np.array(self.cfg.params.spatial_size.cta)
        else:
            spatial_size = np.array(self.cfg.params.spatial_size.mr)

        patch = self.extract_patch({"image": np.expand_dims(vol, 0), "label": np.expand_dims(mask, 0), "spatial_size": spatial_size})[0]

        # 4x128x128
        patch_image = patch["image"][0].numpy()

        # 4x128x128
        patch_mask  = patch["label"][0].numpy()

        # clsx4x128x128 -> clsx128x128
        mask = self.one_hot_encode_mask(patch_mask).max(axis=1)
        patch_label = patch_mask[*(spatial_size // 2)]

        if (patch_label != 0):
            cidx = np.where(loc_labels == patch_label)[0][0]
            z1 = coords[cidx]
            slice = vol[z1]
            mip_slice = vol[z1]

        return 




    def one_hot_encode_mask(self, mask):
        """
        Convert mask of shape (D, H, W) with labels in [0, num_classes]
        into one-hot encoding of shape (num_classes, D, H, W).
        """

        # create one-hot using broadcasting
        onehot = np.eye(self.cfg.params.num_classes, dtype=np.uint8)[mask]   # (D,H,W,num_classes)

        # move channels to front -> (num_classes, D, H, W)
        onehot = np.moveaxis(onehot, -1, 0)

        return onehot




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
