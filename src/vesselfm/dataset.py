import logging
from typing import Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data import generate_transforms
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
import pydicom
from collections import Counter
import nibabel as nib
from utils.io import determine_reader_writer
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    RandCropByLabelClassesd,
    SpatialPadd,
    ConcatItemsd,
    ToTensord,
    Lambdad,
)
from monai.transforms import MapTransform


sitk.ProcessObject.SetGlobalWarningDisplay(False)
logger = logging.getLogger(__name__)

class ModalityIntensityScalingd(MapTransform):
    """
    Modality-agnostic normalization:
      - Robust z-score within 2–98 percentiles
      - Then rescale to [0, 1]
    """
    def __init__(self, keys=("Image",)):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]

        # robust stats
        p2, p98 = np.percentile(img, (2, 98))
        mask = (img >= p2) & (img <= p98)
        mean = np.mean(img[mask])
        std = np.std(img[mask]) + 1e-6

        img = (img - mean) / std
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-6), 0, 1)

        d[self.keys[0]] = img.astype(np.float32)
        return d


def _generate_transforms(vol_size, input_size, mode):
    if mode == "train":
        return Compose([
            EnsureChannelFirstd(keys=["Image", "Mask"], channel_dim="no_channel"),
            EnsureTyped(keys=["Image", "Mask"]),
            Resized(keys=["Image", "Mask"], spatial_size=vol_size, mode=["trilinear", "nearest"]),
            RandCropByLabelClassesd(
                keys=["Image", "Mask"],
                label_key="Mask",
                spatial_size=input_size,
                num_classes=13,
                ratios=[1] * 13,
                num_samples=4,
                image_key="Image",
                allow_smaller=True,
            ),
            #ModalityIntensityScalingd(keys=["Image"]),
            SpatialPadd(keys=["Image", "Mask"], spatial_size=input_size, mode="constant", method="symmetric"),
            ToTensord(keys=["Image", "Mask"]),
        ])
    else:  # val / test
        return Compose([
            EnsureChannelFirstd(keys=["Image", "Mask"], channel_dim="no_channel"),
            EnsureTyped(keys=["Image", "Mask"]),
            Resized(keys=["Image", "Mask"], spatial_size=vol_size, mode=["trilinear", "nearest"]),
            #ModalityIntensityScalingd(keys=["Image"]),
            ToTensord(keys=["Image", "Mask"]),
        ])



class RSNASegDataset(Dataset):
    def __init__(self, uids, dataset_config, mode):
        super().__init__()
        # init datasets
        print(dataset_config)
        self.data_path = dataset_config.path
        self.uids = uids
        #self.reader = determine_reader_writer(dataset_config.file_format)()
        self.transforms = _generate_transforms(
            dataset_config['vol_size'], dataset_config['input_size'], mode) #generate_transforms(dataset_config.transforms[mode])
        self.mode = mode

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        vol_path = f'{self.data_path}/seg_vols/{uid}.npz'
        mask_path = f'{self.data_path}/segmentations/{uid}_cowseg.nii'
        # vol = self.reader.read_images(vol_path)[0].astype(np.float32)
        # mask = self.reader.read_images(mask_path)[0]

        #mask
        mask_data = nib.load(mask_path)
        mask = mask_data.get_fdata()
        affine = mask_data.affine
        mask = np.flip(mask, axis=0)
        mask = mask.transpose(2, 1, 0)
        mask = np.flip(mask, axis=1) #z, y, x

        #volume
        vol_data = np.load(vol_path)['vol'][..., 0] #z, y, x
        transformed = self.transforms({'Image': vol, 'Mask': mask})
        if self.mode == 'train':
            return transformed
        return transformed['Image'], transformed['Mask']
