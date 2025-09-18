import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from configs.data_config import *
from typing import List, Tuple, Dict, Optional
import os
import pydicom
import cv2
from scipy import ndimage

torch.set_float32_matmul_precision('medium')


class DICOMPreprocessorKaggle:
    """DICOM preprocessing system for EfficientNet"""

    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """Load DICOM series"""
        series_path = Path(series_path)
        series_name = series_path.name

        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")

        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception as e:
                continue

        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")

        return datasets, series_name

    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """Extract position information for each slice"""
        slice_info = []

        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }

            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    info['z_position'] = float(info['instance_number'])
            except Exception as e:
                info['z_position'] = float(i)

            slice_info.append(info)

        return slice_info

    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """Sort slices by z-coordinate"""
        return sorted(slice_info, key=lambda x: x['z_position'])

    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[
        Optional[float], Optional[float]]:
        """Get windowing parameters based on modality"""
        modality = getattr(ds, 'Modality', 'CT')

        if modality == 'CT':
            return "CT", "CT"
        else:
            return None, None

    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float],
                                     width: Optional[float]) -> np.ndarray:
        """Apply windowing or statistical normalization"""
        if center is not None and width is not None:
            p1, p99 = 0, 500

            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    return np.zeros_like(img, dtype=np.uint8)
        else:
            p1, p99 = np.percentile(img, [1, 99])

            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    return np.zeros_like(img, dtype=np.uint8)

    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """Extract 2D pixel array from DICOM"""
        img = ds.pixel_array.astype(np.float32)

        if img.ndim == 3:
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]

        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)

        return img

    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target size"""
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)

        if current_shape == target_shape:
            return volume

        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]

        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]

        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]

        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')

        return resized_volume.astype(np.uint8)

    def process_series(self, series_path: str) -> np.ndarray:
        """Process DICOM series and return as NumPy array"""
        try:
            datasets, series_name = self.load_dicom_series(series_path)

            first_ds = datasets[0]
            first_img = first_ds.pixel_array

            if len(datasets) == 1 and first_img.ndim == 3:
                return self._process_single_3d_dicom(first_ds, series_name)
            else:
                return self._process_multiple_2d_dicoms(datasets, series_name)

        except Exception as e:
            raise

    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """Process single 3D DICOM file"""
        volume = ds.pixel_array.astype(np.float32)

        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            volume = volume * float(slope) + float(intercept)

        window_center, window_width = self.get_windowing_params(ds)

        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)

        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)

        return final_volume

    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """Process multiple 2D DICOM files"""
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        first_img = self.extract_pixel_array(sorted_slices[0]['dataset'])
        window_center, window_width = self.get_windowing_params(sorted_slices[0]['dataset'], first_img)
        processed_slices = []

        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            # resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
            processed_slices.append(processed_img)

        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)

        return final_volume


class VolumeSliceDataset(Dataset):
    """
    Dataset to load .npz image volumes and serve random 2D slices.
    """

    def __init__(self, uids, cfg, transform=None, mode="train"):

        self.uids = uids
        self.cfg = cfg

        self.data_path = Path(self.cfg.params.data_dir)

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")
        self.label_df = pd.read_csv(self.data_path / "label_df.csv")

        self.num_classes = cfg.params.num_classes
        self.transform = transform

        self.mode = mode
        self.depth_mr = self.cfg.params.depth.mr
        self.depth_ct = self.cfg.params.depth.ct
        self.preprocessor = DICOMPreprocessorKaggle()

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]
        labeldf = self.label_df[self.label_df["SeriesInstanceUID"] == uid]
        #modality = rowdf["Modality"].iloc[0]

        series_path = self.data_path / f"series/{uid}"
        volume = self.preprocessor.process_series(series_path)
        volume = volume.transpose(1, 2, 0) # (D,H,W) -> (H,W,D)


        labels = np.zeros(self.num_classes)
        if int(rowdf["Aneurysm Present"].iloc[0]) == 1:
            labels[0] = 1
            class_idxs = [LABELS_TO_IDX[loc] + 1 for loc in labeldf["location"].tolist()]
            labels[class_idxs] = 1

        if self.transform:
            # BxDxHxW
            volume = self.transform(image=volume)["image"]
        return uid, volume, labels


class VolumeDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.train_transforms = A.Compose([
            A.Resize(384, 384),
            A.Normalize(),
            ToTensorV2(),
        ])

        self.val_transforms = A.Compose([
            A.Resize(384, 384),
            A.Normalize(),
            ToTensorV2(),
        ])

    def setup(self, stage: str = None):
        data_path = Path(self.cfg.params.data_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = VolumeSliceDataset(uids=list(train_uids), cfg=self.cfg, transform=self.train_transforms)
        self.val_dataset = VolumeSliceDataset(uids=list(val_uids), cfg=self.cfg, transform=self.val_transforms,
                                                 mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.params.batch_size, shuffle=True,
                          num_workers=self.cfg.params.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset
                          , batch_size=1
                          , num_workers=1
                          , pin_memory=True)

if __name__ == '__main__':
    proc = DICOMPreprocessorKaggle()
    series_path = '../data/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647'
    o = proc.process_series(series_path)
    print(o.shape)
