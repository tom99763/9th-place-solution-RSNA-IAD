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
sitk.ProcessObject.SetGlobalWarningDisplay(False)
logger = logging.getLogger(__name__)

def load_series2vol(series_path, series_id=None, spacing_tolerance=1e-3, resample=False, default_thickness=1.0, max_workers=20):
    reader = sitk.ImageSeriesReader()

    # Get all series IDs
    series_ids = reader.GetGDCMSeriesIDs(series_path)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {series_path}")

    # Pick first if not specified
    series_id = str(series_ids[0] if series_id is None else series_id)

    # Get file names for the series
    all_files = reader.GetGDCMSeriesFileNames(series_path, series_id)

    # --- Parallel metadata read (fast size check) ---
    def get_size(f):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        return (int(ds.Rows), int(ds.Columns)), f

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        sizes = list(ex.map(get_size, all_files))

    # Pick the most common size
    most_common_size = Counter(s[0] for s in sizes).most_common(1)[0][0]
    files = [f for (sz, f) in sizes if sz == most_common_size]

    # --- Now read the actual image series ---
    reader.SetFileNames(files)
    image = reader.Execute()

    # --- Fix zero thickness ---
    spacing = list(image.GetSpacing())
    if spacing[2] == 0:
        spacing[2] = default_thickness
        image.SetSpacing(spacing)

    # --- Optional resample ---
    if resample and abs(spacing[2] - spacing[0]) > spacing_tolerance:
        new_spacing = [spacing[0], spacing[1], spacing[0]]
        new_size = [
            int(round(image.GetSize()[0] * spacing[0] / new_spacing[0])),
            int(round(image.GetSize()[1] * spacing[1] / new_spacing[1])),
            int(round(image.GetSize()[2] * spacing[2] / new_spacing[2]))
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        image = resampler.Execute(image)

    # Convert to numpy array
    volume = sitk.GetArrayFromImage(image)
    return volume


class RSNASegDataset(Dataset):
    def __init__(self, uids, dataset_config, mode):
        super().__init__()
        # init datasets
        self.data_path = dataset_config.path
        self.uids = uids
        self.transforms = generate_transforms(dataset_config.transforms[mode])
        self.mode = mode

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        #load volume
        vol_path = f'{self.data_path}/series/{uid}'
        vol = load_series2vol(vol_path).astype(np.float32)

        #load mask
        mask_path = f'{self.data_path}/segmentations/{uid}_cowseg.nii'
        nii_image = nib.load(mask_path)
        mask = nii_image.get_fdata()
        mask = np.transpose(mask, (2, 1, 0)) #(D, H, W)
        mask = np.flip(np.flip(mask, axis=1), axis=2)

        #vol = vol.copy()
        #mask = mask.copy()
        vol = torch.as_tensor(vol.copy()).contiguous()
        mask = torch.as_tensor(mask.copy()).contiguous()



        #transforms
        transformed = self.transforms({'Image': vol, 'Mask': mask})
        if self.mode == 'train':
            return transformed
        return transformed['Image'], transformed['Mask']
