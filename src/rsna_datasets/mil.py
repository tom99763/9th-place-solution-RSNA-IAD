import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from pathlib import Path
from configs.data_config import *
from pathlib import Path

from monai import transforms as T

torch.set_float32_matmul_precision('medium')

def z_to_slice_index(z_coord, depth):
    """Assume z_coord is in voxel coordinates where 0 <= z < depth."""
    return int(round(float(z_coord)))

def make_triplet(volume: np.ndarray, i: int):
    """
    volume: np.ndarray shape (D, H, W)
    i: center slice index
    returns: np.ndarray shape (3, H, W), stacked in channel order
    """
    D, H, W = volume.shape
    trip = []
    for k in [i-1, i, i+1]:
        kk = min(max(k, 0), D-1)
        trip.append(volume[kk])
    arr = np.stack(trip, axis=0).astype(np.float32)  # (3, H, W)
    return arr

# -------------------------
# Dataset
# -------------------------
class AneurysmVolumeDataset(Dataset):
    """
    volumes_meta: list of dicts, each dict contains:
        - 'volume': np.ndarray (D,H,W) or path to load (if you want lazy loading)
        - 'points': list of (z,y,x, artery_label_index) OR list of dicts {'coord':(z,y,x), 'label':int}
        - 'label': volume-level class index or multi-hot vector (len 13)
    N_slices: number of slices to sample per volume (memory control)
    pos_radius: expand positive slice indices by +/- pos_radius
    """
    def __init__(self,
                 uids,
                 cfg,
                 N_slices: int = 32,
                 pos_radius: int = 1,
                 transforms = None,
                 mode: str = 'train'):
        self.uids = uids
        self.N_slices = N_slices
        self.pos_radius = pos_radius
        self.transforms = transforms
        self.mode = mode

        self.cfg = cfg
        self.data_path = Path(cfg.params.data_path)

        self.D = 96
        self.n_classes = 13

    def __len__(self):
        return len(self.uids)

    def _get_positive_slice_indices(self, metadata):
        """
        points: list of dicts or tuples containing z coord and artery label
        returns: dict mapping slice_idx -> list of artery labels (could be multiple)
        """

        (orig_depth,_,_) = metadata["Shape"]
        
        slice_to_labels = {}
        for coord, lab in zip(metadata["Coords"], metadata["Label"]):
            z = self.D * (coord[0] / orig_depth)
            idx = z_to_slice_index(z, self.D)
            for s in range(idx - self.pos_radius, idx + self.pos_radius + 1):
                if s < 0 or s >= self.D:
                    continue
                slice_to_labels.setdefault(s, set()).add(lab)
        return {k: list(v) for k,v in slice_to_labels.items()}

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.get_train_sample(idx)
        else:
            return self.get_val_sample(idx)

    def get_val_sample(self, idx):
        uid = self.uids[idx]

        sample = np.load(self.data_path / f"{uid}.npz", allow_pickle=True)
        volume, metadata = sample["vol"], sample["metadata"].item()
        chosen = [i for i in range(self.D) if i % 3 == 1]

        triplets = []
        for s in chosen:
            trip = make_triplet(volume, s)  # (3,H,W)
            triplets.append(trip)

        X = np.stack(triplets, axis=0)
        if self.transforms:
            X = self.transforms({"images": X})["images"]
        volume_label = np.zeros(self.n_classes)
        volume_label[metadata["Label"]] = 1

        return {
            'X': X,  # (N_slices, 3, H, W)
            'volume_label': volume_label,
            'slice_indices': chosen,
        }

    def get_train_sample(self, idx):
        uid = self.uids[idx]

        sample = np.load(self.data_path / f"{uid}.npz", allow_pickle=True)
        volume, metadata = sample["vol"], sample["metadata"].item()
       
        # map slice -> labels
        slice_to_labels = self._get_positive_slice_indices(metadata)
        
        # choose slices
        pos_indices = sorted(list(slice_to_labels.keys()))
        neg_indices = [i for i in range(self.D) if i not in slice_to_labels]

        # sample negatives to fill N_slices
        chosen = []
        # include all positives (or sample if too many)
        if len(pos_indices) >= self.N_slices:
            chosen = list(np.random.choice(pos_indices, size=self.N_slices, replace=False))
        else:
            chosen = list(pos_indices)
            n_needed = self.N_slices - len(chosen)
            if len(neg_indices) == 0:
                neg_sample = []
            else:
                neg_sample = list(np.random.choice(neg_indices, size=n_needed, replace=(len(neg_indices) < n_needed)))
            chosen += neg_sample
    
        # shuffle order (so attention can't trivially learn pos indices)
        np.random.shuffle(chosen)

        # build data tensors
        triplets = []
        slice_multilabels = []
    
        for s in chosen:
            trip = make_triplet(volume, s)  # (3,H,W)
            triplets.append(trip)

            # build slice label: if slice contains any aneurysm points, create target(s)
            if s in slice_to_labels:
                labs = slice_to_labels[s]
                # if multiple artery labels at same slice, we create a multi-hot vector
                m = np.zeros(self.n_classes, dtype=np.float32)
                for L in labs:
                    m[L] = 1.0
                slice_multilabels.append(m)
            else:
                slice_multilabels.append(np.zeros(self.n_classes))
        
        # stack into tensors
        # X: (N_slices, 3, H, W)

        X = np.stack(triplets, axis=0)
        if self.transforms:
            X = self.transforms({"images": X})["images"]
        slice_multilabels = np.stack(slice_multilabels, axis=0)  # (N_slices, n_classes)
        volume_label = np.zeros(self.n_classes)
        volume_label[metadata["Label"]] = 1

        return {
            'X': X,  # (N_slices, 3, H, W)
            'slice_labels': slice_multilabels,  # (N_slices, n_classes) float
            'volume_label': volume_label,
            'slice_indices': chosen,
        }


class NpzDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
       
        self.cfg = cfg

        # Define augmentations with normalization
        self.train_transforms = T.Compose ([
            T.EnsureChannelFirstd(keys=["images"], channel_dim=1),
            T.Orientationd(keys=["images"], axcodes="RAS"), # Standardize orientation to RAS+
            T.ScaleIntensityd(keys="images"),  # normalize volume intensity
            T.OneOf(
                [
                    # Affine 1: slight, focus on XY dims
                    T.RandAffined(
                        keys=["images"],
                        rotate_range=((0, 0), (0, 0), (0, 360)),
                        scale_range=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)),
                        prob=0.67,
                    ),
                    # Affine2: heavier, focus on all XYZ dims
                    T.RandAffined(
                        keys=["images"],
                        rotate_range=((-15, 15), (-15, 15), (0, 360)),
                        shear_range=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
                        scale_range=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)),
                        prob=0.33,
                    ),
                ],
            ),

             # --- Spatial Augmentations ---
            T.RandFlipd(keys=["images"], prob=0.5, spatial_axis=1),  # flip along height
            T.RandFlipd(keys=["images"], prob=0.5, spatial_axis=2),  # flip along width

            T.RandGaussianNoised(keys=["images"], prob=0.2, std=0.05),
            T.RandStdShiftIntensityd(keys=["images"], factors=0.1, prob=0.2),
            T.Lambdad(keys=["images"], func=lambda x: x.permute(1, 0, 2, 3)), # 3x32x384x384 -> 32x3x384x384

        ])
        self.val_transforms = T.Compose ([
            T.EnsureChannelFirstd(keys=["images"], channel_dim=1),
            T.Orientationd(keys=["images"], axcodes="RAS"), # Standardize orientation to RAS+
            T.ScaleIntensityd(keys="images"),  # normalize volume intensity
            T.Lambdad(keys=["images"], func=lambda x: x.permute(1, 0, 2, 3)), # 3x32x384x384 -> 32x3x384x384
        ])


    def setup(self, stage: str = None):

        data_path = Path(self.cfg.params.data_path)
        df = pd.read_csv(data_path / "train.csv")
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        print(f"train_uids: {len(train_uids)}, val_uids: {len(val_uids)}")

        self.train_dataset = AneurysmVolumeDataset(uids=list(train_uids), cfg=self.cfg, transforms=self.train_transforms)
        self.val_dataset = AneurysmVolumeDataset(uids=list(val_uids), cfg=self.cfg, transforms=self.val_transforms, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.params.batch_size, shuffle=True, num_workers=self.cfg.params.num_workers, pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.params.batch_size, num_workers=4, pin_memory=False)
