import torch
#from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
from torch_geometric.data import Dataset, Data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from configs.data_config import *
from torch_geometric.transforms import AddRandomWalkPE
import os

torch.set_float32_matmul_precision('medium')


class GraphDataset(Dataset):
    def __init__(self, uids, cfg, transform=None):
        super().__init__()
        self.uids = uids
        self.cfg = cfg

        self.data_path = Path(self.cfg.data_dir)

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")

        self.num_classes_ = 14
        self.transform = transform
        self.peTransform = AddRandomWalkPE(walk_length=cfg.walk_length, attr_name=None)

    def len(self):
        return len(self.uids)

    def get(self, idx):

        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]

        #process
        data_path = os.path.join(self.cfg.data_path, uid)
        feat_path = os.path.join(data_path, 'point_feats.npy')
        edge_path = os.path.join(data_path, f'edge_index_k{self.cfg.num_neighbs}.npy')
        feat = torch.from_numpy(np.load(feat_path).astype('float32'))
        edge_index = torch.from_numpy(np.load(edge_path))

        #labels
        cls_labels = torch.tensor(rowdf['Aneurysm Present'].values, dtype=torch.float32)
        loc_labels = torch.tensor(rowdf[list(LABELS_TO_IDX.keys())].values[0], dtype=torch.float32)

        #build data
        data = Data(x=feat, edge_index=edge_index, cls_labels=cls_labels, loc_labels=loc_labels)
        if self.cfg.use_pe:
            data = self.peTransform(data)
        return data


class NpzDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = None):
        data_path = Path(self.cfg.data_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        df = df[df["Modality"].isin(self.cfg.modality)]
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = GraphDataset(uids=list(train_uids), cfg=self.cfg)
        self.val_dataset = GraphDataset(uids=list(val_uids), cfg=self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=1, pin_memory=True)