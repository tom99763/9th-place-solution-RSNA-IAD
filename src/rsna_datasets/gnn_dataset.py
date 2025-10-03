import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Data
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pathlib import Path
from torch_geometric.transforms import AddRandomWalkPE
import os

torch.set_float32_matmul_precision('medium')


class GraphDataset(Dataset):
    def __init__(self, uids, cfg, fold_index, transform=None):
        super().__init__()
        self.fold_index = fold_index
        self.uids = uids
        self.cfg = cfg
        self.data_path = Path(self.cfg.data_dir)
        self.transform = transform
        self.peTransform = AddRandomWalkPE(walk_length=cfg.walk_length, attr_name=None)

    def len(self):
        return len(self.uids)

    def get(self, idx):
        uid = self.uids[idx]
        fold_index = self.fold_index[idx]
        data_path = self.data_path/f'extract_data/fold{fold_index}/{uid}'
        point_path = os.path.join(data_path, f'{uid}_points_fold.npy')
        feat_path = os.path.join(data_path, f'{uid}_extract_feat_fold.npy')
        label_path = os.path.join(data_path, f'{uid}_label_fold.npy')
        if self.cfg.graph_type == 'knn_graph':
            edge_path = os.path.join(data_path, f'{uid}_edge_index_k{self.cfg.k_neibs}_fold.npy')
        elif self.cfg.graph_type == 'delaunay_graph':
            edge_path = os.path.join(data_path, f'{uid}_edge_index_delaunay_fold.npy')
        else:
            raise Exception('invalid graph type')

        points = torch.from_numpy(np.load(point_path, mmap_mode="r").astype('float32'))
        feat = torch.from_numpy(np.load(feat_path, mmap_mode="r").astype('float32'))
        edge_index = torch.from_numpy(np.load(edge_path, mmap_mode="r"))
        labels = torch.from_numpy(np.load(label_path, mmap_mode="r"))
        cls_labels = labels.max()

        # build data
        data = Data(x=feat, edge_index=edge_index, y=labels,
                    cls_labels = cls_labels, points = points)
        if self.cfg.use_pe:
            data = self.peTransform(data)
        return data



class GraphDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = None):
        #select oof features and points
        data_path = Path(self.cfg.data_dir)
        uids = os.listdir(data_path / f'extract_data/fold{self.cfg.fold_id}')
        df = pd.read_csv(data_path / "train_df.csv")
        df = df[df["SeriesInstanceUID"].isin(uids)].copy()
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]
        fold_index_train = df[df["fold_id"] != self.cfg.fold_id].fold_id.tolist()
        fold_index_val = df[df["fold_id"] == self.cfg.fold_id].fold_id.tolist()
        self.train_dataset = GraphDataset(uids=list(train_uids), cfg=self.cfg, fold_index = fold_index_train)
        self.val_dataset = GraphDataset(uids=list(val_uids), cfg=self.cfg, fold_index = fold_index_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=True, persistent_workers=True)