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
    def __init__(self, uids, cfg, transform=None):
        super().__init__()
        self.uids = uids
        self.cfg = cfg
        self.data_path = Path(self.cfg.data_dir)
        self.transform = transform
        self.peTransform = AddRandomWalkPE(walk_length=cfg.walk_length, attr_name=None)

    def len(self):
        return len(self.uids)

    def get(self, idx):
        uid = self.uids[idx]
        data_path = self.data_path/f'extract_data/{uid}'
        point_path = os.path.join(data_path, f'{uid}_points.npy')
        feat_path = os.path.join(data_path, f'{uid}_extract_feat.npy')
        label_path = os.path.join(data_path, f'{uid}_label.npy')
        if self.cfg.graph_type == 'knn_graph':
            edge_path = os.path.join(data_path, f'{uid}_edge_index_k{self.cfg.k_neibs}.npy')
        elif self.cfg.graph_type == 'delaunay_graph':
            edge_path = os.path.join(data_path, f'{uid}_edge_index_delaunay.npy')
        else:
            raise Exception('invalid graph type')

        points = torch.from_numpy(np.load(point_path).astype('float32'))
        feat = torch.from_numpy(np.load(feat_path).astype('float32'))
        edge_index = torch.from_numpy(np.load(edge_path))
        labels = torch.from_numpy(np.load(label_path))
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
        data_path = Path(self.cfg.data_dir)
        uids = os.listdir(data_path /'extract_data')
        df = pd.read_csv(data_path / "train_df.csv")
        df = df[df["SeriesInstanceUID"].isin(uids)].copy()
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]
        self.train_dataset = GraphDataset(uids=list(train_uids), cfg=self.cfg)
        self.val_dataset = GraphDataset(uids=list(val_uids), cfg=self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=True, persistent_workers=True)