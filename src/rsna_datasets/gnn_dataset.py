import torch
#from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
from torch_geometric.data import Dataset, Data
import pytorch_lightning as pl
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from configs.data_config import *
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.transforms import RootedRWSubgraph
import os

torch.set_float32_matmul_precision('medium')


# class RandomSubgraph:
#     """Random node subgraph sampler."""
#     def __init__(self, num_nodes):
#         self.num_nodes = num_nodes
#
#     def __call__(self, data: Data):
#         n_nodes = min(self.num_nodes, data.num_nodes)
#         subset = torch.randperm(data.num_nodes)[:n_nodes]
#
#         # mask edges
#         mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#         mask[subset] = True
#         node_idx_map = torch.zeros(data.num_nodes, dtype=torch.long)
#         node_idx_map[subset] = torch.arange(len(subset))
#
#         row, col = data.edge_index
#         mask_edges = mask[row] & mask[col]
#         edge_index = torch.stack([node_idx_map[row[mask_edges]], node_idx_map[col[mask_edges]]], dim=0)
#
#         x = data.x[subset]
#         cls_labels = data.cls_labels
#         loc_labels = data.loc_labels
#
#         return Data(x=x, edge_index=edge_index, cls_labels=cls_labels, loc_labels=loc_labels)


class GraphDataset(Dataset):
    def __init__(self, uids, cfg, transform=None, mode='train'):
        super().__init__()
        self.uids = uids
        self.cfg = cfg

        self.data_path = Path(self.cfg.meta_dir)
        self.mode = mode

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")

        self.num_classes_ = 13
        self.transform = transform
        self.peTransform = AddRandomWalkPE(walk_length=cfg.walk_length, attr_name=None)
        #self.subgraph_transform = RandomSubgraph(num_nodes=cfg.num_nodes)

    def len(self):
        return len(self.uids)

    def get(self, idx):
        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]

        # load features
        data_path = os.path.join(self.cfg.data_dir, uid)
        feat_path = f'{data_path}/{uid}_point_feats.npy'
        edge_path = f'{data_path}/{uid}_edge_index_k{self.cfg.num_neighbs}.npy'
        feat = torch.from_numpy(np.nan_to_num(np.load(feat_path, mmap_mode='r'), nan=0.0).astype('float32'))
        edge_index = torch.from_numpy(np.load(edge_path, mmap_mode='r'))

        # labels
        cls_labels = torch.tensor(rowdf['Aneurysm Present'].values, dtype=torch.float32)
        loc_labels = torch.tensor(rowdf[list(LABELS_TO_IDX.keys())].values[0], dtype=torch.float32)

        # build data
        data = Data(x=feat, edge_index=edge_index, cls_labels=cls_labels, loc_labels=loc_labels)

        if self.cfg.use_pe:
            data = self.peTransform(data)
        # if self.mode == 'train':
        #     data = self.subgraph_transform(data)
        return data



class GraphDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = None):
        data_path = Path(self.cfg.meta_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        df = df[df["Modality"].isin(self.cfg.modality)]
        train_uids = df[df["fold_id"] != self.cfg.fold_id]["SeriesInstanceUID"]
        val_uids = df[df["fold_id"] == self.cfg.fold_id]["SeriesInstanceUID"]

        self.train_dataset = GraphDataset(uids=list(train_uids), cfg=self.cfg)
        self.val_dataset = GraphDataset(uids=list(val_uids), cfg=self.cfg, mode='valid')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=True, persistent_workers=True)