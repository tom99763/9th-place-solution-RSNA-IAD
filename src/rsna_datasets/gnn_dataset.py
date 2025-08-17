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
import ast
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')

def add_dcm_idx(df_loc, cfg):
    series_maps = {}
    for series_uid in tqdm(df_loc["SeriesInstanceUID"].unique()):
        series_path_ = f"{cfg.meta_dir}/series/{series_uid}"
        files = sorted(os.listdir(series_path_))  # ensure consistent order
        # strip .dcm to get SOPInstanceUID
        series_maps[series_uid] = {
            f.replace(".dcm", ""): idx for idx, f in enumerate(files)
        }
    mapping = [
        {"SeriesInstanceUID": series_uid, "SOPInstanceUID": sop_uid, "dcm_idx": int(idx)}
        for series_uid, uid_map in series_maps.items()
        for sop_uid, idx in uid_map.items()
    ]
    df_map = pd.DataFrame(mapping)
    df_loc = df_loc.merge(df_map, on=["SeriesInstanceUID", "SOPInstanceUID"], how="left")
    return df_loc


def generate_labels(points, locdf, radius):
    #within redius->assign label
    #out of raidus->assign zeros label
    #points: (N, 3); (z, y, x)
    #loc: (k, 2); (y, x)
    loc = locdf[['y', 'x']].values

    pass


class GraphDataset(Dataset):
    def __init__(self, uids, cfg, transform=None, mode='train'):
        super().__init__()
        self.uids = uids
        self.cfg = cfg

        self.data_path = Path(self.cfg.meta_dir)
        self.mode = mode

        self.train_df = pd.read_csv(self.data_path / "train_df.csv")
        self.loc_df = pd.read_csv(self.data_path / "train_localizers.csv")
        self.loc_df['x'] = self.loc_df['coordinates'].map(lambda x: ast.literal_eval(x)['x'])
        self.loc_df['y'] = self.loc_df['coordinates'].map(lambda x: ast.literal_eval(x)['y'])
        self.loc_df = add_dcm_idx(self.loc_df, self.cfg)

        self.num_classes_ = 13
        self.transform = transform
        self.peTransform = AddRandomWalkPE(walk_length=cfg.walk_length, attr_name=None)

    def len(self):
        return len(self.uids)

    def get(self, idx):
        uid = self.uids[idx]
        rowdf = self.train_df[self.train_df["SeriesInstanceUID"] == uid]
        locdf = self.loc_df[self.loc_df["SeriesInstanceUID"] == uid]

        # load features
        data_path = os.path.join(self.cfg.data_dir, uid)
        feat_path = f'{data_path}/{uid}_point_feats.npy'
        edge_path = f'{data_path}/{uid}_edge_index_k{self.cfg.num_neighbs}.npy'
        point_path = f'{data_path}/{uid}_points.npy'
        feat = torch.from_numpy(np.nan_to_num(np.load(feat_path, mmap_mode='r'), nan=0.0).astype('float32'))
        edge_index = torch.from_numpy(np.load(edge_path, mmap_mode='r'))
        points = torch.from_numpy(np.load(point_path, mmap_mode='r'))

        # labels
        cls_labels = torch.tensor(rowdf['Aneurysm Present'].values, dtype=torch.float32)
        loc_labels = torch.tensor(rowdf[list(LABELS_TO_IDX.keys())].values[0], dtype=torch.float32)
        labels = generate_labels(points, locdf, self.cfg.radius)

        # build data
        data = Data(x=feat, edge_index=edge_index, y=labels,
                    cls_labels = cls_labels, loc_labels = loc_labels)

        if self.cfg.use_pe:
            data = self.peTransform(data)
        return data



class GraphDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = None):
        data_path = Path(self.cfg.meta_dir)
        df = pd.read_csv(data_path / "train_df.csv")
        df = df[df["Modality"].isin(self.cfg.modality)]
        dfloc = pd.read_csv(data_path / "train_localizers.csv")
        df = df[df["SeriesInstanceUID"].isin(dfloc["SeriesInstanceUID"])]
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