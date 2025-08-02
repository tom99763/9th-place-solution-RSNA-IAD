
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torchmetrics
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path

from hydra.utils import instantiate
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')

@torch.no_grad()
def eval_one_series(volume, model):

    volume = torch.from_numpy(volume).cuda()
    volume = torch.stack([volume,volume,volume], dim=1)

    pred_cls = []
    pred_locs = []

    for batch_idx in range(0,volume.shape[0], 64):
        pc,pl = model(volume[batch_idx:batch_idx+64])
        pred_cls.append(pc)
        pred_locs.append(pl)

    pred_cls = torch.vstack(pred_cls)
    pred_locs = torch.vstack(pred_locs)

    pred_cls = pred_cls.squeeze()

    return pred_cls.max().sigmoid().item(), pred_locs.max(dim=0).values.sigmoid().cpu().numpy()


class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters() # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def validation(cfg: DictConfig) -> None:

    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    model = instantiate(cfg.model, pretrained=False)
    pl_model = LitTimmClassifier.load_from_checkpoint("./models/efficient_b2-epoch=27-val_loss=0.7049.ckpt")
    # pl_model = LitTimmClassifier(model, cfg).cuda()
    pl_model.eval()


    data_path = Path(cfg.data_dir)
    df = pd.read_csv(data_path / "train_df.csv")

    val_uids = list(df[df["fold_id"] == cfg.fold_id]["SeriesInstanceUID"])


    cls_labels = []
    loc_labels = []
    pred_cls_probs = []
    pred_loc_probs = []

    for uid in tqdm(val_uids):
        rowdf = df[df["SeriesInstanceUID"] == uid]

        loc_label = np.zeros(13)

        for idx, label in enumerate(LABELS):
            loc_label[idx] = int(rowdf[label].iloc[0])

        cls_labels.append(rowdf["Aneurysm Present"].iloc[0])
        loc_labels.append(loc_label)

        with np.load(f"./data/processed/{uid}.npz") as data:
            volume = data['vol'].astype(np.float32)
            cls_prob, loc_probs =  eval_one_series(volume, pl_model)

            pred_cls_probs.append(cls_prob)
            pred_loc_probs.append(loc_probs)

    cls_labels = np.array(cls_labels)
    pred_cls_probs = np.array(pred_cls_probs)

    loc_labels = np.stack(loc_labels)
    pred_loc_probs = np.stack(pred_loc_probs)


    loc_auc_macro = roc_auc_score(loc_labels, pred_loc_probs, average="micro")
    cls_auc = roc_auc_score(cls_labels, pred_cls_probs)

    print(f"Fold: {cfg.fold_id}, cls_auc: {cls_auc}, loc_auc_macro: {loc_auc_macro}")

        

if __name__ == "__main__":

    LABELS_TO_IDX = {
        'Anterior Communicating Artery': 0,
        'Basilar Tip': 1,
        'Left Anterior Cerebral Artery': 2,
        'Left Infraclinoid Internal Carotid Artery': 3,
        'Left Middle Cerebral Artery': 4,
        'Left Posterior Communicating Artery': 5,
        'Left Supraclinoid Internal Carotid Artery': 6,
        'Other Posterior Circulation': 7,
        'Right Anterior Cerebral Artery': 8,
        'Right Infraclinoid Internal Carotid Artery': 9,
        'Right Middle Cerebral Artery': 10,
        'Right Posterior Communicating Artery': 11,
        'Right Supraclinoid Internal Carotid Artery': 12
    }

    
    LABELS = sorted(list(LABELS_TO_IDX.keys()))


    validation()


