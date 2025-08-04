
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from configs import data_config
from hydra.utils import instantiate
from tqdm import tqdm

import torch.cuda.amp as amp

torch.set_float32_matmul_precision('medium')

val_transforms = A.Compose([ A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) ])


@torch.no_grad()
def eval_one_series(volume, model):

    volume = np.stack([volume,volume,volume],axis=1)
    volume = val_transforms(image=volume)["image"]
    volume = torch.from_numpy(volume).cuda()
  
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
    model_ckpt_name="efficient_b2_gaussian_sample-epoch=47-val_loss=0.4765_fold_id=0"
    # pl_model = LitTimmClassifier.load_from_checkpoint(f"./models/{model_ckpt_name}.ckpt", model=model)
    # torch.save(pl_model.model.state_dict(), f"{model_ckpt_name}.pth")
    # return

   
    model.load_state_dict(torch.load(f"{model_ckpt_name}.pth"))
    model.cuda().eval()

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
            cls_prob, loc_probs =  eval_one_series(volume, model)

            pred_cls_probs.append(cls_prob)
            pred_loc_probs.append(loc_probs)

    cls_labels = np.array(cls_labels)
    pred_cls_probs = np.array(pred_cls_probs)

    loc_labels = np.stack(loc_labels)
    pred_loc_probs = np.stack(pred_loc_probs)

    np.savez("labels.npz", cls_probs=pred_cls_probs, loc_probs=pred_loc_probs)

    pred_loc_probs = np.nan_to_num(pred_loc_probs, nan=0.1)
    pred_cls_probs = np.nan_to_num(pred_cls_probs, nan=0.1)

    loc_auc_macro = roc_auc_score(loc_labels, pred_loc_probs, average="micro")
    cls_auc = roc_auc_score(cls_labels, pred_cls_probs)

    print(f"Fold: {cfg.fold_id}, cls_auc: {cls_auc}, loc_auc_macro: {loc_auc_macro}, cv: {(cls_auc + loc_auc_macro) / 2}")

        

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


