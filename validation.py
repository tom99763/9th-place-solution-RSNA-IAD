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
import cv2

torch.set_float32_matmul_precision('medium')

def preprocess_slice(slice_img):
    """
    Apply the same preprocessing as training pipeline.
    The slice is already processed by prepare_data_slices.py, so we just need to:
    1. Create 3-channel image (matches slice_datasets.py)
    2. Convert to tensor format (matches validation transforms)
    """
    # Create 3-channel image (required for most models) - matches slice_datasets.py line 115
    img = np.stack([slice_img] * 3, axis=-1)
    
    # Convert to tensor if no transforms (matches slice_datasets.py line 140)
    img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC to CHW
    
    return img

@torch.no_grad()
def eval_one_series(slice_files, model, data_path):
    """
    Evaluate individual slices for a series using slice-based approach.
    Loads and preprocesses individual slice files like the training pipeline.
    """
    slices = []
    
    # Load and preprocess individual slices
    for slice_filename in slice_files:
        slice_path = data_path / "individual_slices" / slice_filename
        
        with np.load(slice_path) as data:
            slice_img = data['slice'].astype(np.float32)
        
        # Apply same preprocessing as training (CRITICAL for accurate validation)
        processed_slice = preprocess_slice(slice_img)
        slices.append(processed_slice)
    
    if not slices:
        # Return default predictions if no slices
        return 0.1, np.array([0.1] * 13)
    
    # Stack slices into batch
    volume = torch.stack(slices).cuda()
    
    # Debug: Print tensor info for first batch
    if len(slices) > 0:
        print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}, device: {volume.device}")

    pred_cls = []
    pred_locs = []

    # Process slices in batches
    for batch_idx in range(0, volume.shape[0], 64):
        batch_slices = volume[batch_idx:batch_idx+64]
        pc, pl = model(batch_slices)
        pred_cls.append(pc)
        pred_locs.append(pl)

    pred_cls = torch.vstack(pred_cls)
    pred_locs = torch.vstack(pred_locs)

    pred_cls = pred_cls.squeeze()

    # Use max aggregation across slices (same as inference)
    return pred_cls.max().sigmoid().item(), pred_locs.max(dim=0).values.sigmoid().cpu().numpy()


class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model=None, cfg=None):
        super().__init__()
        
        # If loading from checkpoint, model and cfg will be None initially
        if model is not None:
            self.model = model
        if cfg is not None:
            self.cfg = cfg
            self.save_hyperparameters()  # Saves args to checkpoint

    def forward(self, x):
        return self.model(x)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def validation(cfg: DictConfig) -> None:

    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    # Load slice-based model - alternative approach to avoid checkpoint loading issues
    try:
        pl_model = LitTimmClassifier.load_from_checkpoint(
            "/home/sersasj/RSNA-IAD-Codebase/models/slice_based_efficientnet_v2_cls-epoch=39-val_loss=1.2832fold_id=0 copy.ckpt"
        )
    except Exception as e:
        print(f"Failed to load from checkpoint: {e}")
        print("Using alternative loading method...")
        
        # Alternative: manually instantiate model and load state dict
        from hydra.utils import instantiate
        model = instantiate(cfg.model, pretrained=False)
        pl_model = LitTimmClassifier(model, cfg)
        
        # Load only the state dict
        checkpoint = torch.load("/home/sersasj/RSNA-IAD-Codebase/models/slice_based_efficientnet_v2_cls-epoch=39-val_loss=1.2832fold_id=0 copy.ckpt")
        pl_model.load_state_dict(checkpoint['state_dict'])
    
    # CRITICAL: Move model to GPU to match input tensors
    pl_model = pl_model.cuda()
    pl_model.eval()

    data_path = Path(cfg.data_dir)
    
    # Load slice-based data instead of volume-based data
    slice_df = pd.read_csv(data_path / "slice_df.csv")
    train_df = pd.read_csv(data_path / "train_df_slices.csv")  # For series-level labels

    # Get validation series UIDs
    val_slice_df = slice_df[slice_df["fold_id"] == cfg.fold_id]
    val_series_uids = val_slice_df["series_uid"].unique()

    print(f"Validating on {len(val_series_uids)} series with {len(val_slice_df)} individual slices")

    cls_labels = []
    loc_labels = []
    pred_cls_probs = []
    pred_loc_probs = []

    for uid in tqdm(val_series_uids, desc="Validating series"):
        # Get series-level labels from train_df
        series_row = train_df[train_df["SeriesInstanceUID"] == uid]
        if series_row.empty:
            print(f"Warning: No labels found for series {uid}")
            continue
            
        series_row = series_row.iloc[0]

        # Get location labels
        loc_label = np.zeros(13)
        for idx, label in enumerate(LABELS):
            loc_label[idx] = int(series_row[label])

        cls_labels.append(series_row["Aneurysm Present"])
        loc_labels.append(loc_label)

        # Get all slice files for this series
        series_slices = val_slice_df[val_slice_df["series_uid"] == uid]
        slice_files = series_slices["slice_filename"].tolist()
        
        if not slice_files:
            print(f"Warning: No slices found for series {uid}")
            # Use default predictions
            pred_cls_probs.append(0.1)
            pred_loc_probs.append(np.array([0.1] * 13))
            continue

        # Evaluate using slice-based approach
        cls_prob, loc_probs = eval_one_series(slice_files, pl_model, data_path)

        pred_cls_probs.append(cls_prob)
        pred_loc_probs.append(loc_probs)

    cls_labels = np.array(cls_labels)
    pred_cls_probs = np.array(pred_cls_probs)

    loc_labels = np.stack(loc_labels)
    pred_loc_probs = np.stack(pred_loc_probs)

    # Calculate metrics
    loc_auc_macro = roc_auc_score(loc_labels, pred_loc_probs, average="micro")
    cls_auc = roc_auc_score(cls_labels, pred_cls_probs)

    print(f"Fold: {cfg.fold_id}, cls_auc: {cls_auc}, loc_auc_macro: {loc_auc_macro}")
    print(f"Total validation samples: {len(cls_labels)}")
    print(f"Positive samples: {np.sum(cls_labels)}")
    print(f"Negative samples: {len(cls_labels) - np.sum(cls_labels)}")

        

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


