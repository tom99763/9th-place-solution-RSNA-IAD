import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
import sys
sys.path.append('./src')
from configs.data_config import *
from hydra.utils import instantiate
from tqdm import tqdm
import cv2

torch.set_float32_matmul_precision('medium')

def preprocess_slice_2d(slice_img):
    """
    Apply 2D preprocessing - single slice replicated 3 times.
    """
    # Create 3-channel image (required for most models)
    img = np.stack([slice_img] * 3, axis=-1)
    # Convert to tensor format (HWC to CHW)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    return img

def preprocess_slice_2_5d(slice_files, data_path, slice_idx, num_adjacent_slices=1):
    """
    Apply 2.5D preprocessing - load adjacent slices like training.
    """
    # Build lookup for this series
    slice_lookup = {}
    for filename in slice_files:
        # Extract slice index from filename (format: series_uid_XXX.npz)
        slice_num = int(filename.split('_')[-1].replace('.npz', ''))
        slice_lookup[slice_num] = filename
    
    # Get adjacent slice indices (same as training logic)
    slice_indices = list(range(slice_idx - num_adjacent_slices, 
                             slice_idx + num_adjacent_slices + 1))
    
    # Load current slice first as fallback
    current_filename = None
    for filename in slice_files:
        if filename.endswith(f"_{slice_idx:03d}.npz"):
            current_filename = filename
            break
    
    if not current_filename:
        # Fallback to 2D if we can't find the slice
        slice_path = data_path / "individual_slices" / slice_files[0]
        with np.load(slice_path) as data:
            slice_img = data['slice'].astype(np.float32)
        return preprocess_slice_2d(slice_img)
    
    # Load current slice
    slice_path = data_path / "individual_slices" / current_filename
    with np.load(slice_path) as data:
        current_slice = data['slice'].astype(np.float32)
    
    # Load adjacent slices
    all_slices = []
    for adj_idx in slice_indices:
        if adj_idx in slice_lookup:
            adj_filename = slice_lookup[adj_idx]
            adj_path = data_path / "individual_slices" / adj_filename
            with np.load(adj_path) as data:
                adj_slice = data['slice'].astype(np.float32)
        else:
            # Use current slice as fallback
            adj_slice = current_slice
        all_slices.append(adj_slice)
    
    # Stack as channels (same as training)
    img = np.stack(all_slices, axis=-1)
    
    # Reduce channels to RGB if needed (same as training)
    if img.shape[-1] > 3:
        num_channels = img.shape[-1]
        if num_channels == 5:
            # For 5 channels, group as: [0,1] -> R, [2] -> G, [3,4] -> B
            r_channel = np.mean(img[:, :, :2], axis=-1)
            g_channel = img[:, :, 2]
            b_channel = np.mean(img[:, :, 3:], axis=-1)
            img = np.stack([r_channel, g_channel, b_channel], axis=-1)
        else:
            # For other numbers, divide into 3 equal groups
            third = num_channels // 3
            r_channel = np.mean(img[:, :, :third], axis=-1)
            g_channel = np.mean(img[:, :, third:2*third], axis=-1)
            b_channel = np.mean(img[:, :, 2*third:], axis=-1)
            img = np.stack([r_channel, g_channel, b_channel], axis=-1)
    
    # Convert to tensor
    img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC to CHW
    return img

@torch.no_grad()
def eval_one_series(slice_files, model, data_path, image_mode="2D", num_adjacent_slices=1):
    """
    Evaluate individual slices for a series using slice-based approach.
    Matches exactly what happens in training validation.
    """
    # Extract slice indices from filenames for 2.5D mode
    slice_indices = []
    if image_mode == "2.5D":
        for filename in slice_files:
            slice_idx = int(filename.split('_')[-1].replace('.npz', ''))
            slice_indices.append(slice_idx)
    
    slices = []
    
    if image_mode == "2D":
        # 2D mode: process each slice individually
        for slice_filename in slice_files:
            slice_path = data_path / "individual_slices" / slice_filename
            
            with np.load(slice_path) as data:
                slice_img = data['slice'].astype(np.float32)
            
            processed_slice = preprocess_slice_2d(slice_img)
            slices.append(processed_slice)
    
    elif image_mode == "2.5D":
        # 2.5D mode: process with adjacent slices like training
        for slice_idx in slice_indices:
            processed_slice = preprocess_slice_2_5d(
                slice_files, data_path, slice_idx, num_adjacent_slices
            )
            slices.append(processed_slice)
    
    if not slices:
        # Return default predictions if no slices
        return 0.1, np.array([0.1] * 13)
    
    # Stack slices into batch
    volume = torch.stack(slices).cuda()

    pred_cls = []
    pred_locs = []

    # Process slices in batches (same as training)
    batch_size = 64
    for batch_idx in range(0, volume.shape[0], batch_size):
        batch_slices = volume[batch_idx:batch_idx+batch_size]
        pc, pl = model(batch_slices)
        pred_cls.append(pc.squeeze(-1))  # Remove last dimension like training
        pred_locs.append(pl)

    pred_cls = torch.cat(pred_cls)
    pred_locs = torch.cat(pred_locs)

    # Apply sigmoid to get probabilities (same as training validation)
    pred_cls_probs = torch.sigmoid(pred_cls)
    pred_locs_probs = torch.sigmoid(pred_locs)
    
    # Series-level aggregation using max pooling (same as training)
    series_cls_prob = pred_cls_probs.max().item()
    series_loc_probs = pred_locs_probs.max(dim=0).values.cpu().numpy()
    
    return series_cls_prob, series_loc_probs


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


    checkpoint_path = "/home/sersasj/RSNA-IAD-Codebase/models/slice_based_efficientnet_v2_baseline_25d-epoch=119-val_kaggle_score=0.8410fold_id=0.ckpt"
    # Load slice-based model
    try:
        pl_model = LitTimmClassifier.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Failed to load from checkpoint: {e}")
        print("Using alternative loading method...")
        
        # Alternative: manually instantiate model and load state dict
        model = instantiate(cfg.model, pretrained=False)
        pl_model = LitTimmClassifier(model, cfg)
        
        # Load only the state dict
        checkpoint = torch.load(checkpoint_path)
        pl_model.load_state_dict(checkpoint['state_dict'])
    
    pl_model = pl_model.cuda()
    pl_model.eval()

    data_path = Path(cfg.data_dir)
    
    # Load data exactly like training
    slice_df = pd.read_csv(data_path / "slice_df.csv")
    train_df_slices = pd.read_csv(data_path / "train_df_slices.csv")  # For series-level labels
    label_df_slices = pd.read_csv(data_path / "label_df_slices.csv")  # For location labels

    # Get validation series UIDs (same as training)
    val_slice_df = slice_df[slice_df["fold_id"] == cfg.fold_id]
    val_series_uids = val_slice_df["series_uid"].unique()

    print(f"Validating on {len(val_series_uids)} series with {len(val_slice_df)} individual slices")

    cls_labels = []
    loc_labels = []
    pred_cls_probs = []
    pred_loc_probs = []

    for uid in tqdm(val_series_uids, desc="Validating series"):
        # Get series-level binary label
        series_row = train_df_slices[train_df_slices["SeriesInstanceUID"] == uid]
        if series_row.empty:
            print(f"Warning: No labels found for series {uid}")
            continue
        
        series_row = series_row.iloc[0]
        has_aneurysm = int(series_row["Aneurysm Present"])
        cls_labels.append(has_aneurysm)
        
        # Get location labels (same logic as training)
        loc_label = np.zeros(13, dtype=np.float32)
        if has_aneurysm:
            # Get location labels from label_df_slices
            series_locations = label_df_slices[label_df_slices["SeriesInstanceUID"] == uid]
            for _, loc_row in series_locations.iterrows():
                location = loc_row["location"]
                if location in LABELS_TO_IDX:
                    loc_label[LABELS_TO_IDX[location]] = 1
        
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

        # Evaluate using slice-based approach (match training mode)
        cls_prob, loc_probs = eval_one_series(
            slice_files, pl_model, data_path, 
            image_mode=cfg.image_mode,
            num_adjacent_slices=getattr(cfg, 'num_adjacent_slices', 1)
        )

        pred_cls_probs.append(cls_prob)
        pred_loc_probs.append(loc_probs)

    cls_labels = np.array(cls_labels)
    pred_cls_probs = np.array(pred_cls_probs)

    loc_labels = np.stack(loc_labels)
    pred_loc_probs = np.stack(pred_loc_probs)

    # Calculate metrics exactly like training
    loc_auc_macro = roc_auc_score(loc_labels, pred_loc_probs, average="macro")
    cls_auc = roc_auc_score(cls_labels, pred_cls_probs)
    
    # Calculate Kaggle score (same formula as training)
    kaggle_score = (cls_auc * 13 + loc_auc_macro * 1) / 14

    print(f"\n📊 Validation Results (Fold {cfg.fold_id}):")
    print(f"Classification AUC: {cls_auc:.4f}")
    print(f"Localization AUC (macro): {loc_auc_macro:.4f}")
    print(f"Kaggle Score: {kaggle_score:.4f}")
    print(f"Total validation samples: {len(cls_labels)}")
    print(f"Positive samples: {np.sum(cls_labels)}")
    print(f"Negative samples: {len(cls_labels) - np.sum(cls_labels)}")
    
    return kaggle_score

        

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


