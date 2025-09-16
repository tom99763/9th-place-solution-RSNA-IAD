import logging
import sys
import warnings

import hydra
import torch
import torch.utils
import pandas as pd
from omegaconf import OmegaConf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataset import RSNASegDataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import torch.nn as nn

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def flatten_batch_collate_fn(batch):
    # batch is a list of items, each can be a dict or list of dicts
    flat = []
    for item in batch:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    images = torch.stack([d['Image'] for d in flat], dim=0)
    masks = torch.stack([d['Mask'] for d in flat], dim=0)
    return images, masks


@hydra.main(config_path="configs", config_name="finetune", version_base="1.3.2")
def main(cfg):
    #set seed
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")

    #load data
    df = pd.read_csv('../data/train.csv')
    uids = os.listdir('../data/segmentations')
    uids = [uid.split('.nii')[0] for uid in uids if 'cowseg' not in uid]
    df_seg = df[df.SeriesInstanceUID.isin(uids)].copy()
    df_seg.reset_index(inplace=True)

    #split
    df_seg['fold'] = -1
    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_seg, df_seg['Aneurysm Present'])):
        df_seg.loc[val_idx, 'fold'] = fold

    train_uids = df_seg[df_seg.fold != cfg.fold_idx].SeriesInstanceUID.values.tolist()
    val_uids = df_seg[df_seg.fold == cfg.fold_idx].SeriesInstanceUID.values.tolist()
    run_name = cfg.run_name

    # init logger
    wnb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        offline=cfg.offline,
    )

    # callbacks
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_dice"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.wandb_project + "/" + run_name,
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename = f"{run_name}-{{val_dice:.4f}}",
        #auto_insert_metric_name=True,
        save_last=True
    )
    # checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    # checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    # init trainer
    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": wnb_logger,
        "callbacks": [lr_monitor, checkpoint_callback],
        "devices": cfg.devices
    }
    trainer = trainer(**trainer_additional_kwargs)

    # init dataloader
    train_dataset = RSNASegDataset(train_uids, cfg.data.RSNA, 'train')
    train_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=train_dataset,
        persistent_workers=True,
        collate_fn=flatten_batch_collate_fn
    )
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    val_dataset = RSNASegDataset(val_uids, cfg.data.RSNA, 'val')
    val_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=val_dataset, batch_size=1, persistent_workers=True)
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Instantiate model from hydra config
    model = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load(
        './pretrained/vesselfm_13_classes_dynunet-val_dice0.5047.ckpt',
        map_location=f'cuda:{cfg.devices[0]}',
        weights_only=False
    )
    ckpt = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(ckpt)


    # init lightning module
    lightning_module = hydra.utils.instantiate(
        cfg.trainer.lightning_module)(
        model=model,
        dataset_name='rsna',
        threshold=cfg.threshold
    )

    # train loop and eval
    wnb_logger.watch(model, log="all", log_freq=20)
    logger.info("Starting training")
    #trainer.validate(lightning_module, val_loader)
    trainer.fit(lightning_module, train_loader, val_loader)

if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()