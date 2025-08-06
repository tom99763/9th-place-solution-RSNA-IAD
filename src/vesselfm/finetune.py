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

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="finetune", version_base="1.3.2")
def main(cfg):
    #set seed
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")

    #load data
    df = pd.read_csv('../rsna_data/train.csv')
    uids = os.listdir('../rsna_data/segmentations')
    df_seg = df[df.SeriesInstanceUID.isin(uids)].copy()
    df_seg.reset_index(inplace=True)

    #split
    df_seg['fold'] = -1
    gkf = StratifiedGroupKFold(n_splits = 5, shuffle=True, random_state=42)
    groups = df_seg['Modality'].values
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_seg, df_seg['Aneurysm Present'], groups)):
        df_seg.loc[val_idx, 'fold'] = fold

    train_uids = df_seg[df_seg.fold != cfg.fold_idx].SeriesInstanceUID.values.tolist()
    val_uids = df_seg[df_seg.fold == cfg.fold_idx].SeriesInstanceUID.values.tolist()

    dataset_name = 'rsna'
    run_name = f'finetune_{dataset_name}_' + cfg.run_name

    # init logger
    wnb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        offline=cfg.offline,
    )

    # callbacks
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_volumetric_recall"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.wandb_project + "/" + run_name,
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename=f"{run_name}_" + "{step}_{" + monitor_metric + ":.2f}",
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

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
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    val_dataset = RSNASegDataset(val_uids, cfg.data.RSNA, 'val')
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # init model
    hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml')  # required to track downloads
    ckpt = torch.load(
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
        map_location=f'cuda:{cfg.devices[0]}', weights_only=True
    )
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt)


    # init lightning module
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, dataset_name=dataset_name, threshold = cfg.threshold
    )

    # train loop and eval
    wnb_logger.watch(model, log="all", log_freq=20)
    logger.info("Starting training")
    trainer.validate(lightning_module, val_loader)
    # trainer.fit(lightning_module, train_loader, val_loader)

if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()