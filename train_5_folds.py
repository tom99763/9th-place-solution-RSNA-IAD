import hydra
from omegaconf import DictConfig, OmegaConf
from src.trainers.cnn_25D import *
from src.rsna_datasets.cnn_seg_cls import *
from src.nnmodels.resnet_segmentation_classification import *

from lightning.pytorch.loggers import WandbLogger
import pytorch_lightning as pl

def run_one_fold(cfg: DictConfig):
    fold_id = cfg.fold_id
    print(f"\n===== 🚀 Running fold {fold_id} =====\n")


    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    wnb_logger = WandbLogger(
        project=cfg.project_name,
        name=f"{cfg.experiment}_fold{fold_id}",
        config=OmegaConf.to_container(cfg),
        offline=cfg.offline,
    )

    pl.seed_everything(cfg.seed)
    datamodule = VolumeDataModule(cfg)
    model = SegmentationClassifier()
    pl_model = LitTimmClassifier(model, cfg)

    loss_ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath="./models",
        filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_loss:.4f}'+f"_fold_id={fold_id}",
        save_top_k=2
    )
    kaggle_score_ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_cls_auroc",
        mode="max",
        dirpath="./models",
        filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_cls_auroc:.4f}'+f"_fold_id={fold_id}",
        save_top_k=2
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=wnb_logger,
        callbacks=[lr_monitor, loss_ckpt_callback, kaggle_score_ckpt_callback]
    )
    wnb_logger.watch(model, log="all", log_freq=20)

    trainer.fit(pl_model, datamodule=datamodule)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    run_one_fold(cfg)


if __name__ == "__main__":
    train()
