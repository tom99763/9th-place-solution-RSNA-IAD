import hydra
from omegaconf import DictConfig, OmegaConf
from src.trainers.segmentation import *
from src.rsna_datasets.segmentation import *

import pytorch_lightning as pl
from hydra.utils import instantiate
from src.unet import get_model

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Your main training function.
    The 'cfg' object contains all your configuration.
    """
    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    datamodule = NpzDataModule(cfg)

    model = get_model("resunet")

    pl_model = LitSegmentationCls(model, cfg)

    loss_ckpt_callback = pl.callbacks.ModelCheckpoint(
                          monitor="val_loss"
                        , mode="min"
                        , dirpath="./models"
                        , filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_loss:.4f}'+f"_fold_id={cfg.fold_id}"
                        , save_top_k=2
                        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=pl.loggers.TensorBoardLogger("logs/", name=cfg.experiment),
        callbacks=[lr_monitor, loss_ckpt_callback]
    )

    trainer.fit(pl_model, datamodule=datamodule, ckpt_path="./models/segmentation_cls-epoch=08-val_loss=0.0012_fold_id=1.ckpt")
    # trainer.validate(pl_model, datamodule=datamodule)

if __name__ == "__main__":
    train()
