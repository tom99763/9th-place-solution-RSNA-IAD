import hydra
from omegaconf import DictConfig, OmegaConf
from src.trainers.cnn_25D import *
from src.rsna_datasets.cnn_25D import *

import pytorch_lightning as pl
from hydra.utils import instantiate

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

    model = instantiate(cfg.model)

    pl_model = LitTimmClassifier(model, cfg)

    loss_ckpt_callback = pl.callbacks.ModelCheckpoint(
                          monitor="val_loss"
                        , mode="min"
                        , dirpath="./models"
                        , filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_loss:.4f}'+f"_fold_id={cfg.fold_id}"
                        , save_top_k=2
                        )
    kaggle_score_ckpt_callback = pl.callbacks.ModelCheckpoint(
                          monitor="kaggle_score"
                        , mode="max"
                        , dirpath="./models"
                        , filename=f'{cfg.experiment}'+'-{epoch:02d}-{kaggle_score:.4f}'+f"_fold_id={cfg.fold_id}"
                        , save_top_k=2
                        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=pl.loggers.TensorBoardLogger("logs/", name=cfg.experiment),
        callbacks=[lr_monitor, loss_ckpt_callback, kaggle_score_ckpt_callback]
    )

    # trainer.fit(pl_model, datamodule=datamodule)
    trainer.validate(pl_model, datamodule=datamodule, ckpt_path="./models/ch32_effb2-epoch=08-kaggle_score=0.6675_fold_id=3.ckpt")

if __name__ == "__main__":
    train()
