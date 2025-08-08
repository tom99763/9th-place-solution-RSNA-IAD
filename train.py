import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.rsna_datasets.datasets import *
from src.rsna_datasets.slice_datasets import *
from src.trainers.effnet_trainer import *
from hydra.utils import instantiate

import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Your main training function.
    The 'cfg' object contains all your configuration.
    """
    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    datamodule = SliceDataModule(cfg)

    model = instantiate(cfg.model)
    pl_model = LitTimmClassifier(model, cfg)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
                          monitor="val_kaggle_score"
                        , mode="max"
                        , dirpath="./models"

                        , filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_kaggle_score:.4f}'+f"fold_id={cfg.fold_id}",
                        save_top_k=3,


    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    loggers = [pl.loggers.TensorBoardLogger("logs/", name=cfg.experiment)]
    
    if cfg.use_wandb:
        import wandb
        wandb_logger = pl.loggers.WandbLogger(
            project=cfg.project_name, 
            name=cfg.experiment,
            entity=cfg.wandb.get('entity', None),
            tags=cfg.wandb.get('tags', []),
            notes=cfg.wandb.get('notes', '')
        )
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=loggers,
        callbacks=[lr_monitor, ckpt_callback]
    )
    
    # tuner = Tuner(trainer)
    # # 3. Call lr_find on the Tuner instance
    # lr_find_results = tuner.lr_find(pl_model, datamodule=datamodule)
    #
    # # 4. Get the suggested learning rate and plot
    # suggested_lr = lr_find_results.suggestion()
    # print(f"Suggested LR: {suggested_lr}")
    # fig = lr_find_results.plot(suggest=True)
    # fig.show()

    trainer.fit(pl_model, datamodule=datamodule)
    
    if cfg.use_wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    train()
