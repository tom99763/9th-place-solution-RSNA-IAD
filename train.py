import hydra
from omegaconf import DictConfig, OmegaConf
from src.trainers.mil import *
from src.rsna_datasets.mil import *
from src.models.mil import *

import pytorch_lightning as pl

from src.unet import get_model

# def debug_hook(module, inp, out):
#     if torch.isnan(out).any():
#         print(f"NaN in {module}")

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

    model = AneurysmClassifier()

    # for _, module in model.named_modules():
    #     module.register_forward_hook(debug_hook)

    pl_model = LitMil(model, cfg)

    loss_ckpt_callback = pl.callbacks.ModelCheckpoint(
                          monitor="val_loss"
                        , mode="min"
                        , dirpath="./models"
                        , filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_loss:.4f}'+f"_fold_id={cfg.fold_id}"
                        , save_top_k=2
                        , save_last=True
                        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=pl.loggers.TensorBoardLogger("logs/", name=cfg.experiment),
        callbacks=[lr_monitor, loss_ckpt_callback]
    )

    trainer.fit(pl_model, datamodule=datamodule)
    # trainer.validate(pl_model, datamodule=datamodule)

if __name__ == "__main__":
    train()
