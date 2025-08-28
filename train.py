import hydra
from omegaconf import DictConfig, OmegaConf


import torchvision
from src.trainers.crop_classification import LitWindowedMIP
from src.rsna_datasets.crop_classification import WindowedMIPDataModule
from src.model import MultiBackboneModel

import pytorch_lightning as pl


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Your main training function.
    The 'cfg' object contains all your configuration.
    """
    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    datamodule = WindowedMIPDataModule(cfg)

    model = MultiBackboneModel(
        model_name="efficientnet_b2",
        in_chans=4,
        img_size=256,
        num_classes=14,
        drop_rate=0.3,
        drop_path_rate=0.2,
        pretrained=True
    )

    pl_model = LitWindowedMIP(model, cfg)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
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

    # trainer.fit(pl_model, datamodule=datamodule)
    trainer.validate(pl_model, datamodule=datamodule, ckpt_path="./models/windowed_mip_classification-epoch=03-val_loss=0.9985_fold_id=0.ckpt")

if __name__ == "__main__":
    train()
