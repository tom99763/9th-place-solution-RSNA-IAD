import hydra
from omegaconf import DictConfig, OmegaConf


from src.trainers.cnn_25D import *
from src.rsna_datasets.cnn_25D import *
from src.model import MultiBackboneModel

import pytorch_lightning as pl
import timm

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

    model = MultiBackboneModel(
        model_name="efficientnet_b0",
        in_chans=16,
        img_size=512,
        num_classes=14,
        drop_rate=0.3,
        drop_path_rate=0.2,
        pretrained=True
    )
    # model = timm.create_model(
    #     model_name="efficientnet_b2",
    #     pretrained=True,
    #     in_chans=16,
    #     drop_rate=0.3,
    #     drop_path_rate=0.2,
    #     num_classes=14
    # ) 

    pl_model = LitTimmClassifier(model, cfg)

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

    trainer.fit(pl_model, datamodule=datamodule)
    # trainer.validate(pl_model, datamodule=datamodule, ckpt_path="./models/25D_classification-epoch=04-val_loss=0.3963_fold_id=0.ckpt")

if __name__ == "__main__":
    train()
