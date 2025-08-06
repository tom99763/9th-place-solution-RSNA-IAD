import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('./src')
from src.rsna_datasets.datasets import *
from src.trainers.effnet_trainer import *
from hydra.utils import instantiate
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
    datamodule = NpzDataModule(cfg)

    model = instantiate(cfg.model)
    pl_model = LitTimmClassifier(model, cfg)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
                          monitor="val_loss"
                        , mode="min"
                        , dirpath="./models"
                        , filename=f'{cfg.experiment}'+'-{epoch:02d}-{val_loss:.4f}-{val_loc_auroc:.4f}'
                                                       '-{val_cls_auroc:.4f}'+\
                                   f"fold_id={cfg.fold_id}"
                        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=pl.loggers.TensorBoardLogger("logs/", name=cfg.experiment),
        callbacks=[lr_monitor, ckpt_callback]
    )
    trainer.fit(pl_model, datamodule=datamodule)

if __name__ == "__main__":
    train()
    
