import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('./src')
from src.rsna_datasets.gnn_dataset import *
from src.trainers.gnn_trainer import *
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")
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
    datamodule = GraphDataModule(cfg)

    model = instantiate(cfg.model)
    pl_model = GNNClassifier(model, cfg)

    wnb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.experiment,
        config=OmegaConf.to_container(cfg),
        offline=True,
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_cls_auroc"
        , mode="max"
        , dirpath="./models"
        , filename=f'{cfg.experiment}' + '-{epoch:02d}-{val_loss:.4f}'
                                         '-{val_cls_auroc:.4f}' + \
                   f"fold_id={cfg.fold_id}"
        , save_top_k = 2
        , save_last = True
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        **cfg.trainer,
        logger= wnb_logger,   #pl.loggers.TensorBoardLogger("logs/", name=cfg.experiment),
        callbacks=[lr_monitor, ckpt_callback]
    )
    wnb_logger.watch(model, log="all", log_freq=20)
    #trainer.validate(pl_model, datamodule=datamodule)
    trainer.fit(pl_model, datamodule=datamodule)

if __name__ == "__main__":
    train()