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
def run(cfg: DictConfig) -> None:
    """
    Train sequentially over folds 0..4.
    """
    for fold_id in range(5):
        print(f"\n🚀 Starting training for fold {fold_id}...\n")

        # override fold_id in config
        cfg.fold_id = fold_id

        # seed everything
        pl.seed_everything(cfg.seed)

        # datamodule
        datamodule = GraphDataModule(cfg)

        # model
        model = instantiate(cfg.model)
        pl_model = GNNClassifier(model, cfg)

        # logger
        wnb_logger = WandbLogger(
            project=cfg.project_name,
            name=f"{cfg.experiment}_fold{fold_id}",
            config=OmegaConf.to_container(cfg),
            offline=True,
        )

        # callbacks
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_cls_auroc",
            mode="max",
            dirpath="./models",
            filename=f"{cfg.experiment}-fold{fold_id}"
                     + "-{epoch:02d}-{val_loss:.4f}-{val_cls_auroc:.4f}",
            save_top_k=2,
            save_last=True,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        # trainer
        trainer = pl.Trainer(
            **cfg.trainer,
            logger=wnb_logger,
            callbacks=[lr_monitor, ckpt_callback],
        )
        wnb_logger.watch(model, log="all", log_freq=20)

        # training
        trainer.fit(pl_model, datamodule=datamodule)


if __name__ == "__main__":
    run()