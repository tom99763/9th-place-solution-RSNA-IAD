import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from hydra.utils import instantiate
from src.trainers.multi_view import *
from src.rsna_datasets.patch_datasets import *

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Main training with 5-fold cross validation.
    """
    print("✨ Base configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))
    for fold_id in range(5):
        print(f"\n🚀 Starting Fold {fold_id}...\n")

        # Update fold_id in cfg (deepcopy to avoid mutation issues)
        cfg_fold = cfg.copy()
        cfg_fold.fold_id = fold_id

        wnb_logger = WandbLogger(
            project=cfg_fold.project_name,
            name=f"{cfg_fold.experiment}_fold{fold_id}",
            config=OmegaConf.to_container(cfg_fold),
            offline=cfg_fold.offline,
        )

        pl.seed_everything(cfg_fold.seed + fold_id)  # optional: different seed per fold
        datamodule = NpzPatchDataModule(cfg_fold)

        model = instantiate(cfg_fold.model)
        pl_model = LitTimmClassifier(model, cfg_fold)

        # Callbacks
        loss_ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath="./models",
            filename=f'{cfg_fold.experiment}--{cfg_fold.model.model_name}'
                     '-{epoch:02d}-{val_loss:.4f}' + f"_fold_id={fold_id}"
                                                     f"_fold_id_yolo={fold_id}",
            save_top_k=1,
        )

        auroc_callbacks = []
        for cls_id in range(4):  # 0,1,2,3
            auroc_callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    monitor=f"val_cls_auroc_{cls_id}",
                    mode="max",
                    dirpath="./models",
                    filename=f'{cfg_fold.experiment}--{cfg_fold.model.model_name}'
                             f'-{{epoch:02d}}-{{val_cls_auroc_{cls_id}:.4f}}_fold_id={fold_id}',
                    save_top_k=1,
                )
            )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            **cfg_fold.trainer,
            logger=wnb_logger,
            callbacks=[lr_monitor, loss_ckpt_callback] + auroc_callbacks,
        )

        wnb_logger.watch(model, log="all", log_freq=20)

        trainer.fit(pl_model, datamodule=datamodule)
        # trainer.validate(pl_model, datamodule=datamodule)

if __name__ == "__main__":
    train()