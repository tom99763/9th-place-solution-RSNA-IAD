import logging
import sys
import warnings

import hydra
import torch
import torch.utils
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, Subset

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataset import UnionDataset
from utils.evaluation import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="finetune", version_base="1.3.2")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    dataset_name = list(cfg.data.keys())[0]
    run_name = f'finetune_{cfg.num_shots}shot_{dataset_name}_' + cfg.run_name

    # init logger
    wnb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        offline=cfg.offline,
    )

    # callbacks
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_DiceMetric"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.wandb_project + "/" + run_name,
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename=f"{run_name}_" + "{step}_{" + monitor_metric + ":.2f}",
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    # init trainer
    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": wnb_logger,
        "callbacks": [lr_monitor, checkpoint_callback],
        "devices": cfg.devices
    }
    trainer = trainer(**trainer_additional_kwargs)

    # init dataloader
    train_dataset = UnionDataset(cfg.data, "train", finetune=True)
    train_dataset = Subset(train_dataset, range(cfg.num_shots))
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=int(1e6))
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset, sampler=random_sampler)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    val_dataset = UnionDataset(cfg.data, "val", finetune=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
    logger.info(f"Val dataset size: {len(val_dataset)}")

    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # init model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.path_to_chkpt is not None:
        chkpt = torch.load(cfg.path_to_chkpt, map_location=f'cuda:{cfg.devices[0]}')
        model_chkpt = {k.replace("model.", ""): e for k, e in chkpt["state_dict"].items() if "model" in k}
        model.load_state_dict(model_chkpt)

    # init lightning module
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=evaluator, dataset_name=dataset_name
    )

    # train loop and eval
    wnb_logger.watch(model, log="all", log_freq=20)
    if cfg.num_shots == 0:
        trainer.test(lightning_module, test_loader)  # eval on test set
    else:
        logger.info("Starting training")
        trainer.validate(lightning_module, val_loader)
        trainer.fit(lightning_module, train_loader, val_loader)
        logger.info("Finished training")
        trainer.test(lightning_module, test_loader, ckpt_path="best")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()