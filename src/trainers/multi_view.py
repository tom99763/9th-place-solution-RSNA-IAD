import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision('medium')


class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.num_classes = self.cfg.params.num_classes

        # Metrics for patches (3) + merged (1) = 4 AUROCs
        self.train_cls_aurocs = nn.ModuleList([
            torchmetrics.AUROC(task="binary") for _ in range(4)
        ])
        self.val_cls_aurocs = nn.ModuleList([
            torchmetrics.AUROC(task="binary") for _ in range(4)
        ])

    def forward(self, x):
        return self.model(x)  # returns list of [logit0, logit1, logit2]

    def compute_losses_and_metrics(self, logits_list, labels, stage="train"):
        # Merge logits: sum of 3 patches
        merged_logits = sum(logits_list)

        # Compute per-patch + merged losses
        cls_losses = [self.cls_loss_fn(logit.squeeze(-1), labels) for logit in logits_list]
        cls_losses.append(self.cls_loss_fn(merged_logits.squeeze(-1), labels))  # merged

        total_loss = sum(cls_losses) / len(cls_losses)

        # Update metrics
        if stage == "train":
            for i, logit in enumerate(logits_list + [merged_logits]):
                self.train_cls_aurocs[i].update(logit.squeeze(-1), labels.long())
        else:
            for i, logit in enumerate(logits_list + [merged_logits]):
                self.val_cls_aurocs[i].update(logit.squeeze(-1), labels.long())

        return total_loss

    def training_step(self, batch, _):
        x, labels = batch
        logits_list = self(x)
        loss = self.compute_losses_and_metrics(logits_list, labels, stage="train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits_list = self(x)
        loss = self.compute_losses_and_metrics(logits_list, labels, stage="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        for i, metric in enumerate(self.train_cls_aurocs):
            self.log(f"train_cls_auroc_{i}", metric.compute(), on_epoch=True, prog_bar=True)
            metric.reset()

    def on_validation_epoch_end(self):
        cls_aucs = [metric.compute() for metric in self.val_cls_aurocs]

        for i, auc in enumerate(cls_aucs):
            self.log(f"val_cls_auroc_{i}", auc, on_epoch=True, prog_bar=True)

        for metric in self.val_cls_aurocs:
            metric.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())

        frequency = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.trainer.max_steps / frequency,
            eta_min=0.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": frequency,
            }
        }


class Mixup(object):
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0
