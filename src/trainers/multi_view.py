import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision('medium')

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
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.num_classes = self.cfg.params.num_classes
        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        return self.model(x)  # (B, 1)

    def compute_loss_and_metrics(self, logits, labels, stage="train"):
        loss = self.loss_fn(logits.squeeze(-1), labels.float())

        if stage == "train":
            self.train_cls_auroc.update(logits.squeeze(-1), labels.long())
        else:
            self.val_cls_auroc.update(logits.squeeze(-1), labels.long())

        return loss

    def training_step(self, batch, _):
        x, labels = batch
        logits = self(x)
        loss = self.compute_loss_and_metrics(logits, labels, stage="train")
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        x, labels = batch
        logits = self(x)
        loss = self.compute_loss_and_metrics(logits, labels, stage="val")
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_cls_auroc", self.train_cls_auroc.compute(), on_epoch=True, prog_bar=True)
        self.train_cls_auroc.reset()

    def on_validation_epoch_end(self):
        self.log("val_cls_auroc", self.val_cls_auroc.compute(), on_epoch=True, prog_bar=True)
        self.val_cls_auroc.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())

        frequency = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.cfg.trainer.max_steps / frequency),
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


class Mixup:
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        self.do_mixup = np.random.rand() < self.p
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0

    def __call__(self, x, y):
        if not self.do_mixup:
            return x, y
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = self.lam * x + (1 - self.lam) * x[index]
        mixed_y = self.lam * y + (1 - self.lam) * y[index]
        return mixed_x, mixed_y