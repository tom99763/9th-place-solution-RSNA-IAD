import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_float32_matmul_precision('medium')

class PairwiseRankingLoss(nn.Module):
    """
    Pairwise logistic ranking loss (surrogate for AUROC).
    Encourages positive samples to score higher than negative samples.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (N,) raw model scores (logits)
            y_true: (N,) binary labels {0,1}
        """
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # Compute all pairwise differences: s_pos - s_neg
        diff = pos_scores[:, None] - neg_scores[None, :]  # shape (n_pos, n_neg)

        # Pairwise logistic loss
        loss = torch.log1p(torch.exp(-diff)).mean()
        return loss

class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.cfg = cfg
        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = PairwiseRankingLoss()

        self.num_classes = self.cfg.params.num_classes

        # Single AUROC (since we now only have one final logit)
        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        return self.model(x)  # returns (B, 1)

    def compute_loss_and_metrics(self, logits, labels, stage="train"):
        loss = self.loss_fn(logits.squeeze(-1), labels)

        if stage == "train":
            self.train_cls_auroc.update(logits.squeeze(-1), labels.long())
        else:
            self.val_cls_auroc.update(logits.squeeze(-1), labels.long())

        return loss

    def training_step(self, batch, _):
        x, labels = batch
        logits = self(x)
        loss = self.compute_loss_and_metrics(logits, labels, stage="train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits = self(x)
        loss = self.compute_loss_and_metrics(logits, labels, stage="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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