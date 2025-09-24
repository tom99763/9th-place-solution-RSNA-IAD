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
        self.save_hyperparameters(ignore=['model'])  # Saves args to checkpoint

        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        # BCELoss * 0.2 + DICELoss * 0.8.
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.num_classes = self.cfg.params.num_classes

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes - 1)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes - 1)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, labels = batch
        preds = self(x)

        # Split labels
        cls_labels, loc_labels = labels[:, 0], labels[:, 1:]
        pred_cls, pred_locs = preds[:, 0], preds[:, 1:]

        # BCE supports soft labels
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels)
        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)

        loss = (cls_loss + loc_loss) / 2

        # Metrics: AUROC expects hard labels
        self.train_cls_auroc.update(pred_cls, (cls_labels > 0.5).long())
        self.train_loc_auroc.update(pred_locs, (loc_labels > 0.5).long())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, sample, batch_idx):
        x, labels = sample
        # print(f"{uid}")
        preds = self(x)
        # print(f"{preds.sigmoid()}")

        cls_labels, loc_labels = labels[:, 0], labels[:, 1:]
        pred_cls, pred_locs = preds[:, 0], preds[:, 1:]

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels)

        loss = (cls_loss + loc_loss) / 2

        self.val_loc_auroc.update(pred_locs, loc_labels.long())
        self.val_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.log('train_loc_auroc', self.train_loc_auroc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_loc_auroc.reset()
        self.train_cls_auroc.reset()

    def on_validation_epoch_end(self):
        loc_auc = self.val_loc_auroc.compute()
        cls_auc = self.val_cls_auroc.compute()
        kaggle_score = (loc_auc + cls_auc) / 2

        self.log("kaggle_score", kaggle_score, prog_bar=True)

        self.log('val_loc_auroc', loc_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cls_auroc', cls_auc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_loc_auroc.reset()
        self.val_cls_auroc.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())

        frequency = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.trainer.max_steps / frequency,
            eta_min=0.0
        )
        return {"optimizer": optimizer
            , "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": frequency
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
