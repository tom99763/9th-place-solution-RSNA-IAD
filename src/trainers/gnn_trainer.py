import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

torch.set_float32_matmul_precision('medium')


class GNNClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])  # Saves args to checkpoint

        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = nn.BCEWithLogitsLoss()
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

        self.num_classes = 14

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")
        self.automatic_optimization = False

    def training_step(self, data, _):
        cls_labels = data.cls_labels
        loc_labels = data.loc_labels

        pred_cls, pred_locs = self.model(data)

        # Compute losses
        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels)

        loss = 0.5 * (cls_loss + loc_loss)

        # Update metrics
        self.train_loc_auroc.update(pred_locs, loc_labels.long())
        self.train_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loc_auroc', self.train_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)

        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, data, batch_idx):
        cls_labels = data.cls_labels
        loc_labels = data.loc_labels

        with torch.no_grad():
            pred_cls, pred_locs = self.model(data)

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels)

        loss = 0.5 * (cls_loss + loc_loss)

        self.val_loc_auroc.update(pred_locs, loc_labels.long())
        self.val_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loc_auroc', self.val_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cls_auroc', self.val_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and (
                self.current_epoch + 1) % self.cfg.trainer.check_val_every_n_epoch == 0:
            sch.step(self.trainer.callback_metrics["val_loss"])

    def on_train_epoch_start(self):
        self.train_loc_auroc.reset()
        self.train_cls_auroc.reset()

    def on_validation_epoch_start(self):
        self.val_loc_auroc.reset()
        self.val_cls_auroc.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}