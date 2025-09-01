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
        pos_weight = torch.ones([1]) * cfg.pos_weight
        self.node_loss_fn  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")
        self.automatic_optimization = False

    def training_step(self, data, _):
        node_labels = data.y
        cls_labels = data.cls_labels

        node_logits, pred_cls = self.model(data)

        cls_labels = cls_labels.view(pred_cls.shape)

        # Compute losses
        loss = self.node_loss_fn(node_logits, node_labels)

        # Update metrics
        self.train_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)

        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss

    def validation_step(self, data, batch_idx):
        node_labels = data.y
        cls_labels = data.cls_labels

        node_logits, pred_cls = self.model(data)
        cls_labels = cls_labels.view(pred_cls.shape)

        loss = self.node_loss_fn(node_logits, node_labels)
        self.val_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cls_auroc', self.val_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_start(self):
        self.train_cls_auroc.reset()

    def on_validation_epoch_start(self):
        self.val_cls_auroc.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4)
        return {"optimizer": optimizer}