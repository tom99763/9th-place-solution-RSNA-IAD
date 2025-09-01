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

        # Node classification loss
        pos_weight = torch.ones([1]) * cfg.pos_weight
        self.node_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Graph-level metrics
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")

        # Node-level metrics
        self.val_node_auroc = torchmetrics.AUROC(task="binary")
        self.val_node_acc = torchmetrics.Accuracy(task="binary")
        self.val_node_f1 = torchmetrics.F1Score(task="binary")

        self.automatic_optimization = False

    def training_step(self, data, _):
        node_labels = data.y
        node_logits, pred_cls = self.model(data)

        # Loss
        loss = self.node_loss_fn(node_logits[:, 0], node_labels)

        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, data, _):
        node_labels = data.y
        cls_labels = data.cls_labels.view(-1, 1)

        # Forward pass
        node_logits, pred_cls = self.model(data)

        # Loss
        loss = self.node_loss_fn(node_logits[:, 0], node_labels)

        # --- Graph-level metrics ---
        self.val_cls_auroc.update(pred_cls, cls_labels.long())

        # --- Node-level metrics ---
        self.val_node_auroc.update(node_logits, node_labels.long())
        self.val_node_acc.update(torch.sigmoid(node_logits[:, 0]), node_labels.int())
        self.val_node_f1.update(torch.sigmoid(node_logits[:, 0]), node_labels.int())

        # log loss per-batch
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        # reset all metrics at start of val epoch
        self.val_cls_auroc.reset()
        self.val_node_auroc.reset()
        self.val_node_acc.reset()
        self.val_node_f1.reset()

    def on_validation_epoch_end(self):
        # compute metrics over the whole val set
        self.log("val_cls_auroc", self.val_cls_auroc.compute(), prog_bar=True)
        self.log("val_node_auroc", self.val_node_auroc.compute(), prog_bar=True)
        self.log("val_node_acc", self.val_node_acc.compute(), prog_bar=True)
        self.log("val_node_f1", self.val_node_f1.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        return {"optimizer": optimizer}