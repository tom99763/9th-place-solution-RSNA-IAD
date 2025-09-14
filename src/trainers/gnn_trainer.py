import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

torch.set_float32_matmul_precision('medium')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class BCEFocalLoss(nn.Module):
    def __init__(self, pos_weight=None, alpha=1.0, gamma=2.0, bce_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets.float())
        loss_focal = self.focal(logits, targets)
        return self.bce_weight * loss_bce + self.focal_weight * loss_focal

class GNNClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])  # Saves args to checkpoint

        self.model = model
        self.cfg = cfg

        # Node classification loss
        pos_weight = torch.ones([1]) * cfg.pos_weight
        self.node_loss_fn = BCEFocalLoss(
            pos_weight=pos_weight,
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
            bce_weight=cfg.bce_weight,
            focal_weight=cfg.focal_weight,
        )

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
        del node_logits, pred_cls
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, data, _):
        node_labels = data.y
        cls_labels = data.cls_labels.view(-1, 1)

        # Forward pass
        node_logits, pred_cls = self.model(data)

        # Loss
        loss = self.node_loss_fn(node_logits[:, 0], node_labels)

        self.val_cls_auroc.update(pred_cls.detach(), cls_labels.long())
        self.val_node_auroc.update(node_logits.detach(), node_labels.long())
        self.val_node_acc.update(torch.sigmoid(node_logits[:, 0].detach()), node_labels.int())
        self.val_node_f1.update(torch.sigmoid(node_logits[:, 0].detach()), node_labels.int())

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