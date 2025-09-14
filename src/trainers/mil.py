import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from monai.metrics.meandice import DiceMetric
import torchmetrics
import torch.nn as nn

torch.set_float32_matmul_precision('medium')



class LitMil(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.num_classes = 13

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        X,slice_labels, labels = batch["X"], batch["slice_labels"], batch["volume_label"]
        logits, losses, _ = self.model(X, slice_labels=slice_labels)

        loss_main = self.loss_fn(logits, labels)

        if "aux_loss" in losses:
            loss = 0.75 * loss_main + 0.25 * losses["aux_loss"]
        else:
            print("aux_loss not present in the batch")
            loss = loss_main

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_main_loss', loss_main, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_loc_auroc.update(logits, labels.long())

        cls_logits = logits.max(dim=-1).values
        cls_labels = labels.max(dim=-1).values.long()
        self.train_cls_auroc.update(cls_logits, cls_labels)
    
        return loss

    def on_train_epoch_end(self):
        cls_auc = self.train_cls_auroc.compute()
        self.log('train_cls_auroc', cls_auc, on_step=False, on_epoch=True, prog_bar=True)

        loc_auc = self.train_loc_auroc.compute()
        self.log('train_loc_auroc', loc_auc, on_step=False, on_epoch=True, prog_bar=True)

        self.train_cls_auroc.reset()
        self.train_loc_auroc.reset()

    def validation_step(self, batch, batch_idx):
        X,labels = batch["X"], batch["volume_label"]
        logits, _, _ = self.model(X, slice_labels=None)
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_loc_auroc.update(logits, labels.long())

        cls_logits = logits.max(dim=-1).values
        cls_labels = labels.max(dim=-1).values.long()
        self.val_cls_auroc.update(cls_logits, cls_labels)
      
        return loss

    def on_validation_epoch_end(self):
        cls_auc = self.val_cls_auroc.compute()
        self.log('val_cls_auroc', cls_auc, on_step=False, on_epoch=True, prog_bar=True)

        loc_auc = self.val_loc_auroc.compute()
        self.log('val_loc_auroc', loc_auc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_cls_auroc.reset()
        self.val_loc_auroc.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())

        frequency = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.trainer.max_steps / frequency,
            eta_min=0.0
        )
        return { "optimizer": optimizer
                , "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": frequency
                    }
                }
