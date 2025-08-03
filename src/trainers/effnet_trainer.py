import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate
torch.set_float32_matmul_precision('medium')


class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters()  # Saves args to checkpoint

        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=13)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=13)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        self.model.train()
        x, cls_labels, loc_labels = batch

        pred_cls, pred_locs = self(x)
        pred_cls = pred_cls.squeeze()

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels.float())

        loss = 2 * cls_loss + loc_loss

        self.train_loc_auroc.update(pred_locs, loc_labels.long())
        self.train_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        # Log the metric object. Lightning computes and logs it at epoch end.
        self.log('train_loc_auroc', self.train_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, sample, batch_idx):
        self.model.eval()
        x, cls_labels, loc_labels = sample
        x.squeeze_()
        cls_labels.squeeze_()
        loc_labels.squeeze_()

        pred_cls = []
        pred_locs = []

        for batch_idx in range(0, x.shape[0], 64):
            pc, pl = self(x[batch_idx:batch_idx + 64])
            pred_cls.append(pc)
            pred_locs.append(pl)

        pred_cls = torch.vstack(pred_cls)
        pred_locs = torch.vstack(pred_locs)

        pred_cls = pred_cls.squeeze()

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels)

        loss = 2 * cls_loss + loc_loss

        self.val_loc_auroc.update(pred_locs, loc_labels.long())
        self.val_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=False)
        # Log the metric object. Lightning computes and logs it at epoch end.
        self.log('val_loc_auroc', self.val_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cls_auroc', self.val_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())