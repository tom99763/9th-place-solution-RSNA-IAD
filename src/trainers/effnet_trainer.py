import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

torch.set_float32_matmul_precision('medium')

class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        smoothing = float(getattr(self.cfg, 'label_smoothing', 0.0))
        # For binary classification we can emulate label smoothing by blending targets
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()
        self._label_smoothing = smoothing

        self.num_classes = self.cfg.model.num_classes

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")
        self.automatic_optimization = False

       
        self.validation_outputs = []
        self._did_unfreeze = False


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, cls_labels, loc_labels, series_uids = batch

        pred_cls, pred_locs = self.model(x)
        pred_cls = pred_cls.squeeze(-1)  

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        if self._label_smoothing and self._label_smoothing > 0:
            cls_targets = cls_labels.float() * (1 - self._label_smoothing) + 0.5 * self._label_smoothing
        else:
            cls_targets = cls_labels.float()
        cls_loss = self.cls_loss_fn(pred_cls, cls_targets)


        if self.global_step %10 == 0:
            self.train_loc_auroc.update(torch.sigmoid(pred_locs.detach()), loc_labels.int())
            self.train_cls_auroc.update(torch.sigmoid(pred_cls.detach()), cls_labels.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Manual backward pass
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, sample, batch_idx):
        x, cls_labels, loc_labels = sample
        x.squeeze_()
        cls_labels.squeeze_()
        loc_labels.squeeze_()

        pred_cls = []
        pred_locs = []

        
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        # Manual gradient clipping because we're in manual optimization mode
        try:
            max_norm = getattr(self.cfg.trainer, 'gradient_clip_val', 1.0)
        except Exception:
            max_norm = 0.0
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        opt.step()


        return loss
