import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

torch.set_float32_matmul_precision('medium')

class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.num_classes = self.cfg.params.num_classes

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")
        self.automatic_optimization = False


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, cls_labels, loc_labels = batch

        preds =self(x)
        pred_cls, pred_locs = preds[:,0], preds[:, 1:]

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels.float())

        loss = (cls_loss + loc_loss) / 2

        self.train_loc_auroc.update(pred_locs, loc_labels.long())
        self.train_cls_auroc.update(pred_cls, cls_labels.long())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Log the metric object. Lightning computes and logs it at epoch end.
        self.log('train_loc_auroc', self.train_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)

        
        # Manual backward pass
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, sample, batch_idx):
        x, cls_labels, loc_labels = sample

        preds =self(x)
        pred_cls, pred_locs = preds[:,0], preds[:, 1:]
       
        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels.float())
  
        loss = (cls_loss + loc_loss) / 2
       
        self.val_loc_auroc.update(pred_locs, loc_labels.long())
        self.val_cls_auroc.update(pred_cls, cls_labels.long())
      
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # Log the metric object. Lightning computes and logs it at epoch end.
        self.log('val_loc_auroc', self.val_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cls_auroc', self.val_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and (self.current_epoch + 1) % self.cfg.trainer.check_val_every_n_epoch == 0:
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
