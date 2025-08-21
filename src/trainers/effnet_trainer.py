import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

torch.set_float32_matmul_precision('medium')

class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Saves args to checkpoint
        self.automatic_optimization = False

        
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

       
        self.validation_outputs = []
        self._did_unfreeze = False


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, cls_labels, loc_labels, series_uids = batch

        pred_cls, pred_locs =self(x)
        pred_cls = pred_cls.squeeze(-1)  

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        if self._label_smoothing and self._label_smoothing > 0:
            cls_targets = cls_labels.float() * (1 - self._label_smoothing) + 0.5 * self._label_smoothing
        else:
            cls_targets = cls_labels.float()
        cls_loss = self.cls_loss_fn(pred_cls, cls_targets)

        loss = 3*cls_loss + loc_loss

        if self.global_step %10 == 0:
            self.train_loc_auroc.update(torch.sigmoid(pred_locs.detach()), loc_labels.int())
            self.train_cls_auroc.update(torch.sigmoid(pred_cls.detach()), cls_labels.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loc_auroc', self.train_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)

        
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

    def validation_step(self, batch, batch_idx):
        x, cls_labels, loc_labels, series_uids = batch
        
        pred_cls, pred_locs = self(x)
        pred_cls = pred_cls.squeeze(-1)

        cls_labels_float = cls_labels.float()
        loc_labels_float = loc_labels.float()

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels_float)
        if self._label_smoothing and self._label_smoothing > 0:
            cls_targets = cls_labels_float * (1 - self._label_smoothing) + 0.5 * self._label_smoothing
        else:
            cls_targets = cls_labels_float
        cls_loss = self.cls_loss_fn(pred_cls, cls_targets)
        loss = 3*cls_loss + loc_loss 

        self.val_loc_auroc.update(torch.sigmoid(pred_locs.detach()), loc_labels.int())
        self.val_cls_auroc.update(torch.sigmoid(pred_cls.detach()), cls_labels.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loc_auroc_slice', self.val_loc_auroc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_cls_auroc_slice', self.val_cls_auroc, on_step=False, on_epoch=True, prog_bar=False)
        
        batch_outputs = {
            'pred_cls': pred_cls.detach(),  # Keep on GPU, apply sigmoid later
            'pred_locs': pred_locs.detach(),  # Keep on GPU, apply sigmoid later
            'cls_labels': cls_labels,
            'loc_labels': loc_labels,
            'series_uids': series_uids
        }
        self.validation_outputs.append(batch_outputs)
        
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        # Concatenate on GPU first, then apply sigmoid once
        all_pred_cls = torch.cat([batch['pred_cls'] for batch in self.validation_outputs])
        all_pred_locs = torch.cat([batch['pred_locs'] for batch in self.validation_outputs])
        all_cls_labels = torch.cat([batch['cls_labels'] for batch in self.validation_outputs])
        all_loc_labels = torch.cat([batch['loc_labels'] for batch in self.validation_outputs])
        
        # Apply sigmoid once and move to CPU
        all_pred_cls = torch.sigmoid(all_pred_cls).cpu()
        all_pred_locs = torch.sigmoid(all_pred_locs).cpu()
        all_cls_labels = all_cls_labels.cpu()
        all_loc_labels = all_loc_labels.cpu()
        
        all_series_uids = [uid for batch in self.validation_outputs for uid in batch['series_uids']]
        
        data_dict = {
            'series_uid': all_series_uids,
            'pred_cls': all_pred_cls.numpy(),
            'cls_label': all_cls_labels.numpy()
        }
        
        pred_locs_np = all_pred_locs.numpy()
        loc_labels_np = all_loc_labels.numpy()
        
        for i in range(self.num_classes):
            data_dict[f'pred_loc_{i}'] = pred_locs_np[:, i]
            data_dict[f'loc_label_{i}'] = loc_labels_np[:, i]
        
        import pandas as pd
        df = pd.DataFrame(data_dict)
        
        agg_dict = {
            'pred_cls': 'max',
            'cls_label': 'max',
        }
        agg_dict.update({f'pred_loc_{i}': 'max' for i in range(self.num_classes)})
        agg_dict.update({f'loc_label_{i}': 'max' for i in range(self.num_classes)})
        
        series_agg = df.groupby('series_uid').agg(agg_dict).reset_index()
        
        # Calculate series-level metrics
        from sklearn.metrics import roc_auc_score
        import numpy as np
        
        try:
            # Classification AUC (series-level)
            cls_labels_series = series_agg['cls_label'].values
            cls_preds_series = series_agg['pred_cls'].values

            cls_auc_series = 0.5
            if len(np.unique(cls_labels_series)) > 1:
                cls_auc_series = roc_auc_score(cls_labels_series, cls_preds_series)
            self.log('val_cls_auroc_series', cls_auc_series, on_epoch=True, prog_bar=True)

            # Location AUCs per label (series-level, columnwise)
            loc_labels_series = series_agg[[f'loc_label_{i}' for i in range(self.num_classes)]].values
            loc_preds_series = series_agg[[f'pred_loc_{i}' for i in range(self.num_classes)]].values

            loc_aucs = []
            for i in range(self.num_classes):
                y_true_i = loc_labels_series[:, i]
                y_pred_i = loc_preds_series[:, i]
                if len(np.unique(y_true_i)) > 1:
                    auc_i = roc_auc_score(y_true_i, y_pred_i)
                else:
                    auc_i = 0.5
                loc_aucs.append(auc_i)

            mean_loc_auc_series = float(np.mean(loc_aucs)) if len(loc_aucs) > 0 else 0.5
            self.log('val_loc_auroc_series', mean_loc_auc_series, on_epoch=True, prog_bar=True)

            # Kaggle score: simple average of aneurysm-present AUC and the mean of the other 13 AUCs
            kaggle_score = 0.5 * (cls_auc_series + mean_loc_auc_series)
            self.log('val_kaggle_score', kaggle_score, on_epoch=True, prog_bar=True)
                
            if self.trainer.is_global_zero: 
                print(f"\nSeries-level validation | Count: {len(series_agg)} | "
                    f"Positive: {cls_labels_series.sum()}")
                if len(np.unique(cls_labels_series)) > 1:
                    print(f"Classification AUC (series): {cls_auc_series:.4f}")
                if len(loc_aucs) > 0:
                    print(f"Location AUC (series mean): {mean_loc_auc_series:.4f}")
                print(f"Kaggle score: {kaggle_score:.4f}")
                    
        except Exception as e:
            print(f"Error calculating series-level metrics: {e}")
        
        # Step LR scheduler (ReduceLROnPlateau) after validation, when val metrics are available
        try:
            sch = self.lr_schedulers()
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = self.trainer.callback_metrics.get("val_loss")
                if val_loss is not None:
                    sch.step(val_loss)
        except Exception as _:
            pass

        # Clear validation outputs
        self.validation_outputs.clear()

    #def on_train_epoch_end(self):
    #    sch = self.lr_schedulers()
#
    #    # If the selected scheduler is a ReduceLROnPlateau scheduler.
    #    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and (self.current_epoch + 1) % self.cfg.trainer.check_val_every_n_epoch == 0:
    #        sch.step(self.trainer.callback_metrics["val_loss"])
    #    
#
    #
    #def configure_optimizers(self):
    #    optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
    #    
    #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=40)
    #    return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step() 