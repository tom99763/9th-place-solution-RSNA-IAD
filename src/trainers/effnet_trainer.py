import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

torch.set_float32_matmul_precision('medium')

class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) 
        
        self.model = model
        self.cfg = cfg
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.num_classes = self.cfg.model.num_classes

        self.train_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)
        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_classes)

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")
       
        self.validation_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, cls_labels, loc_labels, series_uids = batch

        pred_cls, pred_locs =self(x)
        pred_cls = pred_cls.squeeze(-1)  

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels.float())

        loss = 3*cls_loss + loc_loss

        if self.global_step %10 == 0:
            self.train_loc_auroc.update(pred_locs.detach(), loc_labels.int())
            self.train_cls_auroc.update(torch.sigmoid(pred_cls.detach()), cls_labels.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loc_auroc', self.train_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cls_auroc', self.train_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)

        
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, cls_labels, loc_labels, series_uids = batch
        
        pred_cls, pred_locs = self(x)
        pred_cls = pred_cls.squeeze(-1)

        cls_labels_float = cls_labels.float()
        loc_labels_float = loc_labels.float()

        loc_loss = self.loc_loss_fn(pred_locs, loc_labels_float)
        cls_loss = self.cls_loss_fn(pred_cls, cls_labels_float)
        loss = 3*cls_loss + loc_loss 

        self.val_loc_auroc.update(pred_locs.detach(), loc_labels.int())
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
            # Classification AUC
            cls_labels_series = series_agg['cls_label'].values
            cls_preds_series = series_agg['pred_cls'].values
            
            cls_auc_series = 0.0
            if len(np.unique(cls_labels_series)) > 1:
                cls_auc_series = roc_auc_score(cls_labels_series, cls_preds_series)
                self.log('val_cls_auroc_series', cls_auc_series, on_epoch=True, prog_bar=True)
            
            # Location AUC - more efficient array construction
            loc_labels_series = series_agg[[f'loc_label_{i}' for i in range(self.num_classes)]].values
            loc_preds_series = series_agg[[f'pred_loc_{i}' for i in range(self.num_classes)]].values
            
            loc_auc_series = 0.0
            if np.any(loc_labels_series.sum(axis=0) > 0):
                loc_auc_series = roc_auc_score(loc_labels_series, loc_preds_series, average="micro")
                self.log('val_loc_auroc_series', loc_auc_series, on_epoch=True, prog_bar=True)
            
            kaggle_score = (cls_auc_series * 13 + loc_auc_series * 1) / 14 # kaggle_score = (cls_auc_series * 13 + loc_auc_series * 1) / 14
            self.log('val_kaggle_score', kaggle_score, on_epoch=True, prog_bar=True)
                
            if self.trainer.is_global_zero: 
                print(f"\nSeries-level validation | Count: {len(series_agg)} | "
                    f"Positive: {cls_labels_series.sum()}")
                if len(np.unique(cls_labels_series)) > 1:
                    print(f"Classification AUC (series): {cls_auc_series:.4f}")
                if np.any(loc_labels_series.sum(axis=0) > 0):
                    print(f"Location AUC (series): {loc_auc_series:.4f}")
                print(f"Kaggle score: {kaggle_score:.4f}")
                    
        except Exception as e:
            print(f"Error calculating series-level metrics: {e}")
        
        # Clear validation outputs
        self.validation_outputs.clear()

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and (self.current_epoch + 1) % self.cfg.trainer.check_val_every_n_epoch == 0:
            sch.step(self.trainer.callback_metrics["val_loss"])
    
    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
