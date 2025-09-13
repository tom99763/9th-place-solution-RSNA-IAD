import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
import torch.nn.functional as F
import numpy as np

from monai.losses.dice import DiceLoss
from monai.metrics.meandice import DiceMetric
import torchmetrics
import torch.nn as nn
from monai.inferers import sliding_window_inference

torch.set_float32_matmul_precision('medium')


class DiceBCECombined(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        # DiceLoss expects probabilities or logits depending on use_sigmoid flag.
        self.dice = DiceLoss(include_background=False, to_onehot_y=False, sigmoid=True)
        # Use BCEWithLogits so model outputs raw logits
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([32.0]))

    def forward(self, logits, target):
        """
        logits: tensor [B, 1, D, H, W] raw logits from model
        target: tensor [B, 1, D, H, W] binary {0,1}
        """
        bce_loss = self.bce(logits, target.float())
        return bce_loss


class LitSegmentationCls(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg
        self.loss_fn = DiceBCECombined()
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x = torch.vstack([batch[i]["volume"] for i in range(len(batch))])
        mask = torch.vstack([batch[i]["mask"] for i in range(len(batch))])
        predmask =self(x)
        loss = self.loss_fn(predmask, mask)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # mask = (mask > 0).float()
        # dice = self.dice_metric(y_pred=predmask, y=mask)
        # self.dice_metric.reset()
        #
        # labels = mask.amax(dim=(1,2,3,4))
        # dice_mean = dice[labels == 1].mean()
        #
        # if not torch.isnan(dice_mean):
        #     self.log("train_dice", dice_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x,mask = batch["volume"], batch["mask"]
        predmask = sliding_window_inference(
            inputs=x,
            roi_size=(128,256,256),
            sw_batch_size=4,
            predictor=self,
            overlap=0.0,
            mode="gaussian",   # blending mode (gaussian/constant/mean)
        )


        loss = self.loss_fn(predmask, mask)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pred_cls_logits = predmask.amax(dim=(1,2,3,4))
        labels = mask.amax(dim=(1,2,3,4))
        self.val_cls_auroc.update(pred_cls_logits, labels.long())
        return loss

        # predmask = (predmask.sigmoid() > 0.1).float()
        # mask = (mask > 0).float()
        # dice = self.dice_metric(y_pred=predmask, y=mask)
        # self.dice_metric.reset()
        #
        # dice_mean = dice[labels == 1].mean()
        #
        # if not torch.isnan(dice_mean):
        #     self.log("val_dice", dice_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        cls_auc = self.val_cls_auroc.compute()
        self.log('val_cls_auroc', cls_auc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_cls_auroc.reset()

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
