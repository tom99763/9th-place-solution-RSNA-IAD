import torch
import pytorch_lightning as pl
import torchmetrics
from hydra.utils import instantiate

import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')

class DiceLoss(nn.Module):
    """
    Calculates the Dice Loss, a common metric for segmentation tasks
    that measures overlap.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten the tensors to treat each pixel as an item
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # The loss is 1 minus the Dice Score
        return 1 - dice_score

def calculate_combined_loss(class_logits, seg_logits, class_targets, seg_targets, class_weight=0.5):
    """
    Calculates the combined loss for the multi-task model.
    
    Args:
        class_logits (torch.Tensor): Raw output from the classification head.
        seg_logits (torch.Tensor): Raw output from the segmentation head.
        class_targets (torch.Tensor): Ground truth for classification (0s and 1s).
        seg_targets (torch.Tensor): Ground truth for segmentation masks (0s and 1s).
        class_weight (float): The weight (alpha) for the classification loss.
                              The segmentation weight will be (1 - class_weight).
    
    Returns:
        A dictionary containing the total loss and its individual components.
    """

    
    loss_class = F.binary_cross_entropy_with_logits(class_logits.view(-1), class_targets.float().view(-1))

    # 2. Segmentation Loss (BCE + Dice)
    dice_loss_fn = DiceLoss()
    loss_seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets.unsqueeze(1))
    loss_seg_dice = dice_loss_fn(seg_logits, seg_targets)
    
    # Combine the two segmentation losses (common practice)
    loss_seg = loss_seg_bce + loss_seg_dice

    # 3. Total Combined Loss
    total_loss = class_weight * loss_class + (1 - class_weight) * loss_seg
    
    return {
        'total_loss': total_loss,
        'classification_loss': loss_class,
        'segmentation_loss': loss_seg
    }

class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg
        self.loss_fn = calculate_combined_loss

        self.train_cls_auroc = torchmetrics.AUROC(task="binary")
        self.val_cls_auroc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        xs, masks, labels  = batch

        cls_logits, seg_logits = self(xs)

        all_losses = self.loss_fn(cls_logits, seg_logits, labels, masks)

        loss = all_losses["total_loss"]

        self.train_cls_auroc.update(cls_logits, labels.long())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_cls_loss', all_losses["classification_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_seg_loss', all_losses["segmentation_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, sample, batch_idx):
        xs, masks, labels = sample

        cls_logits, seg_logits =self(xs)

        all_losses = self.loss_fn(cls_logits, seg_logits, labels, masks)
        loss = all_losses["total_loss"]
      
        self.val_cls_auroc.update(cls_logits, labels.long())

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_cls_loss', all_losses["classification_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_seg_loss', all_losses["segmentation_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        
        self.log('train_cls_auroc', self.train_cls_auroc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_cls_auroc.reset()

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
            eta_min=1e-6
        )
        return { "optimizer": optimizer
                , "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": frequency
                    }
                }
