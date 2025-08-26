import logging
import os
import lightning
from monai.inferers.inferer import SlidingWindowInfererAdapt
from utils.metrics import *
import monai
import torch.nn.functional as F
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
import torch.nn as nn

logger = logging.getLogger(__name__)
ce_fn = torch.nn.CrossEntropyLoss()


def dice_loss(pred, target, num_classes=14,
              lambda_ce=0.05, lambda_dice=0.95,
              bg_weight=0.05,  # very small background weight
              smooth=1e-6):
    """
    pred: (B, C, D, H, W) - raw logits
    target: (B, D, H, W) - long class indices [0..C-1]
    """

    B, C, D, H, W = pred.shape
    target = target.long()

    # ---- Compute class weights from target for CE ----
    with torch.no_grad():
        flat_target = target.view(-1)
        counts = torch.bincount(flat_target, minlength=num_classes).float()

        weights = torch.zeros_like(counts)
        nonzero = counts > 0
        weights[nonzero] = counts.sum() / (counts[nonzero] * len(counts[nonzero]))
        weights = weights / weights.sum()

        # reduce background weight
        weights[0] = bg_weight * weights[1:].mean()

    ce_fn = nn.CrossEntropyLoss(weight=weights.to(pred.device))
    ce_loss = ce_fn(pred, target[:, 0] if target.ndim == 5 else target)

    # ---- Dice Loss ----
    pred_soft = F.softmax(pred, dim=1)  # (B, C, D, H, W)

    # ensure target shape is (B, D, H, W)
    if target.ndim == 5 and target.shape[1] == 1:
        target = target.squeeze(1)  # remove channel dim

    # one-hot encode
    target_onehot = F.one_hot(target.long(), num_classes=C)  # (B, D, H, W, C)
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

    # Flatten
    pred_flat = pred_soft.view(B, C, -1)
    target_flat = target_onehot.view(B, C, -1)

    # Intersection & union
    intersection = (pred_flat * target_flat).sum(-1)  # (B, C)
    union = pred_flat.sum(-1) + target_flat.sum(-1)  # (B, C)

    # ---- Ignore background class 0 ----
    dice_score = (2. * intersection + smooth) / (union + smooth)  # (B, C)
    dice_loss_per_class = 1 - dice_score  # (B, C)

    # Apply class weights
    class_weights = torch.ones(C, device=pred.device)
    class_weights[0] = bg_weight
    dice_loss_val = (dice_loss_per_class * class_weights).sum(dim=1) / class_weights.sum()
    dice_loss_val = dice_loss_val.mean()  # average over batch

    # ---- Combine ----
    return lambda_ce * ce_loss + lambda_dice * dice_loss_val



class RSNAModuleFinetune(lightning.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            loss,
            optimizer_factory,
            prediction_threshold: float,
            scheduler_configs=None,
            dataset_name: str = None,
            input_size: tuple = None,
            batch_size: int = None,
            threshold: float = None,
            *args,
            **kwargs
    ):
        super().__init__()
        print('threshold:', threshold)
        self.model = model
        self.loss = dice_loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.prediction_threshold = prediction_threshold
        self.rank = 0 if "LOCAL_RANK" not in os.environ else os.environ["LOCAL_RANK"]
        self.dataset_name = dataset_name
        logger.info(f"Dataset name: {self.dataset_name}")
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=input_size, sw_batch_size=batch_size, overlap=0,
        )
        self.threshold = threshold
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    def training_step(self, batch, batch_idx):
        image, mask = batch
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)  # no extra .long() here; handled inside loss
        self.log("train_loss", loss.item(), logger=(self.rank == 0), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch

        with torch.no_grad():
            pred_logits = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_logits, mask)
            self.log(f"{self.dataset_name}_val_loss", loss.item(), prog_bar=True)

            # convert to predicted labels
            pred_labels = torch.argmax(pred_logits, dim=1, keepdim=True)  # (B,1,D,H,W)

            # one-hot encode target and prediction for metric
            num_classes = pred_logits.shape[1]
            target_oh = one_hot(mask, num_classes=num_classes)
            pred_oh = one_hot(pred_labels, num_classes=num_classes)

            # update dice metric
            self.dice_metric(y_pred=pred_oh, y=target_oh)
        return loss

    def on_validation_epoch_end(self):
        # aggregate
        dice_score = self.dice_metric.aggregate().mean().item()
        self.log("val_dice", dice_score, prog_bar=True)

        # reset for next epoch
        self.dice_metric.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=self.parameters())

        if self.scheduler_configs is not None:
            schedulers = []
            logger.info(f"Initializing schedulers: {self.scheduler_configs}")
            for scheduler_name, scheduler_config in self.scheduler_configs.items():
                if scheduler_config is None:
                    continue  # skip empty configs during finetuning

                logger.info(f"Initializing scheduler: {scheduler_name}")
                scheduler_config["scheduler"] = scheduler_config["scheduler"](optimizer=optimizer)
                scheduler_config = dict(scheduler_config)
                schedulers.append(scheduler_config)
            return [optimizer], schedulers
        return optimizer