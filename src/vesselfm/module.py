import logging
import os
import lightning as L
from monai.inferers.inferer import SlidingWindowInfererAdapt
from utils.metrics import *
import monai
import torch.nn.functional as F
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
import torch.nn as nn

logger = logging.getLogger(__name__)
ce_fn = torch.nn.CrossEntropyLoss()

class ForegroundHitRate:
    def __init__(self, num_classes: int, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background
        self.reset()

    def reset(self):
        self.hits = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)

    def __call__(self, y_pred, y):
        # y_pred and y: one-hot encoded (B, C, D, H, W)
        for c in range(int(self.include_background), self.num_classes):
            gt_present = (y[:, c].sum() > 0).item()
            pred_present = (y_pred[:, c].sum() > 0).item()
            if gt_present:
                self.total[c] += 1
                if pred_present:
                    self.hits[c] += 1

    def aggregate(self):
        return (self.hits / (self.total + 1e-8)).mean().item()


# -----------------------------
# Loss Function
# -----------------------------
def dice_loss(pred, target, num_classes=14, lambda_ce=0.1, lambda_dice=0.9,
              bg_weight=0.05, smooth=1e-6):
    """
    pred: (B, C, D, H, W) - raw logits
    target: (B, D, H, W) - long class indices [0..C-1]
    """
    B, C, D, H, W = pred.shape
    target = target.long()

    # Compute class weights for CE
    with torch.no_grad():
        flat_target = target.view(-1)
        counts = torch.bincount(flat_target, minlength=num_classes).float()
        weights = torch.zeros_like(counts)
        nonzero = counts > 0
        weights[nonzero] = counts.sum() / (counts[nonzero] * len(counts[nonzero]))
        weights = weights / weights.sum()
        weights[0] = bg_weight * weights[1:].mean()

    ce_fn = nn.CrossEntropyLoss(weight=weights.to(pred.device))
    ce_loss = ce_fn(pred, target)

    # Softmax for dice
    pred_soft = F.softmax(pred, dim=1)

    # One-hot target
    target_onehot = F.one_hot(target, num_classes=C).permute(0, 4, 1, 2, 3).float()

    # Flatten
    pred_flat = pred_soft.view(B, C, -1)
    target_flat = target_onehot.view(B, C, -1)

    # Intersection and union
    intersection = (pred_flat * target_flat).sum(-1)
    union = pred_flat.sum(-1) + target_flat.sum(-1)

    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss_per_class = 1 - dice_score

    # Ignore background
    dice_loss_val = dice_loss_per_class[:, 1:].mean()

    return lambda_ce * ce_loss + lambda_dice * dice_loss_val


# -----------------------------
# Lightning Module
# -----------------------------
class RSNAModuleFinetune(L.LightningModule):
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
        self.loss = loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.prediction_threshold = prediction_threshold
        self.rank = 0 if "LOCAL_RANK" not in os.environ else os.environ["LOCAL_RANK"]
        self.dataset_name = dataset_name
        logger.info(f"Dataset name: {self.dataset_name}")
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=input_size, sw_batch_size=batch_size, overlap=0.5,
        )
        self.threshold = threshold

        # Metrics
        self.dice_metric = DiceMetric(include_background=False,
                                      reduction="mean",
                                      ignore_empty=True)
        self.hit_metric = ForegroundHitRate(num_classes=14, include_background=False)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        pred = self.model(image)
        loss = self.loss(pred, mask)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        with torch.no_grad():
            pred_logits = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_logits, mask)

            # One-hot encode
            pred_labels = torch.argmax(pred_logits, dim=1, keepdim=True)
            pred_oh = one_hot(pred_labels, num_classes=14)
            target_oh = one_hot(mask, num_classes=14)

            # Update metrics
            self.dice_metric(y_pred=pred_oh, y=target_oh)
            self.hit_metric(pred_oh, target_oh)

            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        dice_score = self.dice_metric.aggregate().mean().item()
        hit_score = self.hit_metric.aggregate()
        self.log("val_dice", dice_score, prog_bar=True)
        self.log("val_hit_rate", hit_score, prog_bar=True)

        self.dice_metric.reset()
        self.hit_metric.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_configs:
            schedulers = []
            for _, cfg in self.scheduler_configs.items():
                if cfg is None:
                    continue
                cfg["scheduler"] = cfg["scheduler"](optimizer=optimizer)
                schedulers.append(dict(cfg))
            return [optimizer], schedulers
        return optimizer