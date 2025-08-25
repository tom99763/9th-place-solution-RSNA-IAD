import logging
import os
import lightning
from monai.inferers.inferer import SlidingWindowInfererAdapt
from utils.metrics import *
import monai
import torch.nn.functional as F
from monai.networks import one_hot

logger = logging.getLogger(__name__)
ce_fn = torch.nn.CrossEntropyLoss()

def dice_loss(pred, target, num_classes=14, lambda_ce=0.1, lambda_dice=0.9):
    """
    pred: (B, C, D, H, W)
    target: (B, D, H, W), long class indices 0..C-1
    """
    # CrossEntropyLoss (expects class indices)
    ce_loss = ce_fn(pred, target[:, 0].long())

    # DiceLoss memory-efficient: gather predicted probabilities per target class
    pred_soft = F.softmax(pred, dim=1)  # (B, C, D, H, W)

    # Compute per-class Dice in a vectorized way
    # pred_flat: (B, C, D*H*W), target_flat: (B, D*H*W)
    B, C, D, H, W = pred.shape
    pred_flat = pred_soft.view(B, C, -1)
    target_flat = target.view(B, -1)

    dice_loss_total = 0.0
    for c in range(C):
        # create binary mask for this class
        target_c = (target_flat == c).float()  # (B, D*H*W)
        pred_c = pred_flat[:, c, :]  # (B, D*H*W)
        intersection = (pred_c * target_c).sum(dim=1)
        union = pred_c.sum(dim=1) + target_c.sum(dim=1)
        dice_c = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss_total += dice_c.mean()
    dice_loss_total /= C
    return lambda_ce * ce_loss + lambda_dice * dice_loss_total



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

    def training_step(self, batch, batch_idx):
        image, mask = batch
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)  # no extra .long() here; handled inside loss
        self.log("train_loss", loss.item(), logger=(self.rank == 0), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch

        with torch.no_grad():
            # raw logits
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(f"{self.dataset_name}_val_loss", loss.item(), prog_bar=True)

            # ensure target is (B,D,H,W)
            if mask.dim() == 5 and mask.size(1) == 1:
                target_idx = mask[:, 0].long()
            else:
                target_idx = mask.long()

            # compute recall using raw logits
            recall, tp, fn = volumetric_recall(pred_mask, target_idx, already_classes=False)
            self.log(f"{self.dataset_name}_val_volumetric_recall", recall.item(), prog_bar=True)

        return loss

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