import logging
import os
import lightning
from monai.inferers.inferer import SlidingWindowInfererAdapt
from utils.metrics import *

logger = logging.getLogger(__name__)

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

    def training_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask.long()  # make sure labels are class indices, not float
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)
        self.log("train_loss", loss.item(), logger=(self.rank == 0), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask.long()
        with torch.no_grad():
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(f"{self.dataset_name}_val_loss", loss.item(), prog_bar=True)

            # Convert predictions
            pred_classes = pred_mask.softmax(dim=1).argmax(dim=1)

            # Example metric: per-class recall
            recall, tp, fn = volumetric_recall(pred_classes, mask)
            self.log("val_volumetric_recall", recall.item(), prog_bar=True)

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