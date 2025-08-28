import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
import torch.nn.functional as F

from monai.losses.dice import DiceCELoss
from monai.inferers.inferer import Inferer
from monai.inferers.splitter import SlidingWindowSplitter
from monai.inferers.merger import AvgMerger
from monai.metrics.meandice import DiceMetric
import torchmetrics

torch.set_float32_matmul_precision('medium')


class WindowedMIPInferer(Inferer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.validation_params = self.cfg.params.validation
        self.splitter = SlidingWindowSplitter(self.validation_params.patch_size)
        self.merger = AvgMerger(merged_shape=tuple(self.validation_params.output_img), device="cuda")

    def __call__(self, x, model):
        img_width,img_height = (self.cfg.params.img_width, self.cfg.params.img_height)

        slices_resized = F.interpolate(x, size=(img_width, img_height), mode='bilinear', align_corners=False)[0]

        # Bx2x128x128
        patches, locs = [], []
        for patch, loc in self.splitter(x):
            patches.append(patch)  # keep whole (B,C,H,W)
            locs.append(loc)

        patches = torch.vstack(patches)
        patches = F.interpolate(patches, size=(img_width, img_height), mode="bilinear", align_corners=False)

        xs = torch.stack([torch.vstack([patch, slices_resized]) for patch in patches])

        # BSx14x256x256
        ys = model(xs)

        # BSx14x128x128
        ys = F.interpolate(ys, size=tuple(self.validation_params.patch_size))

        for y,loc in zip(ys, locs):
            self.merger.aggregate(y.unsqueeze(0), loc)

        # 1x14x512x512
        y = self.merger.finalize()
        return y


class LitWindowedMIP(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg
        self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)
        self.automatic_optimization = False
        self.dice_metric_fn = DiceMetric(include_background=False, num_classes=self.cfg.params.num_classes, ignore_empty=True)

        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.cfg.params.num_classes - 1, average="macro")
        # self.val_cls_auroc = torchmetrics.AUROC(task="binary")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, mask = batch
        mask = self.mask_to_onehot(mask)

        predmask =self(x)
        loss = self.loss_fn(predmask, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Manual backward pass
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    
    def mask_to_onehot(self, mask):
        """
        Convert mask (B, H, W) to one-hot (B, num_classes, H, W).
        """
        one_hot = F.one_hot(mask, num_classes=self.cfg.params.num_classes)  # (B, H, W, num_classes)
        return one_hot.permute(0, 3, 1, 2)  # (B, num_classes, H, W)

    def validation_step(self, sample, batch_idx):
        # return
        uid, xs,masks,labels = sample
        xs = xs.squeeze(dim=0)
        masks = masks.squeeze(dim=0)
        labels = labels.squeeze(dim=0).long()

        predmasks = []

        for x,_ in zip(xs,masks):
            inferer = WindowedMIPInferer(self.cfg)
            # 1x2x512x512 -> 1x14x512x512
            predmask = inferer(x.unsqueeze(0), self.model)
            predmasks.append(predmask)

        # BSx14x512x512
        predmasks = torch.vstack(predmasks)
        masks = self.mask_to_onehot(masks)

        loss = self.loss_fn(predmasks, masks)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        predmasks = predmasks.softmax(dim=1)

        self.dice_metric_fn(predmasks.argmax(1), masks.argmax(1))
        dice_score = self.dice_metric_fn.aggregate()
        self.log('dice_score', dice_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # probs of 13 classes: 0th class is BG
        cls_probs = predmasks.amax(dim=(0,2,3))[1:]
        # self.val_cls_auroc.update(cls_probs.max().reshape(1,1), labels[0:1].reshape(1,1))
        self.val_loc_auroc.update(cls_probs.unsqueeze(0), labels[1:].unsqueeze(0))
        self.log('val_loc_auroc', self.val_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_cls_auroc', self.val_cls_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and (self.current_epoch + 1) % self.cfg.trainer.check_val_every_n_epoch == 0:
            sch.step(self.trainer.callback_metrics["val_loss"])
        self.dice_metric_fn.reset()
        self.val_loc_auroc.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
