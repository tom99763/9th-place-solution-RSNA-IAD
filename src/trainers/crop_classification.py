import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
import torch.nn.functional as F
import torch.nn as nn

from monai.inferers.inferer import Inferer
from monai.inferers.splitter import SlidingWindowSplitter
from monai.inferers.merger import AvgMerger
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

        # BSx14
        ys = model(xs)
        return ys.max(dim=0, keepdim=True).values


class LitWindowedMIP(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Saves args to checkpoint
        
        self.model = model
        self.cfg = cfg
        self.automatic_optimization = False

        self.val_loc_auroc = torchmetrics.AUROC(task="multilabel", num_labels=self.cfg.params.num_classes - 1, average="macro")

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.loc_loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.val_cls_auroc = torchmetrics.AUROC(task="binary")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, labels = batch

        predlabels =self(x)
        loss = self.cls_loss_fn(predlabels, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Manual backward pass
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss
    
    def validation_step(self, sample, batch_idx):
        # return
        uid, xs, labels = sample
        xs = xs.squeeze(dim=0)
        labels = labels[:,1:]

        predlabels = []

        for x in xs:
            inferer = WindowedMIPInferer(self.cfg)
            # 1x2x512x512 -> 1x14x512x512
            predlabel = inferer(x.unsqueeze(0), self.model)
            predlabels.append(predlabel)

        # BSx14 -> 1x13
        predlabels = torch.vstack(predlabels).max(dim=0, keepdim=True).values[:,1:]
        loss = self.loc_loss_fn(predlabels, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        cls_probs = predlabels.sigmoid()

        self.val_loc_auroc.update(cls_probs, labels.long())
        self.log('val_loc_auroc', self.val_loc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
   
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and (self.current_epoch + 1) % self.cfg.trainer.check_val_every_n_epoch == 0:
            sch.step(self.trainer.callback_metrics["val_loss"])
        self.val_loc_auroc.reset()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
