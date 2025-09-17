import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones"""

    def __init__(self, model_name, in_chans, img_size, num_classes=13, pretrained=True,
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()

        self.model_name = model_name

        self.loc_classifier = timm.create_model(
            model_name,
            num_classes=14,
            pretrained=False,
            in_chans=32
        )

    def forward(self, image):
        return self.loc_classifier(image)
