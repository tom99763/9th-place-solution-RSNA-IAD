import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.networks.nets import DenseNet


# class MultiBackboneModel(nn.Module):
#     """Flexible model that can use different backbones"""
#
#     def __init__(self, model_name, in_chans, img_size, num_classes=13, pretrained=True,
#                  drop_rate=0.3, drop_path_rate=0.2):
#         super().__init__()
#
#         self.model_name = model_name
#
#          self.loc_classifier = timm.create_model(
#              model_name,
#              num_classes=14,
#              pretrained=False,
#              in_chans=32
#         )
#
#
#     def forward(self, image):
#         return self.loc_classifier(image)
#

class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones"""

    def __init__(self, model_name, in_chans, img_size, num_classes=13, pretrained=True,
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()

        self.loc_classifier = DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=14,
            init_features=64,  # double initial features
            growth_rate=32,  # double growth rate
            block_config=(3, 6, 6, 3),  # deeper network
            bn_size=4,  # same bottleneck factor
            dropout_prob=0.3  # slightly higher dropout for larger net
        )

    def forward(self, image):
        return self.loc_classifier(image)



