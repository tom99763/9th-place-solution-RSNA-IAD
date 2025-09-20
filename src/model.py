import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.networks.nets import DenseNet


class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones"""

    def __init__(self, model_name, **kwargs):
        super().__init__()

        self.model_name = model_name
        self.loc_classifier = timm.create_model(
             model_name,
             num_classes=num_classes,
             pretrained=pretrained,
             in_chans=in_chans
        )

    def forward(self, image):
        return self.loc_classifier(image)
#

# class MultiBackboneModel(nn.Module):
#     """Flexible model that can use different backbones"""
#
#     def __init__(self, model_name, **kwargs):
#         super().__init__()
#
#         self.loc_classifier = DenseNet(
#             spatial_dims=3,
#             in_channels=1,
#             out_channels=14,
#             init_features=48,  # increase initial channels
#             growth_rate=24,  # increase growth rate
#             block_config=(3, 4, 4, 3),  # slightly deeper network
#             bn_size=4,  # keep the same bottleneck factor
#             dropout_prob=0.25  # slightly higher dropout
#         )
#
#     def forward(self, image):
#         return self.loc_classifier(image)



