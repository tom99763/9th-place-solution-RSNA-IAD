import monai.networks.nets as mnn
import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, model_name, spatial_dims, in_channels, pretrained, num_classes):
        super(Net, self).__init__()

        self.backbone = mnn.EfficientNetBNFeatures(model_name=model_name
                                          , spatial_dims=spatial_dims
                                          , in_channels=in_channels
                                          , pretrained=pretrained)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = torch.nn.Linear(320,num_classes)        

    def forward(self, x):
        out = self.backbone(x)[-1]
        cls_logits = self.cls_head(self.global_pool(out).flatten(1))
        return cls_logits
