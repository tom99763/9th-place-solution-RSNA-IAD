import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

def encode_for_resnet(e, x, B, depth_scaling=[2,2,2,2,1]):

    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape


        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode=[]
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)

    x, x1 = pool_in_depth(x, depth_scaling[0])

    x = F.avg_pool2d(x,kernel_size=2,stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])

    return x

class Net(nn.Module):
    def __init__(self, arch="resnet34d", pretrained=False, drop_rate=0.3, num_classes=13):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        self.arch = arch

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(self.arch, [768])

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )

        self.global_pool = nn.MaxPool2d(4)
        

        self.loc_classifier = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        self.aneurysm_classifier = nn.Sequential(
            nn.Linear(8192, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 1)
        )


    def forward(self, image):

        B, D, H, W = image.shape
        image = image.reshape(B*D, 1, H, W)

        x = (image.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        x = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,2])
        x = self.global_pool(x)
        x = x.reshape(B, -1)

        loc_logits = self.loc_classifier(x)
        cls_logits = self.aneurysm_classifier(x)

        return cls_logits, loc_logits
