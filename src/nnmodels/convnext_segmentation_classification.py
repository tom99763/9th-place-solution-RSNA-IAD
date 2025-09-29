# %load ../src/models/segmentation_classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# You may need to install timm and torchinfo
# !pip install timm torchinfo

def encode_for_resnet(e, x, B, depth_scaling=[2,2,2,2,1]):

    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode=[]
    x = e.stem_0(x)
    x = e.stem_1(x)
    
   
    x = e.stages_0(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)

    x = e.stages_1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)
    
    x = e.stages_2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)
    
    x = e.stages_3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)
    

    return encode

class DecoderBlock(nn.Module):
    """
    U-Net-style decoder block with skip connections.
    It upsamples the feature map and concatenates it with the corresponding
    feature map from the encoder, followed by two convolutional layers.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsampling layer followed by a convolution to adjust channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # We combine the upsampled channels with the skip connection channels
        combined_in_channels = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(combined_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate along the channel dimension
        if skip is not None:
             x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class SegmentationClassifier(nn.Module):
    """
    A multi-task model for binary classification and binary segmentation.
    Uses a timm model as the encoder.
    """
    def __init__(self, model_name='convnext_tiny.fb_in22k', pretrained=True):
        super().__init__()
        
        # 1. Encoder (from timm)
        # We use features_only=True to get intermediate feature maps
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=3
        )
        
        # Get the channel sizes of the feature maps from the encoder
        encoder_channels = self.encoder.feature_info.channels()
        # e.g., for resnet34: [64, 64, 128, 256, 512]
        
        # 2. Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # The input to the linear layer is the number of channels in the last feature map
        self.classification_head = nn.Linear(encoder_channels[-1], 1)

        # 3. Segmentation Head (Decoder)
        # We work backwards from the last feature map
        self.decoder_blocks = nn.ModuleList()
        
        reversed_encoder_channels = list(reversed(encoder_channels))
        # Example for resnet34: [512, 256, 128, 64, 64]
        
        # The first decoder block takes the last feature map
        in_ch = reversed_encoder_channels[0] 
        # The first decoder block does not have a skip connection from a deeper layer
        skip_ch = 0 
        out_ch = in_ch // 2 # Halve the channels
        
        self.center = nn.Identity() # Placeholder for the deepest features
        
        for i in range(len(reversed_encoder_channels) - 1):
            in_ch = reversed_encoder_channels[i] if i == 0 else out_ch
            skip_ch = reversed_encoder_channels[i+1]
            out_ch = reversed_encoder_channels[i+1]
            
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))

        # Final 1x1 convolution to get a single-channel mask
        # The number of input channels is the output of the last decoder block
        last_decoder_out_channels = reversed_encoder_channels[-1]
        self.segmentation_head = nn.Conv2d(last_decoder_out_channels, 1, kernel_size=1)

    def forward(self, x):

        # Bx96x384x384 -> 
        B, D, H, W = x.shape
        x = x.reshape(B*D, 1, H, W)
        x = x.expand(-1, 3, -1, -1)
        features = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2])

        for i in range(len(features)):
            features[i] = features[i].amax(dim=2)

        # --- Classification Path ---
        # Use the last and most abstract feature map for classification
        last_feature = features[-1]
        pooled_features = self.avgpool(last_feature)
        flat_features = self.flatten(pooled_features)
        classification_output = self.classification_head(flat_features)
        
        # --- Segmentation Path ---
        # Reverse features to be in order for the decoder (from deep to shallow)
        features_reversed = list(reversed(features))
        
        # Start with the deepest feature map
        seg_path = self.center(features_reversed[0])
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # The skip connection is the next feature map in the reversed list
            skip_feature = features_reversed[i+1]
            seg_path = decoder_block(seg_path, skip_feature)
            
        # Apply the final convolution
        segmentation_output = self.segmentation_head(seg_path)
        
        # Upsample the final mask to match the input image size
        segmentation_output = nn.functional.interpolate(
            segmentation_output, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        
        return classification_output, segmentation_output

