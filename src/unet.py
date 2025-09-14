
import torch
import torch.nn as nn
from monai.networks.nets import DynUNet, UNet, AttentionUnet, BasicUNet
from monai.networks.blocks import ResidualUnit
from typing import Tuple


def make_dynunet(
    in_ch: int = 1,
    out_ch: int = 1,
    roi_size: Tuple[int, int, int] = (64, 128, 128),
    strides: Tuple[Tuple[int, int, int], ...] = ((1,2,2),(2,2,2),(2,2,2)),
    kernels: Tuple[Tuple[int, int, int], ...] = ((3,3,3),(3,3,3),(3,3,3))
):
    """
    Create a DynUNet tuned to reasonable depths for volume patches.
    - strides and kernels should match the number of spatial levels.
    - roi_size: the patch size used at inference/training (for convenience/documentation).
    """
    # Typical channel progression (tweak to fit GPU mem): [16,32,64,128,256]
    # Number of levels equals len(strides) + 1
    channels = (16, 32, 64, 128)  # adjust / expand if needed
    model = DynUNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[::-1],  # common choice
        filters=channels,
        norm_name="instance",
        deep_supervision=False,
    )
    return model


def make_attention_unet(in_ch=1, out_ch=1, features=(16, 32, 64, 128)):
    """
    MONAI AttentionUnet. features controls number of channels at each level.
    """
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        features=features,
        norm="instance",
    )
    return model


def make_vanilla_unet(in_ch=1, out_ch=1, channels=(16, 32, 64, 128)):
    """Simple UNet from MONAI (can be tuned)"""
    model = UNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        channels=channels,
        strides=(2,2,2)[: len(channels)-1] if len(channels)>1 else (),
        num_res_units=0,
        norm='instance',
    )
    return model


def make_resunet(in_ch=1, out_ch=1, channels=(64, 128, 256, 256)):
    """
    Small Residual UNet using ResidualUnit blocks.
    Build encoder-decoder manually using BasicUNet for simplicity, but replace conv blocks with residual units
    (This is a lightweight custom ResUNet example.)
    """
    # Use BasicUNet with num_res_units>0 to get residual units inside MONAI UNet implementation:
    model = UNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        channels=channels,
        strides=(2,2,2)[: len(channels)-1],
        num_res_units=1,   # will use ResidualUnit internally
        norm='instance',
    )
    return model


def get_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "dynunet":
        return make_dynunet(**kwargs)
    if name == "attentionunet":
        return make_attention_unet(**kwargs)
    if name == "vanilla":
        return make_vanilla_unet(**kwargs)
    if name == "resunet":
        return make_resunet(**kwargs)
    raise ValueError(f"Unsupported model name: {name}")
