import argparse
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
# from torch import amp

from monai.transforms import (
    Compose, MapTransform, EnsureTyped, ToTensord,
    RandFlipd, RandRotate90d, RandGaussianNoised,
    RandCropByPosNegLabeld,SpatialPadd
)
import pydicom
# from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet,DynUNet
# from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
# from monai.metrics import DiceMetric
# from monai.utils import set_determinism
from tqdm.auto import tqdm
import cc3d


def get_volume(base_dir,sid):
    files = glob.glob(f"{base_dir}/{sid}/*.dcm")
    slices = [pydicom.dcmread(f) for f in files]
    
    
    if len(slices)==1:
        ds=slices[0]
        posz_list=[]
        for frame in ds.PerFrameFunctionalGroupsSequence:
            # Plane position sequence
            pos = frame.PlanePositionSequence[0].ImagePositionPatient
            posz_list.append(float(pos[2]))       
        dy,dx=ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        volume = slices[0].pixel_array.astype(np.float32)
    else:
    
        posz_list=[]
        for s in slices:
            # Plane position sequence
            pos = s.ImagePositionPatient[2]
            
            posz_list.append(pos)      
    
    
        dy,dx=slices[0].PixelSpacing
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

        
        volume_list = []
        for s in slices:
            image = s.pixel_array.astype(np.float32)
            volume_list.append(image)#.astype(np.uint8))  # 轉 uint8 節省空間
    
        # 組成 3D array (z, y, x)
        volume = np.stack(volume_list, axis=0)
    
    
    vol=volume#np.load(f"input/npy/{sid}.npy")
    vol_t = torch.from_numpy(vol).to(dtype=torch.float32)[None, None, ...]  # [N=1, C=1, Z, Y, X]
    
    
    dz=np.ptp(posz_list)/(vol.shape[0]-1)
    in_sz = np.array(vol.shape, float)       # [Z, Y, X]
    in_sp = np.array([dz,dy,dx], float)   # [sz, sy, sx]
    out_sp = np.array([0.7, 0.7, 0.7], float)
    
    out_sz = (in_sz * (in_sp / out_sp)).round().astype(int)
    out_sz = tuple(int(x) for x in out_sz)   # (Z, Y, X)
    
    # grid_sample 需要目標 size（注意順序 D,H,W）
    vol_resampled = F.interpolate(
        vol_t, size=out_sz, mode="trilinear", align_corners=False
    )  # [1, 1, Z, Y, X]
    
    img = vol_resampled[0, 0]  # [Z, Y, X]
    if img.shape[0]<100:
        return None
    

    img = percentile_clip_minmax_np(img, pmin=0.5, pmax=99.5)

    return img

def predictor_highest(x):
    out = model(x)
    return out[0] if isinstance(out, (list, tuple)) else out


def percentile_clip_minmax_np(img: torch.Tensor, pmin=1.0, pmax=99.0):
    # 保證是 float32，放到 CPU
    arr = img.detach().cpu().numpy().astype(np.float32)

    # 算分位數
    low = np.percentile(arr, pmin)
    high = np.percentile(arr, pmax)

    # clip + min-max normalize
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-8)

    # 回 torch.float32，放回原本裝置
    return torch.from_numpy(arr).to(img.device)
def save_seg_pred(img,save_dir,sid):
    os.makedirs(save_dir,exist_ok=True)
    if img.ndim == 3:
        img = img[None, ...]  # -> [C,H,W,D]
        
    sample = {"image": img}
    sample = val_tf(sample)
    
    img = sample["image"].to(device)  
    # === sliding-window 推論 ===
    with torch.no_grad():
        logits = sliding_window_inference(
            img.unsqueeze(0), roi_size=ROI_SIZE, sw_batch_size=2,
            predictor=predictor_highest, overlap=0.5, mode="gaussian"
        )
        prob = torch.sigmoid(logits)
    pred=(prob>0.5).squeeze().cpu().numpy().astype(int)
    
    labels, N = cc3d.connected_components(pred, connectivity=26, return_N=True)
    # if N == 0:
    #     return None, bin_
    
    counts = np.bincount(labels.ravel())
    keep_ids = np.where(counts >= 800)[0]; keep_ids = keep_ids[keep_ids != 0]
    pred = np.isin(labels, keep_ids).astype(np.uint8)
    
    np.save(f"{save_dir}/{sid}.npy",pred)
    return pred


def resize_volume_3d(volume,target_shape=(64,448,448)):
    target_depth, target_height, target_width = target_shape[0], target_shape[1],target_shape[2]
    current_shape = volume.shape
    target_shape = (target_depth, target_height, target_width)
    if current_shape == target_shape:
        return volume
    zoom_factors = [
        target_shape[i] / current_shape[i] for i in range(3)
    ]
    resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
    resized_volume = resized_volume[:target_depth, :target_height, :target_width]
    pad_width = [
        (0, max(0, target_depth - resized_volume.shape[0])),
        (0, max(0, target_height - resized_volume.shape[1])),
        (0, max(0, target_width - resized_volume.shape[2]))
    ]
    if any(pw[1] > 0 for pw in pad_width):
        resized_volume = np.pad(resized_volume, pad_width, mode='edge')
    return resized_volume.astype(np.uint8)



def main(args):

    base_dir = args.base_dir
    save_dir = args.save_dir
    device = args.device
    model_pth = args.model_pth
    target_shape = tuple(args.target_shape)
    pred_resize_dir = args.pred_resize_dir
    axis_csv = args.axis_csv

    print(f"Base directory     : {base_dir}")
    print(f"Save directory     : {save_dir}")
    print(f"Device             : {device}")
    print(f"Model path         : {model_pth}")
    print(f"Target shape       : {target_shape}")
    print(f"Resize output dir  : {pred_resize_dir}")
    print(f"Axis CSV path      : {axis_csv}")


    axis_df=pd.read_csv(axis_csv)
    strides = [
        (1, 1, 1),   # level 0 (no downsample)
        (2, 2, 2),   # level 1
        (2, 2, 2),   # level 2
        (2, 2, 2),   # level 3
        (2, 2, 2),   # level 4
    ]
    kernel_size = [(3,3,3)] * len(strides)
    upsample_kernel_size = strides[1:]

    filters = [32, 64, 128, 256, 320]   # nnU-Net 風格；你有 96GB，這組很穩

    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        dropout=0.0,
        deep_supervision=True,   # 啟用深度監督
        deep_supr_num=3,         # 最高層之外再用 3 個低解析輸出做監督
    ).to(device)

    ckpt = torch.load(model_pth, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    ROI_SIZE    = (96,256,256) 
    val_tf = Compose([
        SpatialPadd(
            keys=["image"], 
            spatial_size=ROI_SIZE,
            method="symmetric", 
            mode="constant", 
            constant_values=0
        ),
        EnsureTyped(keys=["image"]),
        ToTensord(keys=["image"]),
    ])


    z_df=axis_df[axis_df["axis"]=="z"].reset_index(drop=True)
    #z_df=z_df.merge(df[["SeriesInstanceUID","Modality"]],how="left",on="SeriesInstanceUID")

    for sid in tqdm(z_df["SeriesInstanceUID"].values):
        # volume=get_volume(sid)
        # save_seg_pred(volume,save_dir,sid)
        
        
        try:
            volume=get_volume(base_dir,sid)
            seg_pred=save_seg_pred(volume,save_dir,sid)
            
            resized_v = resize_volume_3d(seg_pred,target_shape=target_shape)
            
            np.save(f"{pred_resize_dir}/{sid}.npy",resized_v)
        except:
            pass
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Segmentation prediction pipeline for RSNA 2025 dataset"
    )

    parser.add_argument(
        "--base_dir", type=str,
        default="data/series",
        help="Path to the folder containing DICOM or NIfTI series for prediction."
    )

    parser.add_argument(
        "--save_dir", type=str,
        default="output/seg_pred",
        help="Directory to save raw segmentation prediction results."
    )

    parser.add_argument(
        "--device", type=str,
        default="cuda:0",
        help="Computation device to run inference (e.g., 'cuda:0', 'cuda:1', or 'cpu')."
    )

    parser.add_argument(
        "--model_pth", type=str,
        default="model/seg/job13_fold1_best.pt",
        help="Path to the trained segmentation model checkpoint."
    )

    parser.add_argument(
        "--target_shape", type=int, nargs=3,
        default=[64, 448, 448],
        help="Target 3D shape (D, H, W) for model input, typically after resampling."
    )

    parser.add_argument(
        "--pred_resize_dir", type=str,
        default="output/seg_pred_448_s64",
        help="Directory to save resized prediction results."
    )

    parser.add_argument(
        "--axis_csv", type=str,
        default="output/axis_df.csv",
        help="CSV file containing orientation and axis information."
    )

    args = parser.parse_args()
    main(args)