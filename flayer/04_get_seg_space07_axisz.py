
import argparse
import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage, EnsureChannelFirst, Orientation, Spacing, ScaleIntensityRange,ScaleIntensityRangePercentiles
)
import pydicom
from tqdm.auto import tqdm


from monai.transforms import (
    Compose, EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRangePercentiles
)
from monai.data import NibabelReader
import numpy as np
import torch
from typing import Tuple, Optional, Dict

# ---- 共同工具 ----
def _to_DHW_rot90(t: torch.Tensor, k: int = 1):
    arr = t.numpy() if isinstance(t, torch.Tensor) else t
    arr = np.squeeze(arr, axis=0)        # [H,W,D]
    arr = np.transpose(arr, (2, 0, 1))   # -> [D,H,W]
    arr = np.rot90(arr, k=k, axes=(1, 2))
    return arr

def _clean_mask_labels(msk: np.ndarray) -> np.ndarray:
    msk = (msk > 0).astype(np.uint8)
    return msk.astype(np.uint8)


# ---- 主函式 ----
def process_mri_case(
    seg_folder: str,
    sid: str,
    modality: str = "mri",   # "mri" 或 "ct"
    img_path_tpl: str = "{seg_folder}/{sid}.nii",  # 影像 NIfTI 路徑樣板
    msk_path_tpl: Optional[str] = "{seg_folder}/{sid}_cowseg.nii",# 標註 NIfTI 路徑樣板
    rot_k: int = 1,
    return_meta: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
    """
    回傳:
      img_arr: (D,H,W) float32, 已做強度標準化
      msk_arr: (D,H,W) uint8, 已將 >13 壓為 1；若無 mask，回傳 None
      info:    可選，包含 affine 等資訊
    """
    loader = LoadImage(image_only=False,reader=NibabelReader())

    # --- 影像 ---
    #print(sid)
    img_p = img_path_tpl.format(seg_folder=seg_folder,sid=sid)
    img, img_meta = loader(img_p)            # img: torch.Tensor or np
    # ---- 轉換（MRI 影像 / Mask）----
    # MRI 預處理
    _img_t_mri = Compose([
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(0.7, 0.7, 0.7), mode="bilinear"),
        # ScaleIntensityRangePercentiles(
        #     lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        # ),
        ScaleIntensityRangePercentiles(
            lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
        ),
    ])

    # CT 預處理
    _img_t_ct = Compose([
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(0.7, 0.7, 0.7), mode="bilinear"),
        # ScaleIntensityRange(
        #     a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True
        # ),
        ScaleIntensityRangePercentiles(
            lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
        ),
    ])

    _msk_t = Compose([
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(0.7, 0.7, 0.7), mode="nearest"),
    ])



    
    if modality.lower() == "ct":
        img = _img_t_ct(img)
    else:  # 預設 mri
        img = _img_t_mri(img)
        
    img_arr = _to_DHW_rot90(img, k=rot_k).astype(np.float32)
    # 保險處理 NaN/Inf
    img_arr = np.nan_to_num(img_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 標註（可選） ---
    msk_arr = None
    if msk_path_tpl is not None:
        msk_p = msk_path_tpl.format(seg_folder=seg_folder,sid=sid)
        try:
            msk, msk_meta = loader(msk_p)
            msk = _msk_t(msk)
            msk_arr = _to_DHW_rot90(msk, k=rot_k) 
            msk_arr_b = _clean_mask_labels(msk_arr)  # >13 → 1，且轉 uint8
        except FileNotFoundError:
            msk_meta = None
            msk_arr = None
            msk_arr_b = None

    if not return_meta:
        return img_arr, msk_arr_b, msk_arr, None

    info = {
        "sid": sid,
        "img_path": img_p,
        "msk_path": (msk_path_tpl.format(seg_folder=seg_folder,sid=sid) if msk_path_tpl else None),
        "img_affine": img_meta.get("affine", None) if isinstance(img_meta, dict) else None,
        "img_original_affine": img_meta.get("original_affine", None) if isinstance(img_meta, dict) else None,
        "msk_affine": (msk_meta.get("affine", None) if (msk_arr is not None and isinstance(msk_meta, dict)) else None),
        "msk_original_affine": (msk_meta.get("original_affine", None) if (msk_arr is not None and isinstance(msk_meta, dict)) else None),
        "rot_k": rot_k,
        "spacing_mm": (0.7, 0.7, 0.7),
        "orientation": "RAS",
    }
    return img_arr, msk_arr_b,msk_arr, info

    
def main(args):
        
    seg_folder = args.seg_folder
    train_csv = args.train_csv
    out_folder = args.out_folder

    df = pd.read_csv(train_csv)

    df=df[df["axis"]=="z"].reset_index(drop=True)

    files = glob.glob(f"{seg_folder}/*.nii")
    files = [f for f in files if not f.endswith("cowseg.nii")]
    nii_sid = [Path(f).stem for f in files]  
    print(len(nii_sid))
    nii_sid = [sid for sid in nii_sid if sid in df["SeriesInstanceUID"].unique()]
    print(len(nii_sid))

    ct_nii_sids=[sid for sid in nii_sid if sid in df[df["Modality"]=="CTA"]["SeriesInstanceUID"].unique()]
    mri_nii_sids=[sid for sid in nii_sid if sid in df[df["Modality"]!="CTA"]["SeriesInstanceUID"].unique()]
    
    os.makedirs(out_folder,exist_ok=True)
    os.makedirs(f"{out_folder}/img",exist_ok=True)
    os.makedirs(f"{out_folder}/mask_b",exist_ok=True)

    for sid in tqdm(ct_nii_sids):
        img_arr,msk_arr_b,msk_arr,_=process_mri_case(seg_folder,sid,"ct")
        np.save(f"{out_folder}/img/{sid}.npy",img_arr)
        np.save(f"{out_folder}/mask_b/{sid}.npy",msk_arr_b)
        
    for sid in tqdm(mri_nii_sids):
        img_arr,msk_arr_b,msk_arr,_=process_mri_case(seg_folder,sid,"mri")
        np.save(f"{out_folder}/img/{sid}.npy",img_arr)
        np.save(f"{out_folder}/mask_b/{sid}.npy",msk_arr_b)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="RSNA 2025 preprocessing pipeline: generate standardized spacing volumes."
    )
    parser.add_argument(
        "--seg_folder",
        type=str,
        default="data/segmentations",
        help="Path to the folder containing segmentation masks."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="output/train_with_folds_optimized_axis_v1.csv",
        help="Path to the CSV file containing training data and fold assignments."
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="output/spacing_07_p1_p99_v1",
        help="Output folder to save the preprocessed data with spacing=0.7mm (default)."
    )
    
    args = parser.parse_args()
    main(args)
    
    