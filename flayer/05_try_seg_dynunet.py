import argparse

import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from torch import amp

from monai.transforms import (
    Compose, MapTransform, EnsureTyped, ToTensord,
    RandFlipd, RandRotate90d, RandGaussianNoised,
    RandCropByPosNegLabeld,SpatialPadd,RandAffined
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet, DynUNet
from monai.losses import DiceFocalLoss, DeepSupervisionLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils import set_determinism




class LoadNPYd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = np.load(d[k])
            if arr.ndim == 3:
                arr = arr[None, ...]  # -> [1,H,W,D]
            if k == "label":
                arr = (arr > 0.5).astype(np.float32)  # np.unit8
            else:
                arr = arr.astype(np.float32)
            d[k] = arr
        return d

# ---------- 這個 helper 把 df 轉成 MONAI 需要的 list[dict] ----------
def df_to_monai_list(df, img_col="img_path", mask_col="mask_path"):
    df = df.copy()
    df = df.rename(columns={img_col: "image", mask_col: "label"})
    # 只保留有標註的列（若你有無標註資料，另做測試集）
    df = df.dropna(subset=["image", "label"])
    return df[["image", "label"]].to_dict(orient="records")



# ====== Validate (sliding-window 全圖) ======





@torch.no_grad()
def validate(model,val_loader, device, ROI_SIZE, SW_BATCH, THRESH_LIST):
    model.eval()
    meters = {t: DiceMetric(include_background=False, reduction="mean") for t in THRESH_LIST}
    has_fg = 0
    
    def predictor_highest(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out
    
    for batch in val_loader:
        img, msk = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        
        

        logits = sliding_window_inference(
            img, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH,
            predictor=predictor_highest, overlap=0.5, mode="gaussian"
        )
        prob = torch.sigmoid(logits)

        if msk.sum().item() == 0: continue
        has_fg += 1
        for t,m in meters.items():
            m(y_pred=(prob>t).float(), y=msk)

    if has_fg==0: return {"best_t": None, "dice_by_t": {t:0.0 for t in THRESH_LIST}}
    dice_by_t = {t:m.aggregate().item() for t,m in meters.items()}
    best_t = max(dice_by_t, key=dice_by_t.get)
    return {"best_t": best_t, "dice_by_t": dice_by_t}




def main(args):


    device = args.device
    data_root = args.data_root
    out_dir = args.out_dir
    train_csv = args.train_csv

    print(f"Device      : {device}")
    print(f"Data root   : {data_root}")
    print(f"Output dir  : {out_dir}")
    print(f"Train CSV   : {train_csv}")

    is_amp = True
    IMG_DIR     = f"{data_root}/img"
    MASK_DIR    = f"{data_root}/mask_b"
    os.makedirs(out_dir, exist_ok=True)
    npy_paths=glob.glob(f"{IMG_DIR}/*.npy")

    ROI_SIZE    = (96,256,256)   # ()若還有餘裕，可試 (224, 256, 256) 或 (224, 288, 288)
    BATCH_SIZE  = 6                 # 96G 通常可到 4~6；不夠就降到 3/2
    NUM_WORKERS = 16
    MAX_EPOCHS  = 600 #450
    LR          = 5e-4

    # 驗證時 sliding window 批量
    SW_BATCH    = 2                  # 96G 可設 2~4；推論只有 16G 時請改為 1
    SW_ROI      = ROI_SIZE           # 也可設更大 (224,256,256) 看顯存調整
    SW_OVERLAP  = 0.5

    fold_df=pd.read_csv(train_csv)
    fold_df = fold_df[fold_df["axis"]=="z"].reset_index(drop=True)

    img_paths=glob.glob(f"{data_root}/img/*.npy")
    sid_list=[Path(p).stem for p in img_paths]

    df=pd.DataFrame({"SeriesInstanceUID":sid_list})
    df = df.merge(fold_df[["SeriesInstanceUID","Modality","fold"]],on="SeriesInstanceUID",how="left")

    df["img_path"] = df["SeriesInstanceUID"].apply(
        lambda x: f"{data_root}/img/{x}.npy"
    )
    df["mask_path"] = df["SeriesInstanceUID"].apply(
        lambda x: f"{data_root}/mask_b/{x}.npy"
    )


    fold=1 #0
    train_df=df[df["fold"]!=fold].reset_index(drop=True)
    valid_df=df[df["fold"]==fold].reset_index(drop=True)

    train_files = df_to_monai_list(train_df, "img_path", "mask_path")
    val_files   = df_to_monai_list(valid_df,   "img_path", "mask_path")

    probe = np.load(train_files[0]["image"])
    in_channels = probe.shape[0] if probe.ndim == 4 else 1


    # ====== Transforms ======
    train_tf = Compose([
        LoadNPYd(keys=["image", "label"]),
        SpatialPadd(keys=["image","label"], spatial_size=ROI_SIZE,method="symmetric", mode="constant", constant_values=0),
        RandCropByPosNegLabeld(
            keys=["image","label"], label_key="label",
            spatial_size=ROI_SIZE, pos=4.0, neg=0.25, num_samples=2,
            image_key="image", image_threshold=0.0
        ),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[2]),
        
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),   # 弧度制 ≈ 5.7 度
            scale_range=(0.05, 0.05, 0.05), # 輕微縮放 ±5%
            translate_range=(10, 10, 10),   # 平移最多 10 voxel
            mode=("bilinear", "nearest")
        ),
        
        
        RandRotate90d(keys=["image","label"], prob=0.5, spatial_axes=(1,2)),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=["image","label"]),
        ToTensord(keys=["image","label"]),
    ])

    val_tf = Compose([
        LoadNPYd(keys=["image", "label"]),
        SpatialPadd(keys=["image","label"], spatial_size=ROI_SIZE,method="symmetric", mode="constant", constant_values=0),
        EnsureTyped(keys=["image","label"]),
        ToTensord(keys=["image","label"]),
    ])

    # ====== Datasets / Loaders ======
    train_ds = Dataset(train_files, transform=train_tf)
    val_ds   = Dataset(val_files,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

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
        in_channels=in_channels,
        out_channels=1,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        dropout=0.0,
        deep_supervision=True,   # 啟用深度監督
        deep_supr_num=3,         # 最高層之外再用 3 個低解析輸出做監督
    ).to(device)


    base_loss = DiceFocalLoss(
        sigmoid=True,
        lambda_dice=0.7,      # 初期偏 Dice 更穩；之後可回 0.7/0.3
        lambda_focal=0.3,
        gamma=2.0,
        smooth_nr=1e-5, smooth_dr=1e-5,
        include_background=False,
    )

    # 權重給「高解析 > 低解析」，exp 會自動衰減（也可傳自訂 weights list）
    loss_fn = DeepSupervisionLoss(base_loss, weight_mode="exp")

    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scaler = amp.GradScaler(enabled = is_amp)
    dice_meter= DiceMetric(include_background=False, reduction="mean")

    THRESH_LIST = [0.1, 0.3, 0.5]

    # ====== Train Loop ======

    from torch.optim.lr_scheduler import CosineAnnealingLR

    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-6)



    best_score = 0.0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            msk = batch["label"].to(device, non_blocking=True)
            #print(msk.unique())
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=is_amp):
                out = model(img)  # DynUNet deep supervision 輸出
        
                if isinstance(out, torch.Tensor) and out.dim() == 6:
                    # 拆成 list: [ [B,1,H,W,D], ...共4個]
                    logits_list = [out[:, i, ...] for i in range(out.shape[1])]
                elif isinstance(out, (list, tuple)):
                    logits_list = list(out)
                else:
                    logits_list = [out]

                # 確保 label 是 float32 且 shape [B,1,H,W,D]
                msk = msk.float()
                loss = loss_fn(logits_list, msk)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * img.size(0)

        if scheduler: scheduler.step()

        tr_loss = running / len(train_loader.dataset)
        val_stats = validate(model,val_loader, device, ROI_SIZE, SW_BATCH, THRESH_LIST)  # ← 每次調用都會各自 reset meters
        
        line = " ".join([f"{t:.1f}:{val_stats['dice_by_t'][t]:.4f}" for t in THRESH_LIST])
        print(f"Epoch {ep:03d} | lr={opt.param_groups[0]['lr']:.2e} | train_loss={tr_loss:.4f} | val_dice@ {line} | best_t={val_stats['best_t']}")

        # 用最佳閾值的 Dice 當 early-select 指標
        if val_stats["best_t"] is not None:
            cur_best = val_stats["dice_by_t"][val_stats["best_t"]]
            if cur_best > best_score:
                best_score = cur_best
                torch.save(model.state_dict(), os.path.join(out_dir, "dynunet_model_best.pth"))
                
    print(best_score)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentation training pipeline for RSNA 2025"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Computation device for training (e.g., 'cuda:0', 'cuda:1', or 'cpu')."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="output/spacing_07_p1_p99_v1",
        help="Root directory containing preprocessed training data (NIfTI/NumPy format)."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="model/seg",
        help="Directory to save model checkpoints, logs, and training outputs."
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default="output/train_with_folds_optimized_axis_v1.csv",
        help="Path to the CSV file containing training metadata and fold splits."
    )

    args = parser.parse_args()
    main(args)