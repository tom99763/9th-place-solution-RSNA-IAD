import argparse
import math
import os
import random
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
#import torchio as tio

try:
    import monai  # noqa: F401
except ImportError as exc:
    raise SystemExit("Please install MONAI: pip install monai") from exc

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]
VESSEL_LABELS = LABEL_COLS[:-1]
PRESENCE_LABEL = LABEL_COLS[-1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class TrainConfig:
    npy_dir: str = "./npy"
    seg_dir: str = "/ssd3/IAD/atom1231/input/seg_pred_448_s64"
    train_csv: str = "/ssd3/IAD/atom1231/input/train_with_folds_optimized_axis_v1.csv"
    localizer_csv: str = "/ssd3/IAD/atom1231/input/train_locale_fnum.csv"
    output_dir: str = "/ssd3/IAD/atom1231/output/outputs_heatmap"
    fold: int = 0
    num_folds: int = 5
    seed: int = 42
    batch_size: int = 2
    accumulate_steps: int = 2
    val_batch_size: int = 2
    num_workers: int = 4
    epochs: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    amp: bool = True
    grad_clip: Optional[float] = 1.0
    heatmap_loss_weight: float = 1.0
    offset_loss_weight: float = 1.0
    cls_loss_weight: float = 0.5
    aux_loss_weight: float = 0.5
    gaussian_sigma: float = 1.5
    default_depth_factor: float = 0.5
    save_best: bool = True
    resume: Optional[str] = None
    checkpoint_every: int = 0
    log_interval: int = 20

    # model
    in_channels: int = 1
    base_channels: int = 32
    device: str = "cuda:2"
    encoder_name: str = "r3d_18"
    pretrained_encoder: bool = True
    freeze_encoder: bool = False
    output_stride_depth: int = 1
    output_stride_height: int = 32
    output_stride_width: int = 32
    scheduler: str = "none"
    cosine_tmax: int = 0
    cosine_min_lr: float = 1e-6
    step_lr_step_size: int = 1
    step_lr_gamma: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_min_lr: float = 1e-6
    plateau_mode: str = "min"
    scheduler_monitor: str = "val_loss"


def load_metadata(cfg: TrainConfig) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, object]]]]:
    df = pd.read_csv(cfg.train_csv)
    if 'fold' not in df.columns:
        raise KeyError("train CSV must contain a 'fold' column")
    df = df.set_index('SeriesInstanceUID')
    locale = pd.read_csv(cfg.localizer_csv)
    required_cols = {
        'SeriesInstanceUID',
        'SOPInstanceUID',
        'label_x',
        'label_y',
        'label_frame_num',
        'orig_height',
        'orig_width',
        'orig_depth',
        'location',
    }
    missing = required_cols - set(locale.columns)
    if missing:
        raise KeyError(f"Missing columns in localizer CSV: {sorted(missing)}")

    locator_map: Dict[str, List[Dict[str, object]]] = {}
    for _, row in locale.iterrows():
        series_id = str(row['SeriesInstanceUID'])
        location = str(row['location']).strip()
        if location not in LABEL_COLS:
            continue
        label_x = row['label_x']
        label_y = row['label_y']
        label_frame = row['label_frame_num']
        orig_h = row['orig_height']
        orig_w = row['orig_width']
        orig_d = row['orig_depth']
        if any(pd.isna(v) for v in (label_x, label_y, label_frame, orig_h, orig_w, orig_d)):
            continue
        entry = {
            'location': location,
            'label_x': float(label_x),
            'label_y': float(label_y),
            'label_frame': float(label_frame),
            'orig_height': float(orig_h),
            'orig_width': float(orig_w),
            'orig_depth': float(orig_d),
        }
        locator_map.setdefault(series_id, []).append(entry)
    return df, locator_map


def draw_gaussian_3d(heatmap: np.ndarray, center: Tuple[float, float, float], sigma: float) -> None:
    cz, cy, cx = center
    depth, height, width = heatmap.shape
    radius = max(int(round(sigma * 3)), 1)
    z_min = max(int(cz) - radius, 0)
    z_max = min(int(cz) + radius + 1, depth)
    y_min = max(int(cy) - radius, 0)
    y_max = min(int(cy) + radius + 1, height)
    x_min = max(int(cx) - radius, 0)
    x_max = min(int(cx) + radius + 1, width)
    if z_min >= z_max or y_min >= y_max or x_min >= x_max:
        return
    zz = np.arange(z_min, z_max, dtype=np.float32)
    yy = np.arange(y_min, y_max, dtype=np.float32)
    xx = np.arange(x_min, x_max, dtype=np.float32)
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing='ij')
    gaussian = np.exp(-((Z - cz) ** 2 + (Y - cy) ** 2 + (X - cx) ** 2) / (2 * sigma ** 2))
    patch = heatmap[z_min:z_max, y_min:y_max, x_min:x_max]
    np.maximum(patch, gaussian, out=patch)
    heatmap[z_min:z_max, y_min:y_max, x_min:x_max] = patch


def build_targets(
    lesions: List[Dict[str, object]],
    volume_shape: Tuple[int, int, int],
    out_shape: Tuple[int, int, int],
    stride_depth: int,
    stride_height: int,
    stride_width: int,
    num_classes: int,
    sigma: float,
    default_depth_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    heatmap = np.zeros((num_classes, *out_shape), dtype=np.float32)
    offset = np.zeros((3, *out_shape), dtype=np.float32)
    offset_mask = np.zeros((1, *out_shape), dtype=np.float32)
    center_index: List[Tuple[int, int, int, int]] = []
    depth, height, width = volume_shape
    out_depth, out_height, out_width = out_shape
    for lesion in lesions:
        location = lesion['location']
        if location not in VESSEL_LABELS:
            continue
        class_idx = VESSEL_LABELS.index(location)
        orig_w = lesion['orig_width']
        orig_h = lesion['orig_height']
        orig_d = lesion['orig_depth']
        scale_x = width / max(orig_w, 1.0)
        scale_y = height / max(orig_h, 1.0)
        scale_z = depth / max(orig_d, 1.0)
        x = float(lesion['label_x']) * scale_x
        y = float(lesion['label_y']) * scale_y
        z = float(lesion['label_frame']) * scale_z
        z = np.clip(z, 0.0, depth - 1.0)
        y = np.clip(y, 0.0, height - 1.0)
        x = np.clip(x, 0.0, width - 1.0)
        z_scaled = z / stride_depth
        y_scaled = y / stride_height
        x_scaled = x / stride_width
        draw_gaussian_3d(heatmap[class_idx], (z_scaled, y_scaled, x_scaled), sigma)
        center_z = min(int(round(z_scaled)), out_depth - 1)
        center_y = min(int(round(y_scaled)), out_height - 1)
        center_x = min(int(round(x_scaled)), out_width - 1)
        sub = (float(z_scaled) - center_z, float(y_scaled) - center_y, float(x_scaled) - center_x)
        offset[:, center_z, center_y, center_x] = np.asarray(sub, dtype=np.float32)
        offset_mask[0, center_z, center_y, center_x] = 1.0
        center_index.append((class_idx, center_z, center_y, center_x))
        heatmap[class_idx, center_z, center_y, center_x] = 1.0
    return heatmap, offset, offset_mask, center_index


class RSNACenterNetDataset(Dataset):
    def __init__(
        self,
        ids: List[str],
        meta: pd.DataFrame,
        locator_map: Dict[str, List[Dict[str, object]]],
        cfg: TrainConfig,
        is_train: bool,
    ) -> None:
        self.ids = ids
        self.meta = meta
        self.locator_map = locator_map
        self.cfg = cfg
        self.is_train = is_train
        self.labels = meta.loc[ids, LABEL_COLS].astype(np.float32).values
        self.affine_prob = 0.5
        # self.rotate_range = (-15.0, 15.0) #(-10.0, 10.0)
        self.scale_range = (0.8, 1.2)#(0.9, 1.1)
        # self.translate_frac = (-0.1, 0.1)#(-0.05, 0.05)
        self.rotate_range = (-10.0, 10.0)
        #self.scale_range = (0.9, 1.1)
        self.translate_frac = (-0.05, 0.05)        
        self.flip_prob = 0.0
        #self.transform = A.Compose([A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE), A.Normalize(), ToTensorV2()])
        self.transform = A.Compose([A.Normalize(), ToTensorV2()])
        #self.mask_transform = A.Compose([ToTensorV2()])

        # 可選：是否對 NaN/Inf 做處理與是否裁切極端值
        self.handle_invalid = True
        self.clip_percentile = None  # 例如 (0.5, 99.5) 可啟用分位裁切；None 表

        #for 3d torio normalize
        # 維持原本結構與順序：Resize -> Normalize -> ToTensorV2
        # 將 Normalize 設為灰階 no-op，避免 TorchIO 做完 3D Normalize 後又被改動
        # self.transform = A.Compose([
        #     #A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        #     A.Normalize(mean=(0.0,), std=(1.0,)),  # no-op for 1-channel
        #     ToTensorV2()
        # ])

        # # TorchIO：對 3D 體積做符合語意的 Z-score Normalize（整體均值0、方差1）
        # self.tio_norm = tio.Compose([
        #     tio.ZNormalization()  # 可視需求改成 RescaleIntensity
        # ])

    # def _tio_normalize(self, volume_np: np.ndarray) -> np.ndarray:
    #     """
    #     volume_np: (D, H, W) float32/float64/… -> 回傳 (D, H, W) float32
    #     """
    #     # TorchIO 需要張量 shape: (C, D, H, W)
    #     tensor = torch.from_numpy(volume_np.astype(np.float32))[None, ...]  # (1, D, H, W)
    #     img = tio.ScalarImage(tensor=tensor)
    #     img = self.tio_norm(img)
    #     norm_vol = img.data[0].cpu().numpy()  # (D, H, W)
    #     return norm_vol

    # ---- 純 NumPy/PyTorch 3D Normalize（等價 ZNormalization） ----
    def _zscore_normalize_volume(self, volume_np: np.ndarray) -> np.ndarray:
        """
        針對整個 3D 體積 (D, H, W) 做 Z-score normalize：
        x' = (x - mean) / std
        """
        v = volume_np.astype(np.float32, copy=False)

        if self.handle_invalid:
            # 將 NaN/Inf 先轉成有限值（以中位數替代）
            finite_mask = np.isfinite(v)
            if not finite_mask.all():
                med = np.nanmedian(v[finite_mask]) if finite_mask.any() else 0.0
                v = np.where(finite_mask, v, med).astype(np.float32)

        if self.clip_percentile is not None:
            lo, hi = self.clip_percentile
            lo_val, hi_val = np.percentile(v, [lo, hi])
            v = np.clip(v, lo_val, hi_val)

        mean = float(v.mean())
        std = float(v.std())

        # 避免 std 為 0
        v = (v - mean) / (std + 1e-8)
        return v

    def _apply_albu_per_slice(self, volume_np: np.ndarray) -> torch.Tensor:
        """
        對每個 slice 做 2D Resize + (no-op) Normalize + ToTensorV2
        輸出 shape: (1, D, H', W') 方便 3D 模型食用
        """
        D, H, W = volume_np.shape
        slices = []

        for d in range(D):
            img2d = volume_np[d]  # (H, W), float32
            # Albumentations 對灰階建議傳入 (H, W, 1)
            transformed = self.transform(image=img2d[..., None])
            # transformed['image']: torch.Tensor, shape (1, H', W')
            slices.append(transformed['image'])

        vol_tensor = torch.stack(slices, dim=0)       # (D, 1, H', W')
        vol_tensor = vol_tensor.permute(1, 0, 2, 3)   # (1, D, H', W')
        return vol_tensor


    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series_id = self.ids[idx]
        volume_path = os.path.join(self.cfg.npy_dir, f"{series_id}.npy")
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"Volume not found: {volume_path}")
        volume = np.load(volume_path)
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]
        elif volume.ndim == 4:
            if volume.shape[0] != self.cfg.in_channels:
                volume = volume.transpose(3, 0, 1, 2)
        else:
            raise ValueError(f"Unexpected volume shape {volume.shape} for {series_id}")
        volume = volume.astype(np.float32)
        
        seg_pth = os.path.join(self.cfg.seg_dir, f"{series_id}.npy")
        
        if not os.path.exists(seg_pth):
            seg_img = np.zeros(volume.shape)
        else:
            seg_img = np.load(seg_pth)
            seg_img = seg_img[np.newaxis, ...]
        
        
        depth, height, width = volume.shape[1:]
        lesions = self.locator_map.get(series_id, [])
        lesions = [dict(les) for les in lesions]
        if lesions:
            for lesion in lesions:
                orig_w = max(float(lesion['orig_width']), 1.0)
                orig_h = max(float(lesion['orig_height']), 1.0)
                orig_d = max(float(lesion['orig_depth']), 1.0)
                lesion['label_x'] = float(lesion['label_x']) * width / orig_w
                lesion['label_y'] = float(lesion['label_y']) * height / orig_h
                lesion['label_frame'] = float(lesion['label_frame']) * depth / orig_d
                lesion['orig_width'] = float(width)
                lesion['orig_height'] = float(height)
                lesion['orig_depth'] = float(depth)

        if self.is_train:
            volume, seg_img, affine_mat = self.apply_affine(volume,seg_img)
            if affine_mat is not None and lesions:
                for lesion in lesions:
                    x = float(lesion['label_x'])
                    y = float(lesion['label_y'])
                    new_x = affine_mat[0, 0] * x + affine_mat[0, 1] * y + affine_mat[0, 2]
                    new_y = affine_mat[1, 0] * x + affine_mat[1, 1] * y + affine_mat[1, 2]
                    lesion['label_x'] = np.clip(new_x, 0, width - 1)
                    lesion['label_y'] = np.clip(new_y, 0, height - 1)
            # if np.random.rand() < self.flip_prob:
            #     volume = volume[..., ::-1]
            #     width = volume.shape[-1]
            #     if lesions:
            #         for lesion in lesions:
            #             lesion['label_x'] = width - 1 - float(lesion['label_x'])
            if np.random.rand() < self.flip_prob:
                volume = volume[..., ::-1, :]
                height = volume.shape[-2]
                if lesions:
                    for lesion in lesions:
                        lesion['label_y'] = height - 1 - float(lesion['label_y'])
        out_shape = (
            max(1, depth // self.cfg.output_stride_depth),
            max(1, height // self.cfg.output_stride_height),
            max(1, width // self.cfg.output_stride_width),
        )
        #print("hihi")
        #print(volume.shape)  # (1, 32, 448, 448)

        # # Albumentations 逐切片轉 tensor
        # slices = []
        # for z in range(depth):
        #     slice_img = volume[:, z]
        #     slice_img = slice_img.transpose(1, 2, 0)  # (H,W,C)
        #     transformed = self.transform(image=slice_img)
        #     tensor_slice = transformed['image']  # (C, H, W)
        #     slices.append(tensor_slice)
        # volume_tensor = torch.stack(slices, dim=1)  # (C, D, H, W)
        # print(volume_tensor.shape)  # torch.Size([1, 32, 448, 448])
        #raise

        #3d ok1
        # volume shape: (D, H, W)
        
        volume = volume.squeeze(0)  # (D, H, W)
        image = volume.transpose(1, 2, 0)
        #print(image.shape)
        transformed = self.transform(image=image)
        tensor = transformed['image']  # (D, H, W)
        volume_tensor = tensor.unsqueeze(0)  # (d, 1, H, W)
        # print(volume_tensor.shape)  # torch.Size([1, 32, 448, 448])
        # # raise

        # #3d+2d
        # volume = volume.squeeze(0) 
        # # 1) 先做 TorchIO 3D Normalize（語意正確）
        # #volume = self._tio_normalize(volume)   # (D, H, W) float32
        # volume = self._zscore_normalize_volume(volume)  # (D, H, W) float32
        # #print(volume.shape)  # (32, 448, 448)
        # # 2) 再用原本的 Albumentations pipeline（逐 slice 處理）
        # volume_tensor = self._apply_albu_per_slice(volume)  # (1, D, H', W')
        # #print(volume_tensor.shape)  # torch.Size([1, 32, 448, 448])

        heatmap, offset, offset_mask, center_index = build_targets(
            lesions=lesions,
            volume_shape=(depth, height, width),
            out_shape=out_shape,
            stride_depth=self.cfg.output_stride_depth,
            stride_height=self.cfg.output_stride_height,
            stride_width=self.cfg.output_stride_width,
            num_classes=len(VESSEL_LABELS),
            sigma=self.cfg.gaussian_sigma,
            default_depth_factor=self.cfg.default_depth_factor,
        )
        sample = {
            'id': series_id,
            'image': volume_tensor,
            'heatmap': torch.from_numpy(heatmap),
            'offset': torch.from_numpy(offset),
            'offset_mask': torch.from_numpy(offset_mask),
            'seg_mask': torch.from_numpy(seg_img).long(),
            'labels': torch.from_numpy(self.labels[idx]),
        }
        return sample

    def apply_affine(self, volume: np.ndarray,mask: Optional[np.ndarray]) -> Tuple[np.ndarray,Optional[np.ndarray], Optional[np.ndarray]]:
        if np.random.rand() >= self.affine_prob:
            return volume, mask,None
        channels, depth, height, width = volume.shape
        angle = np.random.uniform(*self.rotate_range)
        scale = np.random.uniform(*self.scale_range)
        tx = np.random.uniform(*self.translate_frac) * width
        ty = np.random.uniform(*self.translate_frac) * height
        center = (width / 2.0, height / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        warped = np.empty_like(volume)
        for z in range(depth):
            for c in range(channels):
                warped[c, z] = cv2.warpAffine(
                    volume[c, z],
                    mat,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                    )
        warped_mask = None
        if mask is not None:
            #print(mask.shape)
            # 將 mask 視為 (M, D, H, W) 統一處理（M=K 或 1 或 1 from (D,H,W)）
            if mask.ndim == 3:
                m = mask[None, ...]             # (1, D, H, W)
            elif mask.ndim == 4:
                m = mask
            else:
                raise ValueError("mask must be (D,H,W) or (M,D,H,W)")

            M = m.shape[0]
            # 保存 dtype（如整數/布林）
            mask_dtype = m.dtype
            warped_m = np.empty_like(m)
            for z in range(depth):
                for k in range(M):
                    warped_m[k, z] = cv2.warpAffine(
                        m[k, z],
                        mat,
                        (width, height),
                        flags=cv2.INTER_NEAREST,           # 最重要：避免類別值被插值
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
            warped_mask = warped_m
        else:
            warped_mask = mask
                
        return warped,warped_mask, mat


class CenterNet3D(nn.Module):
    def __init__(self, cfg: TrainConfig, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        # self.backbone = timm.create_model(
        #     cfg.encoder_name,
        #     pretrained=cfg.pretrained_encoder,
        #     features_only=True,
        #     out_indices=(-1,),
        # )

        self.backbone = timm.create_model(
            cfg.encoder_name,
            pretrained=cfg.pretrained_encoder,
            features_only=True,
            out_indices=(1,-2),
        )

        # self.backbone = timm.create_model(
        #     cfg.encoder_name,
        #     pretrained=cfg.pretrained_encoder,
        #     features_only=True,
        #     out_indices=(-1,),
        #     in_chans=32
        # )


        if cfg.freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        info = self.backbone.feature_info
        self.feature_channels = info.channels()[-1]
        self.spatial_reduction = info.reduction()[-1]
        self.encoder_in_channels = getattr(self.backbone, 'in_chans', None)
        #print(self.encoder_in_channels) #none
        #print("hihhihihihihi")
        if self.encoder_in_channels is None:
            default_input = self.backbone.default_cfg.get('input_size', (3))
            #print(default_input) # (3, 300, 300)
            #raise
            self.encoder_in_channels = default_input[0] if isinstance(default_input, (list, tuple)) else int(default_input)


        c_aux  = self.backbone.feature_info.channels()[0]
        self.aux_head2d = nn.Sequential(
            nn.Conv2d(c_aux, cfg.base_channels, 3, padding=1),
            nn.BatchNorm2d(cfg.base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.base_channels, 1, 1)
        )
        
        head_channels = cfg.base_channels
        self.temporal_head = nn.Sequential(
            nn.Conv3d(self.feature_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv3d(head_channels, num_classes, kernel_size=1)
        self.offset_head = nn.Conv3d(head_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, d, h, w = x.shape
        #print(x.shape)  # torch.Size([4, 1, 32, 448, 448])
        x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        #x = x.permute(0, 2, 1, 3, 4).reshape(b * c, d, h, w) #ytt try
        if x.shape[1] != self.encoder_in_channels:
            if x.shape[1] == 1 and self.encoder_in_channels == 3:
                x = x.repeat(1, 3, 1, 1)
            else:
                raise ValueError(
                    f"Input has {x.shape[1]} channels but encoder expects {self.encoder_in_channels}"
                )

        #print(x.shape)  # torch.Size([128, 3, 148, 148])
        feat_aux2d, features = self.backbone(x) 
        
        aux_logits_2d = self.aux_head2d(feat_aux2d)      # (B*D,1,h',w')
        
        aux_logits_2d = F.interpolate(
            aux_logits_2d, size=(h, w), mode='bilinear', align_corners=False
        )
        aux_logits_3d = aux_logits_2d.view(b, d, 1, h, w).permute(0, 2, 1, 3, 4)  
        
        feat_c, feat_h, feat_w = features.shape[1:]
        #print(features.shape)  # torch.Size([128, 256, 14, 14])  l2 :[128, 160, 28, 28]
        #print(feat_c, feat_h, feat_w)  # 256 14 14
        features = features.view(b, d, feat_c, feat_h, feat_w).permute(0, 2, 1, 3, 4)
        #features = features.view(b, 1, feat_c, feat_h, feat_w).permute(0, 2, 1, 3, 4) #ytt try

        #print(features.shape)  # torch.Size([4, 256, 32, 14, 14])
        feat3d = self.temporal_head(features)
        #print(feat3d.shape)  # torch.Size([4, 32, 32, 14, 14])
        #raise
        heatmap = self.heatmap_head(feat3d)
        #print(heatmap.shape)  # torch.Size([4, 13, 32, 14, 14])
        offset = self.offset_head(feat3d)
        #print(offset.shape)  # torch.Size([4, 3, 32, 14, 14])
        #raise
        return {'heatmap': heatmap, 'offset': offset,'aux': aux_logits_3d}


def centernet_focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred_sigmoid = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
    pos_inds = target.eq(1.0)
    neg_inds = target.lt(1.0)
    pos_loss = torch.log(pred_sigmoid) * torch.pow(1 - pred_sigmoid, alpha)
    neg_loss = torch.log(1 - pred_sigmoid) * torch.pow(pred_sigmoid, alpha) * torch.pow(1 - target, beta)
    num_pos = pos_inds.sum()
    pos_loss = pos_loss[pos_inds]
    neg_loss = neg_loss[neg_inds]
    loss = 0.0
    if num_pos > 0:
        loss += -pos_loss.sum() / num_pos
    else:
        loss += 0.0
    loss += -neg_loss.mean()
    return loss


def dice_bce_loss_with_logits(pred, target, smooth=1e-5):
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum()
    dice = 1 - (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
    bce = F.binary_cross_entropy_with_logits(pred, target.float())
    return bce + dice


def compute_class_logits(heatmap_logits: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = heatmap_logits.shape
    flat = heatmap_logits.view(b, c, -1)
    class_logits = flat.max(dim=2).values
    presence_logits = class_logits.max(dim=1, keepdim=True).values
    return torch.cat([class_logits, presence_logits], dim=1)


def compute_auc(targets: List[np.ndarray], preds: List[np.ndarray]) -> Dict[str, float]:
    if not preds:
        return {name: float('nan') for name in LABEL_COLS}
    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    scores: Dict[str, float] = {}
    for idx, name in enumerate(LABEL_COLS):
        try:
            score = roc_auc_score(y_true[:, idx], y_pred[:, idx])
        except ValueError:
            score = float('nan')
        scores[name] = score
    valid_scores = [s for s in scores.values() if not math.isnan(s)]
    weights = [1]*13 + [13]
    total_weight = sum(weights)
    weighted_scores = []
    for w, s in zip(weights, scores.values()):
        if not math.isnan(s):
            weighted_scores.append((w/total_weight) * s)
    scores['weighted_auc'] = float(sum(weighted_scores)) if weighted_scores else float('nan')
    scores['mean_auc'] = float(np.mean(valid_scores)) if valid_scores else float('nan')
    return scores


def save_checkpoint(state: Dict[str, object], is_best: bool, cfg: TrainConfig, epoch: int, fold: int) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    epoch_path = os.path.join(
        cfg.output_dir,
        f"{cfg.encoder_name}_fold{fold}_epoch{epoch}.pth"
    )
    torch.save(state, epoch_path)
    if is_best and cfg.save_best:
        best_path = os.path.join(
            cfg.output_dir,
            f"{cfg.encoder_name}_fold{fold}_best.pth"
        )
        torch.save(state, best_path)


def train_one_epoch(
    model: CenterNet3D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler,
    cfg: TrainConfig,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        images = batch['image'].to(cfg.device, non_blocking=True)
        heatmap_target = batch['heatmap'].to(cfg.device)
        offset_target = batch['offset'].to(cfg.device)
        offset_mask = batch['offset_mask'].to(cfg.device)
        seg_mask = batch['seg_mask'].to(cfg.device)
        labels = batch['labels'].to(cfg.device)
        #optimizer.zero_grad(set_to_none=True)
        # print("train loop")
        # print(images.shape) #[2, 1, 32, 448, 448]) 2->batch size
        # print(heatmap_target.shape)#torch.Size([2, 13, 8, 112, 112])
        # print(offset_target.shape) #torch.Size([2, 3, 8, 112, 112])
        # print(offset_mask.shape) #torch.Size([2, 1, 8, 112, 112])
        # print(labels.shape)#torch.Size([2, 14])
        #raise
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            outputs = model(images)
            heat_loss = centernet_focal_loss(outputs['heatmap'], heatmap_target)
            if offset_mask.sum() > 0:
                pred_offset = outputs['offset'].permute(0, 2, 3, 4, 1)
                tgt_offset = offset_target.permute(0, 2, 3, 4, 1)
                mask = offset_mask.squeeze(1) > 0
                offset_loss = F.l1_loss(pred_offset[mask], tgt_offset[mask])
            else:
                offset_loss = torch.tensor(0.0, device=cfg.device)
                
                
            #if segqev
            class_logits = compute_class_logits(outputs['heatmap'])
            cls_loss = F.binary_cross_entropy_with_logits(class_logits, labels)
            
            if seg_mask.sum()>0: 
                aux_loss = dice_bce_loss_with_logits(outputs["aux"],seg_mask)
            else:
                aux_loss = torch.tensor(0.0, device=cfg.device)
            loss = cfg.heatmap_loss_weight * heat_loss + cfg.offset_loss_weight * offset_loss + cfg.cls_loss_weight * cls_loss + cfg.aux_loss_weight * aux_loss
        loss = loss / cfg.accumulate_steps
        scaler.scale(loss).backward()
        if (step + 1) % cfg.accumulate_steps == 0:
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) 
        total_loss += loss.item()* cfg.accumulate_steps
        if cfg.log_interval and (step + 1) % cfg.log_interval == 0:
            current_lr = optimizer.param_groups[0].get('lr', cfg.learning_rate)
            print(f"train step {step + 1}/{len(loader)} loss {loss.item():.4f} lr {current_lr:.2e}")
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_epoch(model: CenterNet3D, loader: DataLoader, cfg: TrainConfig) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    targets: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    for batch in loader:
        images = batch['image'].to(cfg.device, non_blocking=True)
        heatmap_target = batch['heatmap'].to(cfg.device)
        offset_target = batch['offset'].to(cfg.device)
        offset_mask = batch['offset_mask'].to(cfg.device)
        labels = batch['labels'].to(cfg.device)
        outputs = model(images)
        heat_loss = centernet_focal_loss(outputs['heatmap'], heatmap_target)
        if offset_mask.sum() > 0:
            pred_offset = outputs['offset'].permute(0, 2, 3, 4, 1)
            tgt_offset = offset_target.permute(0, 2, 3, 4, 1)
            mask = offset_mask.squeeze(1) > 0
            offset_loss = F.l1_loss(pred_offset[mask], tgt_offset[mask])
        else:
            offset_loss = torch.tensor(0.0, device=cfg.device)
        class_logits = compute_class_logits(outputs['heatmap'])
        cls_loss = F.binary_cross_entropy_with_logits(class_logits, labels)
        loss = cfg.heatmap_loss_weight * heat_loss + cfg.offset_loss_weight * offset_loss + cfg.cls_loss_weight * cls_loss
        total_loss += loss.item()
        probs = torch.sigmoid(class_logits).cpu().numpy()
        preds.append(probs)
        targets.append(labels.cpu().numpy())
    metrics = compute_auc(targets, preds)
    return total_loss / max(len(loader), 1), metrics


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    name = cfg.scheduler.lower()
    if name == "cosineannealinglr":
        t_max = cfg.cosine_tmax if cfg.cosine_tmax > 0 else cfg.epochs
        t_max = max(1, t_max)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=cfg.cosine_min_lr,
        )
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.step_lr_step_size),
            gamma=cfg.step_lr_gamma,
        )
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.plateau_mode,
            factor=cfg.plateau_factor,
            patience=max(1, cfg.plateau_patience),
            min_lr=cfg.plateau_min_lr,
            verbose=True,
        )
    return None


def scheduler_step_on_plateau(scheduler, metrics: Dict[str, float], val_loss: float, cfg: TrainConfig):
    monitor = cfg.scheduler_monitor.lower()
    if monitor == "val_loss":
        value = val_loss
    elif monitor == "val_auc":
        value = metrics.get('mean_auc', float('nan'))
    elif monitor == "weighted_auc":
        value = metrics.get('mean_auc', float('nan'))
    else:
        raise ValueError(f"Unsupported scheduler monitor: {cfg.scheduler_monitor}")
    scheduler.step(value)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="3D CenterNet-style training for RSNA aneurysm classification")
    #parser.add_argument('--npy_dir', type=str, default="/home/ubuntu/work/data1/kaggle/2025_rsna/precomputed_volumes")
    #parser.add_argument('--npy_dir', type=str, default="/home/ubuntu/work/data1/kaggle/2025_rsna/pre_volumes_withlabel_448")
    parser.add_argument('--npy_dir', type=str, default="/ssd3/rsna_2025_flayer/input/pre_volumes_withlabel_448_64")
    parser.add_argument('--seg_dir', type=str, default="/ssd3/rsna_2025_flayer//input/seg_pred_448_s64")
 
    parser.add_argument('--train_csv', type=str, default="/ssd3/rsna_2025_flayer/input/train_with_folds_optimized_axis_v1.csv")
    parser.add_argument('--localizer_csv', type=str, default="/ssd3/rsna_2025_flayer/input/train_locale_fnum.csv")
    parser.add_argument('--output_dir', type=str, default="/ssd3/rsna_2025_flayer/output/outputs_heatmap_aux_v1_acc2")
    parser.add_argument('--fold', type=int, default=-1) #<0 means training all folds
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4) #4
    parser.add_argument('--accumulate_steps', type=int, default=2) #4
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=16)#16 #12
    parser.add_argument('--learning_rate', type=float, default=2.5e-4) #2e-4
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--heatmap_loss_weight', type=float, default=1.0)
    parser.add_argument('--offset_loss_weight', type=float, default=1.0)
    parser.add_argument('--cls_loss_weight', type=float, default=1) #0.5
    parser.add_argument('--aux_loss_weight', type=float, default=0.5) #0.5
    parser.add_argument('--gaussian_sigma', type=float, default=0.2)
    parser.add_argument('--default_depth_factor', type=float, default=0.5)
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--no-save-best', dest='save_best', action='store_false')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--base_channels', type=int, default=32) #32
    parser.add_argument('--device', type=str, default='cuda:2')
    #parser.add_argument('--encoder_name', type=str, default='r3d_18')
    parser.add_argument('--encoder_name', type=str, default='tf_efficientnetv2_s.in21k_ft_in1k')
    #parser.add_argument('--encoder_name', type=str, default='caformer_m36.sail_in22k_ft_in1k_384')
    #parser.add_argument('--encoder_name', type=str, default='tiny_vit_21m_384.dist_in22k_ft_in1k')
    #parser.add_argument('--encoder_name', type=str, default='convnext_tiny.in12k_ft_in1k_384')


    parser.add_argument('--no-pretrained-encoder', dest='pretrained_encoder', action='store_false')
    parser.add_argument('--pretrained_encoder', dest='pretrained_encoder', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false')
    parser.add_argument('--output_stride_depth', type=int, default=1)
    parser.add_argument('--output_stride_height', type=int, default=16) #32
    parser.add_argument('--output_stride_width', type=int, default=16) #32
    parser.add_argument('--scheduler', type=str, default='cosineannealinglr')  #none, cosineannealinglr, steplr, reducelronplateau
    parser.add_argument('--cosine_tmax', type=int, default=16) #0,12 #16
    parser.add_argument('--cosine_min_lr', type=float, default=1e-6)
    parser.add_argument('--step_lr_step_size', type=int, default=1)
    parser.add_argument('--step_lr_gamma', type=float, default=0.1)
    parser.add_argument('--plateau_factor', type=float, default=0.5)
    parser.add_argument('--plateau_patience', type=int, default=2)
    parser.add_argument('--plateau_min_lr', type=float, default=1e-6)
    parser.add_argument('--plateau_mode', type=str, default='min')
    parser.add_argument('--scheduler_monitor', type=str, default='val_loss')
    parser.set_defaults(amp=True, save_best=True)
    parser.set_defaults(pretrained_encoder=True, freeze_encoder=False)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    seed_everything(cfg.seed)
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        cfg.device = 'cpu'
    meta, locator_map = load_metadata(cfg)
    if 'fold' not in meta.columns:
        raise KeyError("train CSV must include a 'fold' column")
    #fold<0 means training all folds
    fold_indices = [cfg.fold] if cfg.fold >= 0 else list(range(cfg.num_folds))
    all_results = []

    for fold in fold_indices:
        print(f"\n===== Fold {fold} =====")
        fold_cfg = replace(cfg, fold=fold)
        meta_folds = meta['fold'].astype(int)
        train_mask = meta_folds != fold
        valid_mask = meta_folds == fold
        train_ids = meta.index[train_mask].tolist()
        valid_ids = meta.index[valid_mask].tolist()
        print(f"Train {len(train_ids)} series, Valid {len(valid_ids)} series")

        train_ds = RSNACenterNetDataset(train_ids, meta, locator_map, fold_cfg, is_train=True)
        valid_ds = RSNACenterNetDataset(valid_ids, meta, locator_map, fold_cfg, is_train=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=fold_cfg.batch_size,
            shuffle=True,
            num_workers=fold_cfg.num_workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=fold_cfg.val_batch_size,
            shuffle=False,
            num_workers=max(1, fold_cfg.num_workers // 2),
            pin_memory=True,
        )

        model = CenterNet3D(fold_cfg, num_classes=len(VESSEL_LABELS))
        model.to(fold_cfg.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=fold_cfg.learning_rate, weight_decay=fold_cfg.weight_decay)
        scheduler = build_scheduler(optimizer, fold_cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=fold_cfg.amp)
        best_auc = -float('inf')
        start_epoch = 0

        if fold_cfg.resume and os.path.exists(fold_cfg.resume):
            checkpoint = torch.load(fold_cfg.resume, map_location=fold_cfg.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint.get('epoch', 0)
            best_auc = checkpoint.get('best_auc', best_auc)
            print(f"Resumed from {fold_cfg.resume} epoch {start_epoch}")

        for epoch in range(start_epoch, fold_cfg.epochs):
            print(f"Epoch {epoch + 1}/{fold_cfg.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, fold_cfg)
            val_loss, metrics = validate_epoch(model, valid_loader, fold_cfg)
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_step_on_plateau(scheduler, metrics, val_loss, fold_cfg)
            mean_auc = metrics.get('mean_auc', float('nan'))
            print(f"train_loss {train_loss:.4f} val_loss {val_loss:.4f} mean_auc {mean_auc:.4f}")
            for name in LABEL_COLS:
                score = metrics.get(name)
                if score is None or math.isnan(score):
                    continue
                print(f"  {name}: {score:.4f}")

            is_best = mean_auc > best_auc
            if is_best:
                best_auc = mean_auc

            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_auc': best_auc,
                'config': asdict(fold_cfg),
            }

            if fold_cfg.checkpoint_every and (epoch + 1) % fold_cfg.checkpoint_every == 0:
                save_checkpoint(state, False, fold_cfg, epoch + 1, fold)
            if is_best:
                save_checkpoint(state, True, fold_cfg, epoch + 1, fold)

        print(f"Fold {fold} best mean AUC: {best_auc:.4f}")
        all_results.append((fold, best_auc))

    if len(all_results) > 1:
        avg_auc = np.mean([auc for _, auc in all_results])
        print("\n===== Summary =====")
        for fold, auc in all_results:
            print(f"Fold {fold}: {auc:.4f}")
        print(f"Average AUC: {avg_auc:.4f}")
    elif all_results:
        print(f"Best mean AUC: {all_results[0][1]:.4f}")


if __name__ == '__main__':
    main()
