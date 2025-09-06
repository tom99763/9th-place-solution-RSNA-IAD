"""Prepare nnU-Net v2 compatible dataset from RSNA IAD series.

This script converts per-series DICOM slices into nnU-Net raw dataset structure:

  DatasetXXX_NAME/
    dataset.json
    imagesTr/  <case_id>_0000.nii.gz  (channel 0 = CT)
    labelsTr/  <case_id>.nii.gz       (segmentation mask)
    imagesTs/  (optional, if --test-uids provided)

Labels are generated as hard 3D balls (binary) around provided x,y coordinates per SOPInstanceUID.
The radius is dynamically scaled based on the original image dimensions (reusing logic from
`prepare_unet_data.py`). If --no-masks is specified, labels will be skipped and only images
will be generated (useful for later weak / pseudo labeling). When labels are disabled, the
script will not include them in dataset.json and will not create labelsTr.

Fold splits (for nnU-Net cross-validation) are produced as `splits_final.json` using the
`fold_id` column in `train_df.csv` (created by the earlier data prep). If missing, folds are
generated on the fly with stratified KFold on the series-level aneurysm presence label.

Requirements:
  - The project `configs.data_config` must define `data_path`, `N_FOLDS`, `SEED`.
  - DICOM series located at: {data_path}/series/<SeriesInstanceUID>/*.dcm
  - `train_localizers.csv` with columns: SeriesInstanceUID, SOPInstanceUID, x, y (or coordinates json)
  - `train_df.csv` with columns: SeriesInstanceUID, Aneurysm Present, (optional) fold_id

Example:
    python prepare_nnunet_dataset.py \
            --dataset-id 901 \
            --dataset-name RSNAAneurysm \
            --overwrite \
            --workers 8 \
            --ignore-series-csv multiframe_dicoms.csv \
            --ignore-series-uids 1.2.3,4.5.6 \
            --ignore-sop-uids badsop1,badsop2

"""

from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib  # type: ignore
from scipy.ndimage import zoom  # type: ignore
from sklearn.model_selection import StratifiedKFold

# Project configs
from configs.data_config import data_path, N_FOLDS, SEED  # type: ignore


# ------------------------------ Utility Functions ------------------------------ #

def ordered_dcm_paths(series_dir: Path) -> Tuple[List[Path], Dict[str, int], List[float]]:
    """Return ordered DICOM slice paths, sop->index map, and list of z positions.

    Sorting priority: SliceLocation > ImagePositionPatient(z) > InstanceNumber.
    Returns empty lists if no dicoms.
    """
    dicom_files = list(series_dir.glob("*.dcm"))
    if not dicom_files:
        return [], {}, []
    tmp = []
    z_positions: List[float] = []
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            if hasattr(ds, "SliceLocation"):
                sort_val = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                sort_val = float(ds.ImagePositionPatient[2])
            else:
                sort_val = float(getattr(ds, "InstanceNumber", 0))
            tmp.append((sort_val, fp, ds))
        except Exception:
            continue
    if not tmp:
        return [], {}, []
    tmp.sort(key=lambda x: x[0])
    paths = [t[1] for t in tmp]
    # collect z for spacing estimation if available
    for _, _, ds in tmp:
        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            z_positions.append(float(ds.ImagePositionPatient[2]))
        elif hasattr(ds, "SliceLocation"):
            z_positions.append(float(ds.SliceLocation))
    sop_map = {p.stem: i for i, p in enumerate(paths)}
    return paths, sop_map, z_positions


def estimate_spacing(first_ds, z_positions: Sequence[float]) -> Tuple[float, float, float]:
    """Estimate (spacing_z, spacing_y, spacing_x) in mm.

    PixelSpacing provides (row_spacing, col_spacing) = (dy, dx). Slice spacing is median
    diff of sorted z positions; fallback to SliceThickness then 1.0.
    """
    # Pixel spacing
    if hasattr(first_ds, "PixelSpacing") and len(first_ds.PixelSpacing) >= 2:
        dy = float(first_ds.PixelSpacing[0])
        dx = float(first_ds.PixelSpacing[1])
    else:
        dy = dx = 1.0
    # Slice spacing
    if len(z_positions) >= 2:
        diffs = np.diff(sorted(z_positions))
        diffs = diffs[np.abs(diffs) > 1e-6]
        if len(diffs):
            dz = float(np.median(np.abs(diffs)))
        else:
            dz = float(getattr(first_ds, "SliceThickness", 1.0))
    else:
        dz = float(getattr(first_ds, "SliceThickness", 1.0))
    return dz, dy, dx


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float16)
    norm = (img - mn) / (mx - mn)
    return norm.astype(np.float16)

def create_hard_ball(
    shape: Tuple[int, int, int],
    center_z: int,
    center_y: int,
    center_x: int,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Create a dynamic hard 3D ellipsoidal ball (binary) with validation.

    Enhanced vs original:
      - Guards against zero/negative shape dimensions.
      - Enforces minimum radii (>=3) to avoid single voxel artifacts.
      - Clamps center coordinates into valid bounds.
    """
    # Shape validation & safe allocation
    if any(s <= 0 for s in shape):
        safe_shape = (max(1, shape[0]), max(1, shape[1]), max(1, shape[2]))
        return np.zeros(safe_shape, dtype=np.uint8)

    mask = np.zeros(shape, dtype=np.uint8)
    depth, height, width = shape
    orig_depth, orig_height, orig_width = original_shape

    base_radius = 6
    radius_x = max(3, int(base_radius * orig_width / 512))  # minimum 3
    radius_y = max(3, int(base_radius * orig_height / 512))  # minimum 3
    z_extent = max(2, 6)  # keep default 6, minimum 2 safeguard

    # Clamp centers within bounds
    center_z = max(0, min(depth - 1, center_z))
    center_y = max(0, min(height - 1, center_y))
    center_x = max(0, min(width - 1, center_x))

    z_min = max(0, center_z - z_extent)
    z_max = min(depth, center_z + z_extent + 1)
    y_min = max(0, center_y - radius_y)
    y_max = min(height, center_y + radius_y + 1)
    x_min = max(0, center_x - radius_x)
    x_max = min(width, center_x + radius_x + 1)

    for z in range(z_min, z_max):
        for y in range(y_min, y_max):
            dy = y - center_y
            for x in range(x_min, x_max):
                dz = z - center_z
                dx = x - center_x
                dist = math.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2 + (dz / z_extent) ** 2)
                if dist <= 1.0:
                    mask[z, y, x] = 1
    return mask


def load_labels(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in df.columns or "y" not in df.columns:
        df["x"] = df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        df["y"] = df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
    df["SOPInstanceUID"] = df["SOPInstanceUID"].astype(str)
    return df


def ensure_dirs(base: Path, with_labels: bool):
    (base / "imagesTr").mkdir(parents=True, exist_ok=True)
    if with_labels:
        (base / "labelsTr").mkdir(parents=True, exist_ok=True)
    (base / "imagesTs").mkdir(parents=True, exist_ok=True)


def read_series(series_dir: Path, ignore_sops: Optional[Set[str]] = None) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]], Dict[str, int]]:
    paths, sop_map, z_positions = ordered_dcm_paths(series_dir)
    if not paths:
        return None, None, {}
    slices: List[np.ndarray] = []
    first_ds = None
    for p in paths:
        if ignore_sops and p.stem in ignore_sops:
            continue
        try:
            ds = pydicom.dcmread(str(p), force=True)
            if first_ds is None:
                first_ds = ds
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] != 3:
                arr = arr[..., 0]
            if arr.ndim != 2:
                continue
            hu = arr * slope + intercept
            slices.append(hu)
        except Exception:
            continue
    if not slices or first_ds is None:
        return None, None, {}
    vol = np.stack(slices, axis=0)  # (Z, Y, X)
    spacing = estimate_spacing(first_ds, z_positions)  # (dz, dy, dx)
    return vol, spacing, sop_map


@dataclass
class SeriesJob:
    uid: str
    is_train: bool


def process_series_job(args_tuple):
    (
        job,
        root,
        out_base,
        series_to_labels,
        generate_masks,
        dtype,
        test_uids,
        verbose,
        target_shape,
    ignore_sops,
    min_mask_voxels,
    ) = args_tuple
    uid = job.uid
    series_dir = root / "series" / uid
    if not series_dir.exists():
        if verbose:
            print(f"[MISS] {uid}")
        return False
    vol, spacing, sop_map = read_series(series_dir, ignore_sops=ignore_sops)
    if vol is None:
        if verbose:
            print(f"[EMPTY] {uid}")
        return False
    original_shape = vol.shape

    # Per-slice min-max normalization (replaces previous global HU clipping)
    vol = vol.astype(np.float32)
    for z in range(vol.shape[0]):
        sl = vol[z]
        mn = float(sl.min())
        mx = float(sl.max())
        if mx - mn < 1e-6:
            vol[z] = 0.0
        else:
            vol[z] = (sl - mn) / (mx - mn)

    # nnU-Net expects (C, Z, Y, X) for saving single channel -> we save (Z,Y,X) then nib with proper affine.
    # We do not normalize; keep float32 or cast to int16.
    if dtype == "int16":
        # Preserve dynamic range by scaling to int16 range instead of truncating to {0,1}
        vol_to_save = (vol * 32767.0).clip(0, 32767).astype(np.int16)
    else:
        vol_to_save = vol.astype(np.float32)

    # ------------------------------------------------------------------
    # FIX 1 & 2: Early validation and robust mask construction
    # ------------------------------------------------------------------
    if generate_masks and job.is_train:
        labels = series_to_labels.get(uid, [])
        if not labels:
            if verbose:
                print(f"[SKIP_NO_LABELS] {uid} - no annotations found")
            return False
        mask = np.zeros_like(vol, dtype=np.uint8)
        valid_annotations = 0
        for sop, x, y in labels:
            idx = sop_map.get(sop)
            if idx is None:
                continue
            y_i = int(np.clip(y, 0, vol.shape[1] - 1))
            x_i = int(np.clip(x, 0, vol.shape[2] - 1))
            ball = create_hard_ball(vol.shape, idx, y_i, x_i, original_shape)
            mask = np.maximum(mask, ball)
            valid_annotations += 1
        if mask.sum() == 0:
            if verbose:
                print(f"[SKIP_EMPTY_INITIAL_MASK] {uid} - no valid mask regions created")
            return False
        if valid_annotations == 0:
            if verbose:
                print(f"[SKIP_NO_VALID_ANNOTATIONS] {uid} - no SOPs matched")
            return False
        if mask.sum() < min_mask_voxels:
            if verbose:
                print(f"[SKIP_TINY_MASK] {uid} mask voxels={int(mask.sum())} < {min_mask_voxels}")
            return False
    else:
        mask = None

    # ------------------------------------------------------------------
    # FIX 3: Enhanced mask resizing with stronger fallbacks
    # ------------------------------------------------------------------
    if target_shape is not None:
        tz, ty, tx = target_shape
        if tz > 0 and ty > 0 and tx > 0:
            fz = tz / vol_to_save.shape[0]
            fy = ty / vol_to_save.shape[1]
            fx = tx / vol_to_save.shape[2]
            vol_to_save = zoom(vol_to_save, (fz, fy, fx), order=1)
            if mask is not None:
                orig_mask = mask.copy()
                orig_nonzero = np.argwhere(orig_mask > 0)
                mask_float = zoom(mask.astype(np.float32), (fz, fy, fx), order=1)
                mask_resized = (mask_float >= 0.5).astype(np.uint8)
                if mask_resized.sum() == 0:
                    mask_resized = zoom(mask, (fz, fy, fx), order=0).astype(np.uint8)
                if mask_resized.sum() == 0 and len(orig_nonzero) > 0:
                    centroid = orig_nonzero.mean(axis=0)
                    new_z = int(np.clip(centroid[0] * fz, 0, tz - 1))
                    new_y = int(np.clip(centroid[1] * fy, 0, ty - 1))
                    new_x = int(np.clip(centroid[2] * fx, 0, tx - 1))
                    if mask_resized.size == 0:
                        mask_resized = np.zeros((tz, ty, tx), dtype=np.uint8)
                    mask_resized[new_z, new_y, new_x] = 1
                    if verbose:
                        print(f"[FALLBACK_CENTROID] {uid} - placed single voxel at ({new_z},{new_y},{new_x})")
                mask = mask_resized
            if spacing is not None:
                dz_old, dy_old, dx_old = spacing
                spacing = (dz_old / fz, dy_old / fy, dx_old / fx)
        else:
            if verbose:
                print(f"[WARN] Skipping resize due to non-positive target dims: {target_shape}")

    # ------------------------------------------------------------------
    # FIX 4 & 5: Final mask validation & sanitization
    # ------------------------------------------------------------------
    if mask is not None:
        if mask.size == 0:
            if verbose:
                print(f"[SKIP_ZERO_SIZE_MASK] {uid} - mask has zero size")
            return False
        if mask.sum() == 0:
            if verbose:
                print(f"[SKIP_EMPTY_FINAL_MASK] {uid} - mask is completely empty")
            return False
        if mask.sum() < min_mask_voxels:
            if verbose:
                print(f"[SKIP_TINY_FINAL_MASK] {uid} mask voxels={int(mask.sum())} < {min_mask_voxels}")
            return False
        mask = np.clip(mask, 0, 1).astype(np.uint8)
        if verbose:
            print(f"[MASK_STATS] {uid} - final mask: {int(mask.sum())} voxels, shape {tuple(mask.shape)}")

    # Convert to nibabel NIfTI and save
    dz, dy, dx = spacing if spacing is not None else (1.0, 1.0, 1.0)
    # Our numpy array is (Z,Y,X). Nib expects (X,Y,Z) by default -> transpose.
    vol_nib = np.transpose(vol_to_save, (2, 1, 0))
    affine = np.diag([dx, dy, dz, 1.0])
    img_nifti = nib.Nifti1Image(vol_nib, affine=affine)

    if job.is_train:
        out_img = out_base / "imagesTr" / f"{uid}_0000.nii.gz"
    else:
        out_img = out_base / "imagesTs" / f"{uid}_0000.nii.gz"
    if not out_img.parent.exists():
        out_img.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img_nifti, str(out_img))

    if mask is not None:
        mask_nib = np.transpose(mask.astype(np.uint8), (2, 1, 0))
        mask_img = nib.Nifti1Image(mask_nib, affine=affine)
        out_lab = out_base / "labelsTr" / f"{uid}.nii.gz"
        nib.save(mask_img, str(out_lab))
    if verbose:
        print(f"[OK] {uid} -> {'train' if job.is_train else 'test'}")
    return True


# ------------------------------ Main Orchestration ------------------------------ #

def build_dataset_json(out_base: Path, dataset_id: int, dataset_name: str, have_labels: bool):
    num_tr = len(list((out_base / "imagesTr").glob("*_0000.nii.gz")))
    j = {
        "channel_names": {"0": "ct"},
        "modality": {"0": "CT"},
        "labels": {"background": 0, "aneurysm": 1} if have_labels else {"background": 0},
        "numTraining": num_tr,
        "file_ending": ".nii.gz",
        "dataset_id": dataset_id,
        "name": dataset_name,
        "description": "RSNA Intracranial Aneurysm Detection (converted)",
        "reference": "",
        "licence": "",
        "release": "1.0",
    }
    (out_base / "dataset.json").write_text(json.dumps(j, indent=2))


def build_splits(out_base: Path, folds_map: Dict[str, int]):
    # case ids = filenames without _0000.nii.gz
    train_cases = [p.name.replace("_0000.nii.gz", "") for p in (out_base / "imagesTr").glob("*_0000.nii.gz")]
    splits: List[Dict[str, List[str]]] = []
    for f in range(N_FOLDS):
        tr = [cid for cid in train_cases if folds_map.get(cid) != f]
        vl = [cid for cid in train_cases if folds_map.get(cid) == f]
        splits.append({"train": tr, "val": vl})
    (out_base / "splits_final.json").write_text(json.dumps(splits, indent=2))


def load_or_make_folds(root: Path) -> Dict[str, int]:
    df_path = root / "train_df.csv"
    df = pd.read_csv(df_path)
    if "fold_id" in df.columns and not df["fold_id"].isna().all():
        return dict(zip(df["SeriesInstanceUID"].astype(str), df["fold_id"].astype(int)))
    # create stratified folds
    series_df = df[["SeriesInstanceUID", "Aneurysm Present"]].drop_duplicates().reset_index(drop=True)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds_map: Dict[str, int] = {}
    for fold, (_, val_idx) in enumerate(skf.split(series_df["SeriesInstanceUID"], series_df["Aneurysm Present"])):
        for uid in series_df.loc[val_idx, "SeriesInstanceUID"].tolist():
            folds_map[uid] = fold
    df["fold_id"] = df["SeriesInstanceUID"].astype(str).map(folds_map)
    df.to_csv(df_path, index=False)
    return folds_map


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare nnU-Net v2 dataset from RSNA IAD series")
    ap.add_argument("--dataset-id", type=int, default=901, help="Dataset ID (e.g., 901)")
    ap.add_argument("--dataset-name", type=str, default="RSNAAneurysm", help="Dataset name (e.g., RSNAAneurysm)")
    ap.add_argument("--nnunet-raw-root", type=Path, default=None, help="Root for nnUNet_raw (default: <data_path>/nnUNet_raw)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--no-masks", action="store_true", help="Do not generate labelsTr (images only)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--dtype", choices=["float32", "int16"], default="float32", help="Output voxel dtype after per-slice min-max (int16 scales 0..1 to 0..32767)")
    ap.add_argument("--test-uids", type=str, default="", help="Comma-separated SeriesInstanceUID for test set placement (optional)")
    ap.add_argument("--target-shape", type=int, nargs=3, default=(48, 384, 384), metavar=("Z","Y","X"), help="Resize to (Z Y X). Use -1 to keep a dimension unchanged.")
    ap.add_argument("--fold-as-test", type=int, default=0, help="Fold id to exclude from training and place into imagesTs (overrides its presence in train_df)")
    ap.add_argument("--ignore-series-uids", type=str, default="", help="Comma-separated SeriesInstanceUIDs to skip entirely")
    ap.add_argument("--ignore-series-csv", type=Path, default=None, help="CSV with column 'SeriesInstanceUID' listing series to skip")
    ap.add_argument("--ignore-sop-uids", type=str, default="", help="Comma-separated SOPInstanceUIDs (slice files) to skip")
    ap.add_argument("--ignore-sop-csv", type=Path, default=None, help="CSV with column 'SOPInstanceUID' listing slice ids to skip")
    ap.add_argument("--sample-percent", type=int, default=100, help="Percentage of training series to sample (for smaller dataset)")
    ap.add_argument("--min-mask-voxels", type=int, default=2, help="Minimum number of positive voxels required to keep a training case (skip if mask smaller; default=2 to drop single-voxel masks)")
    ap.add_argument("--verbose", action="store_true", default=False)
    return ap.parse_args()


def load_ignore_sets(root: Path, args) -> Tuple[Set[str], Set[str]]:
    series_ignore: Set[str] = set()
    sop_ignore: Set[str] = set()
    # Direct comma lists
    if args.ignore_series_uids:
        series_ignore.update([s for s in args.ignore_series_uids.split(',') if s])
    if args.ignore_sop_uids:
        sop_ignore.update([s for s in args.ignore_sop_uids.split(',') if s])
    # CSVs
    try:
        import pandas as _pd  # local import
        if args.ignore_series_csv and args.ignore_series_csv.exists():
            df = _pd.read_csv(args.ignore_series_csv)
            if 'SeriesInstanceUID' in df.columns:
                series_ignore.update(df['SeriesInstanceUID'].astype(str).tolist())
        if args.ignore_sop_csv and args.ignore_sop_csv.exists():
            df = _pd.read_csv(args.ignore_sop_csv)
            if 'SOPInstanceUID' in df.columns:
                sop_ignore.update(df['SOPInstanceUID'].astype(str).tolist())
        # Auto-load multiframe list if present
        mf = root / 'multiframe_dicoms.csv'
        if mf.exists():
            df = _pd.read_csv(mf)
            if 'SeriesInstanceUID' in df.columns:
                series_ignore.update(df['SeriesInstanceUID'].astype(str).tolist())
    except Exception:
        pass
    return series_ignore, sop_ignore


def main():
    args = parse_args()
    root = Path(data_path)
    nnunet_raw_root = args.nnunet_raw_root or (root / "nnUNet_raw")
    out_base = nnunet_raw_root / f"Dataset{args.dataset_id}_{args.dataset_name}"

    if out_base.exists() and args.overwrite:
        # Overwrite flag present: we do not delete automatically to be safe; existing files may be replaced.
        # (No action needed here.)
        ...
    ensure_dirs(out_base, with_labels=not args.no_masks)

    # Ignore sets (series & slice-level)
    ignore_series, ignore_sops = load_ignore_sets(root, args)
    if args.verbose and ignore_series:
        print(f"Ignoring {len(ignore_series)} series (bad/multiframe)")
    if args.verbose and ignore_sops:
        print(f"Ignoring {len(ignore_sops)} individual SOP slices")

    # Load labels
    label_df = load_labels(root)
    series_to_labels: Dict[str, List[Tuple[str, float, float]]] = {}
    for _, r in label_df.iterrows():
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append((str(r.SOPInstanceUID), float(r.x), float(r.y)))

    # All training series from train_df
    train_df_path = root / "train_df.csv"
    train_series: List[str]
    if train_df_path.exists():
        tdf = pd.read_csv(train_df_path)
        train_series = tdf["SeriesInstanceUID"].astype(str).unique().tolist()
    else:
        train_series = sorted(series_to_labels.keys())

    # Filter ignored series before sampling
    train_series = [s for s in train_series if s not in ignore_series]

    # Sample percentage of training series (apply ONCE). Safeguard to keep at least 1 case.
    if args.sample_percent < 100:
        import random
        random.seed(SEED)
        sample_size = int(len(train_series) * args.sample_percent / 100)
        sample_size = max(1, sample_size)
        if sample_size < len(train_series):
            train_series = random.sample(train_series, sample_size)
        if args.verbose:
            print(f"Sampled {sample_size} training series ({args.sample_percent}%)")

    test_uids = [s for s in args.test_uids.split(",") if s] if args.test_uids else []
    # Remove test from training list if provided
    train_series = [s for s in train_series if s not in test_uids]

    # Load or create folds early (needed if using --fold-as-test)
    folds_map = load_or_make_folds(root)
    # Filter folds_map to only sampled series
    folds_map = {uid: f for uid, f in folds_map.items() if uid in train_series}

    if args.fold_as_test is not None:
        fold_id = int(args.fold_as_test)
        fold_series = [uid for uid, f in folds_map.items() if f == fold_id]
        # Add to test set if not already present
        added = 0
        for uid in fold_series:
            if uid not in test_uids:
                test_uids.append(uid)
                added += 1
        # Remove from training list
        before = len(train_series)
        train_series = [s for s in train_series if s not in fold_series]
        if args.verbose:
            print(f"Moved fold {fold_id} to test set: {added} added, {before - len(train_series)} removed from training")

    # Build job list (filter ignored series)
    jobs: List[SeriesJob] = [SeriesJob(uid=s, is_train=True) for s in train_series if s not in ignore_series]
    jobs += [SeriesJob(uid=s, is_train=False) for s in test_uids]

    # Folds map already loaded above

    # Target shape handling (allow -1 to keep original dimension)
    raw_target = tuple(args.target_shape)
    target_shape: Optional[Tuple[int,int,int]] = raw_target
    if any(d == -1 for d in raw_target):
        # we will substitute -1 inside worker, so pass raw_target
        target_shape = raw_target

    # Prepare multiprocessing args
    mp_args = [
        (
            job,
            root,
            out_base,
            series_to_labels,
            not args.no_masks,
            args.dtype,
            test_uids,
            args.verbose,
            target_shape,
            ignore_sops,
            args.min_mask_voxels,
        )
        for job in jobs
    ]

    print(f"Preparing nnU-Net dataset at: {out_base}")
    print(f"  Train cases (after filtering): {sum(j.is_train for j in jobs)} | Test cases: {sum((not j.is_train) for j in jobs)} | Masks: {not args.no_masks}")

    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(process_series_job, mp_args)
    else:
        results = [process_series_job(a) for a in mp_args]

    succeeded = sum(bool(r) for r in results)
    print(f"Completed series: {succeeded}/{len(jobs)}")

    # dataset.json & splits
    build_dataset_json(out_base, args.dataset_id, args.dataset_name, have_labels=not args.no_masks)
    if not args.no_masks:
        build_splits(out_base, folds_map)
        print("Created dataset.json and splits_final.json")
        # FIX 7: Post-creation validation
        validate_dataset(out_base, args.verbose)
    else:
        print("Created dataset.json (no labels mode)")
    print("Done.")


def validate_dataset(out_base: Path, verbose: bool = False):
    """Validate the created dataset for common nnU-Net issues.

    Checks:
      - Empty label volumes
      - All-zero label volumes
      - Out-of-range label values
    Prints a concise report (first 10 issues) to help user prune problematic cases.
    """
    issues: List[str] = []
    label_dir = out_base / "labelsTr"
    if label_dir.exists():
        for label_file in label_dir.glob("*.nii.gz"):
            try:
                img = nib.load(str(label_file))
                data = img.get_fdata()
                if data.size == 0:
                    issues.append(f"Empty label file: {label_file.name}")
                elif data.sum() == 0:
                    issues.append(f"All-zero label file: {label_file.name}")
                else:
                    mx = data.max()
                    mn = data.min()
                    if mx > 1 or mn < 0:
                        issues.append(f"Invalid label values in {label_file.name}: min={mn}, max={mx}")
            except Exception as e:  # pragma: no cover - defensive
                issues.append(f"Cannot read label file {label_file.name}: {e}")

    if issues:
        print(f"\nWARNING: Found {len(issues)} potential issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        print("\nConsider removing these files before running nnU-Net preprocessing.")
    else:
        if verbose:
            print("\nDataset validation passed - no obvious issues found.")


if __name__ == "__main__":  # standard entrypoint
    main()



#python3 /home/sersasj/RSNA-IAD-Codebase/prepare_nnunet_dataset.py \
#  --dataset-id 901 \
#  --dataset-name RSNAAneurysm \
#  --workers 8 \
#  --overwrite \
#  --ignore-series-csv /path/to/multiframe_dicoms.csv \
#  --ignore-series-uids 1.2.3,4.5.6 \
#  --ignore-sop-csv /path/to/bad_sops.csv \
#  --target-shape 32 384 384 \
#  --dtype float32 \
#  --verbose


#export nnUNet_raw=/home/sersasj/RSNA-IAD-Codebase/data/nnUNet_raw
#export nnUNet_preprocessed=/home/sersasj/RSNA-IAD-Codebase/data/nnUNet_preprocessed
#export nnUNet_results=/home/sersasj/RSNA-IAD-Codebase/data/nnUNet_results