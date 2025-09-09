"""Simplified nnU-Net dataset preparation from RSNA IAD series."""

import argparse
import ast
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold

from configs.data_config import data_path, SEED

def sanitize_uid(uid: str) -> str:
    """Convert UID to safe filename using hash."""
    return hashlib.md5(uid.encode()).hexdigest()

def ordered_dcm_paths(series_dir: Path) -> List[Path]:
    """Return ordered DICOM paths.

    (SOP->index map is built later once multi-frame expansion is known.)
    """
    dicom_files = list(series_dir.glob("*.dcm"))
    if not dicom_files:
        return []

    slices: List[Tuple[float, Path]] = []
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True)
            sort_val = float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0)))
            slices.append((sort_val, fp))
        except Exception:
            # Skip unreadable headers
            continue
    if not slices:
        return []
    slices.sort()
    return [s[1] for s in slices]

def create_ball_mask(shape: Tuple[int, int, int], center_z: int, center_y: int, center_x: int, 
                    target_shape: Tuple[int, int, int] = (48, 384, 384)) -> np.ndarray:
    """Create ball mask sized to be 6px radius, 3 slices thick after resize."""
    mask = np.zeros(shape, dtype=np.uint8)
    depth, height, width = shape
    target_z, target_h, target_w = target_shape
    
    # Calculate resize factors
    fz = target_z / depth
    fy = target_h / height  
    fx = target_w / width
    
    # Work backwards: want 6px radius and 3 slices in final image
    desired_radius_final = 6
    desired_z_extent_final = 1
    
    # Calculate required original size
    radius_x = max(1, int(desired_radius_final / fx))
    radius_y = max(1, int(desired_radius_final / fy))
    radius_z = max(1, int(desired_z_extent_final / fz))
    
    print(f"Creating ball: orig_shape={shape}, resize_factors=({fz:.2f},{fy:.2f},{fx:.2f}), radii=({radius_z},{radius_y},{radius_x})")
    
    for z in range(max(0, center_z - radius_z), min(depth, center_z + radius_z + 1)):
        for y in range(max(0, center_y - radius_y), min(height, center_y + radius_y + 1)):
            for x in range(max(0, center_x - radius_x), min(width, center_x + radius_x + 1)):
                dz, dy, dx = z - center_z, y - center_y, x - center_x
                # Ellipsoid equation 
                if np.sqrt((dx/radius_x)**2 + (dy/radius_y)**2 + (dz/radius_z)**2) <= 1.0:
                    mask[z, y, x] = 1
    return mask

def load_labels(root: Path) -> pd.DataFrame:
    """Load localizer labels.

    Expected CSV columns include SeriesInstanceUID, SOPInstanceUID, coordinates.
    coordinates is a stringified dict with at least x,y and optionally f (1-based frame index for multi-frame DICOM).
    """
    df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in df.columns or "y" not in df.columns or "f" not in df.columns:
        def _parse_coord(s: str):
            try:
                return ast.literal_eval(s)
            except Exception:
                return {}
        coords = df["coordinates"].map(_parse_coord)
        if "x" not in df.columns:
            df["x"] = coords.map(lambda d: d.get("x", np.nan))
        if "y" not in df.columns:
            df["y"] = coords.map(lambda d: d.get("y", np.nan))
        if "f" not in df.columns:
            df["f"] = coords.map(lambda d: d.get("f"))  # may be None
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
    df["SOPInstanceUID"] = df["SOPInstanceUID"].astype(str)
    return df

def read_series(series_dir: Path) -> Tuple[Optional[np.ndarray], Dict[str, int], Dict[str, Tuple[int, int]]]:
    """Read DICOM series and return (volume, sop->middle_slice_index, frame_info).

    frame_info maps SOPInstanceUID -> (base_index, n_frames). For single-frame objects n_frames=1.
    This allows later use of explicit frame index ("f" in labels, 1-based) for multi-frame.
    We still keep a middle slice fallback mapping for backward compatibility.
    """
    paths = ordered_dcm_paths(series_dir)
    if not paths:
        return None, {}, {}

    volume_slices: List[np.ndarray] = []
    sop_map: Dict[str, int] = {}
    frame_info: Dict[str, Tuple[int, int]] = {}
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p))
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            sop_uid = str(getattr(ds, "SOPInstanceUID", p.stem))

            # RGB: take first channel
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = arr[..., 0]

            if arr.ndim == 2:
                hu = arr * slope + intercept
                slice_index = len(volume_slices)
                volume_slices.append(hu)
                sop_map[sop_uid] = slice_index
                frame_info[sop_uid] = (slice_index, 1)
            elif arr.ndim == 3:  # multi-frame (frames, H, W)
                n_frames = arr.shape[0]
                base_index = len(volume_slices)
                for f in range(n_frames):
                    hu = arr[f] * slope + intercept
                    volume_slices.append(hu)
                # Middle frame fallback
                sop_map[sop_uid] = base_index + n_frames // 2
                frame_info[sop_uid] = (base_index, n_frames)
                print(f"Expanded multi-frame DICOM {p.name}: {n_frames} frames")
            # Else: unsupported shape ignored
        except Exception as e:
            print(f"Failed reading {p.name}: {e}")
            continue

    if not volume_slices:
        return None, {}, {}

    return np.stack(volume_slices, axis=0), sop_map, frame_info

def process_series(uid: str, root: Path, out_base: Path, series_to_labels: Dict,
                  folds_map: Dict, fold_as_test: int, target_shape: Tuple) -> bool:
    """Process single series. Returns True if saved (train or test)."""
    series_dir = root / "series" / uid
    if not series_dir.exists():
        return False
    
    vol, sop_map, frame_info = read_series(series_dir)
    if vol is None:
        return False
    
    # Normalize per slice
    for z in range(vol.shape[0]):
        sl = vol[z]
        mn, mx = float(sl.min()), float(sl.max())
        if mx - mn > 1e-6:
            vol[z] = (sl - mn) / (mx - mn)
    
    vol = vol.astype(np.float32)
    
    # Create mask for training
    is_train = folds_map.get(uid, 0) != fold_as_test
    mask = None
    
    had_positives = False
    if is_train:
        labels = series_to_labels.get(uid, [])
        mask = np.zeros_like(vol, dtype=np.uint8)
        if labels:  # only build mask if we have positive annotations
            had_positives = True
            placed = 0
            missing_sop = 0
            for sop, x, y, f in labels:
                idx = None
                # Clean frame index (may be NaN or string)
                if f is not None:
                    try:
                        if isinstance(f, str) and f.strip() == '':
                            f_clean = None
                        else:
                            f_clean = int(float(f))
                    except Exception:
                        f_clean = None
                else:
                    f_clean = None

                if f_clean is not None and sop in frame_info:
                    base_index, n_frames = frame_info[sop]
                    if 1 <= f_clean <= n_frames:
                        idx = base_index + f_clean - 1
                if idx is None:
                    idx = sop_map.get(sop)
                if idx is None:
                    missing_sop += 1
                    continue
                y_i = int(np.clip(y, 0, vol.shape[1] - 1))
                x_i = int(np.clip(x, 0, vol.shape[2] - 1))
                ball = create_ball_mask(vol.shape, idx, y_i, x_i, target_shape)
                mask = np.maximum(mask, ball)
                placed += 1
            print(f"[mask-debug] {uid}: labels={len(labels)} placed={placed} missing_sop={missing_sop} mask_sum_pre_resize={int(mask.sum())}")
            pos_vox = int(mask.sum())
            if pos_vox < 10:
                print(f"Skipping {uid} - tiny positive mask ({pos_vox} voxels)")
                return False
        else:
            # Negative case: keep empty mask (all zeros) so nnU-Net has negative samples
            pass
    
    # Resize
    if target_shape:
        tz, ty, tx = target_shape
        fz = tz / vol.shape[0]
        fy = ty / vol.shape[1]  
        fx = tx / vol.shape[2]
        vol = zoom(vol, (fz, fy, fx), order=1)
        if mask is not None:
            mask = zoom(mask.astype(np.float32), (fz, fy, fx), order=1)
            mask = (mask >= 0.5).astype(np.uint8)
    
    # Print shapes and mask info
    mask_sum = mask.sum() if mask is not None else 0
    print(f"Image shape: {vol.shape}, Mask shape: {mask.shape if mask is not None else None}, Mask sum: {mask_sum}")
    
    # Check minimum mask size after resizing for training cases
    if is_train and had_positives and mask is not None and mask_sum < 10:
        print(f"Skipping {uid} - mask too small after resize: {mask_sum} voxels (positive case)")
        return False
    
    # Save as NIfTI
    safe_id = sanitize_uid(uid)
    vol_nib = np.transpose(vol, (2, 1, 0))
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img_nifti = nib.Nifti1Image(vol_nib, affine=affine)
    
    if is_train:
        out_img = out_base / "imagesTr" / f"{safe_id}_0000.nii.gz"
        out_lab = out_base / "labelsTr" / f"{safe_id}.nii.gz"
        
        nib.save(img_nifti, str(out_img))
        
        if mask is not None:
            mask_nib = np.transpose(mask, (2, 1, 0))
            mask_img = nib.Nifti1Image(mask_nib, affine=affine)
            nib.save(mask_img, str(out_lab))
        
        print(f"Saved {uid} -> train ({safe_id})")
    else:
        out_img = out_base / "imagesTs" / f"{safe_id}_0000.nii.gz"
        nib.save(img_nifti, str(out_img))
        print(f"Saved {uid} -> test ({safe_id})")
    
    return True

def create_folds(root: Path) -> Dict[str, int]:
    """Create stratified folds."""
    df = pd.read_csv(root / "train_df.csv")
    series_df = df[["SeriesInstanceUID", "Aneurysm Present"]].drop_duplicates()
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds_map = {}
    for fold, (_, val_idx) in enumerate(skf.split(series_df["SeriesInstanceUID"], series_df["Aneurysm Present"])):
        for uid in series_df.loc[val_idx, "SeriesInstanceUID"]:
            folds_map[uid] = fold
    
    return folds_map

def build_dataset_json(out_base: Path, dataset_id: int, dataset_name: str):
    """Build nnU-Net dataset.json."""
    num_tr = len(list((out_base / "imagesTr").glob("*_0000.nii.gz")))
    dataset_json = {
        "channel_names": {"0": "ct"},
        "modality": {"0": "CT"},
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": num_tr,
        "file_ending": ".nii.gz",
        "dataset_id": dataset_id,
        "name": dataset_name,
        "description": "RSNA Intracranial Aneurysm Detection",
    }
    (out_base / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

def build_splits(out_base: Path, folds_map: Dict, uid_mapping: Dict):
    """Build nnU-Net splits."""
    train_cases = [p.name.replace("_0000.nii.gz", "") for p in (out_base / "imagesTr").glob("*_0000.nii.gz")]
    safe_folds_map = {uid_mapping.get(uid, uid): fold for uid, fold in folds_map.items()}
    
    splits = []
    for f in range(5):
        tr = [cid for cid in train_cases if safe_folds_map.get(cid) != f]
        vl = [cid for cid in train_cases if safe_folds_map.get(cid) == f]
        splits.append({"train": tr, "val": vl})
    
    (out_base / "splits_final.json").write_text(json.dumps(splits, indent=2))

def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Prepare nnU-Net dataset")
    ap.add_argument("--dataset-id", type=int, required=True, help="Dataset ID")
    ap.add_argument("--dataset-name", type=str, required=True, help="Dataset name")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--workers", type=int, default=1, help="Number of workers")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[48, 384, 384], help="Target shape (Z Y X)")
    ap.add_argument("--fold-as-test", type=int, default=0, help="Fold to use as test")
    return ap.parse_args()

def write_progress_json(out_base: Path, dataset_id: int, dataset_name: str, processed: int,
                        processed_train: int, uid_mapping: Dict[str, str]):
    """Write incremental progress JSON (every 10 processed series)."""
    progress = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "processed_series_total": processed,
        "processed_train_series": processed_train,
        "uid_mapping_count": len(uid_mapping),
    }
    (out_base / "progress.json").write_text(json.dumps(progress, indent=2))


def main():
    """Main function."""
    args = parse_args()
    root = Path(data_path)
    out_base = root / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    
    if out_base.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out_base)
    
    # Create directories
    (out_base / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out_base / "labelsTr").mkdir(parents=True, exist_ok=True)
    (out_base / "imagesTs").mkdir(parents=True, exist_ok=True)

    # Ensure an initial dataset.json exists early (numTraining may be 0 initially)
    # This prevents downstream nnUNet planning from failing if the script is interrupted
    try:
        build_dataset_json(out_base, args.dataset_id, args.dataset_name)
    except Exception as e:
        print(f"Warning: initial dataset.json creation failed: {e}")
    
    # Load data
    folds_map = create_folds(root)
    label_df = load_labels(root)
    
    # Group labels by series
    series_to_labels = {}
    for _, r in label_df.iterrows():
        series_to_labels.setdefault(r.SeriesInstanceUID, []).append(
            (r.SOPInstanceUID, float(r.x), float(r.y), r.get('f', None))
        )
    
    # Get all series
    train_df = pd.read_csv(root / "train_df.csv")
    all_series = train_df["SeriesInstanceUID"].astype(str).unique().tolist()
    
    print(f"Processing {len(all_series)} series...")
    
    # Process series
    uid_mapping: Dict[str, str] = {}
    total = 0
    total_train = 0

    lock = threading.Lock()

    def _worker(uid: str):
        safe_id = sanitize_uid(uid)
        is_train_local = folds_map.get(uid, 0) != args.fold_as_test
        success = process_series(uid, root, out_base, series_to_labels,
                                 folds_map, args.fold_as_test, tuple(args.target_shape))
        # Only update shared structures under lock when needed
        with lock:
            uid_mapping[uid] = safe_id
        return uid, success, is_train_local

    if args.workers > 1:
        print(f"Using threading with {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker, uid) for uid in all_series]
            for fut in as_completed(futures):
                uid, success, is_train_flag = fut.result()
                if success:
                    total += 1
                    if is_train_flag:
                        total_train += 1
                    if total % 1 == 0:
                        try:
                            build_dataset_json(out_base, args.dataset_id, args.dataset_name)
                        except Exception:
                            pass
                        write_progress_json(out_base, args.dataset_id, args.dataset_name, total, total_train, uid_mapping)
    else:
        for uid in all_series:
            _, success, is_train_flag = _worker(uid)
            if success:
                total += 1
                if is_train_flag:
                    total_train += 1
                if total % 1 == 0:
                    try:
                        build_dataset_json(out_base, args.dataset_id, args.dataset_name)
                    except Exception:
                        pass
                    write_progress_json(out_base, args.dataset_id, args.dataset_name, total, total_train, uid_mapping)

    print(f"Processed {total} series (train: {total_train})")
    # Final progress write
    write_progress_json(out_base, args.dataset_id, args.dataset_name, total, total_train, uid_mapping)
    
    # Build dataset files
    build_dataset_json(out_base, args.dataset_id, args.dataset_name)
    build_splits(out_base, folds_map, uid_mapping)
    
    print(f"nnU-Net dataset created at: {out_base}")

if __name__ == "__main__":
    main()