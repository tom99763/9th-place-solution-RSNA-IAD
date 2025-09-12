import argparse
import ast
import math
import random
import sys
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import cv2
import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold

# Project config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.data_config import data_path, N_FOLDS as CONF_N_FOLDS, SEED, LABELS_TO_IDX  # type: ignore

# Defaults (can be overridden by config)
N_FOLDS = CONF_N_FOLDS if isinstance(CONF_N_FOLDS, int) else 5
rng = random.Random(SEED)


# --------- IO helpers ---------

def read_dicom_frames_hu(path: Path) -> Tuple[List[np.ndarray], Tuple[float, float, float]]:
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    # spacing (x, y, z)
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
    spacing_z = float(getattr(ds, "SpacingBetweenSlices", slice_thickness))
    spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)

    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        frames.append(pix.astype(np.float32) * slope + intercept)
    elif pix.ndim == 3:
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            frames.append(pix[..., 0].astype(np.float32) * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frames.append(pix[i].astype(np.float32) * slope + intercept)
    return frames, spacing


def ordered_dcm_paths(series_dir: Path) -> Tuple[List[Path], Dict[str, int]]:
    files = list(series_dir.glob("*.dcm"))
    if not files:
        return [], {}
    tmp = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True)
            if hasattr(ds, "SliceLocation"):
                key = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                key = float(ds.ImagePositionPatient[-1])
            else:
                key = float(getattr(ds, "InstanceNumber", 0))
        except Exception:
            key = 0.0
        tmp.append((key, fp))
    tmp.sort(key=lambda x: x[0])
    paths = [t[1] for t in tmp]
    sop_to_idx = {p.stem: i for i, p in enumerate(paths)}
    return paths, sop_to_idx


def load_volume(series_dir: Path) -> Tuple[np.ndarray, Tuple[float, float, float], List[Path], Dict[str, int]]:
    paths, sop_to_idx = ordered_dcm_paths(series_dir)
    if not paths:
        raise FileNotFoundError(f"No DICOMs in {series_dir}")
    slices: List[np.ndarray] = []
    spacing = None
    h = w = None
    for p in paths:
        frames, sp = read_dicom_frames_hu(p)
        if spacing is None:
            spacing = sp
        for fr in frames:
            if h is None:
                h, w = fr.shape
            if fr.shape != (h, w):
                continue
            slices.append(fr)
    if not slices:
        raise RuntimeError(f"No readable frames in {series_dir}")
    vol = np.stack(slices, axis=0).astype(np.float32)
    assert spacing is not None
    return vol, spacing, paths, sop_to_idx


# --------- Geometry ---------

def resample_isotropic(vol: np.ndarray, spacing: Tuple[float, float, float], target: float = 1.0) -> np.ndarray:
    sx, sy, sz = spacing
    zoom = (sz / target, sy / target, sx / target)  # Z, Y, X order
    return ndimage.zoom(vol, zoom, order=1, prefilter=False)


def resize_to_cube(vol: np.ndarray, size: int) -> np.ndarray:
    """Deprecated: Avoid resizing whole 3D volume to a cube for speed.

    Kept for backward compatibility if needed elsewhere, but unused in this script now.
    """
    if size <= 0:
        return vol
    Z, H, W = vol.shape
    if Z == size and H == size and W == size:
        return vol
    zoom = (size / Z, size / H, size / W)
    return ndimage.zoom(vol, zoom, order=1, prefilter=False).astype(np.float32)


def normalize_u8(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


def yolo_box_from_point(x: float, y: float, box: int, w: int, h: int) -> Tuple[float, float, float, float]:
    half = box / 2.0
    x0, y0 = max(0.0, x - half), max(0.0, y - half)
    x1, y1 = min(w - 1.0, x + half), min(h - 1.0, y + half)
    bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
    xc, yc = x0 + bw / 2.0, y0 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


# --------- Data/labels ---------

def load_folds(root: Path) -> Dict[str, int]:
    df_path = root / "train_df.csv"
    df = pd.read_csv(df_path)
    series_df = df[["SeriesInstanceUID", "Aneurysm Present"]].drop_duplicates().reset_index(drop=True)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_map: Dict[str, int] = {}
    for i, (_, test_idx) in enumerate(skf.split(series_df["SeriesInstanceUID"], series_df["Aneurysm Present"])):
        for uid in series_df.loc[test_idx, "SeriesInstanceUID"].tolist():
            fold_map[uid] = i
    df["fold_id"] = df["SeriesInstanceUID"].astype(str).map(fold_map)
    df.to_csv(df_path, index=False)
    return fold_map


def load_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in label_df.columns or "y" not in label_df.columns:
        label_df["x"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        label_df["y"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    label_df["SeriesInstanceUID"] = label_df["SeriesInstanceUID"].astype(str)
    label_df["SOPInstanceUID"] = label_df["SOPInstanceUID"].astype(str)
    return label_df


def ensure_yolo_dirs(base: Path):
    for v in ["axial", "coronal", "sagittal"]:
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (base / v / sub).mkdir(parents=True, exist_ok=True)


# --------- Views ---------
@dataclass
class ViewDef:
    name: str  # axial/coronal/sagittal

    def slice_image(self, vol: np.ndarray, idx: int) -> np.ndarray:
        if self.name == "axial":
            return vol[idx, :, :]
        if self.name == "coronal":
            return np.flipud(vol[:, idx, :])
        if self.name == "sagittal":
            return np.flipud(vol[:, :, idx])
        raise ValueError(self.name)

    def point_to_view(self, x: float, y: float, z: float, Z: int) -> Tuple[float, float, int]:
        # return (u, v, slice_index)
        if self.name == "axial":
            return x, y, int(round(z))
        if self.name == "coronal":
            return x, (Z - 1) - z, int(round(y))
        if self.name == "sagittal":
            return y, (Z - 1) - z, int(round(x))
        raise ValueError(self.name)


VIEWS = [ViewDef("axial"), ViewDef("coronal"), ViewDef("sagittal")]


# --------- Main generation ---------

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare YOLO dataset of axial/coronal/sagittal slices from isotropic volumes")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--generate-all-folds", action="store_true")
    ap.add_argument("--mip-img-size", type=int, default=512, help="Output slice size (each 2D image will be resized to this square NxN)")
    ap.add_argument("--target-spacing", type=float, default=1.0, help="Isotropic voxel size (mm)")
    ap.add_argument("--box-size", type=int, default=24)
    ap.add_argument("--image-ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-name", type=str, default="yolo_planes")
    ap.add_argument("--label-scheme", type=str, choices=["locations", "aneurysm_present"], default="aneurysm_present")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel workers for series processing (use >1 to speed up)")
    return ap.parse_args()


def write_label(path: Path, cls_id: int, x: float, y: float, w: int, h: int, box: int):
    xc, yc, bw, bh = yolo_box_from_point(x, y, box, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# --------- Parallel worker context ---------
_CTX_SERIES_TO_LABELS: Dict[str, List[Tuple[str, float, float, int]]] | None = None
_CTX_FOLDS: Dict[str, int] | None = None
_CTX_ARGS: types.SimpleNamespace | None = None
_CTX_OUT_ROOT: Path | None = None
_CTX_VAL_FOLD: int | None = None


def _init_worker(series_to_labels: Dict[str, List[Tuple[str, float, float, int]]],
                 folds: Dict[str, int],
                 args_dict: Dict[str, object],
                 out_root_str: str,
                 val_fold: int):
    global _CTX_SERIES_TO_LABELS, _CTX_FOLDS, _CTX_ARGS, _CTX_OUT_ROOT, _CTX_VAL_FOLD
    _CTX_SERIES_TO_LABELS = series_to_labels
    _CTX_FOLDS = folds
    _CTX_ARGS = types.SimpleNamespace(**args_dict)
    _CTX_OUT_ROOT = Path(out_root_str)
    _CTX_VAL_FOLD = val_fold


def _worker_job(uid: str) -> Tuple[str, str]:
    assert _CTX_SERIES_TO_LABELS is not None and _CTX_FOLDS is not None and _CTX_ARGS is not None
    assert _CTX_OUT_ROOT is not None and _CTX_VAL_FOLD is not None
    split = "val" if _CTX_FOLDS.get(uid, 0) == _CTX_VAL_FOLD else "train"
    series_dir = Path(data_path) / "series" / uid
    if not series_dir.exists():
        return (uid, "MISS")
    try:
        lbl_rows = _CTX_SERIES_TO_LABELS.get(uid, [])
        process_series(uid, split, _CTX_ARGS, _CTX_OUT_ROOT, lbl_rows)
        return (uid, "OK")
    except Exception as e:
        return (uid, f"ERR: {e}")


def _deterministic_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def process_series(uid: str, split: str, args, out_root: Path, label_rows: List[Tuple[str, float, float, int]]):
    series_dir = Path(data_path) / "series" / uid
    vol, spacing, paths, sop_to_idx = load_volume(series_dir)
    # resample 3D isotropic then resize to cube
    vol_iso = resample_isotropic(vol, spacing, args.target_spacing)
    # Avoid expensive 3D cube resizing; do 2D per-slice resize on write instead for speed.
    Z, H, W = vol_iso.shape

    # scale factors from original index space (x=width, y=height, z=slice index among sorted paths)
    kx, ky, kz = (W / max(1, vol.shape[2]), H / max(1, vol.shape[1]), Z / max(1, vol.shape[0]))

    # group labels by SOP slice index
    points_xyz: List[Tuple[float, float, float, int]] = []  # x,y,z,cls
    for sop, x, y, cls in label_rows:
        z0 = sop_to_idx.get(sop)
        if z0 is None:
            continue
        points_xyz.append((x * kx, y * ky, z0 * kz, cls))

    # for each view, process based on whether this is a positive or negative series
    for view in VIEWS:
        img_dir = out_root / view.name / "images" / split
        lbl_dir = out_root / view.name / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # map slice index -> list of points in this view
        view_pts: Dict[int, List[Tuple[float, float, int]]] = {}
        for (xv, yv, zv, cls) in points_xyz:
            u, v, s_idx = view.point_to_view(xv, yv, zv, Z)
            view_pts.setdefault(s_idx, []).append((u, v, cls))

        if label_rows:  # Positive series - only save slices with annotations
            for s_idx in sorted(view_pts.keys()):
                raw = view.slice_image(vol_iso, s_idx)
                h0, w0 = raw.shape
                out_sz = int(args.mip_img_size)
                if out_sz > 0:
                    img = cv2.resize(raw, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)
                    sx, sy = out_sz / w0, out_sz / h0
                    out_h, out_w = out_sz, out_sz
                else:
                    img = raw
                    sx, sy = 1.0, 1.0
                    out_h, out_w = h0, w0
                img = normalize_u8(img)
                stem = f"{uid}_{view.name}_{s_idx}"
                img_path = img_dir / f"{stem}.{args.image_ext}"
                lbl_path = lbl_dir / f"{stem}.txt"
                if img_path.exists() and lbl_path.exists() and not args.overwrite:
                    if args.verbose:
                        print(f"[SKIP] already exists: {img_path}")
                    continue
                if args.verbose:
                    print(f"[PROC] writing: {img_path}")
                cv2.imwrite(str(img_path), img)
                pts = view_pts[s_idx]
                # Write all points on this slice (multi-object)
                with open(lbl_path, "w") as f:
                    for (u, v, cls) in pts:
                        u2, v2 = u * sx, v * sy
                        xc, yc, bw, bh = yolo_box_from_point(u2, v2, args.box_size, out_w, out_h)
                        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        else:  # Negative series - save one random slice
            max_slices = {"axial": Z, "coronal": H, "sagittal": W}[view.name]
            if max_slices > 0:
                # deterministic selection for reproducibility and parallel safety
                s_idx = _deterministic_index(f"{uid}:{view.name}:{SEED}", max_slices)
                raw = view.slice_image(vol_iso, s_idx)
                h0, w0 = raw.shape
                out_sz = int(args.mip_img_size)
                if out_sz > 0:
                    img = cv2.resize(raw, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)
                else:
                    img = raw
                img = normalize_u8(img)
                stem = f"{uid}_{view.name}_{s_idx}"
                img_path = img_dir / f"{stem}.{args.image_ext}"
                lbl_path = lbl_dir / f"{stem}.txt"
                if not (img_path.exists() and lbl_path.exists()) or args.overwrite:
                    if args.verbose:
                        print(f"[PROC] writing (negative): {img_path}")
                    cv2.imwrite(str(img_path), img)
                    # Empty label file for negative
                    lbl_path.touch()
                else:
                    if args.verbose:
                        print(f"[SKIP] already exists (negative): {img_path}")


def build_series_lists(root: Path, labels_df: pd.DataFrame) -> List[str]:
    all_series: List[str] = []
    train_df_path = root / "train_df.csv"
    if train_df_path.exists():
        try:
            train_csv = pd.read_csv(train_df_path)
            if "SeriesInstanceUID" in train_csv.columns:
                all_series = train_csv["SeriesInstanceUID"].astype(str).unique().tolist()
        except Exception:
            pass
    if not all_series:
        all_series = sorted(labels_df["SeriesInstanceUID"].astype(str).unique().tolist())
    return all_series


def generate_for_fold(val_fold: int, args):
    root = Path(data_path)
    out_root = root / f"{args.out_name}_fold{val_fold}"
    ensure_yolo_dirs(out_root)

    folds = load_folds(root)
    labels = load_labels(root)

    # Build series -> list[(SOP, x, y, cls)] with scheme
    series_to_labels: Dict[str, List[Tuple[str, float, float, int]]] = {}
    for _, r in labels.iterrows():
        if args.label_scheme == "locations":
            loc = r.get("location", None)
            if isinstance(loc, str) and loc in LABELS_TO_IDX:
                cls_id = int(LABELS_TO_IDX[loc])
            else:
                continue
        else:
            cls_id = 0
        series_to_labels.setdefault(str(r.SeriesInstanceUID), []).append(
            (str(r.SOPInstanceUID), float(r.x), float(r.y), cls_id)
        )

    all_series = build_series_lists(root, labels)

    print(f"Processing {len(all_series)} series (val fold={val_fold}, workers={args.workers}) -> {out_root}")

    if max(1, int(args.workers)) > 1:
        n_workers = max(1, int(args.workers))
        # Pass minimal picklable context to workers
        args_dict = vars(args).copy()
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(series_to_labels, folds, args_dict, str(out_root), val_fold)) as ex:
            futs = {ex.submit(_worker_job, uid): uid for uid in all_series}
            for i, fut in enumerate(as_completed(futs)):
                uid, status = fut.result()
                if args.verbose and status != "OK":
                    print(f"[{status}] {uid}")
    else:
        for uid in all_series:
            # Run directly without pool
            split = "val" if folds.get(uid, 0) == val_fold else "train"
            series_dir = root / "series" / uid
            if not series_dir.exists():
                status = "MISS"
            else:
                try:
                    lbl_rows = series_to_labels.get(uid, [])
                    process_series(uid, split, args, out_root, lbl_rows)
                    status = "OK"
                except Exception as e:
                    status = f"ERR: {e}"
            uid, status = uid, status
            if args.verbose and status != "OK":
                print(f"[{status}] {uid}")

    # quick stats
    for view in ["axial", "coronal", "sagittal"]:
        for split in ["train", "val"]:
            n_imgs = len(list((out_root / view / "images" / split).glob("*")))
            n_lbls = len(list((out_root / view / "labels" / split).glob("*.txt")))
            print(f"{view}/{split}: images={n_imgs} labels={n_lbls}")

    return out_root


def write_yolo_yaml(yaml_dir: Path, yaml_name: str, dataset_root: Path, label_scheme: str):
    yaml_dir.mkdir(parents=True, exist_ok=True)
    rel = dataset_root.resolve().relative_to(ROOT)
    # names
    names: List[str] = []
    if label_scheme == "aneurysm_present":
        names = ["aneurysm_present"]
    else:
        idx_to_label = {idx: name for name, idx in LABELS_TO_IDX.items()}
        for i in range(max(idx_to_label.keys()) + 1):
            names.append(idx_to_label.get(i, f"class_{i}"))

    for view in ["axial", "coronal", "sagittal"]:
        # Important: train/val must be RELATIVE to 'path' to avoid duplication by Ultralytics
        text = [
            f"path: {rel}/{view}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
        ] + [f"  {i}: {n}" for i, n in enumerate(names)] + [""]
        with open(yaml_dir / yaml_name.format(view=view), "w") as f:
            f.write("# Auto-generated by prepare_yolo_dataset_v2_planes.py\n")
            f.write("\n".join(text))


if __name__ == "__main__":
    args = parse_args()
    if args.generate_all_folds:
        for f in range(N_FOLDS):
            out_root = generate_for_fold(f, args)
            write_yolo_yaml(Path(ROOT / "configs"), f"yolo_planes_{{view}}_fold{f}.yaml", out_root, args.label_scheme)
        print(f"Generated per-view datasets and YAMLs for folds 0..{N_FOLDS-1}")
    else:
        out_root = generate_for_fold(args.val_fold, args)
        write_yolo_yaml(Path(ROOT / "configs"), f"yolo_planes_{{view}}.yaml", out_root, args.label_scheme)
        print("Done. YAMLs written under configs/ as yolo_planes_{view}.yaml")

#python3 -m src.prepare_yolo_dataset_v2_planes --val-fold 0 --mip-img-size 512 --target-spacing 1.0 --label-
#scheme locations --out-name yolo_planes

