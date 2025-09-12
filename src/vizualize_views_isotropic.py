import argparse
import ast
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

import cv2
import numpy as np
from scipy import ndimage
import pandas as pd
import pydicom
import matplotlib.pyplot as plt


def read_dicom_frames_hu(path: Path) -> Tuple[List[np.ndarray], Tuple[float, float, float]]:
    """Read DICOM and return frames + spacing (x, y, z)."""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    
    # Get spacing info
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
    spacing_z = float(getattr(ds, "SpacingBetweenSlices", slice_thickness))
    spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)
    
    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    
    return frames, spacing


def load_volume_from_series(series_dir: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load 3D volume and return spacing info."""
    from pathlib import Path
    paths, _ = ordered_dcm_paths(series_dir)
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")
    
    slices: List[np.ndarray] = []
    spacing = None
    h, w = None, None
    
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
        raise RuntimeError(f"No readable frames in: {series_dir}")
    
    vol = np.stack(slices, axis=0).astype(np.float32)
    return vol, spacing


def resample_to_isotropic(vol: np.ndarray, spacing: Tuple[float, float, float], 
                         target_spacing: float = 1.0) -> np.ndarray:
    """Resample volume to isotropic voxels using scipy.ndimage."""
    sx, sy, sz = spacing
    zoom_factors = (sz / target_spacing, sy / target_spacing, sx / target_spacing)
    return ndimage.zoom(vol, zoom_factors, order=1, prefilter=False)


def extract_planes(vol: np.ndarray, x: int = None, y: int = None, z: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract orthogonal planes from isotropic volume."""
    Z, H, W = vol.shape
    if z is None: z = Z // 2
    if y is None: y = H // 2  
    if x is None: x = W // 2
    
    z = np.clip(z, 0, Z - 1)
    y = np.clip(y, 0, H - 1)
    x = np.clip(x, 0, W - 1)
    
    axial = vol[z, :, :]
    coronal = vol[:, y, :]
    sagittal = vol[:, :, x]
    
    # Flip for proper anatomical orientation
    coronal = np.flipud(coronal)
    sagittal = np.flipud(sagittal)
    
    return axial, coronal, sagittal


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Normalize to [0,255] uint8."""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def ordered_dcm_paths(series_dir: Path) -> Tuple[List[Path], Dict[str, int]]:
    """Collect and sort DICOM files by spatial position."""
    dicom_files = list(series_dir.glob("*.dcm"))
    if not dicom_files:
        return [], {}
    
    temp_slices = []
    for filepath in dicom_files:
        try:
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
            if hasattr(ds, "SliceLocation"):
                sort_val = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                sort_val = float(ds.ImagePositionPatient[-1])
            else:
                sort_val = float(getattr(ds, "InstanceNumber", 0))
            temp_slices.append((sort_val, filepath))
        except:
            temp_slices.append((str(filepath.name), filepath))
    
    temp_slices.sort(key=lambda x: x[0])
    sorted_files = [item[1] for item in temp_slices]
    sop_to_idx = {p.stem: i for i, p in enumerate(sorted_files)}
    return sorted_files, sop_to_idx


def plot_planes_grid(
    axials: List[np.ndarray],
    coronals: List[np.ndarray],
    sagittals: List[np.ndarray],
    save: Path | None = None,
    show: bool = True,
    overlays: Dict[str, List[List[Tuple[float, float]]]] | None = None,
):
    """Plot an Nx3 grid of planes with optional point overlays.

    Supports different number of slices per view by using N = max(len(axials), len(coronals), len(sagittals)).
    Rows with missing views are left blank.
    """
    n = max(len(axials), len(coronals), len(sagittals))
    if n == 0:
        raise ValueError("No slices to plot")

    fig, axes = plt.subplots(n, 3, figsize=(12, 2.2 * n))
    if n == 1:
        axes = np.array([axes])

    headers = ["Axial", "Coronal", "Sagittal"]
    for j, h in enumerate(headers):
        axes[0, j].set_title(h)

    def _maybe_show(ax, imgs, i, key):
        if i < len(imgs):
            ax.imshow(imgs[i], cmap="gray")
            if overlays:
                pts_list = overlays.get(key, [])
                if i < len(pts_list) and pts_list[i]:
                    xs, ys = zip(*pts_list[i])
                    ax.plot(xs, ys, "r.", markersize=6)
        else:
            # leave blank
            pass

    for i in range(n):
        _maybe_show(axes[i, 0], axials, i, "axial")
        _maybe_show(axes[i, 1], coronals, i, "coronal")
        _maybe_show(axes[i, 2], sagittals, i, "sagittal")

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout()
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save), dpi=200, bbox_inches="tight")
    else:
        fig.savefig("grid.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _quantile_indices(n: int, k: int = 8) -> List[int]:
    """Return k evenly spaced indices."""
    if n <= 0:
        return []
    if n == 1:
        return [0] * k
    pos = (np.arange(1, k + 1, dtype=np.float32) * (n - 1) / (k + 1))
    idx = np.clip(np.round(pos).astype(int), 0, n - 1)
    return idx.tolist()


def parse_vis_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", type=str, required=True)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--target-spacing", type=float, default=1, 
                   help="Target isotropic spacing in mm")
    ap.add_argument("--labels", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "train_localizers.csv"),
                   help="Path to train_localizers.csv; if missing, labels are skipped")
    ap.add_argument("--slice-mode", type=str, choices=["all", "quartile"], default="all",
                   help="Whether to plot all slices or only quartile slices (default: all)")
    return ap.parse_args()


def main():
    args = parse_vis_args()
    series_dir = Path(args.series_dir)
    uid = series_dir.name
    paths, sop_to_idx = ordered_dcm_paths(series_dir)
    
    # Load volume with spacing info
    vol, spacing = load_volume_from_series(series_dir)
    print(f"Original volume shape: {vol.shape}, spacing: {spacing}")
    
    # Resample to isotropic voxels
    vol_iso = resample_to_isotropic(vol, spacing, args.target_spacing)
    print(f"Resampled volume shape: {vol_iso.shape}")
    
    Z, H, W = vol_iso.shape
    Z0, H0, W0 = vol.shape
    sz, sy, sx = spacing[2], spacing[1], spacing[0]
    # scale factors original->iso
    kx, ky, kz = (W / max(1, W0), H / max(1, H0), Z / max(1, Z0))

    # Choose indices per view based on mode
    if args.slice_mode == "quartile":
        k = 4  # quartiles
        z_idx = _quantile_indices(Z, k)
        y_idx = _quantile_indices(H, k)
        x_idx = _quantile_indices(W, k)
    else:  # all
        z_idx = list(range(Z))
        y_idx = list(range(H))
        x_idx = list(range(W))

    axials_u8, coronals_u8, sagittals_u8 = [], [], []
    overlays = {
        "axial": [[] for _ in range(len(z_idx))],
        "coronal": [[] for _ in range(len(y_idx))],
        "sagittal": [[] for _ in range(len(x_idx))],
    }

    for zi in z_idx:
        axial = vol_iso[zi, :, :]
        axials_u8.append(min_max_normalize(axial))
    for yi in y_idx:
        coronal = vol_iso[:, yi, :]
        coronal = np.flipud(coronal)
        coronals_u8.append(min_max_normalize(coronal))
    for xi in x_idx:
        sagittal = vol_iso[:, :, xi]
        sagittal = np.flipud(sagittal)
        sagittals_u8.append(min_max_normalize(sagittal))

    # Load labels and map to iso coordinates if available
    lbl_path = Path(args.labels)
    if lbl_path.exists():
        try:
            df = pd.read_csv(lbl_path)
            if {"x", "y"}.issubset(df.columns) is False and "coordinates" in df.columns:
                df["x"] = df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore
                df["y"] = df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore
            df = df[df["SeriesInstanceUID"].astype(str) == uid]
            # Build quick lookup from displayed indices to row position for overlays
            z_to_pos = {val: i for i, val in enumerate(z_idx)}
            y_to_pos = {val: i for i, val in enumerate(y_idx)}
            x_to_pos = {val: i for i, val in enumerate(x_idx)}

            for _, r in df.iterrows():
                sop = str(r.get("SOPInstanceUID", ""))
                if sop not in sop_to_idx:
                    continue
                z0 = sop_to_idx[sop]
                x0 = float(r["x"]) if not pd.isna(r["x"]) else None
                y0 = float(r["y"]) if not pd.isna(r["y"]) else None
                if x0 is None or y0 is None:
                    continue
                print("original (x,y,z):", x0, y0, z0)
                # map to iso indices
                xi, yi, zi = x0 * kx, y0 * ky, z0 * kz
                print("isotropic (x,y,z):", xi, yi, zi)
                zi_flip = (Z - 1) - zi  # for flipped coronal/sagittal
                # assign to nearest displayed planes
                zi_r = int(round(zi))
                yi_r = int(round(yi))
                xi_r = int(round(xi))
                if zi_r in z_to_pos:
                    overlays["axial"][z_to_pos[zi_r]].append((xi, yi))
                if yi_r in y_to_pos:
                    overlays["coronal"][y_to_pos[yi_r]].append((xi, zi_flip))
                if xi_r in x_to_pos:
                    overlays["sagittal"][x_to_pos[xi_r]].append((yi, zi_flip))
        except Exception as e:
            print(f"[labels] skipped due to error: {e}")
    
    save_path = Path(args.save) if args.save else None
    plot_planes_grid(
        axials_u8,
        coronals_u8,
        sagittals_u8,
        save=save_path,
        show=not args.no_show,
        overlays=overlays,
    )


if __name__ == "__main__":
    main()
# Example usage:
#python3 -m src.vizualize_views --series-dir /home/sersasj/RSNA-IAD-Codebase/data/series/1.2.826.0.1.3680043.8.498.98540679971743770244217490650829689406

# python3 -m src.vizualize_views --series-dir /home/sersasj/RSNA-IAD-Codebase/data/series/1.2.826.0.1.3680043.8.498.98671147049544538232951626931886481868

# python3 -m src.vizualize_views --series-dir /home/sersasj/RSNA-IAD-Codebase/data/series/1.2.826.0.1.3680043.8.498.99892990973227842737467360295351276702
# python3 -m src.vizualize_views --series-dir /home/sersasj/RSNA-IAD-Codebase/data/series/1.2.826.0.1.3680043.8.498.98671147049544538232951626931886481868 --slice-mode quartile --no-show --save outputs/series_quartiles.png

#python3 -m src.vizualize_views --series-dir /home/sersasj/RSNA-IAD-Codebase/data/series/1.2.826.0.1.3680043.8.498.98671147049544538232951626931886481868 --slice-mode all --no-show --save outputs/series_quartiles.png