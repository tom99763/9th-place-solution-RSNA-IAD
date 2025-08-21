#!/usr/bin/env python
"""Visualize 3D volume NPZ files (prepared by prepare_data_3d.py) as slice grids.

Features:
 - Select one or multiple series UIDs (explicit list, random sample, or first N).
 - Choose slice indices explicitly or auto-sample (center + quartiles or uniform).
 - Apply one or more HU windows on-the-fly for raw (not normalized) volumes.
 - Save matplotlib PNG grids to an output directory.

Usage examples:
 1) Visualize specific series (center + quartile slices):
    python -m src.visualize_volume3d --data_dir data/processed --series 1.2.3.4 5.6.7.8
 2) Random 3 series, 6 uniformly spaced slices, custom windows:
    python -m src.visualize_volume3d --data_dir data/processed \
        --random 3 --num_slices 6 \
        --windows "(40,80),(50,150),(300,700)" \
        --out_dir volume_grids
 3) Explicit slice indices:
    python -m src.visualize_volume3d --data_dir data/processed --series 1.2.3.4 \
        --slice_indices 0,5,10,15,20,25,31

Notes (updated for new preprocessing):
 - New preprocessing stores volumes as uint8 (D,H,W) already normalized 0..255 after either:
     (a) DICOM window-based branch (meta = [window_center, window_width, 1]) using fixed clipping 0..500 → scaled;
     (b) Percentile statistical normalization branch (meta = [-1, -1, 0]).
 - Raw HU are NOT stored; windowing now operates on an approximate HU reconstruction:
     If meta[2]==1 we map uint8 back to [0,500] HU (the fixed range used in preprocessing) then apply requested windows.
     Else we treat uint8/255 as an already contrast-normalized image; additional windows just re-scale that range.
 - Multiple windows produce panels per requested (center,width) even if approximation.
 - Legacy volumes (float raw HU) are still supported: if volume dtype != uint8 and meta doesn't match new schema,
   we fall back to original behavior treating vol as HU or [0,1] normalized.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

DEFAULT_WINDOWS = [(-1200, 4000), (50,150), (60,300), (300,700)]


def parse_windows(s: str | None):
    if not s:
        return DEFAULT_WINDOWS
    # Accept formats like "(40,80),(50,150)" or "40:80,50:150" or "40-80;50-150"
    s = s.strip()
    if not s:
        return DEFAULT_WINDOWS
    pairs = []
    # Normalize separators
    s_norm = re.sub(r'[;\n]', ',', s)
    # Split by ) if tuple-like
    tuple_like = re.findall(r'\(?\s*[-+]?[0-9]*\.?[0-9]+\s*[,;:\-]\s*[-+]?[0-9]*\.?[0-9]+\s*\)?', s_norm)
    if tuple_like:
        for tk in tuple_like:
            nums = re.split(r'[,:;\-]', tk.strip('() '))
            if len(nums) == 2:
                try:
                    c = float(nums[0]); w = float(nums[1])
                    if w <= 0: continue
                    pairs.append((c, w))
                except Exception:
                    pass
    else:
        # Fallback simple split
        parts = [p for p in re.split(r'[;,]', s_norm) if p.strip()]
        for p in parts:
            nums = re.split(r'[:\-]', p)
            if len(nums)==2:
                try:
                    c = float(nums[0]); w = float(nums[1]);
                    if w>0: pairs.append((c,w))
                except Exception:
                    pass
    return pairs if pairs else DEFAULT_WINDOWS


def load_volume(npz_path: Path):
    """Return a dict with standardized fields regardless of old/new format.

    Returns:
        {
          'vol_uint8': uint8 volume (D,H,W) (created if necessary),
          'approx_hu_vol': float32 approximate HU (if reconstructable) else None,
          'meta': meta ndarray or None,
          'mode': 'new_uint8' | 'legacy_hu' | 'legacy_norm'
        }
    """
    with np.load(npz_path) as data:
        vol = data['vol']
        meta = data.get('meta', None)

    # Detect new format: uint8 + meta where meta[2] in {0,1} and meta[0] possibly -1 or plausible center
    if vol.dtype == np.uint8 and meta is not None and meta.shape[0] >= 3:
        window_center, window_width, used_flag = float(meta[0]), float(meta[1]), float(meta[2])
        vol_uint8 = vol
        approx_hu_vol = None
        if used_flag == 1.0:  # window branch used fixed 0..500 mapping
            approx_hu_vol = (vol_uint8.astype(np.float32) / 255.0) * 500.0  # approximate HU in [0,500]
        else:
            # Statistical normalization branch – treat as generic normalized image, no real HU scale.
            approx_hu_vol = None
        return {
            'vol_uint8': vol_uint8,
            'approx_hu_vol': approx_hu_vol,
            'meta': meta,
            'mode': 'new_uint8'
        }

    # Legacy behavior (previous pipeline): vol stored float (HU clipped or normalized)
    vol_f32 = vol.astype(np.float32)
    if meta is not None and meta.shape[0] >= 3:
        normalized_flag = float(meta[2]) == 1.0
        hu_min, hu_max = float(meta[0]), float(meta[1])
        if normalized_flag:
            # Already [0,1]; create uint8 representation
            vol_uint8 = (np.clip(vol_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
            approx_hu_vol = vol_f32 * (hu_max - hu_min) + hu_min
            mode = 'legacy_norm'
        else:
            # Raw HU clipped range hu_min..hu_max
            vol_uint8 = ((vol_f32 - hu_min) / max(hu_max - hu_min, 1e-6))
            vol_uint8 = (np.clip(vol_uint8, 0.0, 1.0) * 255.0).astype(np.uint8)
            approx_hu_vol = vol_f32
            mode = 'legacy_hu'
    else:
        # Unknown meta; treat as normalized image
        vol_uint8 = (np.clip(vol_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        approx_hu_vol = None
        mode = 'legacy_unknown'
    return {
        'vol_uint8': vol_uint8,
        'approx_hu_vol': approx_hu_vol,
        'meta': meta,
        'mode': mode
    }


def apply_window(vol_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply window on approximate HU volume returning [0,1]."""
    low = center - width/2.0
    high = center + width/2.0
    wv = np.clip(vol_hu, low, high)
    return (wv - low) / max(width, 1e-6)


def choose_slice_indices(depth: int, args) -> list[int]:
    if args.slice_indices:
        return [min(depth-1, max(0, int(x))) for x in args.slice_indices.split(',') if x.strip().isdigit()]
    if args.num_slices:
        n = min(depth, args.num_slices)
        return sorted({int(round(i)) for i in np.linspace(0, depth-1, n)})
    # Default: center + quartiles
    indices = sorted(set([depth//2, depth//4, (3*depth)//4, 0, depth-1]))
    return [i for i in indices if 0 <= i < depth]


def build_grid_for_series(series_uid: str, vol_dir: Path, windows, args, out_dir: Path):
    matches = list(vol_dir.glob(f"{series_uid}*_d*_sz*.npz"))
    if not matches:
        print(f"[WARN] No volume file found for {series_uid}")
        return
    vol_path = matches[0]
    info = load_volume(vol_path)
    vol_uint8 = info['vol_uint8']
    depth = vol_uint8.shape[0]
    slice_idxs = choose_slice_indices(depth, args)
    windowed_sets = []  # List[(window_label, list(slice_imgs))]
    approx_hu = info['approx_hu_vol']
    mode = info['mode']
    meta = info['meta']
    used_flag = None
    if meta is not None and len(meta) >= 3:
        used_flag = int(meta[2])

    if approx_hu is None:
        # Statistical normalization (MRI / no window tags). Allow multiple artificial windows if requested.
        base = (vol_uint8.astype(np.float32) / 255.0)
        if len(windows) <= 1:
            windowed_sets.append(("BASE", [base[i] for i in slice_idxs]))
        else:
            # Simulate window effects by simple contrast windows over the scaled base (treat base*500 as pseudo HU)
            pseudo_hu = base * 500.0
            for (c, w) in windows:
                window_vol = apply_window(pseudo_hu, c, w)
                images = [window_vol[i] for i in slice_idxs]
                windowed_sets.append((f"C{int(c)}W{int(w)}", images))
    else:
        # CT with real window tags reconstructed (approx_hu available). If already clipped to 0-500 (used_flag==1) just show single column.
        if used_flag == 1:
            base = (vol_uint8.astype(np.float32) / 255.0)
            windowed_sets.append(("C0-500", [base[i] for i in slice_idxs]))
        else:
            for (c, w) in windows:
                window_vol = apply_window(approx_hu, c, w)
                images = [window_vol[i] for i in slice_idxs]
                windowed_sets.append((f"C{int(c)}W{int(w)}", images))

    # Figure layout: rows = len(slice_idxs), cols = len(windows)
    rows = len(slice_idxs)
    cols = len(windowed_sets)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if cols == 1:
        axes = np.expand_dims(axes, 1)

    for col, (wlabel, img_list) in enumerate(windowed_sets):
        for row, img in enumerate(img_list):
            ax = axes[row][col]
            ax.imshow(img, cmap='gray')
            if col == 0:
                ax.set_ylabel(f"z={slice_idxs[row]}", fontsize=8)
            if row == 0:
                ax.set_title(wlabel, fontsize=9)
            ax.axis('off')

    if info['mode'] == 'new_uint8' and meta is not None:
        wc, ww, used = meta.tolist()[:3]
        meta_str = f"wc={wc:.1f} ww={ww:.1f} used={int(used)}"
    else:
        meta_str = info['mode']
    fig.suptitle(f"Series {series_uid}\nDepth={depth} Mode={info['mode']} {meta_str}", fontsize=10)
    plt.tight_layout(rect=(0,0,1,0.95))
    out_path = out_dir / f"{series_uid}_slices.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D volume slices")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Processed data directory containing volumes_3d and volume_df.csv')
    parser.add_argument('--series', nargs='*', default=None, help='Explicit list of SeriesInstanceUIDs')
    parser.add_argument('--first', type=int, default=0, help='Take first N series from CSV (after filtering)')
    parser.add_argument('--random', type=int, default=0, help='Random sample N series (after filtering)')
    parser.add_argument('--filter_positive', action='store_true', help='Only visualize positive (aneurysm) series')
    parser.add_argument('--num_slices', type=int, default=0, help='Number of slices uniformly spaced (overrides quartile default)')
    parser.add_argument('--slice_indices', type=str, default='', help='Comma-separated explicit slice indices')
    parser.add_argument('--windows', type=str, default='', help='Custom windows specification')
    parser.add_argument('--out_dir', type=str, default='volume_grids', help='Output directory for PNGs')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    vol_dir = data_dir / 'volumes_3d'
    meta_csv = data_dir / 'volume_df.csv'

    if not vol_dir.exists():
        raise SystemExit(f"Volume directory not found: {vol_dir}")
    if not meta_csv.exists():
        raise SystemExit(f"Metadata CSV not found: {meta_csv}")

    df = pd.read_csv(meta_csv)
    if args.filter_positive:
        df = df[df['series_has_aneurysm'] == 1]

    candidates = df['series_uid'].tolist()

    selected = []
    if args.series:
        selected.extend([s for s in args.series if s in candidates])
    if args.first > 0:
        selected.extend(candidates[:args.first])
    if args.random > 0:
        selected.extend(random.sample(candidates, k=min(args.random, len(candidates))))

    # Deduplicate preserving order
    seen = set()
    final_series = []
    for s in selected:
        if s not in seen:
            seen.add(s); final_series.append(s)

    if not final_series:
        # fallback: take first 3
        # 3 random idx
        random_choices = random.sample(candidates, k=3)
        final_series = random_choices

    windows = parse_windows(args.windows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {len(final_series)} series | windows={windows} | out={out_dir}")
    for uid in final_series:
        build_grid_for_series(uid, vol_dir, windows, args, out_dir)

if __name__ == '__main__':
    main()
