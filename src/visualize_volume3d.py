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

Notes:
 - Volumes stored as (D,H,W). If meta[2]==1 they are already normalized [0,1]; otherwise raw clipped HU
   and windowing will be applied per requested window.
 - For multiple windows we create a panel per window.
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
    with np.load(npz_path) as data:
        vol = data['vol'].astype(np.float32)
        meta = data.get('meta', None)
    normalized = False
    if meta is not None and meta.shape[0] >= 3:
        normalized = float(meta[2]) == 1.0
        hu_min, hu_max = float(meta[0]), float(meta[1])
    else:
        # Fallback values from prepare script
        hu_min, hu_max = -1200.0, 4000.0
    return vol, normalized, (hu_min, hu_max)


def apply_window(vol: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width/2.0
    high = center + width/2.0
    wv = np.clip(vol, low, high)
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
    vol, normalized, (hu_min, hu_max) = load_volume(vol_path)
    depth = vol.shape[0]
    slice_idxs = choose_slice_indices(depth, args)

    # If already normalized, one window only (use original); if multiple windows requested we still show each by re-windowing normalized (harmless)
    windowed_sets = []  # List[ (window_label, list(slice_imgs)) ]
    for (c,w) in windows:
        if normalized:
            window_vol = apply_window(vol * (hu_max - hu_min) + hu_min, c, w)  # reconstruct HU approx then window
        else:
            window_vol = apply_window(vol, c, w)
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

    fig.suptitle(f"Series {series_uid}\nDepth={depth} Normalized={normalized} HU[{hu_min},{hu_max}]", fontsize=10)
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
