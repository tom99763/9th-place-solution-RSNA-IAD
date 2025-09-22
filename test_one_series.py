#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.prepare_yolo_dataset_v3 import (
    process_series_worker, 
    ordered_dcm_paths, 
    load_folds, 
    load_labels
)
from configs.data_config import data_path as _data_path, LABELS_TO_IDX

# Resolve data_path
data_path = str(ROOT / _data_path) if not Path(_data_path).is_absolute() else _data_path
root = Path(data_path)

print(f"Data root: {root}")

# Use a series we know has labels
uid = "1.2.826.0.1.3680043.8.498.31629979420404800139928339434297456334"
series_dir = root / "series" / uid

if not series_dir.exists():
    print(f"Series directory doesn't exist: {series_dir}")
    sys.exit(1)

print(f"Testing series: {uid}")
print(f"Series directory: {series_dir}")

# Load data
folds = load_folds(root)
label_df = load_labels(root)

# Get labels for this specific series
series_labels = []
for _, r in label_df.iterrows():
    if r.SeriesInstanceUID == uid:
        loc = r.get("location", None)
        if isinstance(loc, str) and loc in LABELS_TO_IDX:
            cls_id = int(LABELS_TO_IDX[loc])
            series_labels.append((str(r.SOPInstanceUID), float(r.x), float(r.y), cls_id, None))

print(f"Found {len(series_labels)} labels for this series")

split = "val" if folds.get(uid, 0) == 0 else "train"
paths, sop_to_idx = ordered_dcm_paths(series_dir)

print(f"Split: {split}")
print(f"DICOM files: {len(paths)}")
print(f"SOP to index mapping: {len(sop_to_idx)}")

# Create output directory
out_base = root / "test_one_series_output"
out_base.mkdir(exist_ok=True)

# Setup args tuple for worker
args_tuple = (
    uid,                    # uid
    series_dir,            # series_dir
    split,                 # split
    series_labels,         # labels_for_series
    sop_to_idx,            # sop_to_idx
    out_base,              # out_base
    "locations",           # label_scheme
    24,                    # box_size
    "png",                 # image_ext
    True,                  # overwrite
    True,                  # verbose
    0,                     # pos_neg_ratio
    1,                     # neg_per_series
    42                     # seed
)

print("\n=== Running process_series_worker ===")
result = process_series_worker(args_tuple)
print(f"Worker result: {result}")

# Check what was created
png_files = list(out_base.glob("**/*.png"))
txt_files = list(out_base.glob("**/*.txt"))
print(f"\nFiles created: {len(png_files)} PNG, {len(txt_files)} TXT")

if png_files:
    print("PNG files:")
    for f in png_files:
        print(f"  {f}")

if txt_files:
    print("TXT files:")  
    for f in txt_files[:3]:
        content = f.read_text().strip()
        print(f"  {f}: {content}")