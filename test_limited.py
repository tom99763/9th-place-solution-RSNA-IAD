#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import the generate_for_fold function directly
from src.prepare_yolo_dataset_v3 import generate_for_fold
import argparse

# Create a mock args object with limited processing
class MockArgs:
    def __init__(self):
        self.seed = 42
        self.val_fold = 0
        self.generate_all_folds = False
        self.box_size = 24
        self.image_ext = "png"
        self.overwrite = True
        self.verbose = True
        self.workers = 1  # Use single worker for testing
        self.mip_img_size = 512
        self.fold_output_name = "test_limited_output"
        self.label_scheme = "locations"
        self.neg_per_series = 1
        self.pos_neg_ratio = 0
        self.yaml_out_dir = "configs"
        self.yaml_name_template = "test_limited_fold{fold}.yaml"

args = MockArgs()

# Monkey patch the all_series list to only process first 10 series
import src.prepare_yolo_dataset_v3 as prep_module

original_generate = prep_module.generate_for_fold

def limited_generate_for_fold(val_fold, args):
    # Call original but limit series
    from pathlib import Path
    import pandas as pd
    from src.prepare_yolo_dataset_v3 import data_path
    
    root = Path(data_path)
    
    # Get first 10 series only
    train_df_path = root / "train.csv"
    if train_df_path.exists():
        train_csv = pd.read_csv(train_df_path)
        if "SeriesInstanceUID" in train_csv.columns:
            all_series = train_csv["SeriesInstanceUID"].astype(str).unique().tolist()[:10]  # Limit to 10
            print(f"Limited processing to first {len(all_series)} series")
            
            # Patch the module to use our limited series list
            prep_module.limited_series = all_series
    
    return original_generate(val_fold, args)

# Monkey patch
prep_module.generate_for_fold = limited_generate_for_fold

print("Running limited test with first 10 series...")
out_base, folds = generate_for_fold(0, args)
print(f"Completed. Output base: {out_base}")

# Check results
png_files = list(out_base.glob("**/*.png"))
txt_files = list(out_base.glob("**/*.txt"))
print(f"Created: {len(png_files)} PNG files, {len(txt_files)} TXT files")

if png_files:
    print("Sample files created:")
    for f in png_files[:3]:
        print(f"  {f}")