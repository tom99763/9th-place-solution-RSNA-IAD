#!/usr/bin/env python3
"""
Minimal script to create YOLO classification dataset for view classification.
Classes: axial, sagittal, coronal
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import pydicom
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm

# Configuration
DATA_ROOT = Path("./data")
SERIES_ROOT = DATA_ROOT / "series"
OUTPUT_DIR = DATA_ROOT / "yolo_view_cls"
SERIES_VIEWS_CSV = DATA_ROOT / "series_views.csv"
TRAIN_CSV = DATA_ROOT / "train.csv"
N_FOLDS = 2
SEED = 42
IMAGES_PER_SERIES = 10
AXIAL_DOWNSAMPLE_RATIO = 0.1  # Keep 10% of axial samples

random.seed(SEED)
np.random.seed(SEED)

VIEW_TO_IDX = {"axial": 0, "sagittal": 1, "coronal": 2}


def read_dicom_frame(dcm_path: Path, frame_idx: int = 0) -> np.ndarray | None:
    """Read single frame from DICOM and normalize to uint8."""
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        pix = ds.pixel_array
        
        # Handle different DICOM formats
        if pix.ndim == 3 and pix.shape[-1] == 3:  # RGB
            img = cv2.cvtColor(pix, cv2.COLOR_RGB2GRAY)
        elif pix.ndim == 3:  # Multi-frame
            if frame_idx < pix.shape[0]:
                img = pix[frame_idx]
            else:
                return None
        else:  # 2D
            img = pix
        
        # Normalize to 0-255
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        img = img.astype(np.uint8)
        
        # Resize to 512x512 for YOLO
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        return img
    except Exception as e:
        print(f"Error reading {dcm_path}: {e}")
        return None


def extract_frames_from_series(series_dir: Path, n_frames: int = 10) -> list[np.ndarray]:
    """Extract uniformly spaced frames from a DICOM series."""
    dcm_files = sorted(list(series_dir.glob("*.dcm")))
    if not dcm_files:
        return []
    
    frames = []
    
    # Try to read total frames from first file
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), force=True)
        pix = ds.pixel_array
        
        if pix.ndim == 3 and pix.shape[-1] != 3:  # Multi-frame DICOM
            total_frames = pix.shape[0]
            indices = np.linspace(0, total_frames - 1, min(n_frames, total_frames), dtype=int)
            for idx in indices:
                frame = read_dicom_frame(dcm_files[0], frame_idx=int(idx))
                if frame is not None:
                    frames.append(frame)
        else:  # Multiple single-frame DICOMs
            indices = np.linspace(0, len(dcm_files) - 1, min(n_frames, len(dcm_files)), dtype=int)
            for idx in indices:
                frame = read_dicom_frame(dcm_files[int(idx)])
                if frame is not None:
                    frames.append(frame)
    except Exception as e:
        print(f"Error processing series {series_dir.name}: {e}")
    
    return frames


def create_dataset():
    """Main function to create YOLO classification dataset."""
    # Load series views
    views_df = pd.read_csv(SERIES_VIEWS_CSV)
    
    # Filter for valid views
    views_df = views_df[views_df['view'].isin(['axial', 'sagittal', 'coronal'])].copy()
    views_df['SeriesInstanceUID'] = views_df['SeriesInstanceUID'].astype(str)
    
    # Load train.csv for additional stratification features
    train_df = pd.read_csv(TRAIN_CSV)
    train_df['SeriesInstanceUID'] = train_df['SeriesInstanceUID'].astype(str)
    
    # Merge view info with train metadata (keep only series that exist in both)
    views_df = views_df.merge(
        train_df[['SeriesInstanceUID', 'Modality', 'Aneurysm Present']].drop_duplicates(subset=['SeriesInstanceUID']),
        on='SeriesInstanceUID',
        how='inner'
    )
    
    print(f"Total series with train info: {len(views_df)}")
    print(f"View distribution:\n{views_df['view'].value_counts()}")
    print(f"Modality distribution:\n{views_df['Modality'].value_counts()}")
    print(f"Aneurysm Present distribution:\n{views_df['Aneurysm Present'].value_counts()}")
    
    # Prepare for stratified split on FULL dataset
    series_ids = views_df['SeriesInstanceUID'].values
    
    # Create multi-label matrix for stratification
    # Include: view (one-hot), modality (one-hot), aneurysm present
    view_onehot = pd.get_dummies(views_df['view'], prefix='view').astype(int)
    modality_onehot = pd.get_dummies(views_df['Modality'], prefix='mod').astype(int)
    aneurysm_present = views_df[['Aneurysm Present']].astype(int)
    
    y_stratify = pd.concat([view_onehot, modality_onehot, aneurysm_present], axis=1).values
    
    print(f"\nStratification matrix shape: {y_stratify.shape}")
    
    # Split into train and test using first fold
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(mskf.split(series_ids, y_stratify))
    
    print(f"\nBefore downsampling:")
    print(f"Train: {len(train_idx)} series, Test: {len(test_idx)} series")
    
    # Downsample axial in each split separately
    train_df = views_df.iloc[train_idx].copy()
    test_df = views_df.iloc[test_idx].copy()
    
    # Downsample axial in train
    train_axial = train_df[train_df['view'] == 'axial'].sample(
        frac=AXIAL_DOWNSAMPLE_RATIO, random_state=SEED
    )
    train_other = train_df[train_df['view'] != 'axial']
    train_df = pd.concat([train_axial, train_other], ignore_index=True)
    
    # Downsample axial in test
    test_axial = test_df[test_df['view'] == 'axial'].sample(
        frac=AXIAL_DOWNSAMPLE_RATIO, random_state=SEED
    )
    test_other = test_df[test_df['view'] != 'axial']
    test_df = pd.concat([test_axial, test_other], ignore_index=True)
    
    print(f"\nAfter downsampling axial to {AXIAL_DOWNSAMPLE_RATIO*100:.0f}%:")
    print(f"Train: {len(train_df)} series")
    print(f"  View distribution:\n{train_df['view'].value_counts()}")
    print(f"  Aneurysm Present: {train_df['Aneurysm Present'].sum()}/{len(train_df)} ({train_df['Aneurysm Present'].mean()*100:.1f}%)")
    print(f"  Modality distribution:\n{train_df['Modality'].value_counts()}")
    
    print(f"\nTest: {len(test_df)} series")
    print(f"  View distribution:\n{test_df['view'].value_counts()}")
    print(f"  Aneurysm Present: {test_df['Aneurysm Present'].sum()}/{len(test_df)} ({test_df['Aneurysm Present'].mean()*100:.1f}%)")
    print(f"  Modality distribution:\n{test_df['Modality'].value_counts()}")
    
    splits = {
        'train': (train_df, 'train'),
        'test': (test_df, 'test')
    }
    
    # Create output directories
    for split in ['train', 'test']:
        for view in ['axial', 'sagittal', 'coronal']:
            (OUTPUT_DIR / split / view).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, (split_df, split_label) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        total_images = 0
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"{split_name}"):
            series_id = row['SeriesInstanceUID']
            view = row['view']
            series_dir = SERIES_ROOT / series_id
            
            if not series_dir.exists():
                continue
            
            # Extract frames
            frames = extract_frames_from_series(series_dir, IMAGES_PER_SERIES)
            
            # Save frames
            for i, frame in enumerate(frames):
                out_path = OUTPUT_DIR / split_label / view / f"{series_id}_{i:02d}.jpg"
                cv2.imwrite(str(out_path), frame)
                total_images += 1
        
        print(f"{split_name}: Saved {total_images} images")
    
    # Create YAML config (optional - can also use directory path directly)
    yaml_content = f"""# YOLO Classification Dataset - View Classification
# Note: You can use the directory path directly without this YAML
path: {OUTPUT_DIR.absolute()}
train: train
val: test

names:
  0: axial
  1: sagittal
  2: coronal
"""
    
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nDataset YAML saved to: {yaml_path}")
    print(f"Dataset directory: {OUTPUT_DIR.absolute()}")
    print("\nYou can train using either:")
    print(f"  1. Directory path: {OUTPUT_DIR.absolute()}")
    print(f"  2. YAML file: {yaml_path}")
    print("\nDone!")


if __name__ == "__main__":
    create_dataset()

