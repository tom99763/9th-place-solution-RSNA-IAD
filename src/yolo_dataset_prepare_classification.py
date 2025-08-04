import pandas as pd
import numpy as np
import os
from pathlib import Path
import cv2
from tqdm import tqdm
import sys
import pydicom
sys.path.append('.')
from configs.data_config import LABELS_TO_IDX

def create_yolo_classification_dataset(data_dir: str, output_dir: str, img_size: int = 640, mode: str = "2D"):
    """
    Create YOLO classification dataset from NPZ files and labels for 13 anatomical regions
    Args:
        data_dir: Directory containing NPZ files and label CSVs
        output_dir: Directory to save YOLO classification dataset
        img_size: Target image size (default 640x640)
    """
    # Create output directories for classification
    output_dir = Path(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each class
    for class_name in LABELS_TO_IDX.keys():
        class_idx = LABELS_TO_IDX[class_name]
        (output_dir / f"train/{class_idx}").mkdir(parents=True, exist_ok=True)
        (output_dir / f"val/{class_idx}").mkdir(parents=True, exist_ok=True)

    # Read label data
    label_df = pd.read_csv(Path(data_dir) / "label_df.csv")
    train_df = pd.read_csv(Path(data_dir) / "train_df.csv")

    # Filter for positive samples only
    positive_uids = train_df[train_df["Aneurysm Present"] == 1]["SeriesInstanceUID"].unique()
    print(f"Found {len(positive_uids)} series with aneurisms")

    # Track statistics
    class_counts = {split: {class_idx: 0 for class_idx in LABELS_TO_IDX.values()} for split in ['train', 'val']}
    total_images = {'train': 0, 'val': 0}

    # Process each positive series
    for uid in tqdm(positive_uids, desc="Processing series"):
        # Determine split (train/val) based on fold_id
        fold = train_df[train_df["SeriesInstanceUID"] == uid]["fold_id"].iloc[0]
        split = "val" if fold == 0 else "train"  # Using fold 0 as validation

        # Get labels for this series
        series_labels = label_df[label_df["SeriesInstanceUID"] == uid]
        
        # Get unique slices that contain aneurisms
        positive_slices = series_labels["z"].unique()
        
        # Load volume
        try:
            with np.load(f"{data_dir}/processed/slices/{uid}.npz") as data:
                volume = data['vol']
        except Exception as e:
            print(f"Error loading {uid}: {e}")
            continue

        # Process only slices with aneurisms
        for slice_idx in positive_slices:
            if slice_idx >= len(volume):
                print(f"Warning: Slice index {slice_idx} out of bounds for volume with {len(volume)} slices")
                continue
                
            # Get slice image
            if mode == "2D":
                img = volume[slice_idx]
            elif mode == "2.5D":
                # Create 2.5D image: R=slice-1, G=slice, B=slice+1
                prev_slice = max(0, slice_idx - 1)  # Handle edge case for first slice
                curr_slice = slice_idx
                next_slice = min(len(volume) - 1, slice_idx + 1)  # Handle edge case for last slice
                img = np.stack([volume[prev_slice], volume[curr_slice], volume[next_slice]], axis=-1)
            
            # Resize image if needed
            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = cv2.resize(img, (img_size, img_size))

            # Get all aneurysms for this slice
            slice_labels = series_labels[series_labels["z"] == slice_idx]
            
            # For each aneurysm in this slice, create a separate classified image
            for idx, (_, row) in enumerate(slice_labels.iterrows()):
                location = row["location"]
                
                # Map location to class index
                if location not in LABELS_TO_IDX:
                    print(f"Warning: Unknown location '{location}' for {uid}_{slice_idx}")
                    continue
                
                class_idx = LABELS_TO_IDX[location]
                
                # Create unique filename for this aneurysm
                img_name = f"{uid}_{slice_idx}_{idx}.png"
                img_path = output_dir / f"{split}/{class_idx}/{img_name}"
                
                # Save image in the appropriate class directory
                cv2.imwrite(str(img_path), img)
                
                # Update statistics
                class_counts[split][class_idx] += 1
                total_images[split] += 1

    # Print statistics
    print("\n=== Dataset Statistics ===")
    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET:")
        print(f"Total images: {total_images[split]}")
        for class_name, class_idx in LABELS_TO_IDX.items():
            count = class_counts[split][class_idx]
            print(f"  Class {class_idx} ({class_name}): {count} images")

    # Create dataset.yaml for classification
    class_names = {str(idx): name for name, idx in LABELS_TO_IDX.items()}
    
    yaml_content = f"""# YOLO Classification Dataset Configuration
# Dataset path (relative to this file)
path: {output_dir}
train: train
val: val

# Number of classes
nc: {len(LABELS_TO_IDX)}

# Class names (anatomical regions)
names:
"""
    
    for class_idx in sorted(LABELS_TO_IDX.values()):
        class_name = [name for name, idx in LABELS_TO_IDX.items() if idx == class_idx][0]
        yaml_content += f"  {class_idx}: {class_name}\n"
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset configuration saved to: {output_dir}/dataset.yaml")
    print(f"Classification dataset created in: {output_dir}")

def analyze_class_distribution(data_dir: str):
    """
    Analyze the distribution of aneurysm locations in the dataset
    """
    label_df = pd.read_csv(Path(data_dir) / "label_df.csv")
    train_df = pd.read_csv(Path(data_dir) / "train_df.csv")
    
    # Filter for positive samples only
    positive_uids = train_df[train_df["Aneurysm Present"] == 1]["SeriesInstanceUID"].unique()
    positive_labels = label_df[label_df["SeriesInstanceUID"].isin(positive_uids)]
    
    print("=== Aneurysm Location Distribution ===")
    location_counts = positive_labels["location"].value_counts()
    
    for location, count in location_counts.items():
        class_idx = LABELS_TO_IDX.get(location, "Unknown")
        print(f"Class {class_idx}: {location} - {count} aneurysms")
    
    print(f"\nTotal aneurysms: {len(positive_labels)}")
    print(f"Total unique locations: {len(location_counts)}")

if __name__ == "__main__":
    data_dir = "data"
    output_dir = "/home/sersasj/RSNA-IAD-Codebase/yolo_classification_dataset"
    
    # First analyze the class distribution
    print("Analyzing class distribution...")
    analyze_class_distribution(data_dir)
    mode = "2.5D"
    if mode == "2.5D":
        output_dir = "/home/sersasj/RSNA-IAD-Codebase/yolo_classification_dataset_2.5d"
    
    print("\nCreating classification dataset...")
    create_yolo_classification_dataset(data_dir, output_dir, mode=mode)