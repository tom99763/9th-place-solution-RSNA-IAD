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

def get_original_dicom_dimensions(data_dir: str, uid: str, sop_id: str):
    """
    Get original DICOM image dimensions before any processing
    Args:
        data_dir: Base data directory
        uid: SeriesInstanceUID
        sop_id: SOPInstanceUID
    Returns:
        tuple: (width, height) of original DICOM image
    """
    try:
        dcm_path = Path(data_dir) / "series" / uid / f"{sop_id}.dcm"
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array
        
        if img.ndim == 3 and img.shape[-1] == 3:
            # RGB image, use first two dimensions
            return img.shape[1], img.shape[0]  # width, height
        elif img.ndim == 3:
            # Multi-slice, use first slice dimensions
            return img.shape[2], img.shape[1]  # width, height  
        else:
            # 2D image
            return img.shape[1], img.shape[0]  # width, height
    except Exception as e:
        print(f"Error reading DICOM dimensions for {uid}/{sop_id}: {e}")
        return 640, 640  # fallback to default

def create_yolo_multiclass_dataset(data_dir: str, output_dir: str, img_size: int = 640, mode: str = "2.5D"):
    """
    Create YOLO format dataset for object detection with 13 aneurysm region classes
    Args:
        data_dir: Directory containing NPZ files and label CSVs
        output_dir: Directory to save YOLO dataset
        img_size: Target image size (default 640x640)
        mode: "2D" or "2.5D" dataset mode
    """
    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    # Read label data
    label_df = pd.read_csv(Path(data_dir) / "label_df.csv")
    train_df = pd.read_csv(Path(data_dir) / "train_df.csv")

    # Filter for positive samples only
    positive_uids = train_df[train_df["Aneurysm Present"] == 1]["SeriesInstanceUID"].unique()
    print(f"Found {len(positive_uids)} series with aneurisms")

    # Track statistics for each class
    class_counts = {split: {class_idx: 0 for class_idx in LABELS_TO_IDX.values()} for split in ['train', 'val']}
    total_images = {'train': 0, 'val': 0}
    total_annotations = {'train': 0, 'val': 0}

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
                
                # Stack three slices as RGB channels
                r_channel = volume[prev_slice]
                g_channel = volume[curr_slice] 
                b_channel = volume[next_slice]
                
                # Combine into 3-channel image (H, W, 3)
                img = np.stack([r_channel, g_channel, b_channel], axis=-1)
            
            # Resize image if needed
            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = cv2.resize(img, (img_size, img_size))

            # Save image
            img_path = output_dir / f"images/{split}/{uid}_{slice_idx}.png"
            cv2.imwrite(str(img_path), img)
            total_images[split] += 1

            # Create label file
            label_path = output_dir / f"labels/{split}/{uid}_{slice_idx}.txt"
            
            # Get annotations for this slice
            slice_labels = series_labels[series_labels["z"] == slice_idx]
            
            with open(label_path, 'w') as f:
                # Process each annotation for this slice
                for _, row in slice_labels.iterrows():
                    location = row["location"]
                    
                    # Map location to class index
                    if location not in LABELS_TO_IDX:
                        print(f"Warning: Unknown location '{location}' for {uid}_{slice_idx}")
                        continue
                    
                    class_idx = LABELS_TO_IDX[location]
                    
                    # Get original DICOM dimensions for this specific slice
                    sop_id = label_df[
                        (label_df["SeriesInstanceUID"] == uid) & 
                        (label_df["z"] == slice_idx)
                    ]["SOPInstanceUID"].iloc[0]
                    
                    orig_width, orig_height = get_original_dicom_dimensions(data_dir, uid, sop_id)
                    
                    # Scale coordinates from original DICOM dimensions to target image size
                    x_scaled = row["x"] * (img_size / orig_width)
                    y_scaled = row["y"] * (img_size / orig_height)
                    
                    # Convert to YOLO format (normalized 0-1)
                    x_norm = x_scaled / img_size
                    y_norm = y_scaled / img_size
                    
                    # YOLO format: <class> <x_center> <y_center> <width> <height>
                    # Using fixed size for bounding box, scaled to target image size
                    bbox_size = 48  # Original bbox size in pixels
                    width = bbox_size / img_size  # Normalize to image size
                    height = bbox_size / img_size  # Normalize to image size
                    
                    # Write annotation with class index for the anatomical region
                    f.write(f"{class_idx} {x_norm} {y_norm} {width} {height}\n")
                    
                    # Update statistics
                    class_counts[split][class_idx] += 1
                    total_annotations[split] += 1

    # Print statistics
    print("\n=== Dataset Statistics ===")
    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET:")
        print(f"Total images: {total_images[split]}")
        print(f"Total annotations: {total_annotations[split]}")
        for class_name, class_idx in LABELS_TO_IDX.items():
            count = class_counts[split][class_idx]
            print(f"  Class {class_idx} ({class_name}): {count} annotations")

    # Create dataset.yaml for multi-class object detection
    yaml_content = f"""# YOLO Multi-class Object Detection Dataset Configuration
# Dataset path (relative to this file)
path: {output_dir}
train: images/train
val: images/val

# Number of classes
nc: {len(LABELS_TO_IDX)}

# Class names (anatomical regions for aneurysm detection)
names:
"""
    
    for class_idx in sorted(LABELS_TO_IDX.values()):
        class_name = [name for name, idx in LABELS_TO_IDX.items() if idx == class_idx][0]
        yaml_content += f"  {class_idx}: {class_name}\n"
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset configuration saved to: {output_dir}/dataset.yaml")
    print(f"Multi-class object detection dataset created in: {output_dir}")

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
    output_dir = "/home/sersasj/RSNA-IAD-Codebase/yolo_multiclass_dataset_2.5d"
    
    # First analyze the class distribution
    print("Analyzing class distribution...")
    analyze_class_distribution(data_dir)
    
    # Choose mode: "2D" or "2.5D"
    mode = "2.5D"  # Default to 2.5D as requested
    
    print(f"\nCreating {mode} multi-class object detection dataset...")
    create_yolo_multiclass_dataset(data_dir, output_dir, mode=mode)