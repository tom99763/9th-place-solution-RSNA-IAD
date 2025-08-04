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

def create_yolo_dataset(data_dir: str, output_dir: str, img_size: int = 640, mode: str = "2D"):
    """
    Create YOLO format dataset from NPZ files and labels, using only positive samples
    Args:
        data_dir: Directory containing NPZ files and label CSVs
        output_dir: Directory to save YOLO dataset
        img_size: Target image size (default 640x640)
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

    # Process each positive series
    for uid in tqdm(positive_uids):
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
            #if img.shape[0] != img_size:
            #    img = cv2.resize(img, (img_size, img_size))

            # Save image
            img_path = output_dir / f"images/{split}/{uid}_{slice_idx}.png"
            cv2.imwrite(str(img_path), img)

            # Create label file
            label_path = output_dir / f"labels/{split}/{uid}_{slice_idx}.txt"
            
            # Get annotations for this slice
            slice_labels = series_labels[series_labels["z"] == slice_idx]
            
            with open(label_path, 'w') as f:
                # For binary classification, we just need one class (0 for aneurism)
                # Process each annotation for this slice
                for _, row in slice_labels.iterrows():
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
                    
                    # Class 0 for aneurism
                    f.write(f"0 {x_norm} {y_norm} {width} {height}\n")
        #break
    # Create dataset.yaml
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

# Classes
names:
  0: aneurism  # Binary classification - just aneurism vs no aneurism
"""
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    data_dir = "data"
    output_dir = "/home/sersasj/RSNA-IAD-Codebase/yolo_dataset"
    
    # Choose mode: "2D" or "2.5D"
    mode = "2.5D"  # Change to "2.5D" for 2.5D dataset
    
    if mode == "2.5D":
        output_dir = "/home/sersasj/RSNA-IAD-Codebase/yolo_dataset_2.5d"
    
    create_yolo_dataset(data_dir, output_dir, mode=mode)
