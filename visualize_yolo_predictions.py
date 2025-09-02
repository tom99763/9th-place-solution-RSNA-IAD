"""Visualize YOLO predictions on all slices of a tomogram.

This script loads a specific series (tomogram), runs YOLO inference on all slices,
and creates a visualization showing each slice with bounding boxes overlaid.

Usage:
    python visualize_yolo_predictions.py --series-id <SeriesInstanceUID> [options]
"""
import argparse
from pathlib import Path
import sys
import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Add ultralytics path
sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

# Project root & config imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from configs.data_config import data_path


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize YOLO predictions on tomogram slices")
    ap.add_argument('--series-id', type=str, required=True, help='SeriesInstanceUID to visualize')
    ap.add_argument('--model-path', type=str, 
                    default="/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_binary/fold0_run_fold0/weights/best.pt",
                    help='Path to YOLO weights (.pt)')
    ap.add_argument('--conf-threshold', type=float, default=0.1, help='Confidence threshold for displaying boxes')
    ap.add_argument('--max-slices', type=int, default=0, help='Limit number of slices to display (0=all)')
    ap.add_argument('--cols', type=int, default=6, help='Number of columns in visualization grid')
    ap.add_argument('--save-path', type=str, default='', help='Path to save visualization image')
    ap.add_argument('--output-folder', type=str, default='yolo-predictions', help='Folder to save individual slice predictions')
    ap.add_argument('--slice-step', type=int, default=1, help='Show every Nth slice')
    return ap.parse_args()


def read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    """Return list of HU frames from a DICOM. Handles 2D, multi-frame, and RGB->grayscale."""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    frames: List[np.ndarray] = []
    
    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append(img * slope + intercept)
    elif pix.ndim == 3:
        # RGB or multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                frames.append(frm * slope + intercept)
    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Normalize HU image to uint8 [0, 255]."""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def collect_series_slices(series_dir: Path) -> List[Path]:
    """Collect all DICOM files in series directory and sort by spatial position."""
    dicom_files = list(series_dir.glob('*.dcm'))
    
    if not dicom_files:
        return []
    
    # First pass: collect all slices with their spatial information
    temp_slices = []
    for filepath in dicom_files:
        try:
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
            
            # Get z-position from ImagePositionPatient (most reliable)
            if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                z_val = float(ds.ImagePositionPatient[-1])  # z-coordinate
            else:
                # Fallback to InstanceNumber or filename
                z_val = float(getattr(ds, "InstanceNumber", 0))
            
            # Store filepath with its z-position for sorting
            temp_slices.append((z_val, filepath))
            
        except Exception as e:
            print(f"Error reading DICOM metadata from {filepath.name}: {e}")
            # Fallback: use filename as last resort
            temp_slices.append((str(filepath.name), filepath))
            continue
    
    if not temp_slices:
        return []
    
    # Sort slices by z-position (spatial order)
    temp_slices.sort(key=lambda x: x[0])
    
    # Extract the sorted filepaths
    sorted_files = [item[1] for item in temp_slices]
    return sorted_files


def run_yolo_on_slice(model: YOLO, img_hu: np.ndarray, conf_threshold: float = 0.1) -> Tuple[np.ndarray, List[dict]]:
    """Run YOLO inference on a single slice and return image + detections."""
    # Normalize to uint8 and convert to BGR for YOLO
    img_uint8 = min_max_normalize(img_hu)
    if img_uint8.ndim == 2:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_uint8
    
    # Run inference
    results = model.predict(img_bgr, verbose=False, conf=conf_threshold)
    
    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else np.zeros(len(boxes))
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                detections.append({
                    'bbox': box,  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class': int(cls)
                })
    
    return img_uint8, detections


def plot_slice_with_predictions(ax, img: np.ndarray, detections: List[dict], slice_idx: int, 
                               max_conf: float, title_info: str = "", ground_truths: List[Tuple[float, float]] = None):
    """Plot a single slice with bounding box overlays."""
    if ground_truths is None:
        ground_truths = []
    
    # Display image in grayscale
    ax.imshow(img, cmap='gray', aspect='equal')
    ax.set_title(f'Slice {slice_idx}{title_info}\nMax conf: {max_conf:.3f}', fontsize=10)
    ax.axis('off')
    
    # Draw bounding boxes
    for det in detections:
        bbox = det['bbox']  # [x1, y1, x2, y2]
        conf = det['confidence']
        
        # Create rectangle patch
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Color by confidence (red = high confidence)
        color = plt.cm.Reds(min(conf, 1.0))
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add confidence text
        ax.text(x1, y1 - 5, f'{conf:.3f}', fontsize=8, color=color, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Draw ground truth X marks
    for i, (x, y) in enumerate(ground_truths):
        # Plot X
        ax.plot([x-10, x+10], [y-10, y+10], 'g-', linewidth=3)
        ax.plot([x-10, x+10], [y+10, y-10], 'g-', linewidth=3)
        
        # Add label
        label = f'GT{i}' if len(ground_truths) > 1 else 'GT'
        ax.text(x+15, y, label, fontsize=10, color='green', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


def main():
    args = parse_args()
    
    # Load data and model
    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_df = pd.read_csv(data_root / 'train_df.csv') if (data_root / 'train_df.csv').exists() else pd.read_csv(data_root / 'train.csv')
    label_df = pd.read_csv(data_root / 'label_df.csv')
    
    # Check if series exists
    series_dir = series_root / args.series_id
    if not series_dir.exists():
        raise SystemExit(f"Series directory not found: {series_dir}")
    
    # Load YOLO model
    model = YOLO(args.model_path)
    print(f"Loaded YOLO model from: {args.model_path}")
    
    # Get series info from train_df
    series_info = train_df[train_df['SeriesInstanceUID'] == args.series_id]
    if len(series_info) > 0:
        row = series_info.iloc[0]
        aneurysm_present = int(row.get('Aneurysm Present', -1))
        title_info = f" (GT: {'Pos' if aneurysm_present == 1 else 'Neg' if aneurysm_present == 0 else 'Unknown'})"
    else:
        title_info = " (GT: Unknown)"
    
    # Load ground truth locations for this series
    ground_truths = {}
    series_labels = label_df[label_df['SeriesInstanceUID'] == args.series_id]
    for _, row in series_labels.iterrows():
        slice_num = int(row['z'])
        x = float(row['x'])
        y = float(row['y'])
        if slice_num not in ground_truths:
            ground_truths[slice_num] = []
        ground_truths[slice_num].append((x, y))
    print(f"Loaded {len(series_labels)} ground truth annotations for series {args.series_id}")
    
    # Collect and process slices
    dicoms = collect_series_slices(series_dir)
    if not dicoms:
        raise SystemExit(f"No DICOM files found in {series_dir}")
    
    print(f"Found {len(dicoms)} DICOM files in series {args.series_id}")
    
    # Process slices and collect results
    slice_results = []
    all_confidences = []
    
    for i, dcm_path in enumerate(dicoms[::args.slice_step]):
        try:
            frames = read_dicom_frames_hu(dcm_path)
            # Get instance number for ground truth matching
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            instance_number = int(getattr(ds, 'InstanceNumber', 0))
        except Exception as e:
            print(f"[SKIP] {dcm_path.name}: {e}")
            continue
            
        for frame_idx, frame in enumerate(frames):
            img_uint8, detections = run_yolo_on_slice(model, frame, args.conf_threshold)
            
            # Calculate max confidence for this slice
            max_conf = max([d['confidence'] for d in detections], default=0.0)
            all_confidences.append(max_conf)
            
            slice_results.append({
                'slice_idx': len(slice_results),
                'dcm_path': dcm_path,
                'frame_idx': frame_idx,
                'image': img_uint8,
                'detections': detections,
                'max_conf': max_conf,
                'instance_number': instance_number
            })
            
            result = slice_results[-1]  # Get the just appended result
            if result['slice_idx'] in ground_truths:
                print(f"Ground truth on slice {result['slice_idx']} (instance {instance_number}): {ground_truths[result['slice_idx']]}")
            
            if args.max_slices > 0 and len(slice_results) >= args.max_slices:
                break
        
        if args.max_slices > 0 and len(slice_results) >= args.max_slices:
            break
    
    if not slice_results:
        raise SystemExit("No slices could be processed")
    
    print(f"Processed {len(slice_results)} slices")
    print(f"Overall max confidence: {max(all_confidences):.4f}")
    print(f"Mean confidence: {np.mean(all_confidences):.4f}")
    print(f"Detections found in {sum(1 for r in slice_results if r['detections'])} slices")
    
    # Save individual slice predictions
    if args.output_folder:
        output_dir = Path(args.output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(slice_results):
            fig, ax = plt.subplots(figsize=(8, 8))
            gts = ground_truths.get(result['slice_idx'], [])
            plot_slice_with_predictions(
                ax, result['image'], result['detections'], 
                result['slice_idx'], result['max_conf'], title_info, gts
            )
            plt.tight_layout()
            
            # Save individual slice
            slice_filename = f"slice_{result['slice_idx']:03d}_conf_{result['max_conf']:.3f}.png"
            slice_path = output_dir / slice_filename
            plt.savefig(slice_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        print(f"Saved {len(slice_results)} slice predictions to: {output_dir}")
        
        # Also create and save the grid visualization
        n_slices = len(slice_results)
        cols = args.cols
        rows = math.ceil(n_slices / cols)
        
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        gs = GridSpec(rows, cols, figure=fig)
        
        # Add overall title
        overall_max_conf = max(all_confidences)
        fig.suptitle(f'YOLO Predictions for Series: {args.series_id}{title_info}\n'
                     f'Overall Max Confidence: {overall_max_conf:.4f}, '
                     f'Conf Threshold: {args.conf_threshold}, '
                     f'Total Slices: {n_slices}', 
                     fontsize=16, y=0.98)
        
        # Plot each slice
        for i, result in enumerate(slice_results):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])
            
            gts = ground_truths.get(result['slice_idx'], [])
            plot_slice_with_predictions(
                ax, result['image'], result['detections'], 
                result['slice_idx'], result['max_conf'], title_info, gts
            )
        
        # Hide empty subplots
        for i in range(n_slices, rows * cols):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save the grid
        grid_filename = f"grid_all_slices_conf_{args.conf_threshold:.1f}.png"
        grid_path = output_dir / grid_filename
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved grid visualization to: {grid_path}")
    



if __name__ == '__main__':
    main()