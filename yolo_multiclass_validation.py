#!/usr/bin/env python3
"""
YOLO Multiclass Validation Script
Validates the YOLO11 multiclass detection model on 2.5D medical images
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import argparse
import sys
sys.path.append('.')
from configs.data_config import LABELS_TO_IDX


def create_25d_image(volume, slice_idx, img_size=640):
    """
    Create 2.5D image same as training: R=slice-1, G=slice, B=slice+1
    Args:
        volume: 3D numpy array of shape (D, H, W)
        slice_idx: Index of current slice
        img_size: Target image size
    Returns:
        3-channel image of shape (img_size, img_size, 3)
    """
    # Handle edge cases for first and last slice
    prev_slice = max(0, slice_idx - 1)
    curr_slice = slice_idx
    next_slice = min(len(volume) - 1, slice_idx + 1)
    
    # Stack three slices as RGB channels
    r_channel = volume[prev_slice]
    g_channel = volume[curr_slice] 
    b_channel = volume[next_slice]
    
    # Combine into 3-channel image (H, W, 3)
    img = np.stack([r_channel, g_channel, b_channel], axis=-1)
    
    # Resize if needed
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size))
    
    return img

def process_yolo_predictions(results, confidence_threshold=0.005):
    """
    Process YOLO detection results to extract class predictions and confidence scores
    Args:
        results: YOLO prediction results
        confidence_threshold: Minimum confidence threshold for detections
    Returns:
        tuple: (has_detection, class_probs, max_confidence)
    """
    # Initialize class probabilities for all 13 classes
    class_probs = np.zeros(len(LABELS_TO_IDX))
    max_confidence = 0.0
    has_detection = False
    
    if results and len(results) > 0:
        result = results[0]  # First (and only) image
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Get detection boxes, confidences, and classes
            boxes = result.boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            
            # Filter by confidence threshold
            valid_detections = confidences >= confidence_threshold
            
            if np.any(valid_detections):
                has_detection = True
                valid_confidences = confidences[valid_detections]
                valid_classes = classes[valid_detections]
                
                # Aggregate confidence scores for each class (take maximum)
                for cls_idx, conf in zip(valid_classes, valid_confidences):
                    if cls_idx < len(class_probs):
                        class_probs[cls_idx] = max(class_probs[cls_idx], conf)
                
                max_confidence = np.max(valid_confidences)
    
    return has_detection, class_probs, max_confidence

def validate_series(model, volume, img_size=640, confidence_threshold=0.005, batch_size=32):
    """
    Run validation on a complete series (volume)
    Args:
        model: Loaded YOLO model
        volume: 3D numpy array of shape (D, H, W)
        img_size: Target image size
        confidence_threshold: Confidence threshold for detections
        batch_size: Batch size for inference
    Returns:
        tuple: (series_has_aneurysm, series_class_probs, max_slice_confidence)
    """
    all_class_probs = []
    all_confidences = []
    series_has_detection = False
    
    # Process volume slice by slice
    for slice_idx in range(len(volume)):
        # Create 2.5D image
        img_25d = create_25d_image(volume, slice_idx, img_size)
        
        # Convert to uint8 if needed
        if img_25d.dtype != np.uint8:
            img_25d = (img_25d * 255).astype(np.uint8)
        
        # Run YOLO inference
        results = model(img_25d, conf=confidence_threshold, verbose=False)
        
        # Process predictions
        has_detection, class_probs, max_conf = process_yolo_predictions(results, confidence_threshold)
        
        if has_detection:
            series_has_detection = True
            all_class_probs.append(class_probs)
            all_confidences.append(max_conf)
    
    # Aggregate slice-level predictions to series-level
    if len(all_class_probs) > 0:
        # Take maximum probability across all slices for each class
        series_class_probs = np.maximum.reduce(all_class_probs)
        max_series_confidence = np.max(all_confidences)
    else:
        series_class_probs = np.zeros(len(LABELS_TO_IDX))
        max_series_confidence = 0.0
    
    return series_has_detection, series_class_probs, max_series_confidence

def calculate_aneurysm_present_probability(class_probs):
    """
    Calculate aneurysm present probability from class probabilities
    Multiple strategies to try:
    """
    # Strategy 1: Max of all location classes
    aneurysm_prob_v1 = np.max(class_probs)
    
    # Strategy 2: Weighted sum (give more weight to common locations)
    # You could learn these weights from training data
    weights = np.ones(len(class_probs))  # Start with equal weights
    aneurysm_prob_v2 = np.sum(class_probs * weights) / np.sum(weights)
    
    # Strategy 3: Probability that at least one class is positive
    aneurysm_prob_v3 = 1 - np.prod(1 - class_probs)
    
    # Strategy 4: Use a learned threshold/combination
    # Train a simple model on validation data to combine class probs
    
    return aneurysm_prob_v1  # Start with v1, experiment with others

def validate_series_improved(model, volume, img_size=640, confidence_threshold=0.05):
    """
    Improved series validation with better aggregation
    """
    slice_class_probs = []
    slice_max_confs = []
    slice_detections = []
    
    # Process each slice
    for slice_idx in range(len(volume)):
        img_25d = create_25d_image(volume, slice_idx, img_size)
        if img_25d.dtype != np.uint8:
            img_25d = (img_25d * 255).astype(np.uint8)
        
        results = model(img_25d, conf=confidence_threshold, verbose=False)
        has_detection, class_probs, max_conf = process_yolo_predictions(results, confidence_threshold)
        
        slice_class_probs.append(class_probs)
        slice_max_confs.append(max_conf)
        slice_detections.append(has_detection)
    
    slice_class_probs = np.array(slice_class_probs)
    slice_max_confs = np.array(slice_max_confs)
    
    # Multiple aggregation strategies
    # Strategy 1: Max across slices (current)
    series_class_probs_max = np.max(slice_class_probs, axis=0)
    
    # Strategy 2: 95th percentile (less sensitive to outliers)
    series_class_probs_p95 = np.percentile(slice_class_probs, 95, axis=0)
    
    # Strategy 3: Mean of top K slices
    k = min(5, len(volume))  # Top 5 slices or all if fewer
    top_k_indices = np.argsort(slice_max_confs)[-k:]
    series_class_probs_topk = np.mean(slice_class_probs[top_k_indices], axis=0)
    
    # Strategy 4: Weighted average (weight by confidence)
    weights = slice_max_confs / (np.sum(slice_max_confs) + 1e-8)
    series_class_probs_weighted = np.sum(slice_class_probs * weights[:, np.newaxis], axis=0)
    
    # Choose your aggregation strategy
    series_class_probs = series_class_probs_p95  # Try this first
    
    # Calculate aneurysm present probability
    aneurysm_present_prob = calculate_aneurysm_present_probability(series_class_probs)
    
    return any(slice_detections), series_class_probs, aneurysm_present_prob

def prepare_ground_truth_labels(df, uid):
    """
    Prepare ground truth labels for a series
    Args:
        df: DataFrame with ground truth labels
        uid: SeriesInstanceUID
    Returns:
        tuple: (aneurysm_present, location_labels)
    """
    row = df[df["SeriesInstanceUID"] == uid].iloc[0]
    
    # Binary classification: aneurysm present/absent
    aneurysm_present = int(row["Aneurysm Present"])
    
    # Multi-class location labels
    location_labels = np.zeros(len(LABELS_TO_IDX))
    for location, class_idx in LABELS_TO_IDX.items():
        if location in row and pd.notna(row[location]):
            location_labels[class_idx] = int(row[location])
    
    return aneurysm_present, location_labels

def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """
    Calculate comprehensive metrics
    """
    print("\n=== METRICS ===")
    
    # Binary classification metrics (aneurysm detection)
    if len(np.unique(y_true)) > 1:
        binary_auc = roc_auc_score(y_true, y_probs)
        print(f"Binary AUC (Aneurysm Detection): {binary_auc:.4f}")
    else:
        print("Binary AUC: Cannot calculate (only one class present)")
    
    # Binary classification report
    print("\nBinary Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Aneurysm', 'Aneurysm Present']))
    
    return binary_auc if len(np.unique(y_true)) > 1 else None

def calculate_location_metrics(y_true_locations, y_pred_locations, class_names):
    """
    Calculate location-specific metrics
    """
    print("\n=== LOCATION METRICS ===")
    
    location_aucs = []
    for i, class_name in enumerate(class_names):
        if len(np.unique(y_true_locations[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true_locations[:, i], y_pred_locations[:, i])
                location_aucs.append(auc)
                print(f"Class {i} ({class_name}): AUC = {auc:.4f}")
            except ValueError:
                print(f"Class {i} ({class_name}): AUC = N/A (insufficient data)")
        else:
            print(f"Class {i} ({class_name}): AUC = N/A (only one class)")
    
    if location_aucs:
        mean_auc = np.mean(location_aucs)
        print(f"\nMean Location AUC: {mean_auc:.4f}")
        return mean_auc
    else:
        print("\nMean Location AUC: N/A")
        return None

def main():
    parser = argparse.ArgumentParser(description='YOLO Multiclass Validation')
    parser.add_argument('--model_path', type=str, 
                       default='rsna-iad-multiclass-detection/yolo11s-multiclass-detection-v0_25d/weights/best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--fold_id', type=int, default=0, help='Validation fold ID')
    parser.add_argument('--confidence_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    
    args = parser.parse_args()
    
    print("🔬 YOLO Multiclass Validation")
    print(f"Model: {args.model_path}")
    print(f"Fold: {args.fold_id}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # Load YOLO model
    print("📥 Loading YOLO model...")
    model = YOLO(args.model_path)
    
    # Load data
    print("📊 Loading validation data...")
    data_path = Path(args.data_dir)
    df = pd.read_csv(data_path / "train_df.csv")
    
    # Get validation UIDs
    val_uids = list(df[df["fold_id"] == args.fold_id]["SeriesInstanceUID"])
    print(f"Found {len(val_uids)} validation series")
    
    # Initialize result storage
    y_true_binary = []  # Aneurysm present/absent
    y_pred_binary = []  # Predicted aneurysm present/absent
    y_prob_binary = []  # Predicted aneurysm probability (max confidence)
    
    y_true_locations = []  # Location labels (13 classes)
    y_pred_locations = []  # Predicted location probabilities
    
    # Class names for reporting
    class_names = [name for name, _ in sorted(LABELS_TO_IDX.items(), key=lambda x: x[1])]
    
    # Validate each series
    print("🔍 Running validation...")
    for uid in tqdm(val_uids, desc="Processing series"):
        try:
            # Load volume
            volume_path = data_path / "processed" / "slices" / f"{uid}.npz"
            if not volume_path.exists():
                print(f"Warning: Volume not found for {uid}")
                continue
                
            with np.load(volume_path) as data:
                volume = data['vol'].astype(np.float32)
            
            # Get ground truth labels
            aneurysm_present, location_labels = prepare_ground_truth_labels(df, uid)
            
            # Run validation on series
            has_detection, class_probs, max_confidence = validate_series_improved(
                model, volume, args.img_size, args.confidence_threshold
            )
            
            # Store results
            y_true_binary.append(aneurysm_present)
            y_pred_binary.append(int(has_detection))
            y_prob_binary.append(max_confidence)
            
            y_true_locations.append(location_labels)
            y_pred_locations.append(class_probs)
            
        except Exception as e:
            print(f"Error processing {uid}: {e}")
            continue
    
    # Convert to numpy arrays
    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)
    y_prob_binary = np.array(y_prob_binary)
    
    y_true_locations = np.stack(y_true_locations)
    y_pred_locations = np.stack(y_pred_locations)
    
    print(f"\n✅ Processed {len(y_true_binary)} series")
    
    # Calculate and display metrics
    binary_auc = calculate_metrics(y_true_binary, y_pred_binary, y_prob_binary, class_names)
    location_auc = calculate_location_metrics(y_true_locations, y_pred_locations, class_names)
    
    # Summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    print(f"Fold: {args.fold_id}")
    print(f"Series processed: {len(y_true_binary)}")
    print(f"Binary AUC (Aneurysm Detection): {binary_auc:.4f}" if binary_auc else "Binary AUC: N/A")
    print(f"Mean Location AUC: {location_auc:.4f}" if location_auc else "Mean Location AUC: N/A")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'SeriesInstanceUID': val_uids[:len(y_true_binary)],
        'true_aneurysm': y_true_binary,
        'pred_aneurysm': y_pred_binary,
        'pred_aneurysm_prob': y_prob_binary,
    })
    
    # Add location predictions
    for i, class_name in enumerate(class_names):
        results_df[f'true_{class_name}'] = y_true_locations[:, i]
        results_df[f'pred_{class_name}'] = y_pred_locations[:, i]
    
    output_path = f"yolo_validation_results_fold_{args.fold_id}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()

