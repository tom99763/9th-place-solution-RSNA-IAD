#!/usr/bin/env python3
"""
YOLO Binary Aneurysm Detection Validation Script
Validates the YOLO11 binary detection model on 2.5D medical images
Tests only aneurysm present/absent classification
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg')  

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

def process_binary_yolo_predictions(results, confidence_threshold=0.25):
    """
    Process YOLO binary detection results
    Args:
        results: YOLO prediction results
        confidence_threshold: Minimum confidence threshold for detections
    Returns:
        tuple: (has_aneurysm, max_confidence, num_detections)
    """
    max_confidence = 0.0
    num_detections = 0
    has_aneurysm = False
    
    if results and len(results) > 0:
        result = results[0]  # First (and only) image
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Get detection boxes and confidences
            boxes = result.boxes
            confidences = boxes.conf.cpu().numpy()
            
            # Filter by confidence threshold
            valid_detections = confidences >= confidence_threshold
            
            if np.any(valid_detections):
                has_aneurysm = True
                valid_confidences = confidences[valid_detections]
                num_detections = len(valid_confidences)
                max_confidence = np.max(valid_confidences)
    
    return has_aneurysm, max_confidence, num_detections

def validate_series_binary(model, volume, img_size=640, confidence_threshold=0.25):
    """
    Run binary validation on a complete series (volume)
    Args:
        model: Loaded YOLO model
        volume: 3D numpy array of shape (D, H, W)
        img_size: Target image size
        confidence_threshold: Confidence threshold for detections
    Returns:
        tuple: (series_has_aneurysm, max_series_confidence, total_detections)
    """
    all_confidences = []
    total_detections = 0
    series_has_aneurysm = False
    
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
        has_aneurysm, max_conf, num_det = process_binary_yolo_predictions(results, confidence_threshold)
        
        if has_aneurysm:
            series_has_aneurysm = True
            all_confidences.append(max_conf)
            total_detections += num_det
    
    # Aggregate slice-level predictions to series-level
    if len(all_confidences) > 0:
        max_series_confidence = np.max(all_confidences)
    else:
        max_series_confidence = 0.0
    
    return series_has_aneurysm, max_series_confidence, total_detections

def plot_metrics(y_true, y_probs, output_dir="./"):
    """
    Plot ROC curve and precision-recall curve
    """
    from sklearn.metrics import roc_curve, auc
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Aneurysm Detection')
    plt.legend(loc="lower right")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Binary Aneurysm Detection')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/binary_validation_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc, pr_auc

def plot_confusion_matrix(y_true, y_pred, output_dir="./"):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Aneurysm', 'Aneurysm Present'],
                yticklabels=['No Aneurysm', 'Aneurysm Present'])
    plt.title('Confusion Matrix - Binary Aneurysm Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/binary_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def calculate_optimal_threshold(y_true, y_probs):
    """
    Calculate optimal threshold using Youden's J statistic
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, j_scores[optimal_idx]

def main():
    parser = argparse.ArgumentParser(description='YOLO Binary Aneurysm Detection Validation')
    parser.add_argument('--model_path', type=str, 
                       default='rsna-iad-aneurysm-detection/yolo11s-aneurysm-detection-v2_25d/weights/best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--fold_id', type=int, default=0, help='Validation fold ID')
    parser.add_argument('--confidence_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("🔬 YOLO Binary Aneurysm Detection Validation")
    print(f"Model: {args.model_path}")
    print(f"Fold: {args.fold_id}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load YOLO model
    print("📥 Loading YOLO model...")
    model = YOLO(args.model_path)
    print(f"Model classes: {model.names}")
    
    # Load data
    print("📊 Loading validation data...")
    data_path = Path(args.data_dir)
    df = pd.read_csv(data_path / "train_df.csv")
    
    # Get validation UIDs
    val_uids = list(df[df["fold_id"] == args.fold_id]["SeriesInstanceUID"])
    print(f"Found {len(val_uids)} validation series")
    
    # Initialize result storage
    y_true = []  # Ground truth: aneurysm present/absent
    y_pred = []  # Predicted: aneurysm present/absent
    y_probs = []  # Predicted probabilities (confidence scores)
    detection_counts = []  # Number of detections per series
    
    # Validate each series
    print("🔍 Running validation...")
    for uid in tqdm(val_uids[:10], desc="Processing series"):
        try:
            # Load volume
            volume_path = data_path / "processed" / "slices" / f"{uid}.npz"
            if not volume_path.exists():
                print(f"Warning: Volume not found for {uid}")
                continue
                
            with np.load(volume_path) as data:
                volume = data['vol'].astype(np.float32)
            
            # Get ground truth
            row = df[df["SeriesInstanceUID"] == uid].iloc[0]
            aneurysm_present = int(row["Aneurysm Present"])
            
            # Run validation on series
            has_aneurysm, max_confidence, total_detections = validate_series_binary(
                model, volume, args.img_size, args.confidence_threshold
            )
            
            # Store results
            y_true.append(aneurysm_present)
            y_pred.append(int(has_aneurysm))
            y_probs.append(max_confidence)
            detection_counts.append(total_detections)
            
        except Exception as e:
            print(f"Error processing {uid}: {e}")
            continue
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    detection_counts = np.array(detection_counts)

    # print in a table format 5 results per row
    print(f"\n=== BINARY ANEURYSM DETECTION METRICS ===")
    print(f"SeriesInstanceUID, true_aneurysm, pred_aneurysm, pred_confidence, num_detections")
    for i in range(0, len(y_true), 5):
        print(f"{val_uids[i]}, {y_true[i]}, {y_pred[i]}, {y_probs[i]}, {detection_counts[i]}")
    
    print(f"\n✅ Processed {len(y_true)} series")
    
    # Calculate metrics
    print("\n=== BINARY ANEURYSM DETECTION METRICS ===")
    
    # AUC Score
    if len(np.unique(y_true)) > 1:
        auc_score = roc_auc_score(y_true, y_probs)
        print(f"AUC Score: {auc_score:.4f}")
    else:
        print("AUC Score: Cannot calculate (only one class present)")
        auc_score = None
    
    # Find optimal threshold
    if auc_score is not None:
        optimal_threshold, j_stat = calculate_optimal_threshold(y_true, y_probs)
        print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f} (J = {j_stat:.4f})")
        
        # Predictions with optimal threshold
        y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
        print(f"\nUsing optimal threshold {optimal_threshold:.4f}:")
        print(classification_report(y_true, y_pred_optimal, target_names=['No Aneurysm', 'Aneurysm Present']))
    
    # Classification report with current threshold
    print(f"\nUsing confidence threshold {args.confidence_threshold}:")
    print(classification_report(y_true, y_pred, target_names=['No Aneurysm', 'Aneurysm Present']))
    
    # Detection statistics
    print(f"\n=== DETECTION STATISTICS ===")
    print(f"Total series with detections: {np.sum(y_pred)}")
    print(f"Average detections per positive series: {np.mean(detection_counts[y_pred == 1]):.2f}" if np.sum(y_pred) > 0 else "No positive predictions")
    print(f"Max detections in a series: {np.max(detection_counts)}")
    
    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    print(f"\n=== CLASS DISTRIBUTION ===")
    for cls, count in zip(unique, counts):
        cls_name = "No Aneurysm" if cls == 0 else "Aneurysm Present"
        print(f"{cls_name}: {count} series ({count/len(y_true)*100:.1f}%)")
    
    # Plot metrics
    if auc_score is not None:
        print("\n📈 Generating plots...")
        roc_auc, pr_auc = plot_metrics(y_true, y_probs, args.output_dir)
        plot_confusion_matrix(y_true, y_pred, args.output_dir)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'SeriesInstanceUID': val_uids[:len(y_true)],
        'true_aneurysm': y_true,
        'pred_aneurysm': y_pred,
        'pred_confidence': y_probs,
        'num_detections': detection_counts,
    })
    
    output_path = f"{args.output_dir}/binary_validation_results_fold_{args.fold_id}.csv"
    results_df.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 BINARY VALIDATION SUMMARY")
    print("="*60)
    print(f"Fold: {args.fold_id}")
    print(f"Series processed: {len(y_true)}")
    print(f"AUC Score: {auc_score:.4f}" if auc_score else "AUC Score: N/A")
    if auc_score is not None:
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Confidence Threshold Used: {args.confidence_threshold}")
    print(f"💾 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()