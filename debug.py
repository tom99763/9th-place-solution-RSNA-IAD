#!/usr/bin/env python3
"""
Debug script to identify issues with nnU-Net dataset preparation.
This script checks for common issues that cause preprocessing failures.
"""

import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path

def check_dataset_integrity(dataset_path):
    """Check for common issues in nnU-Net dataset."""
    
    print(f"Checking dataset at: {dataset_path}")
    dataset_path = Path(dataset_path)
    
    # Check directory structure
    required_dirs = ['imagesTr', 'labelsTr']
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            print(f"❌ Missing directory: {dir_name}")
            return
        else:
            print(f"✅ Found directory: {dir_name}")
    
    # Check dataset.json
    dataset_json_path = dataset_path / 'dataset.json'
    if not dataset_json_path.exists():
        print("❌ Missing dataset.json")
        return
    
    with open(dataset_json_path) as f:
        dataset_json = json.load(f)
    print(f"✅ Found dataset.json with {dataset_json.get('numTraining', 0)} training cases")
    
    # Get all image and label files
    images_dir = dataset_path / 'imagesTr'
    labels_dir = dataset_path / 'labelsTr'
    
    image_files = sorted(list(images_dir.glob('*_0000.nii.gz')))
    label_files = sorted(list(labels_dir.glob('*.nii.gz')))
    
    print(f"\nFound {len(image_files)} image files")
    print(f"Found {len(label_files)} label files")
    
    # Check for missing pairs
    case_ids_images = set(f.name.replace('_0000.nii.gz', '') for f in image_files)
    case_ids_labels = set(f.name.replace('.nii.gz', '') for f in label_files)
    
    missing_labels = case_ids_images - case_ids_labels
    missing_images = case_ids_labels - case_ids_images
    
    if missing_labels:
        print(f"❌ Cases missing labels: {sorted(list(missing_labels))[:10]}...")
        print(f"   Total missing labels: {len(missing_labels)}")
    
    if missing_images:
        print(f"❌ Cases missing images: {sorted(list(missing_images))[:10]}...")
        print(f"   Total missing images: {len(missing_images)}")
    
    # Check individual files for issues
    print(f"\nChecking first 10 files for detailed issues...")
    issues = []
    
    for i, (img_file, lbl_file) in enumerate(zip(image_files[:10], label_files[:10])):
        case_id = img_file.name.replace('_0000.nii.gz', '')
        
        try:
            # Load image
            img_nii = nib.load(str(img_file))
            img_data = img_nii.get_fdata()
            
            # Load label
            lbl_nii = nib.load(str(lbl_file))
            lbl_data = lbl_nii.get_fdata()
            
            # Check shapes match
            if img_data.shape != lbl_data.shape:
                issues.append(f"❌ {case_id}: Shape mismatch - Image: {img_data.shape}, Label: {lbl_data.shape}")
                continue
            
            # Check if label is empty
            if lbl_data.size == 0:
                issues.append(f"❌ {case_id}: Empty label array")
                continue
            
            # Check label values
            unique_labels = np.unique(lbl_data)
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                issues.append(f"⚠️  {case_id}: Label contains only background (all zeros)")
            elif np.max(unique_labels) > 1:
                issues.append(f"⚠️  {case_id}: Label values > 1 found: {unique_labels}")
            else:
                print(f"✅ {case_id}: OK - Shape: {img_data.shape}, Labels: {unique_labels}")
                
        except Exception as e:
            issues.append(f"❌ {case_id}: Error loading files - {str(e)}")
    
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ No major issues found in sampled files")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total training cases: {len(image_files)}")
    print(f"Cases with missing labels: {len(missing_labels)}")
    print(f"Cases with missing images: {len(missing_images)}")
    
    return len(missing_labels) == 0 and len(missing_images) == 0 and len(issues) == 0

def fix_empty_labels(dataset_path):
    """Remove cases that have empty labels to prevent preprocessing errors."""
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'imagesTr'
    labels_dir = dataset_path / 'labelsTr'
    
    image_files = sorted(list(images_dir.glob('*_0000.nii.gz')))
    
    removed_cases = []
    
    for img_file in image_files:
        case_id = img_file.name.replace('_0000.nii.gz', '')
        lbl_file = labels_dir / f"{case_id}.nii.gz"
        
        if not lbl_file.exists():
            print(f"Removing {case_id}: no corresponding label file")
            img_file.unlink()
            removed_cases.append(case_id)
            continue
        
        try:
            lbl_nii = nib.load(str(lbl_file))
            lbl_data = lbl_nii.get_fdata()
            
            # Check if label is empty or all zeros
            if lbl_data.size == 0 or np.max(lbl_data) == 0:
                print(f"Removing {case_id}: empty or all-zero label")
                img_file.unlink()
                lbl_file.unlink()
                removed_cases.append(case_id)
                
        except Exception as e:
            print(f"Removing {case_id}: error reading label - {e}")
            img_file.unlink()
            if lbl_file.exists():
                lbl_file.unlink()
            removed_cases.append(case_id)
    
    # Update dataset.json
    if removed_cases:
        dataset_json_path = dataset_path / 'dataset.json'
        with open(dataset_json_path) as f:
            dataset_json = json.load(f)
        
        # Update numTraining
        remaining_images = len(list(images_dir.glob('*_0000.nii.gz')))
        dataset_json['numTraining'] = remaining_images
        
        with open(dataset_json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        print(f"\nRemoved {len(removed_cases)} problematic cases")
        print(f"Updated dataset.json: numTraining = {remaining_images}")
        
        # Update splits if they exist
        splits_path = dataset_path / 'splits_final.json'
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            
            # Remove problematic cases from all folds
            for fold_data in splits:
                fold_data['train'] = [c for c in fold_data['train'] if c not in removed_cases]
                fold_data['val'] = [c for c in fold_data['val'] if c not in removed_cases]
            
            with open(splits_path, 'w') as f:
                json.dump(splits, f, indent=2)
            
            print(f"Updated splits_final.json")
    
    return removed_cases

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug nnU-Net dataset issues")
    # Optional positional argument: if not provided, use default Dataset901 path
    parser.add_argument("dataset_path", nargs='?', default="/home/sersasj/RSNA-IAD-Codebase/data/nnUNet_raw/Dataset901_RSNAAneurysm", help="Path to Dataset901_RSNAAneurysm directory")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues by removing problematic cases")
    
    args = parser.parse_args()
    
    print("=== nnU-Net Dataset Debug Tool ===\n")
    
    # Check dataset
    is_ok = check_dataset_integrity(args.dataset_path)
    
    if not is_ok and args.fix:
        print(f"\n=== FIXING ISSUES ===")
        removed = fix_empty_labels(args.dataset_path)
        
        if removed:
            print(f"\nRe-checking dataset after fixes...")
            check_dataset_integrity(args.dataset_path)
    
    elif not is_ok:
        print(f"\n💡 Run with --fix to automatically remove problematic cases")
        print(f"   python debug_dataset.py {args.dataset_path} --fix")