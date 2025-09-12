#!/usr/bin/env python3
"""
Test script to verify the complete plane pipeline works end-to-end.

This script:
1. Prepares plane datasets for all views (axial, coronal, sagittal)
2. Runs the plane training and validation pipeline
3. Reports results

Usage:
    python test_plane_pipeline.py
"""
import subprocess
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False


def main():
    print("YOLO Plane Pipeline Test")
    print("=" * 50)
    
    fold = 0  # Test with fold 0
    
    # Step 1: Generate plane datasets
    print(f"\nStep 1: Generating plane datasets for fold {fold}")
    prepare_cmd = [
        sys.executable, "-m", "src.prepare_yolo_dataset_v2_planes",
        "--val-fold", str(fold),
        "--mip-img-size", "512",
        "--target-spacing", "1.0",
        "--label-scheme", "aneurysm_present",
        "--out-name", "yolo_planes"
    ]
    
    if not run_command(prepare_cmd, f"Dataset preparation for fold {fold}"):
        print("Dataset preparation failed. Stopping.")
        return False
    
    # Check if config files were generated
    config_files = list(ROOT.glob(f"configs/yolo_planes_*_fold{fold}.yaml"))
    if len(config_files) < 3:
        print(f"✗ Expected 3 config files but found {len(config_files)}")
        print(f"Found: {[f.name for f in config_files]}")
        return False
    
    print(f"✓ Found {len(config_files)} plane config files")
    
    # Step 2: Run the training and validation pipeline (with very short training for testing)
    print(f"\nStep 2: Running plane training and validation pipeline")
    pipeline_cmd = [
        sys.executable, "-m", "src.run_yolo_pipeline_planes",
        "--model", "yolo11n.pt",  # Small model for testing
        "--epochs", "2",  # Very short training for testing
        "--batch", "8",  # Small batch size
        "--fold", str(fold),
        "--project", "yolo_planes_test",
        "--name", "test_run",
        "--exist-ok",
        "--series-limit", "10",  # Process only 10 series for testing
        "--val-batch", "8"
    ]
    
    if not run_command(pipeline_cmd, f"Plane pipeline for fold {fold}"):
        print("Plane pipeline failed. Stopping.")
        return False
    
    # Step 3: Check results
    print(f"\nStep 3: Checking results")
    summary_file = ROOT / f"planes_summary_fold{fold}.json"
    
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            
            print(f"✓ Summary file found: {summary_file}")
            print(f"Individual results:")
            for result in summary['individual_results']:
                status = result['status']
                auc = result.get('auc', 0.0)
                plane = result['plane']
                icon = "✓" if status == 'success' else "✗"
                print(f"  {icon} {plane.upper()}: {auc:.4f}")
            
            ensemble_auc = summary['ensemble_result'].get('ensemble_auc', 0.0)
            ensemble_status = summary['ensemble_result']['status']
            icon = "✓" if ensemble_status == 'success' else "✗"
            print(f"  {icon} Ensemble AUC: {ensemble_auc:.4f}")
            
            successful_planes = summary['successful_planes']
            print(f"Successfully trained and validated: {successful_planes}/3 planes")
            
            if successful_planes >= 2:
                print(f"✓ Pipeline test PASSED (≥2 planes successful)")
                return True
            else:
                print(f"✗ Pipeline test FAILED (only {successful_planes} planes successful)")
                return False
                
        except Exception as e:
            print(f"✗ Failed to read summary file: {e}")
            return False
    else:
        print(f"✗ Summary file not found: {summary_file}")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*50}")
    if success:
        print("🎉 PLANE PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("The pipeline is ready for full training and evaluation.")
    else:
        print("❌ PLANE PIPELINE TEST FAILED!")
        print("Please check the error messages above.")
    print(f"{'='*50}")
    
    sys.exit(0 if success else 1)