#!/usr/bin/env python3
"""
Test script to visualize the difference between raw pixel values and HU conversion.
Generates PNG images for comparison.
"""
import pydicom
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def read_dicom_raw(path: Path):
    """Read DICOM with raw pixel values (no HU conversion)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    if pix.ndim == 2:
        return pix.astype(np.float32)
    elif pix.ndim == 3:
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return gray
        else:
            return pix[0].astype(np.float32)
    return None

def read_dicom_hu(path: Path):
    """Read DICOM with HU conversion (slope + intercept)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

    if pix.ndim == 2:
        return pix.astype(np.float32) * slope + intercept
    elif pix.ndim == 3:
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return gray * slope + intercept
        else:
            return pix[0].astype(np.float32) * slope + intercept
    return None

def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)

def main():
    # Find DICOM files to test
    data_path = Path('./data')
    series_dirs = list(data_path.glob('series/*'))
    if not series_dirs:
        print("No series directories found in ./data/series/")
        return

    # Collect examples of different modalities
    examples = []
    ct_count = 0
    mr_count = 0
    
    for series_dir in series_dirs[:200]:  # Check first 200 series
        dcm_files = list(series_dir.glob('*.dcm'))
        if dcm_files:
            ds = pydicom.dcmread(str(dcm_files[0]), force=True)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            modality = getattr(ds, 'Modality', 'unknown')
            
            if modality == 'CT' and intercept < 0 and ct_count < 3:
                examples.append((dcm_files[0], modality, slope, intercept))
                ct_count += 1
                print(f"Found CT scan {ct_count}: {series_dir.name}")
            elif modality == 'MR' and intercept == 0 and mr_count < 3:
                examples.append((dcm_files[0], modality, slope, intercept))
                mr_count += 1
                print(f"Found MR scan {mr_count}: {series_dir.name}")
                
            if ct_count >= 3 and mr_count >= 3:
                break

    if not examples:
        print("No suitable DICOM files found")
        return

    print(f"\nFound {len(examples)} examples: {ct_count} CT, {mr_count} MR")

    # Process each example
    for i, (dicom_path, modality, slope, intercept) in enumerate(examples):
        print(f"\n--- Processing Example {i+1}: {modality} ---")
        print(f"File: {dicom_path.name}")
        print(f"Slope: {slope}, Intercept: {intercept}")
        
        # Read both versions
        img_raw = read_dicom_raw(dicom_path)
        img_hu = read_dicom_hu(dicom_path)
        
        if img_raw is None or img_hu is None:
            print("Failed to read DICOM")
            continue
            
        print(f"Raw pixel range: {img_raw.min():.1f} to {img_raw.max():.1f}")
        print(f"HU range: {img_hu.min():.1f} to {img_hu.max():.1f}")
        print(f"Mean difference: {(img_hu - img_raw).mean():.1f}")
        
        # Normalize both for visualization
        img_raw_norm = min_max_normalize(img_raw)
        img_hu_norm = min_max_normalize(img_hu)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Raw pixel version
        axes[0].imshow(img_raw_norm, cmap='gray')
        axes[0].set_title(f'{modality} - Raw Pixels\nRange: {img_raw.min():.0f} to {img_raw.max():.0f}')
        axes[0].axis('off')
        
        # HU version
        axes[1].imshow(img_hu_norm, cmap='gray')
        axes[1].set_title(f'{modality} - HU Conversion\nRange: {img_hu.min():.0f} to {img_hu.max():.0f}')
        axes[1].axis('off')
        
        # Difference
        diff = img_hu_norm.astype(np.int16) - img_raw_norm.astype(np.int16)
        im = axes[2].imshow(diff, cmap='bwr', vmin=-100, vmax=100)
        axes[2].set_title(f'Difference\nMean: {(img_hu - img_raw).mean():.0f}')
        axes[2].axis('off')
        
        # Add colorbar for difference
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        filename = f'hu_comparison_{modality.lower()}_example_{i+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        
        # Also save individual PNGs
        cv2.imwrite(f'raw_pixels_{modality.lower()}_{i+1}.png', img_raw_norm)
        cv2.imwrite(f'hu_conversion_{modality.lower()}_{i+1}.png', img_hu_norm)

    print("\nSummary:")
    print("- Raw pixels: Direct pixel values from DICOM")
    print("- HU conversion: pixel_value * slope + intercept")
    print("- For CT scans: intercept = -1024 (shifts values down)")
    print("- For MR scans: intercept = 0 (no change)")
    print("- This affects image contrast and model predictions!")

if __name__ == "__main__":
    main()
