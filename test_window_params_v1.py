
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydicom
import cv2
import glob
from typing import Dict, List, Tuple

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from configs.data_config import *
except ImportError:
    print("Warning: Could not import data_config, using default values")
    IMG_SIZE = 512


def apply_ct_window(image: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    """Apply CT windowing to convert HU values to display range [0, 255]"""
    lower = window_level - (window_width / 2)
    upper = window_level + (window_width / 2)
    image = np.clip(image, lower, upper)
    image = ((image - lower) / (window_width + 1e-7)) * 255.0
    return image

def preprocess_single_slice(image: np.ndarray, dcm: "pydicom.dataset.FileDataset", output_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess a single 2D slice to uint8 and resize to output_size.
    - If modality is CT and rescale is available, convert to HU and apply vascular window.
    - Otherwise, perform min-max normalization.
    """
    image = image.astype(np.float64)

    modality = str(getattr(dcm, "Modality", "")).upper()
    is_ct = 'CT' in dcm.get('Modality', '').upper()
    print("Modalidade é ", modality)
    if is_ct:
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            image = image * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        # Use CTA Vascular default window if not provided
        window_center, window_width = 150, 350
        image = apply_ct_window(image, window_center, window_width)
    else:
        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if image_max > image_min:
            image = ((image - image_min) / (image_max - image_min + 1e-7)) * 255.0
        else:
            image = np.zeros_like(image, dtype=np.float64)

    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    resized = cv2.resize(image_uint8, output_size, interpolation=cv2.INTER_LINEAR)
    return resized

def load_dicom_slice(dicom_path: str) -> Tuple[np.ndarray, pydicom.dataset.FileDataset]:
    """Load a DICOM slice and return the pixel array and metadata"""
    dcm = pydicom.dcmread(dicom_path, force=True)
    px = dcm.pixel_array
    
    # Convert to HU if CT and rescale parameters are available
    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        px = px.astype(np.float64)
        px = px * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    
    return px, dcm


def load_series_slices(series_dir: str) -> Tuple[List[np.ndarray], pydicom.dataset.FileDataset]:
    """Load all slices from a series directory and return list of pixel arrays and sample metadata"""
    dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    
    slices = []
    sample_dcm = None
    
    for dicom_file in sorted(dicom_files):
        try:
            dcm = pydicom.dcmread(dicom_file, force=True)
            px = dcm.pixel_array
            
            # Convert to HU if CT and rescale parameters are available
            if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                px = px.astype(np.float64)
                px = px * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
            
            # Handle multi-dimensional arrays
            if px.ndim == 3:
                if px.shape[-1] == 3:  # RGB
                    px = cv2.cvtColor(px.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
                else:  # Multi-frame, take all slices
                    for slice_idx in range(px.shape[0]):
                        slices.append(px[slice_idx].astype(np.float64))
            else:
                slices.append(px.astype(np.float64))
            
            if sample_dcm is None:
                sample_dcm = dcm
                
        except Exception as e:
            print(f"Error loading {dicom_file}: {e}")
            continue
    
    return slices, sample_dcm


def compute_mip(slices: List[np.ndarray]) -> np.ndarray:
    """Compute Maximum Intensity Projection from list of slices"""
    if not slices:
        return None
    
    # Ensure all slices have the same shape
    target_shape = slices[0].shape
    processed_slices = []
    
    for slice_data in slices:
        if slice_data.shape != target_shape:
            # Resize to match target shape
            slice_resized = cv2.resize(slice_data.astype(np.float32), 
                                     (target_shape[1], target_shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            processed_slices.append(slice_resized.astype(np.float64))
        else:
            processed_slices.append(slice_data)
    
    # Compute MIP
    stack = np.stack(processed_slices, axis=0)
    mip = np.max(stack, axis=0)
    
    return mip


def get_image_type_info(dcm: pydicom.dataset.FileDataset) -> str:
    """Extract image type information from DICOM metadata"""
    info_parts = []
    
    # Modality
    modality = getattr(dcm, 'Modality', 'Unknown')
    info_parts.append(f"Modality: {modality}")
    
    # Image Type
    if hasattr(dcm, 'ImageType'):
        image_type = dcm.ImageType
        if isinstance(image_type, list):
            image_type = '/'.join(image_type[:3])  # Take first 3 elements
        info_parts.append(f"Type: {image_type}")
    
    # Series Description
    if hasattr(dcm, 'SeriesDescription'):
        series_desc = dcm.SeriesDescription[:20]  # Truncate if too long
        info_parts.append(f"Series: {series_desc}")
    
    # Study Description
    if hasattr(dcm, 'StudyDescription'):
        study_desc = dcm.StudyDescription[:15]  # Truncate if too long
        info_parts.append(f"Study: {study_desc}")
    
    return ' | '.join(info_parts)


def test_windowing_parameters(dicom_path: str, output_dir: str = "windowing_tests", use_mip: bool = True) -> None:
    """Test different windowing parameters on a DICOM slice or series MIP"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if path is a file or directory
    if os.path.isfile(dicom_path):
        # Single DICOM file
        try:
            pixel_array, dcm = load_dicom_slice(dicom_path)
            is_mip = False
            
            # Handle multi-dimensional arrays
            if pixel_array.ndim == 3:
                if pixel_array.shape[-1] == 3:  # RGB
                    pixel_array = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
                else:  # Multi-frame, take middle slice
                    pixel_array = pixel_array[pixel_array.shape[0] // 2].astype(np.float64)
            else:
                pixel_array = pixel_array.astype(np.float64)
                
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            return
    
    elif os.path.isdir(dicom_path) and use_mip:
        # Series directory - compute MIP
        try:
            slices, dcm = load_series_slices(dicom_path)
            if len(slices) < 2:
                print(f"Warning: Only {len(slices)} slices found, using single slice instead of MIP")
                pixel_array = slices[0] if slices else None
                is_mip = False
            else:
                pixel_array = compute_mip(slices)
                is_mip = True
                print(f"Computed MIP from {len(slices)} slices")
                
            if pixel_array is None:
                print(f"Error: Could not load any slices from {dicom_path}")
                return
                
        except Exception as e:
            print(f"Error loading series {dicom_path}: {e}")
            return
    
    else:
        print(f"Error: Path {dicom_path} is not a valid file or directory")
        return
    
    # Define different windowing presets for CT
    window_presets = {
        "Brain": (40, 80),         # Brain window
        "Bone": (400, 1000),       # Bone window  
        "Soft Tissue": (40, 400),  # Soft tissue window
        "Lung": (-600, 1200),      # Lung window
        "Liver": (60, 160),        # Liver window
        "CTA Vascular": (150, 350), # CTA vascular (current default)
        "Stroke": (40, 40),        # Stroke window
        "Subdural": (75, 215),     # Subdural window
        "Wide": (50, 500),         # Wide window
        "Narrow": (40, 50),        # Narrow window
    }
    
    # Additional experimental windows
    experimental_windows = {
        "High Contrast": (100, 200),
        "Low Contrast": (50, 800),
        "Aneurysm Focus": (120, 300),
        "Vessel Enhanced": (200, 600),
        "Ultra Wide": (0, 1000),
    }
    
    # Combine all windows
    all_windows = {**window_presets, **experimental_windows}

    # Add percentile-based windows
    percentile_defs = [
        (10, 90),
        (5, 95),
        (1, 99),
        (0.5, 99.5),
        (0, 100)
    ]
    for p_low, p_high in percentile_defs:
        low_val = np.percentile(pixel_array, p_low)
        high_val = np.percentile(pixel_array, p_high)
        level = (low_val + high_val) / 2
        width = high_val - low_val
        name = f"Percentile {p_low}-{p_high}"
        all_windows[name] = (level, width)
    
    # Get image type information
    image_type_info = get_image_type_info(dcm)
    mip_info = "MIP" if is_mip else "Single Slice"
    
    print(f"Testing {len(all_windows)} windowing parameters on {os.path.basename(dicom_path)}")
    print(f"Image type: {mip_info} | {image_type_info}")
    print(f"Original image HU range: [{pixel_array.min():.1f}, {pixel_array.max():.1f}]")
    print(f"Image shape: {pixel_array.shape}")
    
    # **FIXED**: Always add min-max normalization to comparison (for all modalities)
    # Create min-max normalized version
    min_max_img = preprocess_single_slice(pixel_array, dcm, (IMG_SIZE, IMG_SIZE))
    all_windows["Min-Max Normalized"] = None  # Special marker for non-windowed processing
    
    # **FIXED**: Account for the additional min-max image in grid calculation
    n_windows = len(all_windows)
    cols = 5
    rows = int(np.ceil(n_windows / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    title = f'Windowing Parameter Comparison ({mip_info}) - {os.path.basename(dicom_path)}\n{image_type_info}'
    fig.suptitle(title, fontsize=14)
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = [axes] if cols == 1 else [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]
    
    # Apply each windowing preset
    windowed_images = {}
    idx = 0
    
    for name, window_params in all_windows.items():
        if name == "Min-Max Normalized":
            # **FIXED**: Use the pre-computed min-max normalized image
            resized = min_max_img
            windowed_images[name] = resized
            
            # Plot
            ax = axes_flat[idx]
            im = ax.imshow(resized, cmap='gray', vmin=0, vmax=255)
            
            # Get modality info for title
            modality = str(getattr(dcm, "Modality", "Unknown")).upper()
            ax.set_title(f'{name}\n({modality})\n{mip_info}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        else:
            # Apply CT windowing
            level, width = window_params
            windowed = apply_ct_window(pixel_array, level, width)
            windowed_uint8 = np.clip(windowed, 0, 255).astype(np.uint8)
            # Resize to standard size
            resized = cv2.resize(windowed_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            windowed_images[name] = resized
            
            # Plot
            ax = axes_flat[idx]
            im = ax.imshow(resized, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'{name}\nW/L: {width}/{level}\n{mip_info}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        idx += 1
    
    # Hide unused subplots
    for idx in range(n_windows, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    # Save comparison grid
    comparison_path = os.path.join(output_dir, f"windowing_comparison_{os.path.basename(dicom_path).replace('.dcm', '')}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved windowing comparison: {comparison_path}")
    
    # Save individual windowed images
    individual_dir = os.path.join(output_dir, "individual_windows", os.path.basename(dicom_path).replace('.dcm', ''))
    os.makedirs(individual_dir, exist_ok=True)
    
    for name, image in windowed_images.items():
        individual_path = os.path.join(individual_dir, f"{name.replace(' ', '_').replace('/', '_')}.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        
        if name == "Min-Max Normalized":
            modality = str(getattr(dcm, "Modality", "Unknown")).upper()
            plt.title(f'{name} ({mip_info})\nModality: {modality}\n{image_type_info}', fontsize=12)
        else:
            level, width = all_windows[name]
            plt.title(f'{name} Windowing ({mip_info})\nWindow Level: {level}, Window Width: {width}\n{image_type_info}', fontsize=12)
        
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(windowed_images)} individual windowed images to: {individual_dir}")
    


def test_multiple_dicoms(data_dir: str, max_files: int = 5, output_dir: str = "windowing_tests", use_mip: bool = True) -> None:
    """Test windowing on multiple DICOM files or series"""
    
    if use_mip:
        # Find series directories for MIP processing
        series_dirs = []
        series_base = os.path.join(data_dir, "series")
        if os.path.exists(series_base):
            series_dirs = [os.path.join(series_base, d) for d in os.listdir(series_base) 
                          if os.path.isdir(os.path.join(series_base, d))]
        
        if series_dirs:
            print(f"Found {len(series_dirs)} series directories for MIP processing")
            # Limit number of series
            series_dirs = series_dirs[:max_files]
            print(f"Testing windowing with MIP on first {len(series_dirs)} series:")
            
            for i, series_dir in enumerate(series_dirs, 1):
                print(f"\n[{i}/{len(series_dirs)}] Processing series: {os.path.basename(series_dir)}")
                try:
                    test_windowing_parameters(series_dir, output_dir, use_mip=True)
                except Exception as e:
                    print(f"Error processing {series_dir}: {e}")
        else:
            print(f"No series directories found in {data_dir}/series")
            use_mip = False
    
    if not use_mip:
        # Find individual DICOM files
        dicom_patterns = [
            os.path.join(data_dir, "**/*.dcm"),
            os.path.join(data_dir, "series/**/*.dcm"),
        ]
        
        dicom_files = []
        for pattern in dicom_patterns:
            dicom_files.extend(glob.glob(pattern, recursive=True))
        
        if not dicom_files:
            print(f"No DICOM files found in {data_dir}")
            print("Available patterns searched:")
            for pattern in dicom_patterns:
                print(f"  - {pattern}")
            return
        
        print(f"Found {len(dicom_files)} DICOM files")
        
        # Limit number of files
        dicom_files = dicom_files[:max_files]
        print(f"Testing windowing on first {len(dicom_files)} files:")
        
        for i, dicom_path in enumerate(dicom_files, 1):
            print(f"\n[{i}/{len(dicom_files)}] Processing: {dicom_path}")
            try:
                test_windowing_parameters(dicom_path, output_dir, use_mip=False)
            except Exception as e:
                print(f"Error processing {dicom_path}: {e}")
    
    print(f"\nWindowing tests complete! Check '{output_dir}' directory for results.")


def create_windowing_summary_report(output_dir: str = "windowing_tests") -> None:
    """Create a summary report of all windowing tests"""
    
    summary_path = os.path.join(output_dir, "windowing_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("CT Windowing Parameters Test Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Window Presets Tested:\n")
        f.write("-" * 25 + "\n")
        
        window_presets = {
            "Brain": (40, 80),
            "Bone": (400, 1000),
            "Soft Tissue": (40, 400),
            "Lung": (-600, 1200),
            "Liver": (60, 160),
            "CTA Vascular": (150, 350),
            "Stroke": (40, 40),
            "Subdural": (75, 215),
            "Wide": (50, 500),
            "Narrow": (40, 50),
            "High Contrast": (100, 200),
            "Low Contrast": (50, 800),
            "Aneurysm Focus": (120, 300),
            "Vessel Enhanced": (200, 600),
            "Ultra Wide": (0, 1000),
        }
        
        for name, (level, width) in window_presets.items():
            f.write(f"{name:15} - Level: {level:4}, Width: {width:4}\n")
        
        f.write(f"\nRecommendations for Aneurysm Detection:\n")
        f.write("-" * 40 + "\n")
        f.write("1. CTA Vascular (150/350) - Current default, good for vascular structures\n")
        f.write("2. Aneurysm Focus (120/300) - Enhanced contrast for small vessels\n")
        f.write("3. Brain (40/80) - Standard brain tissue visualization\n")
        f.write("4. Vessel Enhanced (200/600) - Higher contrast vessels\n")
        f.write("\nConsider testing combinations of these windows for optimal detection.\n")
    
    print(f"Created windowing summary report: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CT windowing parameters on DICOM files")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Directory containing DICOM files")
    parser.add_argument("--single_file", type=str, default=None,
                       help="Test windowing on a single DICOM file")
    parser.add_argument("--max_files", type=int, default=20,
                       help="Maximum number of DICOM files or series to test")
    parser.add_argument("--output_dir", type=str, default="windowing_tests",
                       help="Output directory for results")
    parser.add_argument("--series_dir", type=str, default=None,
                       help="Test windowing with MIP on a single series directory")
    parser.add_argument("--no_mip", action="store_true",
                       help="Disable MIP processing, use individual slices only")
    
    args = parser.parse_args()
    
    print("CT Windowing Parameter Testing Tool")
    print("=" * 40)
    
    use_mip = not args.no_mip
    
    if args.single_file:
        if os.path.exists(args.single_file):
            print(f"Testing windowing on single file: {args.single_file}")
            test_windowing_parameters(args.single_file, args.output_dir, use_mip=False)
        else:
            print(f"File not found: {args.single_file}")
    elif args.series_dir:
        if os.path.exists(args.series_dir) and os.path.isdir(args.series_dir):
            print(f"Testing windowing with MIP on series: {args.series_dir}")
            test_windowing_parameters(args.series_dir, args.output_dir, use_mip=True)
        else:
            print(f"Series directory not found: {args.series_dir}")
    else:
        print(f"Testing windowing on multiple {'series with MIP' if use_mip else 'files'} from: {args.data_dir}")
        test_multiple_dicoms(args.data_dir, args.max_files, args.output_dir, use_mip=use_mip)
    
    # Create summary report
    create_windowing_summary_report(args.output_dir)