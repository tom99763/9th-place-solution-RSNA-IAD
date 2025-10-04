import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2



def get_spacing_info(ds: pydicom.Dataset) -> Dict[str, float]:
    """
    Extract physical spacing information from DICOM dataset.
    
    Args:
        ds: DICOM dataset
    
    Returns:
        Dictionary with 'row_spacing', 'col_spacing', 'slice_spacing' in mm
    """
    spacing_info = {}
    
    # Get pixel spacing [row_spacing, col_spacing] in mm
    if hasattr(ds, 'PixelSpacing'):
        spacing_info['row_spacing'] = float(ds.PixelSpacing[0])
        spacing_info['col_spacing'] = float(ds.PixelSpacing[1])
    else:
        # Default to 1mm if not available
        spacing_info['row_spacing'] = 1.0
        spacing_info['col_spacing'] = 1.0
    
    # Get slice spacing
    if hasattr(ds, 'SpacingBetweenSlices'):
        spacing_info['slice_spacing'] = float(ds.SpacingBetweenSlices)
    elif hasattr(ds, 'SliceThickness'):
        spacing_info['slice_spacing'] = float(ds.SliceThickness)
    else:
        # Default to 1mm if not available
        spacing_info['slice_spacing'] = 1.0
    
    return spacing_info


def load_volume(series_dir: Path) -> Tuple[np.ndarray, pydicom.Dataset, Dict[str, float]]:
    """
    Load entire 3D volume from a DICOM series.
    
    Args:
        series_dir: Path to directory containing DICOM files
    
    Returns:
        Tuple of (3D volume array, first DICOM dataset for metadata, spacing info dict)
    """
    dcm_files = list(series_dir.glob("*.dcm"))
    
    if not dcm_files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    
    # Read first file to get metadata
    first_ds = pydicom.dcmread(str(dcm_files[0]))
    
    # Get spacing information
    spacing_info = get_spacing_info(first_ds)
    
    # Check if it's a multiframe DICOM
    if hasattr(first_ds, 'NumberOfFrames') and int(first_ds.NumberOfFrames) > 1:
        # Multiframe DICOM - all frames in one file
        volume = first_ds.pixel_array
        if len(volume.shape) == 2:
            volume = volume[np.newaxis, ...]  # Add frame dimension
        return volume, first_ds, spacing_info
    else:
        # Multiple single-frame files
        # Sort by instance number or slice location
        dcm_data = []
        for dcm_file in dcm_files:
            ds = pydicom.dcmread(str(dcm_file))
            slice_loc = float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else 0
            instance_num = int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0
            dcm_data.append((slice_loc, instance_num, ds.pixel_array))
        
        # Sort by slice location, then instance number
        dcm_data.sort(key=lambda x: (x[0], x[1]))
        
        # Stack into 3D volume
        volume = np.stack([data[2] for data in dcm_data], axis=0)
        return volume, first_ds, spacing_info


def transform_spacing_for_reslicing(spacing_info: Dict[str, float], current_view: str) -> Dict[str, float]:
    """
    Transform spacing information when reslicing to axial view.
    
    Args:
        spacing_info: Original spacing dict with 'row_spacing', 'col_spacing', 'slice_spacing'
        current_view: Current view type
    
    Returns:
        New spacing dict for axial orientation
    """
    row_sp = spacing_info['row_spacing']
    col_sp = spacing_info['col_spacing']
    slice_sp = spacing_info['slice_spacing']
    
    if current_view == 'axial':
        # Already axial: [Z, Y, X] with spacing [slice, row, col]
        return {'row_spacing': row_sp, 'col_spacing': col_sp, 'slice_spacing': slice_sp}
    elif current_view == 'sagittal':
        # Sagittal: [X, Z, Y] -> Axial: [Z, Y, X]
        # Original: slice=X, row=Z, col=Y
        # After transpose (1,2,0): slice=Z, row=Y, col=X
        return {'row_spacing': col_sp, 'col_spacing': slice_sp, 'slice_spacing': row_sp}
    elif current_view == 'coronal':
        # Coronal: [Y, Z, X] -> Axial: [Z, Y, X]
        # Original: slice=Y, row=Z, col=X
        # After transpose (1,0,2): slice=Z, row=Y, col=X
        return {'row_spacing': slice_sp, 'col_spacing': col_sp, 'slice_spacing': row_sp}
    else:
        return spacing_info


def reslice_volume_to_axial(volume: np.ndarray, current_view: str, 
                             spacing_info: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Reslice a 3D volume to create axial view and resample to isotropic spacing.
    
    For sagittal volumes (shape: [slices_X, height_Z, width_Y]):
        - Transpose to [height_Z, width_Y, slices_X] then reslice along Z
    
    For coronal volumes (shape: [slices_Y, height_Z, width_X]):
        - Transpose to [height_Z, slices_Y, width_X] then reslice along Z
    
    For axial volumes (shape: [slices_Z, height_Y, width_X]):
        - Already in correct orientation
    
    Args:
        volume: Input 3D volume (slices, height, width)
        current_view: Current view type ('sagittal', 'coronal', or 'axial')
        spacing_info: Physical spacing information
    
    Returns:
        Tuple of (resliced volume, new spacing info)
    """
    # First, transpose to axial orientation
    if current_view == 'axial':
        resliced = volume
    elif current_view == 'sagittal':
        # Sagittal: [X-slices, Z-height, Y-width] -> [Z-slices, Y-height, X-width]
        resliced = np.transpose(volume, (1, 2, 0))
    elif current_view == 'coronal':
        # Coronal: [Y-slices, Z-height, X-width] -> [Z-slices, Y-height, X-width]
        resliced = np.transpose(volume, (1, 0, 2))
    else:
        resliced = volume
    
    # Get new spacing after reslicing
    new_spacing = transform_spacing_for_reslicing(spacing_info, current_view)
    
    # Resample to isotropic voxels (equal spacing in all directions)
    target_spacing = min(new_spacing['row_spacing'], new_spacing['col_spacing'])  # Use finer spacing
    
    zoom_factors = [
        new_spacing['slice_spacing'] / target_spacing,
        new_spacing['row_spacing'] / target_spacing,
        new_spacing['col_spacing'] / target_spacing
    ]
    
    print(f"    Resampling with zoom factors: {zoom_factors}")
    print(f"    Original spacing: slice={new_spacing['slice_spacing']:.2f}, row={new_spacing['row_spacing']:.2f}, col={new_spacing['col_spacing']:.2f} mm")
    print(f"    Target isotropic spacing: {target_spacing:.2f} mm")
    
    # Resample to isotropic spacing
    resampled = zoom(resliced, zoom_factors, order=1)  # order=1 for bilinear interpolation
    
    isotropic_spacing = {
        'slice_spacing': target_spacing,
        'row_spacing': target_spacing,
        'col_spacing': target_spacing
    }
    
    return resampled, isotropic_spacing


def resize_volume_to_target(volume: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Resize each slice of a 3D volume to target_size x target_size.
    Directly resizes without padding (may distort aspect ratio).
    
    Args:
        volume: Input 3D volume (slices, height, width)
        target_size: Target size for height and width (default: 512)
    
    Returns:
        Resized volume (slices, target_size, target_size)
    """
    n_slices, height, width = volume.shape
    resized_slices = []
    
    for i in range(n_slices):
        slice_img = volume[i, :, :]
        
        # Resize directly to target_size x target_size
        resized = cv2.resize(slice_img.astype(np.float32), (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        resized_slices.append(resized)
    
    return np.stack(resized_slices, axis=0)


def plot_volume_comparison(original_volume: np.ndarray, resliced_volume: np.ndarray,
                           series_uid: str, original_view: str, slice_idx: int = None):
    """
    Plot comparison of original volume slice and resliced axial volume slice.
    
    Args:
        original_volume: Original 3D volume
        resliced_volume: Resliced 3D volume in axial orientation
        series_uid: Series UID for title
        original_view: Original view type
        slice_idx: Slice index to display (default: middle slice)
    """
    if slice_idx is None:
        slice_idx = original_volume.shape[0] // 2
    
    # Get middle slice from resliced volume
    resliced_slice_idx = resliced_volume.shape[0] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Original volume
    # Show three orthogonal views of original
    axes[0, 0].imshow(original_volume[slice_idx, :, :], cmap='gray')
    axes[0, 0].set_title(f'Original {original_view.capitalize()} View\n(Slice {slice_idx}/{original_volume.shape[0]})', 
                         fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_volume[:, original_volume.shape[1]//2, :], cmap='gray')
    axes[0, 1].set_title(f'Original Volume - Cross Section 1', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(original_volume[:, :, original_volume.shape[2]//2], cmap='gray')
    axes[0, 2].set_title(f'Original Volume - Cross Section 2', fontsize=10)
    axes[0, 2].axis('off')
    
    # Bottom row: Resliced axial volume (with proper resampling)
    axes[1, 0].imshow(resliced_volume[resliced_slice_idx, :, :], cmap='gray')
    axes[1, 0].set_title(f'axial view\n(Slice {resliced_slice_idx}/{resliced_volume.shape[0]})', 
                         fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(resliced_volume[:, resliced_volume.shape[1]//2, :], cmap='gray')
    axes[1, 1].set_title(f'coronal view - Cross Section 1', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(resliced_volume[:, :, resliced_volume.shape[2]//2], cmap='gray')
    axes[1, 2].set_title(f'sagittal view - Cross Section 2', fontsize=10)
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Volume Transformation: {original_view.capitalize()} → Axial (Isotropic Voxels)\n' +
                 f'Series: {series_uid[:50]}...\n' +
                 f'Original shape: {original_volume.shape} → Resliced shape: {resliced_volume.shape}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()


def process_series_views(series_dir: Path, views_df: pd.DataFrame = None) -> Tuple[str, List[Tuple[str, str, int]]]:
    """
    Process all DICOM files in a series directory and determine views.
    If views_df is provided, looks up view from CSV. Otherwise uses get_dicom_view.
    
    Args:
        series_dir: Path to directory containing DICOM files
        views_df: Optional DataFrame with columns ['SeriesInstanceUID', 'view', 'n_files', 'total_frames']
    
    Returns:
        Tuple of (predominant_view, list of (filename, view, n_frames) tuples)
    """
    dcm_files = list(series_dir.glob("*.dcm"))
    
    if not dcm_files:
        return 'unknown', []
    
    # Extract series UID from directory name
    series_uid = series_dir.name
    
    series_row = views_df[views_df['SeriesInstanceUID'] == series_uid].iloc[0]
    view = series_row['view']
    return view

    


# Example usage
if __name__ == "__main__":
    ROOT = Path("/home/sersasj/RSNA-IAD-Codebase/data")
    csv_path = "/home/sersasj/RSNA-IAD-Codebase/data/series_views.csv"
    
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter for non-axial views
    non_axial_df = df[df['view'].isin(['sagittal', 'coronal'])].copy()
    print(f"\nFound {len(non_axial_df)} non-axial series:")
    print(f"  - Sagittal: {len(non_axial_df[non_axial_df['view'] == 'sagittal'])}")
    print(f"  - Coronal: {len(non_axial_df[non_axial_df['view'] == 'coronal'])}")
    
    # Process first few non-axial series for visualization
    num_to_visualize = min(5, len(non_axial_df))
    print(f"\nVisualizing {num_to_visualize} series with volume reslicing to axial view...\n")
    
    for idx, row in non_axial_df.head(num_to_visualize).iterrows():
        series_uid = row['SeriesInstanceUID']
        view = row['view']
        
        print(f"Processing {series_uid}")
        print(f"  Original View: {view}")
        
        # Find DICOM files for this series
        series_dir = ROOT / "series" / str(series_uid)
        
        if not series_dir.exists():
            print(f"  Warning: Directory not found: {series_dir}")
            continue
        
        try:
            # Load entire 3D volume with spacing info
            print(f"  Loading 3D volume...")
            original_volume, dcm_metadata, spacing_info = load_volume(series_dir)
            print(f"  Original volume shape: {original_volume.shape}")
            print(f"  Original spacing: slice={spacing_info['slice_spacing']:.2f}, " +
                  f"row={spacing_info['row_spacing']:.2f}, col={spacing_info['col_spacing']:.2f} mm")
            
            # Reslice volume to axial orientation isotropic spacing
            print(f"  Reslicing to axial orientation...")
            resliced_volume, new_spacing = reslice_volume_to_axial(original_volume, view, spacing_info)
            print(f"  Resliced volume shape: {resliced_volume.shape}")
            # final resize to 512x512
            print(f"  Resizing to 512x512...")
            resized_volume = resize_volume_to_target(resliced_volume, target_size=512)
            print(f"  Final resized shape: {resized_volume.shape}")
            
            # Plot comparison (using the isotropic resliced volume before 512x512 resize)
            plot_volume_comparison(original_volume, resized_volume, series_uid, view)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    print("Done!")

