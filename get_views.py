import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
from typing import List, Tuple


def get_view_from_orientation(image_orientation: List[float]) -> str:
    """
    Determine view type (axial, sagittal, coronal) from ImageOrientationPatient.
    
    Args:
        image_orientation: 6-element list [row_x, row_y, row_z, col_x, col_y, col_z]
    
    Returns:
        View type: 'axial', 'sagittal', or 'coronal'
    """
    if len(image_orientation) != 6:
        return 'unknown'
    
    # Extract row and column direction vectors
    row_vec = np.array(image_orientation[:3])
    col_vec = np.array(image_orientation[3:6])
    
    # Calculate normal vector (perpendicular to image plane)
    normal = np.cross(row_vec, col_vec)
    
    # Find which axis the normal is most aligned with
    abs_normal = np.abs(normal)
    dominant_axis = np.argmax(abs_normal)
    
    # Map to view type
    # 0 (X-axis) -> Sagittal
    # 1 (Y-axis) -> Coronal  
    # 2 (Z-axis) -> Axial
    view_map = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
    
    return view_map.get(dominant_axis, 'unknown')


def get_dicom_view(dcm_path: Path, frame_idx: int = 0) -> str:
    """
    Read DICOM file and determine its view.
    Handles both single-frame and multiframe DICOMs.
    
    For multiframe DICOMs, orientation is typically in:
    - SharedFunctionalGroupsSequence.PlaneOrientationSequence (shared across all frames)
    - Or PerFrameFunctionalGroupsSequence[i].PlaneOrientationSequence (per-frame)
    
    Args:
        dcm_path: Path to DICOM file
        frame_idx: Frame index for multiframe DICOMs (default: 0)
    
    Returns:
        View type: 'axial', 'sagittal', 'coronal', or 'unknown'
    """
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        
        # Priority 1: Check SharedFunctionalGroupsSequence (multiframe - shared orientation)
        if hasattr(ds, 'SharedFunctionalGroupsSequence') and len(ds.SharedFunctionalGroupsSequence) > 0:
            shared_seq = ds.SharedFunctionalGroupsSequence[0]
            if hasattr(shared_seq, 'PlaneOrientationSequence') and len(shared_seq.PlaneOrientationSequence) > 0:
                plane_orient = shared_seq.PlaneOrientationSequence[0]
                if hasattr(plane_orient, 'ImageOrientationPatient'):
                    orientation = plane_orient.ImageOrientationPatient
                    return get_view_from_orientation(orientation)
        
        # Priority 2: Check PerFrameFunctionalGroupsSequence (multiframe - per-frame orientation)
        if hasattr(ds, 'PerFrameFunctionalGroupsSequence') and len(ds.PerFrameFunctionalGroupsSequence) > frame_idx:
            frame_seq = ds.PerFrameFunctionalGroupsSequence[frame_idx]
            if hasattr(frame_seq, 'PlaneOrientationSequence') and len(frame_seq.PlaneOrientationSequence) > 0:
                plane_orient = frame_seq.PlaneOrientationSequence[0]
                if hasattr(plane_orient, 'ImageOrientationPatient'):
                    orientation = plane_orient.ImageOrientationPatient
                    return get_view_from_orientation(orientation)
        
        # Priority 3: Check main header ImageOrientationPatient (single-frame standard location)
        if hasattr(ds, 'ImageOrientationPatient'):
            orientation = ds.ImageOrientationPatient
            return get_view_from_orientation(orientation)
        
        return 'unknown'
            
    except Exception as e:
        print(f"Error reading {dcm_path}: {e}")
        return 'error'


def get_num_frames(dcm_path: Path) -> int:
    """
    Get the number of frames in a DICOM file.
    
    Args:
        dcm_path: Path to DICOM file
    
    Returns:
        Number of frames (1 for single-frame)
    """
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        
        # Check for NumberOfFrames attribute (multiframe indicator)
        if hasattr(ds, 'NumberOfFrames'):
            return int(ds.NumberOfFrames)
        
        # Check PerFrameFunctionalGroupsSequence length
        if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
            return len(ds.PerFrameFunctionalGroupsSequence)
        
        # Default: single frame
        return 1
        
    except Exception:
        return 1


def process_series_views(series_dir: Path) -> Tuple[str, List[Tuple[str, str, int]]]:
    """
    Process all DICOM files in a series directory and determine views.
    Handles multiframe DICOMs by checking the first frame.
    
    Args:
        series_dir: Path to directory containing DICOM files
    
    Returns:
        Tuple of (predominant_view, list of (filename, view, n_frames) tuples)
    """
    dcm_files = list(series_dir.glob("*.dcm"))
    
    if not dcm_files:
        return 'unknown', []
    
    views = []
    for dcm_file in dcm_files[0:1]:
        view = get_dicom_view(dcm_file, frame_idx=0)  # Check first frame
        n_frames = get_num_frames(dcm_file)
        views.append((dcm_file.name, view, n_frames))
    
    # Determine predominant view (ignoring n_frames for counting)
    view_counts = {}
    for _, view, _ in views:
        view_counts[view] = view_counts.get(view, 0) + 1
    
    predominant_view = max(view_counts, key=view_counts.get) if view_counts else 'unknown'
    
    return predominant_view, views


# Example usage
if __name__ == "__main__":
  
    
    # Example: process all series in train.csv
    ROOT = Path("/home/sersasj/RSNA-IAD-Codebase/data")
    train_csv = "/home/sersasj/RSNA-IAD-Codebase/data/train.csv"
    print(train_csv)
    print("Processing train.csv")
    df = pd.read_csv(train_csv)
    series_uids = df['SeriesInstanceUID'].unique()
    
    results = []
    for uid in series_uids:  # Process first 10 series as example
        series_dir = ROOT / "series" / str(uid)
        if series_dir.exists():
            predominant_view, file_views = process_series_views(series_dir)
            total_frames = sum(n_frames for _, _, n_frames in file_views)
            results.append({
                'SeriesInstanceUID': uid,
                'view': predominant_view,
                'n_files': len(file_views),
                'total_frames': total_frames
            })
            print(f"{uid}: {predominant_view} ({len(file_views)} files, {total_frames} frames)")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(ROOT / ".csv", index=False)
    print(f"\nSaved results to series_views.csv")

