import pydicom
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import ast


def load_slice_from_npz(npz_path):
    """Load a single slice from an NPZ file"""
    try:
        with np.load(npz_path) as data:
            # Try different possible keys
            if 'vol' in data:
                return data['vol']
            elif 'image' in data:
                return data['image']
            elif 'slice' in data:
                return data['slice']
            else:
                # If no known key, try to get the first array
                keys = list(data.keys())
                if keys:
                    print(f"Available keys in {os.path.basename(npz_path)}: {keys}")
                    return data[keys[0]]
                else:
                    print(f"No data found in {npz_path}")
                    return None
    except Exception as e:
        print(f"Error loading {npz_path}: {str(e)}")
        return None


def load_metadata(train_csv_path, localizers_csv_path):
    """Load metadata from CSV files"""
    try:
        # Load train.csv for modality information
        train_df = pd.read_csv(train_csv_path)
        print(f"Loaded train.csv with {len(train_df)} entries")
        
        # Load localizers.csv for location information
        localizers_df = pd.read_csv(localizers_csv_path)
        print(f"Loaded train_localizers.csv with {len(localizers_df)} entries")
        
        # Create a mapping from SeriesInstanceUID to modality
        modality_map = dict(zip(train_df['SeriesInstanceUID'], train_df['Modality']))
        
        # Create a mapping from SeriesInstanceUID to location information
        location_map = {}
        for _, row in localizers_df.iterrows():
            series_uid = row['SeriesInstanceUID']
            if series_uid not in location_map:
                location_map[series_uid] = []
            
            try:
                coords = ast.literal_eval(row['coordinates'])
                location_info = {
                    'location': row['location'],
                    'x': coords['x'],
                    'y': coords['y']
                }
                location_map[series_uid].append(location_info)
            except:
                # If coordinates parsing fails, just store the location name
                location_info = {
                    'location': row['location'],
                    'x': None,
                    'y': None
                }
                location_map[series_uid].append(location_info)
        
        return modality_map, location_map
        
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return {}, {}


def create_grid_from_slices(folder_path, max_files=None, output_name="all_slices_grid", 
                           train_csv_path="data/train.csv", 
                           localizers_csv_path="data/train_localizers.csv"):
    """Load all NPZ slice files from a folder and create a grid visualization with metadata"""
    
    # Load metadata
    modality_map, location_map = load_metadata(train_csv_path, localizers_csv_path)
    
    # Find all NPZ files in the folder
    npz_pattern = os.path.join(folder_path, "*.npz")
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        print(f"No NPZ files found in {folder_path}")
        return
    
    print(f"Found {len(npz_files)} NPZ files in {folder_path}")
    
    # Sort files for consistent ordering
    npz_files.sort()
    
    # Limit number of files if specified
    if max_files:
        npz_files = npz_files[:max_files]
        print(f"Processing first {len(npz_files)} files")
    
    # List all files
    print("\nNPZ files found:")
    for i, file_path in enumerate(npz_files, 1):
        print(f"{i:2d}. {os.path.basename(file_path)}")
    
    # Load all slices with metadata
    print(f"\nLoading {len(npz_files)} slices...")
    slices = []
    valid_files = []
    metadata_list = []
    
    for npz_file in npz_files:
        slice_data = load_slice_from_npz(npz_file)
        if slice_data is not None:
            # Handle both 2D and 3D arrays
            if slice_data.ndim == 3:
                # If 3D, take the first slice or middle slice
                slice_data = slice_data[slice_data.shape[0]//2]
            elif slice_data.ndim > 3:
                # If more than 3D, flatten to 2D
                slice_data = slice_data.reshape(slice_data.shape[-2:])
            
            # Extract SeriesInstanceUID from filename
            filename = os.path.basename(npz_file)
            series_uid = filename.replace('_mip.npz', '')
            
            # Get metadata
            modality = modality_map.get(series_uid, 'Unknown')
            #if modality != 'CTA':
            #    continue
            locations = location_map.get(series_uid, [])
            
            metadata = {
                'series_uid': series_uid,
                'modality': modality,
                'locations': locations,
                'has_aneurysm': len(locations) > 0
            }
            
            slices.append(slice_data)
            valid_files.append(filename)
            metadata_list.append(metadata)
        else:
            print(f"Skipping {os.path.basename(npz_file)} - could not load data")
    
    if not slices:
        print("No valid slices found!")
        return
    
    print(f"Successfully loaded {len(slices)} slices")
    
    # Create output directory
    os.makedirs("slice_grids", exist_ok=True)
    
    # Determine grid size
    num_slices = len(slices)
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))
    
    print(f"Creating {rows}x{cols} grid for {num_slices} slices")
    
    # Create the grid plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(f'Grid of {num_slices} slices from {os.path.basename(folder_path)}', fontsize=16)
    
    # Handle single row or column cases
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each slice
    for i in range(num_slices):
        row = i // cols
        col = i % cols
        
        im = axes[row][col].imshow(slices[i], cmap='gray')
        
        # Create title with metadata
        metadata = metadata_list[i]
        title_parts = [f'{i+1}: {metadata["modality"]}']
        
        if metadata['has_aneurysm']:
            title_parts.append('ANEURYSM')
            # Add location information
            locations = metadata['locations']
            if locations:
                # Get unique location names
                unique_locations = list(set([loc['location'] for loc in locations]))
                title_parts.append(f'Loc: {", ".join(unique_locations[:2])}')  # Show first 2 locations
        
        title = ' | '.join(title_parts)
        axes[row][col].set_title(title, fontsize=8)
        axes[row][col].axis('off')
        
        # Mark aneurysm locations if present
        #if metadata['has_aneurysm']:
        #    for location_info in metadata['locations']:
        #        if location_info['x'] is not None and location_info['y'] is not None:
        #            # Convert coordinates to image coordinates if needed
        #            x, y = location_info['x'], location_info['y']
        #            
        #            # Check if coordinates are within image bounds
        #            if 0 <= x < slices[i].shape[1] and 0 <= y < slices[i].shape[0]:
        #                axes[row][col].plot(x, y, 'r+', markersize=8, markeredgewidth=2)
    
    # Hide unused subplots
    for i in range(num_slices, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    
    # Save the grid
    output_path = f"slice_grids/{output_name}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid visualization as '{output_path}'")
    
    # Also save individual slices for reference
    print("Saving individual slices...")
    individual_dir = f"slice_grids/individual_{output_name}"
    os.makedirs(individual_dir, exist_ok=True)
    
    #for i, (slice_data, filename, metadata) in enumerate(zip(slices, valid_files, metadata_list)):
    #    plt.figure(figsize=(10, 10))
    #    plt.imshow(slice_data, cmap='gray')
    #    
    #    # Create detailed title
    #    title_parts = [f'Slice {i+1}: {metadata["modality"]}']
    #    if metadata['has_aneurysm']:
    #        title_parts.append('ANEURYSM DETECTED')
    #        locations = metadata['locations']
    #        if locations:
    #            unique_locations = list(set([loc['location'] for loc in locations]))
    #            title_parts.append(f'Locations: {", ".join(unique_locations)}')
    #    
    #    plt.title(' | '.join(title_parts), fontsize=12)
    #    plt.axis('off')
    #    
    #    # Mark aneurysm locations if present
    #    #if metadata['has_aneurysm']:
    #    #    for location_info in metadata['locations']:
    #    #        if location_info['x'] is not None and location_info['y'] is not None:
    #    #            x, y = location_info['x'], location_info['y']
    #    #            if 0 <= x < slice_data.shape[1] and 0 <= y < slice_data.shape[0]:
    #    #                plt.plot(x, y, 'r+', markersize=12, markeredgewidth=3, 
    #    #                       label=f"{location_info['location']} ({x:.1f}, {y:.1f})")
    #    #
    #    # Clean filename for saving
    #    clean_name = filename.replace('.npz', '').replace('.', '_')
    #    plt.savefig(f"{individual_dir}/slice_{i+1:03d}_{clean_name}.png", dpi=150, bbox_inches='tight')
    #    plt.close()
    #
    #print(f"Saved {len(slices)} individual slices to {individual_dir}/")
    #
    ## Print summary statistics
    #print("\nSummary Statistics:")
    #modalities = [m['modality'] for m in metadata_list]
    #modality_counts = pd.Series(modalities).value_counts()
    #print("Modality distribution:")
    #for modality, count in modality_counts.items():
    #    print(f"  {modality}: {count}")
    #
    #aneurysm_count = sum(1 for m in metadata_list if m['has_aneurysm'])
    #print(f"\nAneurysm cases: {aneurysm_count}/{len(metadata_list)} ({aneurysm_count/len(metadata_list)*100:.1f}%)")
    #
    #if aneurysm_count > 0:
    #    all_locations = []
    #    for m in metadata_list:
    #        if m['has_aneurysm']:
    #            all_locations.extend([loc['location'] for loc in m['locations']])
    #    location_counts = pd.Series(all_locations).value_counts()
    #    print("\nAneurysm locations:")
    #    for location, count in location_counts.items():
    #        print(f"  {location}: {count}")
    #
    #print(f"\nProcessing complete! Check the 'slice_grids/' directory for outputs.")


if __name__ == "__main__":
    # Specify the folder containing NPZ files
    folder_path = "/home/sersasj/RSNA-IAD-Codebase/data/processed/mip_images/"
    
    max_files_to_process = 200  # Change this or set to None
    
    create_grid_from_slices(folder_path, max_files=max_files_to_process, output_name="mip_images_grid")