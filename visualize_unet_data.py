import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_npz_data(npz_path: Path):
    """Load volume and mask from NPZ file."""
    data = np.load(npz_path)
    volume = data['volume']
    mask = data['mask']
    return volume, mask

def check_labels(mask: np.ndarray):
    """Check if mask has any non-zero labels."""
    has_labels = np.any(mask > 0)
    total_voxels = mask.size
    labeled_voxels = np.sum(mask > 0)
    max_value = np.max(mask)
    print(f"Mask statistics:")
    print(f"  Has labels: {has_labels}")
    print(f"  Total voxels: {total_voxels}")
    print(f"  Labeled voxels: {labeled_voxels}")
    print(f"  Max mask value: {max_value:.6f}")
    if labeled_voxels > 0:
        print(".2f")
    return has_labels

def plot_slices(volume: np.ndarray, mask: np.ndarray, series_uid: str, num_slices: int = 8, output_path: str = None):
    """Plot slices from volume and mask."""
    n_slices, h, w = volume.shape
    fig, axes = plt.subplots(num_slices, 2, figsize=(12, 4 * num_slices))
    
    # Select slices evenly spaced
    slice_indices = np.linspace(0, n_slices - 1, num_slices, dtype=int)
    
    for i, slice_idx in enumerate(slice_indices):
        # Volume slice
        vol_slice = volume[slice_idx]
        axes[i, 0].imshow(vol_slice, cmap='gray')
        axes[i, 0].set_title(f'Volume Slice {slice_idx}')
        axes[i, 0].axis('off')
        
        # Mask slice
        mask_slice = mask[slice_idx]
        axes[i, 1].imshow(vol_slice, cmap='gray')
        axes[i, 1].imshow(mask_slice, cmap='Reds', alpha=0.5)
        axes[i, 1].set_title(f'Mask Overlay Slice {slice_idx}')
        axes[i, 1].axis('off')
    
    plt.suptitle(f'Series: {series_uid}')
    plt.tight_layout()

    plt.savefig("aaaa.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to aaaa.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize U-Net dataset NPZ files")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to the NPZ file")
    parser.add_argument("--num-slices", type=int, default=32, help="Number of slices to plot")
    parser.add_argument("--output", type=str, help="Output PNG file path (if not specified, displays plot)")
    args = parser.parse_args()
    
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"NPZ file not found: {npz_path}")
        return
    
    volume, mask = load_npz_data(npz_path)
    series_uid = npz_path.stem
    
    # Check if mask has labels
    has_labels = check_labels(mask)
    
    plot_slices(volume, mask, series_uid, args.num_slices, args.output)

if __name__ == "__main__":
    main()


#python3 visualize_unet_data.py --npz-path data/unet_dataset/train/1.2.826.0.1.3680043.8.498.76479968083900503150635695368644295015.npz --num-slices 8 --output visualization.png

#python3 visualize_unet_data.py --npz-path data/Dataset001_unet_isotropic_resize/train/1.2.82
#6.0.1.3680043.8.498.76479968083900503150635695368644295015.npz --num-slices 8 --output visualization.png

#data/Dataset001_unet_isotropic_resize/train/1.2.826.0.1.3680043.8.498.10129540112106776730428126836684374398.npz
#python3 visualize_unet_data.py --npz-path data/Dataset001_unet_isotropic/test/1.2.826.0.1.3680043.8.498.10337340834925241563571050156541599503.npz --num-slices 8 --output visualization.png