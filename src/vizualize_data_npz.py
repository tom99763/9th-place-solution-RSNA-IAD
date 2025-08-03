import pydicom
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os


def plot_npz(npz_path):
    with np.load(npz_path) as data:
        volume = data['vol']
        print(f"Volume shape: {volume.shape}")
        print(f"Number of slices: {volume.shape[0]}")
        
        for i in range(volume.shape[0]):
            plt.figure(figsize=(8, 8))
            plt.imshow(volume[i], cmap='gray')
            plt.title(f'Slice {i+1}/{volume.shape[0]}')
            plt.axis('off')
            os.makedirs("all_slices_grid", exist_ok=True)
            plt.savefig(f"all_slices_grid/slice_{i+1:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f"Saved {volume.shape[0]} individual slice images")
        
        num_slices = volume.shape[0]
        cols = int(np.ceil(np.sqrt(num_slices)))
        rows = int(np.ceil(num_slices / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(f'All {num_slices} slices', fontsize=16)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        for i in range(num_slices):
            row = i // cols
            col = i % cols
            axes[row, col].imshow(volume[i], cmap='gray')
            axes[row, col].set_title(f'Slice {i+1}')
            axes[row, col].axis('off')
            
        for i in range(num_slices, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
            
        plt.tight_layout()
        os.makedirs("all_slices_grid", exist_ok=True)
        plt.savefig("all_slices_grid/all_slices_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Saved grid view as 'all_slices_grid.png'")

#1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317,1.2.826.0.1.3680043.8.498.10775329348174902199350466348663848346
if __name__ == "__main__":
    #1.2.826.0.1.3680043.8.498.11466016618035234391071120016712127446
    plot_npz("/home/sersasj/RSNA-IAD-Codebase/processed_data/processed_data/data/processed/1.2.826.0.1.3680043.8.498.11466016618035234391071120016712127446.npz")