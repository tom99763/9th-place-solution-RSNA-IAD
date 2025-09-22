# UNet Training on Aneurysm Cubes

This guide explains how to train a 3D UNet model on the aneurysm cubes dataset using fold-based cross-validation.

## Overview

The aneurysm cubes dataset consists of 3D volumes (32×128×128) extracted from RSNA intracranial aneurysm detection data. Each volume is stored as an NPZ file containing:
- `volume`: uint8 normalized CT scan data (0-255)
- `mask`: float16 Gaussian mask indicating aneurysm locations (0-1)
- `label`: binary label (0=negative, 1=positive)
- `fold`: fold assignment for cross-validation (0-4)

## Dataset Structure

```
aneurysm_cubes/
├── fold_0/
│   ├── pos_<series_id>_<index>.npz  # Positive samples (with aneurysms)
│   └── neg_<series_id>_<index>.npz  # Negative samples (without aneurysms)
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

## Quick Start

### 1. Generate Aneurysm Cubes (if not already done)

```bash
python3 prepare_aneurysm_cubes.py --output-dir aneurysm_cubes --cube-shape 32 128 128
```

### 2. Quick Test Training

```bash
# Test with visualization (exits after generating 5 sample plots)
python3 example_train_aneurysm_cubes.py --test --viz

# Quick test training (1 epoch, small dataset)
python3 example_train_aneurysm_cubes.py --test
```

### 3. Full Training

```bash
# Train using fold 0 as validation
python3 example_train_aneurysm_cubes.py --full --val-fold 0

# Train using fold 1 as validation
python3 example_train_aneurysm_cubes.py --full --val-fold 1
```

## Training Scripts

### Main Training Script: `train_unet_aneurysm_cubes.py`

The core training script with the following key features:

- **Fold-based data loading**: Uses one fold for validation, others for training
- **Data format handling**: Converts uint8 volumes to float32, handles float16 masks
- **3D UNet architecture**: DynUNet with spatial dimensions optimized for 32×128×128 input
- **Data augmentation**: Random rotations, flips, and cropping
- **Mixed precision training**: 16-bit automatic mixed precision for faster training
- **Comprehensive metrics**: Dice score, classification accuracy, AUC-ROC

#### Key Arguments:

```bash
python3 train_unet_aneurysm_cubes.py \
  --cubes-dir aneurysm_cubes \     # Path to cubes directory
  --val-fold 0 \                   # Fold to use for validation (0-4)
  --lr 1e-4 \                      # Learning rate
  --epochs 200 \                   # Number of training epochs
  --pos-weight 128.0 \             # Weight for positive samples in loss
  --label-threshold 0.1 \          # Threshold for binarizing masks
  --pos-thr 0.1 \                  # Threshold for identifying positive volumes
  --small-dataset \                # Use 10% of data for quick testing
  --only-positive \                # Train only on positive samples
  --wandb \                        # Enable Weights & Biases logging
  --viz-samples 5                  # Generate visualization samples and exit
```

### Example Script: `example_train_aneurysm_cubes.py`

Simplified wrapper script with predefined configurations:

```bash
# Quick test (1 epoch, small dataset)
python3 example_train_aneurysm_cubes.py --test

# Quick test with visualizations
python3 example_train_aneurysm_cubes.py --test --viz

# Full training (200 epochs)
python3 example_train_aneurysm_cubes.py --full --val-fold 0

# Train only on positive samples
python3 example_train_aneurysm_cubes.py --positive-only --val-fold 1

# Enable W&B logging
python3 example_train_aneurysm_cubes.py --full --wandb --val-fold 2
```

## Training Configurations

### Standard Training
- Uses all available data (positive and negative samples)
- Batch size: 4 (for 3D data)
- Learning rate: 1e-4 with ReduceLROnPlateau scheduler
- Loss: DiceCE loss with high positive weight (128.0)
- Data augmentation: Random rotations, flips, spatial cropping

### Positive-Only Training
```bash
python3 train_unet_aneurysm_cubes.py --only-positive --only-positive-val
```
- Focuses only on samples with aneurysms
- Useful for learning aneurysm segmentation details
- Lower positive weight (10.0) since no negative samples

### Cross-Validation Training

Train on each fold separately:
```bash
for fold in {0..4}; do
    python3 example_train_aneurysm_cubes.py --full --val-fold $fold
done
```

## Model Architecture

The script uses a 3D DynUNet with:
- **Input**: 1-channel 3D volumes (32×128×128)
- **Output**: 1-channel probability maps (same size)
- **Architecture**: 4-level encoder-decoder with skip connections
- **Filters**: [32, 64, 128, 256]
- **Kernels**: 3×3×3 convolutions
- **Strides**: [1, 2, 2, 2] for downsampling
- **Dropout**: 0.1 for regularization

## Data Preprocessing

### Volume Preprocessing:
1. Convert uint8 to float32: `volume.astype(np.float32) / 255.0`
2. Normalize intensity: Mean=0, Std=1
3. Ensure channel-first format: Add channel dimension
4. Apply RAS orientation

### Mask Preprocessing:
1. Keep as float32: `mask.astype(np.float32)`
2. Binarize with threshold: `(mask > label_threshold).astype(np.float32)`
3. Default threshold: 0.1 (to handle Gaussian masks)

### Data Augmentation:
- Random 90° rotations in axial plane
- Random flips along Y and Z axes
- Random spatial cropping (16×64×64 patches)
- Random center placement for robust training

## Monitoring and Logging

### Metrics Tracked:
- **Training**: Loss per step and epoch
- **Validation**: Loss, Dice score, classification accuracy, AUC-ROC
- **Classification metrics**: Overall accuracy, positive accuracy, negative accuracy

### Weights & Biases Integration:
```bash
python3 train_unet_aneurysm_cubes.py --wandb \
  --wandb-project "unet_aneurysm_cubes" \
  --wandb-name "fold_0_experiment"
```

### Visualization:
```bash
# Generate visualization of training samples
python3 train_unet_aneurysm_cubes.py --viz-samples 10 --viz-dir outputs/viz
```

## Expected Performance

### Dataset Statistics (with 10% subset):
- **Training**: ~368 volumes (171 positive, 197 negative)
- **Validation**: ~90 volumes (40 positive, 50 negative)

### Training Time (RTX 3090):
- **1 epoch**: ~8-10 seconds (small dataset)
- **Full training (200 epochs)**: ~30-40 minutes (small dataset)

### Memory Requirements:
- **GPU Memory**: ~6-8 GB (batch size 4)
- **RAM**: ~8-16 GB depending on caching

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size in script (default: 4)
2. **No positive samples**: Check `--pos-thr` value (default: 0.1)
3. **Missing fold directories**: Run `prepare_aneurysm_cubes.py` first
4. **Low Dice scores**: Try adjusting `--pos-weight` or `--label-threshold`

### Memory Optimization:
- Use `--small-dataset` for testing
- Reduce batch size in DataLoader
- Use CPU instead of GPU for very large datasets

### Performance Tuning:
- Adjust `--pos-weight` based on positive/negative ratio
- Tune `--label-threshold` based on mask distribution
- Experiment with different learning rates and schedulers

## File Structure

```
├── train_unet_aneurysm_cubes.py     # Main training script
├── example_train_aneurysm_cubes.py  # Example configurations
├── prepare_aneurysm_cubes.py        # Data preparation script
├── aneurysm_cubes/                  # Generated dataset
│   ├── fold_0/ ... fold_4/          # Cross-validation folds
│   └── examples/                    # Sample visualizations
├── outputs/                         # Training outputs
│   ├── aneurysm_cubes_viz/          # Visualization outputs
│   └── lightning_logs/              # Training logs
└── README_UNET_ANEURYSM_CUBES.md    # This file
```

## Next Steps

1. **Hyperparameter tuning**: Experiment with learning rates, architectures
2. **Ensemble methods**: Combine predictions from different folds
3. **Post-processing**: Apply morphological operations to predictions
4. **Advanced augmentation**: Add intensity variations, elastic deformations
5. **Model comparison**: Try different architectures (nnU-Net, Swin-UNet, etc.)
