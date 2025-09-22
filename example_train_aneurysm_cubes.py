#!/usr/bin/env python3
"""
Example script demonstrating how to train UNet on aneurysm cubes.

This script shows different training configurations for the aneurysm cubes dataset.
The dataset should be organized in fold directories (fold_0, fold_1, ..., fold_4)
with NPZ files containing 'volume', 'mask', 'label', and 'fold' keys.

Usage examples:
1. Quick test with visualization:
   python3 example_train_aneurysm_cubes.py --test --viz

2. Full training with fold 0 as validation:
   python3 example_train_aneurysm_cubes.py --full --val-fold 0

3. Training only on positive samples:
   python3 example_train_aneurysm_cubes.py --positive-only --val-fold 1

4. Training with W&B logging:
   python3 example_train_aneurysm_cubes.py --wandb --val-fold 2
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd):
    """Run a command and print it first."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Example training configurations for aneurysm cubes")
    parser.add_argument('--test', action='store_true', help='Quick test run (1 epoch, small dataset)')
    parser.add_argument('--viz', action='store_true', help='Generate visualizations (5 samples)')
    parser.add_argument('--full', action='store_true', help='Full training run (200 epochs)')
    parser.add_argument('--positive-only', action='store_true', help='Train only on positive samples')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--val-fold', type=int, default=0, help='Fold to use for validation (0-4)')
    parser.add_argument('--cubes-dir', type=str, default='aneurysm_cubes', help='Path to aneurysm cubes directory')
    
    args = parser.parse_args()
    
    # Check if aneurysm cubes directory exists
    cubes_dir = Path(args.cubes_dir)
    if not cubes_dir.exists():
        print(f"Error: Aneurysm cubes directory '{cubes_dir}' does not exist!")
        print("Please run prepare_aneurysm_cubes.py first to generate the cubes.")
        return 1
    
    # Check if fold directories exist
    fold_dirs = [cubes_dir / f"fold_{i}" for i in range(5)]
    missing_folds = [f for f in fold_dirs if not f.exists()]
    if missing_folds:
        print(f"Error: Missing fold directories: {[str(f) for f in missing_folds]}")
        print("Please ensure all fold directories (fold_0 to fold_4) exist.")
        return 1
    
    # Base command
    base_cmd = [
        'python3', 'train_unet_aneurysm_cubes.py',
        '--cubes-dir', args.cubes_dir,
        '--val-fold', str(args.val_fold),
        '--lr', '1e-4',
    ]
    
    # Configure based on options
    if args.test:
        print("=== Quick Test Configuration ===")
        cmd = base_cmd + [
            '--epochs', '1',
            '--small-dataset',
            '--pos-weight', '50.0',
        ]
        if args.viz:
            cmd += ['--viz-samples', '5']
        
    elif args.full:
        print("=== Full Training Configuration ===")
        cmd = base_cmd + [
            '--epochs', '200',
            '--pos-weight', '128.0',
        ]
        
    elif args.positive_only:
        print("=== Positive-Only Training Configuration ===")
        cmd = base_cmd + [
            '--epochs', '100',
            '--only-positive',
            '--only-positive-val',
            '--pos-weight', '10.0',  # Lower weight since we only have positives
        ]
        
    else:
        print("=== Default Configuration ===")
        cmd = base_cmd + [
            '--epochs', '150',
            '--pos-weight', '100.0',
        ]
    
    # Add W&B logging if requested
    if args.wandb:
        cmd += [
            '--wandb',
            '--wandb-project', 'unet_aneurysm_cubes',
            '--wandb-name', f'fold_{args.val_fold}_experiment',
        ]
    
    # Print configuration summary
    print(f"Validation fold: {args.val_fold}")
    print(f"Training folds: {[i for i in range(5) if i != args.val_fold]}")
    print(f"Cubes directory: {args.cubes_dir}")
    print()
    
    # Run the training
    return run_command(cmd)

if __name__ == '__main__':
    sys.exit(main())
