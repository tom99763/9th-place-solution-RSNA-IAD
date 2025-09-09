


#!/usr/bin/env bash
set -euo pipefail

# Delayed nnU-Net workflow script
# Dataset ID: 902
# This script performs planning/preprocessing and then starts the 3d_fullres training.
# Added logging + timestamps. Safe to re-run; preprocessing will skip existing steps if already done.

export nnUNet_raw=$PWD/data/nnUNet_raw
export nnUNet_preprocessed=$PWD/data/nnUNet_preprocessed
export nnUNet_results=$PWD/data/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" logs/nnunet

echo "[nnUNet] Planning & preprocessing started at $(date)" >&2
nnUNetv2_plan_and_preprocess -d 902 --verify_dataset_integrity

echo "[nnUNet] Training (3d_fullres, all folds) started at $(date)" >&2
nnUNetv2_train 902 3d_fullres 0

echo "[nnUNet] Completed at $(date)" >&2
#1.2.826.0.1.3680043.8.498.7529432539245717936
