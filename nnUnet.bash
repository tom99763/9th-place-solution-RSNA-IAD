


#!/usr/bin/env bash
set -euo pipefail

# Delayed nnU-Net workflow script
# Dataset ID: 901
# This script performs planning/preprocessing and then starts the 3d_fullres training.
# Added logging + timestamps. Safe to re-run; preprocessing will skip existing steps if already done.
sleep 3600
export nnUNet_raw=$PWD/data/nnUNet_raw
export nnUNet_preprocessed=$PWD/data/nnUNet_preprocessed
export nnUNet_results=$PWD/data/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" logs/nnunet

echo "[nnUNet] Planning & preprocessing started at $(date)" >&2
nnUNetv2_plan_and_preprocess -d 901 --verify_dataset_integrity

echo "[nnUNet] Training (3d_fullres, all folds) started at $(date)" >&2
nnUNetv2_train 901 3d_fullres all

echo "[nnUNet] Completed at $(date)" >&2
