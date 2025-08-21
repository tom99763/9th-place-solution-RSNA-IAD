#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_slice_config.sh
# Simple runner for slice-based training using configs/config_slices.yaml
# Customize models or folds by editing arrays below or exporting FOLDS env var.

# List of backbone model names (timm names). Add/remove as needed.
MODELS=(
  resnet18.a1_in1k
  # efficientnet_b2
  # convnext_tiny.fb_in22k_ft_in1k
)

# Space-separated list of folds to run (override via: FOLDS="0 1 2" bash run_slice_config.sh)
FOLDS=${FOLDS:-"0 1"}

# Optional overrides
IMG_SIZE=${IMG_SIZE:-512}
MAX_EPOCHS=${MAX_EPOCHS:-120}
CONFIG_NAME=config_slices

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/[^A-Za-z0-9_-]/_/g')
  for FOLD in $FOLDS; do
    EXPERIMENT="slices_${SAFE_MODEL_NAME}_e${MAX_EPOCHS}_fold${FOLD}"
    echo -e "\n>>> Running slice training: model=$MODEL fold=$FOLD experiment=$EXPERIMENT"
    HYDRA_FULL_ERROR=1 \
    python3 train.py \
      --config-name "$CONFIG_NAME" \
      model.model_name="$MODEL" \
      model.in_chans=8 \
      model.img_size=$IMG_SIZE \
      experiment="$EXPERIMENT" \
      fold_id=$FOLD \
      trainer.max_epochs=$MAX_EPOCHS
  done
done

echo -e "\nAll slice runs completed."
