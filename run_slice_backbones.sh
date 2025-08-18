#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_slice_backbones.sh [FOLD_ID]
# Runs slice-based training on multiple backbones up to epoch 150.
# Uses configs/config_slices.yaml and sets experiment name per backbone for W&B.

FOLD_ID="${1:-0}"

MODELS=(
  # ResNet family
  resnet18.a1_in1k
  #resnet50.a1_in1k
  #resnet26.bt_in1k 
 
  
  # ConvNeXt family
  #convnextv2_nano.fcmae_ft_in22k_in1k
  #convnextv2_nano.fcmae_ft_in22k_in1k

)

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/[^A-Za-z0-9_-]/_/g')
  EXPERIMENT="slice_${SAFE_MODEL_NAME}_e150_fold${FOLD_ID}"

  echo -e "\n>>> Running: $MODEL | experiment=$EXPERIMENT | fold_id=$FOLD_ID"
  HYDRA_FULL_ERROR=1 \
  python3 train.py \
    --config-name config_slices \
    model.model_name="$MODEL" \
    model.in_chans=8 \
    model.img_size=512 \
    +model.global_pool_override=null \
    experiment="$EXPERIMENT" \
    fold_id=$FOLD_ID \
    trainer.max_epochs=100
done

echo -e "\nAll runs completed."
