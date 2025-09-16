#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_mip_backbones.sh
# Runs MIP training on multiple backbones for folds 0-4.
# Uses configs/config_mip.yaml and sets experiment name per backbone for W&B.

MODELS=(
  #tf_efficientnetv2_s.in21k_ft_in1k
  resnet18.a1_in1k
  #seresnet50.a3_in1k
  #efficientnet_b2
)

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/[^A-Za-z0-9_-]/_/g')
  for FOLD_ID in {0..4}; do
    EXPERIMENT="mip_${SAFE_MODEL_NAME}_e150_fold${FOLD_ID}_all_folds_v2"
    echo -e "\n>>> Running: $MODEL | experiment=$EXPERIMENT | fold_id=$FOLD_ID"
    HYDRA_FULL_ERROR=1 \
    python3 train.py \
      --config-name config_mip \
      model.model_name="$MODEL" \
      model.in_chans=5 \
      model.img_size=512 \
      +model.global_pool_override=null \
      experiment="$EXPERIMENT" \
      fold_id=$FOLD_ID \
      trainer.max_epochs=150
  done
done

echo -e "\nAll runs completed."