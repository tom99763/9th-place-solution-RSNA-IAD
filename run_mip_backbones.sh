#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_mip_backbones.sh
# Runs MIP training on multiple backbones for folds 0-4.
# Uses configs/config_mip.yaml and sets experiment name per backbone for W&B.
sleep 4600
MODELS=(
  tf_efficientnetv2_m.in21k_ft_in1k
)

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/[^A-Za-z0-9_-]/_/g')
  for FOLD_ID in {2..4}; do
    EXPERIMENT="mip_${SAFE_MODEL_NAME}_30_fold${FOLD_ID}_v4"
    echo -e "\n>>> Running: $MODEL | experiment=$EXPERIMENT | fold_id=$FOLD_ID"
    HYDRA_FULL_ERROR=1 \
    python3 train.py \
      --config-name config_mip \
      model.model_name="$MODEL" \
      model.in_chans=8 \
      model.img_size=512 \
      +model.global_pool_override=null \
      experiment="$EXPERIMENT" \
      fold_id=$FOLD_ID \
      trainer.max_epochs=30
  done
done

echo -e "\nAll runs completed."