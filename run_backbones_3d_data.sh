#!/usr/bin/env bash
set -euo pipefail


FOLD_ID="${1:-0}"
NUM_WINDOWS="${2:-1}"
DEPTH=32
IN_CHANS=$((DEPTH * NUM_WINDOWS))

# List of timm model names to iterate over
MODELS=(
  "tf_efficientnetv2_s.in21k_ft_in1k"
  "resnet18.a1_in1k"
  "seresnet50.a3_in1k"
)

EPOCHS=120
IMG_SIZE=384

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/[^A-Za-z0-9_-]/_/g')
  EXPERIMENT="vol3d_${SAFE_MODEL_NAME}_w${NUM_WINDOWS}_e${EPOCHS}_fold${FOLD_ID}"

  echo -e "\n>>> Running: $MODEL | experiment=$EXPERIMENT | fold_id=$FOLD_ID | in_chans=$IN_CHANS"
  HYDRA_FULL_ERROR=1 \
  python3 train.py \
    --config-name config_volume3d \
    model.model_name="$MODEL" \
    model.in_chans=$IN_CHANS \
    model.img_size=$IMG_SIZE \
    experiment="$EXPERIMENT" \
    fold_id=$FOLD_ID \
    trainer.max_epochs=$EPOCHS
    # Append extra overrides below if needed, e.g. optimizer.lr=1e-4 batch_size=2

done

echo -e "\nAll 3D volume runs completed."
