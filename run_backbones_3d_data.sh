#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_backbones_3d_data.sh [FOLD_ID] [NUM_WINDOWS]
# Trains multiple backbones on prepared 3D volume data (prepare_data_3d.py output)
# treating depth (and optionally HU windows) as channels.
#
# Arguments:
#   FOLD_ID      : (default 0) validation fold
#   NUM_WINDOWS  : (default 3) number of HU windows applied on-the-fly during training
#                  Each window adds another (depth) set of channels.
#                  Effective input channels = DEPTH * NUM_WINDOWS when raw HU volumes stored.
#                  (Depth is 32 in current prepare script.)
#
# Notes:
# - If volumes were saved with STORE_NORMALIZED=True the dataset will not window augment and
#   input channels will just be DEPTH (override NUM_WINDOWS=1 to match).
# - Some backbones (e.g. ConvNeXt, ResNet, EfficientNet) support large in_chans directly.
#   For transformers expecting 3 channels, you can enable channel compression by adding
#     model.fuse_multi_window=true
#   which applies a 1x1 conv to reduce channels to 3.
# - Adjust trainer.max_epochs, batch_size, etc via additional Hydra overrides if needed.

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
