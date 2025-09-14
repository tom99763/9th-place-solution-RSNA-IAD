#!/usr/bin/env bash
set -euo pipefail

# This script runs axial-only training for folds 0,1,2,3,4
# Usage:
#   scripts/train_axial_all_folds.sh [extra args...]
# Examples:
#   scripts/train_axial_all_folds.sh --epochs 100 --device 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"

EXTRA_ARGS=("$@")

for f in 0 1 2 3 4; do
  echo "\n===== Training axial, fold ${f} ====="
  python3 -m src.run_yolo_pipeline_planes --fold "${f}" --planes axial "${EXTRA_ARGS[@]}"
done

echo "\nAll folds completed."


