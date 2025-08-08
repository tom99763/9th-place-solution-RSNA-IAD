#!/usr/bin/env bash
set -euo pipefail

# Sleep for 5 hours (5 * 60 * 60 = 18000 seconds)
sleep 18000

# Ensure we run from the repo root (where train.py and configs/ live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start training with the desired Hydra config
python3 train.py --config-name=config_slices "$@"

