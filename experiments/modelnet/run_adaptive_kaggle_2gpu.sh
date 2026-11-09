#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_DIR="${DATA_DIR:-data/ModelNetPoint}"
EPOCHS="${EPOCHS:-100}"
POINTS="${POINTS:-1024}"
BS="${BS:-8}"
WORKERS="${WORKERS:-2}"
GPUS="${GPUS:-0,1}"

python -u -m experiments.modelnet.run_adaptive \
  --run-all \
  --gpus "$GPUS" \
  --data-dir "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --points "$POINTS" \
  --batch-size "$BS" \
  --num-workers "$WORKERS"
