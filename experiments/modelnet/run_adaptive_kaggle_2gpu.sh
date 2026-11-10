#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Kaggle Free Tier Optimized Settings
# 13 GB CPU RAM safe, 2x Tesla T4 GPU accelerated
DATASET="${DATASET:-10}"
DATA_DIR="${DATA_DIR:-data/ModelNet10}"
EPOCHS="${EPOCHS:-30}"
POINTS="${POINTS:-512}"
BS="${BS:-16}"
WORKERS="${WORKERS:-0}"
GPUS="${GPUS:-0,1}"
EXPERIMENTS="${EXPERIMENTS:-none uniform curvature random}"

echo "=========================================================="
echo "Running ModelNet${DATASET} Adaptive Multi-Delay Half-Hop on Kaggle"
echo "GPUs: $GPUS | Points: $POINTS | Batch Size: $BS | Epochs: $EPOCHS"
echo "Data Dir: $DATA_DIR | Workers: $WORKERS"
echo "Experiments: $EXPERIMENTS"
echo "=========================================================="

python -u -m experiments.modelnet.run_adaptive \
  --run-all \
  --dataset "$DATASET" \
  --gpus "$GPUS" \
  --data-dir "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --points "$POINTS" \
  --batch-size "$BS" \
  --num-workers "$WORKERS" \
  --experiments $EXPERIMENTS
