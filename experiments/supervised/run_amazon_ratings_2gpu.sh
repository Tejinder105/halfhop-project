#!/bin/bash
# Amazon-ratings: GCN vs HH-GCN on Kaggle 2x T4 (one model per GPU).
# Prepares the dataset once so two GPUs do not race on download/process.
set -euo pipefail
cd "$(dirname "$0")/../.."

EPOCHS="${EPOCHS:-200}"
mkdir -p experiments/results

echo "=== Prepare Amazon-ratings once (clear any corrupted cache) ==="
rm -rf data/Heterophilous
python -c "from halfhop.datasets import load_dataset; ds, data = load_dataset('amazon-ratings'); print(f'ready nodes={data.num_nodes} splits={data.train_mask.size(1)}')"

echo "=== GPU0: gcn | GPU1: hh-gcn  (Amazon-ratings, 10 splits) ==="
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.supervised.run \
  --dataset amazon-ratings --model gcn --epochs "$EPOCHS" \
  | tee experiments/results/amazon_ratings_gcn.txt &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u -m experiments.supervised.run \
  --dataset amazon-ratings --model hh-gcn --epochs "$EPOCHS" \
  | tee experiments/results/amazon_ratings_hh-gcn.txt &
PID1=$!

wait $PID0 $PID1
echo "Done. See experiments/results/amazon_ratings_*.txt"
