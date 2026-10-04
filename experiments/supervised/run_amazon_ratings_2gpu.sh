#!/bin/bash
# Amazon-ratings: GCN vs HH-GCN on Kaggle 2x T4 (one model per GPU).
# Real e-commerce co-purchase graph (heterophilous suite).
set -euo pipefail
cd "$(dirname "$0")/../.."

EPOCHS="${EPOCHS:-200}"
mkdir -p experiments/results

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
