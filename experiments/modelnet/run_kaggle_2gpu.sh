#!/bin/bash
# ModelNet10 Half-Hop on Kaggle 2x T4 (one model per GPU).
# Usage: bash experiments/modelnet/run_kaggle_2gpu.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

EPOCHS="${EPOCHS:-100}"
BS="${BS:-4}"
P="${P:-0.5}"

mkdir -p experiments/results

echo "GPU0: gcn | GPU1: hh-gcn"
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.modelnet.run \
  --dataset 10 --model gcn --epochs "$EPOCHS" --batch-size "$BS" \
  --device cuda --p "$P" \
  --output experiments/results/modelnet10_gcn.txt &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u -m experiments.modelnet.run \
  --dataset 10 --model hh-gcn --epochs "$EPOCHS" --batch-size "$BS" \
  --device cuda --p "$P" \
  --output experiments/results/modelnet10_hh-gcn.txt &
PID1=$!

wait $PID0 $PID1
echo "Done pair 1 (gcn / hh-gcn)"

echo "GPU0: sage | GPU1: hh-sage"
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.modelnet.run \
  --dataset 10 --model sage --epochs "$EPOCHS" --batch-size "$BS" \
  --device cuda --p "$P" \
  --output experiments/results/modelnet10_sage.txt &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u -m experiments.modelnet.run \
  --dataset 10 --model hh-sage --epochs "$EPOCHS" --batch-size "$BS" \
  --device cuda --p "$P" \
  --output experiments/results/modelnet10_hh-sage.txt &
PID1=$!

wait $PID0 $PID1
echo "Done pair 2 (sage / hh-sage)"
echo "Results in experiments/results/modelnet10_*.txt"
