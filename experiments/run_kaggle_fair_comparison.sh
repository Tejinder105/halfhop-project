#!/bin/bash
# ==============================================================================
# run_kaggle_fair_comparison.sh
# Step 2: Equal-Budget Fair Comparison (100 trials per method)
#
# GPU 0: Wisconsin (100 trials, 10 splits)
# GPU 1: Chameleon (100 trials, search_splits=3, then re-evaluates winner on 10 splits)
#
# Usage on Kaggle:
#   bash experiments/run_kaggle_fair_comparison.sh
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Checking GPU Environment ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== 2. Installing Dependencies on Kaggle ==="
pip install -q pyyaml scipy matplotlib torch-geometric

echo "=== 3. Launching Fair Comparison: Wisconsin (GPU 0) and Chameleon (GPU 1) ==="
mkdir -p experiments/results/fair

# Wisconsin on GPU 0 (100 trials, 10 splits)
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.fair_comparison \
    --dataset wisconsin \
    --backbone gcn \
    --trials 100 \
    --search_splits 10 \
    --device cuda:0 \
    | tee experiments/results/fair/wisconsin_fair_log.txt &
PID0=$!

# Chameleon on GPU 1 (100 trials, search_splits=3, re-evaluates winner on 10 splits)
CUDA_VISIBLE_DEVICES=1 python -u -m experiments.fair_comparison \
    --dataset chameleon \
    --backbone gcn \
    --trials 100 \
    --search_splits 3 \
    --device cuda:0 \
    | tee experiments/results/fair/chameleon_fair_log.txt &
PID1=$!

echo "Wisconsin PID: $PID0, Chameleon PID: $PID1"
echo "Waiting for fair comparison runs to complete..."

wait $PID0 $PID1

echo ""
echo "=== 4. Fair Comparison Completed! ==="
ls -la experiments/results/fair/

