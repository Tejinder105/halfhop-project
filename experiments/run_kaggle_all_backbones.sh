#!/bin/bash
# ==============================================================================
# run_kaggle_all_backbones.sh
# Runs full 10-split reproduction across GCN, GraphSAGE, and GAT with and
# without Half-Hop on Wisconsin (GPU 0) and Chameleon (GPU 1).
#
# Usage on Kaggle:
#   bash experiments/run_kaggle_all_backbones.sh
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Checking GPU Environment ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== 2. Installing Dependencies on Kaggle ==="
pip install -q pyyaml scipy
python -c "import torch; print('torch OK, version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo "=== 3. Launching Wisconsin on GPU 0 and Chameleon on GPU 1 ==="
mkdir -p experiments/results/reproduction

CUDA_VISIBLE_DEVICES=0 python -u -m experiments.reproduce_all_backbones \
    --dataset wisconsin \
    --device cuda:0 \
    --epochs 200 \
    | tee experiments/results/reproduction/wisconsin_all_backbones_log.txt &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u -m experiments.reproduce_all_backbones \
    --dataset chameleon \
    --device cuda:0 \
    --epochs 200 \
    | tee experiments/results/reproduction/chameleon_all_backbones_log.txt &
PID1=$!

echo "Wisconsin PID: $PID0, Chameleon PID: $PID1"
echo "Waiting for all-backbones runs to complete..."

wait $PID0 $PID1

echo ""
echo "=== 4. All-Backbones Reproduction Completed! ==="
ls -la experiments/results/reproduction/
