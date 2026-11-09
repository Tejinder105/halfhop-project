#!/bin/bash
# ==============================================================================
# run_kaggle_adahop_benchmark.sh
# Phase 3 / Novel Method: AdaHop vs. Uniform Half-Hop & Baselines
#
# GPU 0: Wisconsin & Texas (Depth L=4)
# GPU 1: Chameleon & Cornell (Depth L=4)
#
# Usage on Kaggle:
#   cd /kaggle/working/halfhop-project
#   git pull
#   bash experiments/run_kaggle_adahop_benchmark.sh
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Checking GPU Environment ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== 2. Installing Dependencies on Kaggle ==="
pip install -q pyyaml scipy matplotlib torch-geometric

echo "=== 3. Launching AdaHop Benchmark on GPU 0 and GPU 1 ==="
mkdir -p experiments/results/adahop

# Job on GPU 0: Wisconsin and Texas
(
    echo "--- Starting Wisconsin on GPU 0 ---"
    CUDA_VISIBLE_DEVICES=0 python -u -m experiments.benchmark_adahop \
        --dataset wisconsin \
        --depth 4 \
        --epochs 200 \
        --patience 50 \
        --device cuda:0 \
        --out experiments/results/adahop

    echo "--- Starting Texas on GPU 0 ---"
    CUDA_VISIBLE_DEVICES=0 python -u -m experiments.benchmark_adahop \
        --dataset texas \
        --depth 4 \
        --epochs 200 \
        --patience 50 \
        --device cuda:0 \
        --out experiments/results/adahop
) | tee experiments/results/adahop/gpu0_benchmark_log.txt &
PID0=$!

# Job on GPU 1: Chameleon and Cornell
(
    echo "--- Starting Chameleon on GPU 1 ---"
    CUDA_VISIBLE_DEVICES=1 python -u -m experiments.benchmark_adahop \
        --dataset chameleon \
        --depth 4 \
        --epochs 200 \
        --patience 50 \
        --device cuda:0 \
        --out experiments/results/adahop

    echo "--- Starting Cornell on GPU 1 ---"
    CUDA_VISIBLE_DEVICES=1 python -u -m experiments.benchmark_adahop \
        --dataset cornell \
        --depth 4 \
        --epochs 200 \
        --patience 50 \
        --device cuda:0 \
        --out experiments/results/adahop
) | tee experiments/results/adahop/gpu1_benchmark_log.txt &
PID1=$!

echo "GPU 0 PID: $PID0 | GPU 1 PID: $PID1"
echo "Waiting for AdaHop benchmark jobs to finish..."

wait $PID0 $PID1

echo ""
echo "=== 4. AdaHop Benchmark Completed Successfully! ==="
ls -la experiments/results/adahop/

