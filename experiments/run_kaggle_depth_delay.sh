#!/bin/bash
# ==============================================================================
# run_kaggle_depth_delay.sh
# Step 3: Depth-Matched Dilation & Multi-Delay Study
#
# GPU 0: Wisconsin (Depths L in {2, 3, 4, 6, 8, 16})
# GPU 1: Chameleon (Depths L in {2, 3, 4, 6, 8, 16})
#
# Usage on Kaggle:
#   cd /kaggle/working/halfhop-project
#   git pull
#   bash experiments/run_kaggle_depth_delay.sh
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Checking GPU Environment ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== 2. Installing Dependencies on Kaggle ==="
pip install -q pyyaml scipy matplotlib torch-geometric

echo "=== 3. Launching Depth-Matched Study on GPU 0 (Wisconsin) and GPU 1 (Chameleon) ==="
mkdir -p experiments/results/depth

# Wisconsin on GPU 0
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.depth_matched_delay \
    --dataset wisconsin \
    --depths 2 3 4 6 8 16 \
    --methods plain lazywalk halfhop_k1 halfhop_k2 halfhop_k3 appnp gcn_res \
    --epochs 200 \
    --patience 50 \
    --device cuda:0 \
    --out experiments/results/depth \
    | tee experiments/results/depth/wisconsin_depth_log.txt &
PID0=$!

# Chameleon on GPU 1
CUDA_VISIBLE_DEVICES=1 python -u -m experiments.depth_matched_delay \
    --dataset chameleon \
    --depths 2 3 4 6 8 16 \
    --methods plain lazywalk halfhop_k1 halfhop_k2 halfhop_k3 appnp gcn_res \
    --epochs 200 \
    --patience 50 \
    --device cuda:0 \
    --out experiments/results/depth \
    | tee experiments/results/depth/chameleon_depth_log.txt &
PID1=$!

echo "Wisconsin PID: $PID0, Chameleon PID: $PID1"
echo "Waiting for Step 3 runs to complete..."

wait $PID0 $PID1

echo ""
echo "=== 4. Step 3 Depth-Matched Study Completed! ==="
ls -la experiments/results/depth/
