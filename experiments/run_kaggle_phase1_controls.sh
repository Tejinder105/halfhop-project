#!/bin/bash
# ==============================================================================
# Run Phase 1 Controls on Kaggle (2x T4 GPU)
# Runs Texas & Cornell on GPU 0, Wisconsin & Chameleon on GPU 1
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Checking GPU Environment ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== 2. Installing Dependencies on Kaggle ==="
pip install -q pyyaml matplotlib tqdm torch-geometric

echo "=== 3. Running Phase 1 Controls across GPU 0 and GPU 1 ==="
mkdir -p experiments/results/phase1_controls

CUDA_VISIBLE_DEVICES=0 python -u -m experiments.phase1_controls \
    --datasets texas,cornell \
    --epochs 200 \
    --seed 42 \
    | tee experiments/results/phase1_controls/gpu0_texas_cornell.txt &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u -m experiments.phase1_controls \
    --datasets wisconsin,chameleon \
    --epochs 200 \
    --seed 42 \
    | tee experiments/results/phase1_controls/gpu1_wisconsin_chameleon.txt &
PID1=$!

wait $PID0 $PID1

echo "=== 4. All Phase 1 Controls runs completed! ==="
echo "Results stored in experiments/results/phase1_controls/"
ls -la experiments/results/phase1_controls/ || true

