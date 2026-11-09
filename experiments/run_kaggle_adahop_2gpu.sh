#!/bin/bash
# ==============================================================================
# Run AdaHop Phase 3 Experiments on Kaggle (2x T4 GPU)
#
# Usage in a Kaggle Notebook code cell:
#   !bash experiments/run_kaggle_adahop_2gpu.sh
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Checking GPU Environment ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== 2. Installing Dependencies on Kaggle ==="
pip install -q pyyaml matplotlib tqdm torch-geometric
python -c "import torch; print('torch OK:', torch.__version__)"

echo "=== 3. Running AdaHop Parallel Multi-GPU Runner ==="
# Automatically uses GPU 0 and GPU 1 simultaneously across datasets
python -u -m experiments.run_parallel_2gpu \
    --datasets texas,cornell,wisconsin,chameleon \
    --epochs 200 \
    --seed 42

echo "=== 4. All Kaggle Experiments Completed Successfully! ==="
echo "Results are stored in experiments/results/adahop/"
ls -la experiments/results/adahop/ || true

