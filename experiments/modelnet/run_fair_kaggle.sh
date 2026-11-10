#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# ============================================================
# Fair ModelNet10 Experiment — Kaggle Free Tier Optimized
# ============================================================

# 1. Clean up stale/leftover datasets and caches to ensure plenty of disk space on Kaggle
echo "=== Disk Space Management ==="
echo "Cleaning up any stale ModelNet40 / temporary files to free disk space..."
rm -rf data/ModelNetPoint data/ModelNet40 /tmp/ModelNet* data/ModelNet10/pts_* data/ModelNet10/*.zip 2>/dev/null || true

# If raw ModelNet10 was downloaded previously in ModelNetPoint, reuse it
if [ -d "data/ModelNetPoint/raw/ModelNet10" ] && [ ! -d "data/ModelNet10/raw/ModelNet10" ]; then
    echo "Reusing existing raw ModelNet10 files..."
    mkdir -p data/ModelNet10/raw
    cp -r data/ModelNetPoint/raw/ModelNet10 data/ModelNet10/raw/ 2>/dev/null || true
fi

df -h .
echo "============================="

# 2. Configuration
DATASET="${DATASET:-10}"
DATA_DIR="${DATA_DIR:-data/ModelNet10}"
POINTS="${POINTS:-512}"
BACKBONE="${BACKBONE:-edgeconv}"
EPOCHS="${EPOCHS:-50}"
BS="${BS:-16}"
HIDDEN="${HIDDEN:-128}"
GPUS="${GPUS:-0,1}"
CONDITIONS="${CONDITIONS:-none sparse_inv uniform curvature random}"
SEEDS="${SEEDS:-0 1 2}"

echo "============================================================"
echo "Fair ModelNet${DATASET} Ablation on Kaggle Free Tier"
echo "Backbone: ${BACKBONE} | Points: ${POINTS} | Epochs: ${EPOCHS} | GPUs: ${GPUS}"
echo "Conditions: ${CONDITIONS}"
echo "Seeds: ${SEEDS}"
echo "============================================================"

# 3. Prepare dataset sequentially first (prevents download race conditions and disk errors)
echo "Step 1: Preparing base dataset once sequentially..."
python -u -m experiments.modelnet.run_fair \
  --dataset "$DATASET" \
  --data-dir "$DATA_DIR" \
  --points "$POINTS" \
  --prepare-only

# 4. Run the ablation experiments in parallel across GPUs
echo "Step 2: Launching ablation matrix across GPUs..."
python -u -m experiments.modelnet.run_fair \
  --run-all \
  --dataset "$DATASET" \
  --data-dir "$DATA_DIR" \
  --points "$POINTS" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BS" \
  --hidden "$HIDDEN" \
  --gpus "$GPUS" \
  --conditions $CONDITIONS \
  --seeds $SEEDS
