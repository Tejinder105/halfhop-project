#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# ============================================================
# Fair ModelNet10 Experiment — Kaggle Free Tier Optimized
# Fixes all 7 flaws from run_adaptive.py:
#   F1: Controlled expansion (sparse_inv ~6.9x vs 21x for curvature)
#   F2: Correct metadata stored per condition (delays=(0,0,0) for none)
#   F3: Dynamic 3D data augmentation per epoch (rotation, jitter, scale)
#   F4: Multi-seed (seeds 0 1 2) with mean ± std and Wilcoxon test
#   F5: 50 epochs + cosine annealing LR + per-epoch JSONL logging
#   F6: Modern EdgeConv backbone (DGCNN-style relative position features)
#   F7: Isolated pre_transform cache per point count (no PyG cache collisions)
#
# New approach: sparse_inv delays only planar (low-curv) edges (m=1),
# leaving boundary/corner edges undelayed (m=0) so diagnostic landmarks
# flow freely across the object.
# ============================================================

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
