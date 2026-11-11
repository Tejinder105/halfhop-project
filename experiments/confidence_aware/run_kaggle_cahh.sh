#!/usr/bin/env bash
# Run CAHH on Kaggle 2×T4 for one category.
#
# Workflow:
#   1.  Train confidence MLP (CPU is fine; runs fast)
#   2.  Run GNN sweep (hh-cahh + baselines) on GPU
#
# Usage:
#   CATEGORY=Chair SEED=0 bash experiments/confidence_aware/run_kaggle_cahh.sh
#
# Environment:
#   CATEGORY : ShapeNet category (default: Chair)
#   SEED     : RNG seed for GNN training (default: 0)
#   GPU      : CUDA device id for GNN training (default: 0)
#   DATA_DIR : path to ShapeNet root (default: /kaggle/input/shapenet-part/data/ShapeNet)

set -euo pipefail

CATEGORY="${CATEGORY:-Chair}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-/kaggle/input/shapenet-part/data/ShapeNet}"
RESULTS_DIR="experiments/results"
MLP_CKPT="${RESULTS_DIR}/cahh_mlp_${CATEGORY}.pt"

export OMP_NUM_THREADS=1

echo "===== CAHH: ${CATEGORY} seed=${SEED} GPU=${GPU} ====="

# --- Step 1: train confidence MLP (CPU) ---
echo "--- Training confidence MLP ---"
python -m halfhop.confidence.train_gate \
    --category "${CATEGORY}" \
    --data-dir "${DATA_DIR}" \
    --k 16 \
    --epochs 50 \
    --lr 1e-3 \
    --batch-size 4096 \
    --dropout 0.2 \
    --budget 0.10 \
    --seed "${SEED}" \
    --device cpu \
    --output "${MLP_CKPT}" \
    --metrics-output "${RESULTS_DIR}/cahh_mlp_metrics.jsonl"

echo "MLP checkpoint saved: ${MLP_CKPT}"

# --- Step 2: GNN sweep (all four arms) ---
echo "--- Running GNN sweep ---"
CUDA_VISIBLE_DEVICES="${GPU}" python -m experiments.confidence_aware.train_cahh \
    --category "${CATEGORY}" \
    --mlp-ckpt "${MLP_CKPT}" \
    --data-dir "${DATA_DIR}" \
    --sweep \
    --seed "${SEED}" \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-3 \
    --hidden 64 \
    --depth 3 \
    --dropout 0.5 \
    --k 16 \
    --budget 0.10 \
    --device auto \
    --output "${RESULTS_DIR}/cahh_segmentation.jsonl"

echo "--- Evaluation ---"
CUDA_VISIBLE_DEVICES="${GPU}" python -m experiments.confidence_aware.evaluate \
    --category "${CATEGORY}" \
    --mlp-ckpt "${MLP_CKPT}" \
    --data-dir "${DATA_DIR}" \
    --k 16 \
    --budget 0.10 \
    --device auto \
    --seg-results "${RESULTS_DIR}/cahh_segmentation.jsonl" \
    --output "${RESULTS_DIR}/cahh_eval_metrics.jsonl"

echo "===== Done: ${CATEGORY} seed=${SEED} ====="

