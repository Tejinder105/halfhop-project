#!/usr/bin/env bash
# Hyperparameter sweep on Boundary Half-Hop geometric thresholding (Chair).
# Runs pairwise experiments across 2 Kaggle GPUs (Tesla T4).
#
# Sweeps:
# - Baseline: none
# - Stricter quantiles with physical crease gate:
#     Q=0.90 (top 10% sharpest) + min_score=0.15
#     Q=0.95 (top 5% sharpest)  + min_score=0.15
#     Q=0.98 (top 2% sharpest)  + min_score=0.15
#     Q=0.95 + min_score=0.20 (very strict crease only)

set -e
cd /kaggle/working/halfhop-project
export OMP_NUM_THREADS=1

OUTPUT=experiments/results/threshold_sweep_chair.jsonl
DATA=data/ShapeNet
EPOCHS=50
BATCH=16
HIDDEN=128
DEPTH=3
K=16
LR=0.001

# Prepare dataset
python -m experiments.shapenet_partseg.run_partseg \
    --prepare-only --category Chair --data-dir "$DATA"

run_gpu() {
    local GPU=$1; shift
    echo "[GPU $GPU] starting: $*"
    CUDA_VISIBLE_DEVICES=$GPU python -m experiments.shapenet_partseg.run_partseg \
        --category Chair --epochs $EPOCHS --batch-size $BATCH --hidden $HIDDEN \
        --depth $DEPTH --k $K --lr $LR \
        --data-dir "$DATA" --output "$OUTPUT" "$@" &
}

echo "=== Pair 1: Q=0.90 vs Q=0.95 (min_score=0.15) ==="
run_gpu 0 --condition boundary --curv-quantile 0.90 --min-score 0.15 --seed 0
run_gpu 1 --condition boundary --curv-quantile 0.95 --min-score 0.15 --seed 0
wait
echo "=== Pair 1 Complete ==="

echo "=== Pair 2: Q=0.98 vs Q=0.95 (min_score=0.20) ==="
run_gpu 0 --condition boundary --curv-quantile 0.98 --min-score 0.15 --seed 0
run_gpu 1 --condition boundary --curv-quantile 0.95 --min-score 0.20 --seed 0
wait
echo "=== Pair 2 Complete ==="

echo "=== All Sweep Runs Complete ==="
python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/threshold_sweep_chair.jsonl")
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"\n{'Condition':<10} {'Quantile':>10} {'MinScore':>10} {'Best Val mIoU':>15} {'Test mIoU':>12} {'Test Acc':>10}")
    print("-" * 72)
    for r in sorted(rows, key=lambda x: x.get('test_miou', 0), reverse=True):
        print(f"{r['condition']:<10} {str(r.get('curv_quantile')):>10} {str(r.get('min_score')):>10} "
              f"{r.get('best_val_miou', 0):>15.4f} {r.get('test_miou', 0):>12.4f} {r.get('test_acc', 0):>10.4f}")
PY
