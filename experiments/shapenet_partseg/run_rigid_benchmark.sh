#!/usr/bin/env bash
# Rigid Category Benchmark: Table & Airplane + Depth Compensation + Boundary-Specific Metrics.
#
# Tests the 3 hypotheses across 2 Kaggle GPUs (Tesla T4):
# 1. Boundary-Specific mIoU: Does Geometry-Aware Half-Hop draw cleaner part seams?
# 2. Rigid Categories: Do Table and Airplane benefit from sharp 90-degree seam detection?
# 3. Hop-Penalty Compensation: Does depth=4 give boundary messages enough hops to propagate?

set -e
cd /kaggle/working/halfhop-project
export OMP_NUM_THREADS=1

OUTPUT=experiments/results/rigid_partseg_benchmark.jsonl
DATA=data/ShapeNet
EPOCHS=50
BATCH=16
HIDDEN=128
K=16
LR=0.001

# Prepare datasets sequentially first
echo "=== Preparing Table and Airplane datasets ==="
python -m experiments.shapenet_partseg.run_partseg --prepare-only --category Table --data-dir "$DATA"
python -m experiments.shapenet_partseg.run_partseg --prepare-only --category Airplane --data-dir "$DATA"
echo "=== Datasets ready ==="

run_gpu() {
    local GPU=$1; shift
    echo "[GPU $GPU] starting: $*"
    CUDA_VISIBLE_DEVICES=$GPU python -m experiments.shapenet_partseg.run_partseg \
        --epochs $EPOCHS --batch-size $BATCH --hidden $HIDDEN \
        --k $K --lr $LR --data-dir "$DATA" --output "$OUTPUT" "$@" &
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. TABLE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
echo "=== TABLE (Pair 1): Baseline vs Boundary (Depth 4 compensated) ==="
run_gpu 0 --category Table --condition none     --depth 3 --seed 0
run_gpu 1 --category Table --condition boundary --depth 4 --curv-quantile 0.98 --min-score 0.15 --seed 0
wait
echo "=== Table Pair 1 Complete ==="

echo "=== TABLE (Pair 2): Boundary (Depth 3) vs Random Control ==="
run_gpu 0 --category Table --condition boundary --depth 3 --curv-quantile 0.98 --min-score 0.15 --seed 0
run_gpu 1 --category Table --condition random   --depth 3 --budget 0.05 --seed 0
wait
echo "=== Table Pair 2 Complete ==="

# ─────────────────────────────────────────────────────────────────────────────
# 2. AIRPLANE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
echo "=== AIRPLANE (Pair 3): Baseline vs Boundary (Depth 4 compensated) ==="
run_gpu 0 --category Airplane --condition none     --depth 3 --seed 0
run_gpu 1 --category Airplane --condition boundary --depth 4 --curv-quantile 0.98 --min-score 0.15 --seed 0
wait
echo "=== Airplane Pair 3 Complete ==="

echo "=== AIRPLANE (Pair 4): Boundary (Depth 3) vs Random Control ==="
run_gpu 0 --category Airplane --condition boundary --depth 3 --curv-quantile 0.98 --min-score 0.15 --seed 0
run_gpu 1 --category Airplane --condition random   --depth 3 --budget 0.05 --seed 0
wait
echo "=== Airplane Pair 4 Complete ==="

echo ""
echo "=========================================================================="
echo "                         BENCHMARK RESULTS SUMMARY"
echo "=========================================================================="
python - <<'PY'
import json
from pathlib import Path

p = Path("experiments/results/rigid_partseg_benchmark.jsonl")
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    header = f"{'Category':<10} {'Condition':<12} {'Depth':>5} {'Overall mIoU':>14} {'Boundary mIoU':>15} {'Interior mIoU':>15} {'Test Acc':>10}"
    print(header)
    print("-" * len(header))
    # Group by category
    for cat in ['Table', 'Airplane']:
        cat_rows = [r for r in rows if r.get('category') == cat]
        for r in sorted(cat_rows, key=lambda x: x.get('test_bnd_miou', 0), reverse=True):
            print(f"{r.get('category', ''):<10} "
                  f"{r.get('condition', ''):<12} "
                  f"{str(r.get('depth', '')):>5} "
                  f"{r.get('test_miou', 0)*100:>13.2f}% "
                  f"{r.get('test_bnd_miou', 0)*100:>14.2f}% "
                  f"{r.get('test_int_miou', 0)*100:>14.2f}% "
                  f"{r.get('test_acc', 0)*100:>9.2f}%")
        if cat == 'Table':
            print("-" * len(header))
PY

