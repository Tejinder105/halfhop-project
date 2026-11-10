#!/usr/bin/env bash
# Two-GPU parallel runner for ShapeNet Part Segmentation ablation.
#
# Strategy: pin each background process to one GPU using CUDA_VISIBLE_DEVICES.
# Two conditions (jobs) run simultaneously, one per GPU, then we wait before
# launching the next pair. This halves wall-clock time vs. a single-GPU run.
#
# Total scope: 4 categories x 2 seeds x 3 conditions = 24 runs
# Estimated time (2x T4, 50 epochs, batch=16): ~6-8 hours.

set -e
cd /kaggle/working/halfhop-project
export OMP_NUM_THREADS=1

OUTPUT=experiments/results/partseg_2gpu.jsonl
EPOCHS=50
BATCH=16
HIDDEN=128
DEPTH=3
K=16
LR=0.001
DATA=data/ShapeNet

CATEGORIES="Chair Airplane Table Lamp"
SEEDS="0 1"

# ── Phase 1: download all categories sequentially (avoid parallel disk writes) ──
echo "=== Downloading datasets ==="
for CAT in $CATEGORIES; do
    python -m experiments.shapenet_partseg.run_partseg \
        --prepare-only --category "$CAT" --data-dir "$DATA"
done
echo "=== All datasets ready ==="

# ── Helper: run one job on a given GPU in the background ──
run_gpu() {
    local GPU=$1; shift
    echo "[GPU $GPU] starting: $*"
    CUDA_VISIBLE_DEVICES=$GPU python -m experiments.shapenet_partseg.run_partseg \
        "$@" \
        --epochs $EPOCHS --batch-size $BATCH --hidden $HIDDEN \
        --depth $DEPTH --k $K --lr $LR \
        --data-dir "$DATA" --output "$OUTPUT" &
}

# ── Phase 2: run jobs in parallel pairs (GPU 0 + GPU 1) ──
echo "=== Starting training across 2 GPUs ==="

for CAT in $CATEGORIES; do
    for SEED in $SEEDS; do
        echo "--- $CAT seed=$SEED ---"

        # Pair 1: none on GPU 0, boundary on GPU 1
        run_gpu 0 --condition none     --category "$CAT" --seed "$SEED"
        run_gpu 1 --condition boundary --category "$CAT" --seed "$SEED"
        wait
        echo "[done] $CAT seed=$SEED: none + boundary"

        # Pair 2: random on GPU 0 (GPU 1 idles — only 3 conditions)
        run_gpu 0 --condition random   --category "$CAT" --seed "$SEED"
        wait
        echo "[done] $CAT seed=$SEED: random"
    done
done

echo ""
echo "=== ALL DONE ==="
echo "Results saved to $OUTPUT"
python - <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path("experiments/results/partseg_2gpu.jsonl").read_text().splitlines() if l.strip()]
print(f"\nTotal runs recorded: {len(rows)}")
by_cond = {}
for r in rows:
    by_cond.setdefault(r['condition'], []).append(r['test_miou'])
print(f"{'Condition':<12}  {'Runs':>5}  {'Mean test mIoU':>14}  {'Best':>8}")
for cond, vals in sorted(by_cond.items()):
    print(f"{cond:<12}  {len(vals):>5}  {sum(vals)/len(vals):>14.4f}  {max(vals):>8.4f}")
PY
