#!/usr/bin/env bash
set -e
cd /kaggle/working/halfhop-project
export OMP_NUM_THREADS=1

CATEGORIES="Chair Airplane Table Lamp"

for CAT in $CATEGORIES; do
  python -m experiments.shapenet_partseg.run_partseg \
    --prepare-only --category "$CAT" --data-dir data/ShapeNet
done

for CAT in $CATEGORIES; do
  for SEED in 0 1 2; do
    python -m experiments.shapenet_partseg.run_partseg \
      --sweep --category "$CAT" --seed "$SEED" --epochs 100 \
      --batch-size 16 --hidden 128 --depth 3 --k 16 \
      --data-dir data/ShapeNet \
      --output experiments/results/partseg.jsonl
  done
done

echo "DONE. Results in experiments/results/partseg.jsonl"
