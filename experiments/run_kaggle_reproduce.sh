#!/bin/bash
# ==============================================================================
# run_kaggle_reproduce.sh
# Step 1: Reproduce paper's Wisconsin (target 72.55%) and Chameleon (target 62.64%).
#
# Runs the two datasets in parallel across GPU 0 and GPU 1.
# Total runtime estimate: ~30-50 min on 2x T4 (1000 epoch x 10 splits x 6 experiments).
#
# Usage on Kaggle:
#   bash experiments/run_kaggle_reproduce.sh
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Checking GPU Environment ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== 2. Installing Dependencies on Kaggle ==="
pip install -q pyyaml scipy
python -c "import torch; print('torch OK, version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo "=== 3. Launching Wisconsin on GPU 0 and Chameleon on GPU 1 ==="
mkdir -p experiments/results/reproduction

CUDA_VISIBLE_DEVICES=0 python -u -m experiments.reproduce_wisconsin_chameleon \
    --dataset wisconsin \
    --device cuda:0 \
    | tee experiments/results/reproduction/wisconsin_repro_log.txt &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u -m experiments.reproduce_wisconsin_chameleon \
    --dataset chameleon \
    --device cuda:0 \
    | tee experiments/results/reproduction/chameleon_repro_log.txt &
PID1=$!

echo "Wisconsin PID: $PID0, Chameleon PID: $PID1"
echo "Waiting for both jobs to complete..."

wait $PID0 $PID1

echo ""
echo "=== 4. Results Summary ==="
python - <<'PYEOF'
import json, os

for ds, target in [("wisconsin", 72.55), ("chameleon", 62.64)]:
    path = f"experiments/results/reproduction/{ds}_reproduction.json"
    if not os.path.exists(path):
        print(f"{ds}: file not found at {path}")
        continue
    with open(path) as f:
        d = json.load(f)
    print(f"\n{ds.upper()} (paper target: {target}%)")
    best_exp = min(d["experiments"].items(), key=lambda item: abs(item[1]["mean"] - target))[0]
    for exp, v in d["experiments"].items():
        gap = v["mean"] - target
        marker = " <-- CLOSEST TO TARGET" if exp == best_exp else ""
        print(f"  {exp:35s}: {v['mean']:5.2f}% +/- {v['std']:.2f}% (gap: {gap:+.2f}%){marker}")
PYEOF
