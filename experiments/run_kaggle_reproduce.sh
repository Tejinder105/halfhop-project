#!/usr/bin/env bash
# run_kaggle_reproduce.sh
# Step 1: Reproduce paper's Wisconsin (target 72.55%) and Chameleon (target 62.64%).
#
# Runs the two datasets in parallel across GPU 0 and GPU 1.
# Total runtime estimate: ~30-50 min on 2×T4 (1000 epoch × 10 splits × 6 experiments).
#
# Usage on Kaggle:
#   bash experiments/run_kaggle_reproduce.sh
#
# Results are saved to:
#   experiments/results/reproduction/wisconsin_reproduction.json
#   experiments/results/reproduction/chameleon_reproduction.json

set -e

echo "============================================="
echo " Step 1: Reproduce Wisconsin & Chameleon"
echo "============================================="
echo "GPU 0 → Wisconsin  (target: 72.55%)"
echo "GPU 1 → Chameleon  (target: 62.64%)"
echo ""

# Install dependencies (Kaggle has torch/pyg pre-installed)
pip install -q pyyaml scipy

# Wisconsin on GPU 0
python experiments/reproduce_wisconsin_chameleon.py \
    --dataset wisconsin \
    --device cuda:0 \
    > experiments/results/reproduction/wisconsin_repro_log.txt 2>&1 &
PID_WI=$!
echo "Wisconsin PID: $PID_WI (logs: experiments/results/reproduction/wisconsin_repro_log.txt)"

# Chameleon on GPU 1
python experiments/reproduce_wisconsin_chameleon.py \
    --dataset chameleon \
    --device cuda:1 \
    > experiments/results/reproduction/chameleon_repro_log.txt 2>&1 &
PID_CH=$!
echo "Chameleon PID: $PID_CH (logs: experiments/results/reproduction/chameleon_repro_log.txt)"

echo ""
echo "Both jobs launched. Waiting..."
wait $PID_WI && echo "Wisconsin DONE" || echo "Wisconsin FAILED"
wait $PID_CH && echo "Chameleon DONE" || echo "Chameleon FAILED"

echo ""
echo "============================================="
echo " Results summary"
echo "============================================="
python - <<'PYEOF'
import json, os

for ds, target in [("wisconsin", 72.55), ("chameleon", 62.64)]:
    path = f"experiments/results/reproduction/{ds}_reproduction.json"
    if not os.path.exists(path):
        print(f"{ds}: file not found")
        continue
    with open(path) as f:
        d = json.load(f)
    print(f"\n{ds.upper()} (paper target: {target}%)")
    for exp, v in d["experiments"].items():
        gap = v["mean"] - target
        marker = " <-- BEST" if abs(v["mean"] - target) == min(
            abs(x["mean"] - target) for x in d["experiments"].values()
        ) else ""
        print(f"  {exp:40s}: {v['mean']:5.2f}% ± {v['std']:.2f}%  (gap {gap:+.2f}){marker}")
PYEOF
