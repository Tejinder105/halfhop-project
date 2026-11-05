#!/bin/bash
# Run Half-Hop Extensions (Option 2 Sensitivity + Option 3 DropEdge) on Kaggle GPU.
# Usage in a Kaggle notebook cell:
#   !bash experiments/run_kaggle_extensions.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Installing dependencies on Kaggle ==="
pip install -q pyyaml matplotlib tqdm torch-geometric

echo "=== Running Option 2 & 3 on Kaggle GPU ==="
python -u -m experiments.run_extensions \
    --mode both \
    --datasets texas,cornell,wisconsin,chameleon \
    --models hh-gcn \
    --epochs 200

echo "=== All Kaggle runs complete! ==="
echo "Results and figures saved in experiments/results/"

