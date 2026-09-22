#!/bin/bash
# Runs all supervised experiments across 6 datasets and 6 models.
# Already-completed runs (those containing "FINAL RESULTS") are skipped
# automatically so the script can be safely resumed after interruption.

if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Create .venv first."
    exit 1
fi

DATASETS=("texas" "wisconsin" "actor" "cornell" "squirrel" "chameleon")
MODELS=("gcn" "sage" "gat" "hh-gcn" "hh-sage" "hh-gat")
EPOCHS="${EPOCHS:-100}"

mkdir -p experiments/results
mkdir -p experiments/results/ablations

# -----------------------------------------------------------------------
# Phase 6 & 7: Supervised experiments
# -----------------------------------------------------------------------
echo "=========================================="
echo "Phase 6 & 7: Supervised Experiments"
echo "=========================================="
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        result_file="experiments/results/${dataset}_${model}.txt"
        # Skip if already completed successfully
        if grep -q "FINAL RESULTS" "$result_file" 2>/dev/null; then
            echo "SKIP  ${model} on ${dataset} (already done)"
            continue
        fi
        echo "RUN   ${model} on ${dataset}..."
        python -u -m experiments.supervised.run \
            --dataset "$dataset" \
            --model "$model" \
            --epochs "$EPOCHS" \
            --verbose \
            > "$result_file" 2>&1
        echo "DONE  ${model} on ${dataset}"
    done
done

# -----------------------------------------------------------------------
# Phase 8: Ablations (connectivity + initialization) on Texas
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Phase 8: Ablation Studies"
echo "=========================================="

if ! grep -q "CONNECTIVITY ABLATION" experiments/results/ablations/connectivity_texas.txt 2>/dev/null; then
    echo "RUN   connectivity ablation on texas..."
    python -u -m experiments.ablations.connectivity \
        --dataset texas \
        > experiments/results/ablations/connectivity_texas.txt 2>&1
    echo "DONE  connectivity ablation"
else
    echo "SKIP  connectivity ablation (already done)"
fi

if ! grep -q "INITIALIZATION ABLATION" experiments/results/ablations/initialization_texas.txt 2>/dev/null; then
    echo "RUN   initialization ablation on texas..."
    python -u -m experiments.ablations.initialization \
        --dataset texas \
        > experiments/results/ablations/initialization_texas.txt 2>&1
    echo "DONE  initialization ablation"
else
    echo "SKIP  initialization ablation (already done)"
fi

# -----------------------------------------------------------------------
# Phase 9: SSL experiments (BGRL + GRACE) on Amazon/Coauthor
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Phase 9: SSL Experiments"
echo "=========================================="
SSL_DATASETS=("amazon_computers" "amazon_photo" "coauthor_cs")

for dataset in "${SSL_DATASETS[@]}"; do
    for ssl_method in "bgrl" "grace"; do
        result_file="experiments/results/${dataset}_${ssl_method}.txt"
        if grep -q "FINAL SSL RESULTS" "$result_file" 2>/dev/null; then
            echo "SKIP  ${ssl_method} on ${dataset} (already done)"
            continue
        fi
        echo "RUN   ${ssl_method} on ${dataset}..."
        python -u -m experiments.ssl.${ssl_method} \
            --dataset "$dataset" \
            > "$result_file" 2>&1
        echo "DONE  ${ssl_method} on ${dataset}"
    done
done

# -----------------------------------------------------------------------
# Phase 10: Generate report
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Phase 10: Generating Report"
echo "=========================================="
python experiments/generate_report.py

echo ""
echo "ALL EXPERIMENTS COMPLETE."
echo "Results are in: experiments/results/"
echo "Report: experiments/results/final_report.md"
