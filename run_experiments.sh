#!/bin/bash
# Runs all supervised experiments across 6 datasets and 6 models.
# Already-completed runs (those containing "FINAL RESULTS") are skipped
# automatically so the script can be safely resumed after interruption.
set -o pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

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
SSL_EPOCHS="${SSL_EPOCHS:-1000}"

has_marker() {
    local result_file="$1"
    local marker="$2"
    grep -Fq -- "$marker" "$result_file" 2>/dev/null
}

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
        if has_marker "$result_file" "FINAL RESULTS"; then
            echo "SKIP  ${model} on ${dataset} (already done)"
            continue
        fi
        if grep -Eq -- "CUDA error: out of memory|KeyboardInterrupt|RUN FAILED" "$result_file" 2>/dev/null; then
            echo "SKIP  ${model} on ${dataset} (previous run failed; delete ${result_file} to retry)"
            continue
        fi
        echo "RUN   ${model} on ${dataset}..."
        if python -u -m experiments.supervised.run \
            --dataset "$dataset" \
            --model "$model" \
            --epochs "$EPOCHS" \
            --verbose \
            > "$result_file" 2>&1; then
            echo "DONE  ${model} on ${dataset}"
        else
            printf '\nRUN FAILED: %s on %s\n' "$model" "$dataset" >> "$result_file"
            echo "FAILED ${model} on ${dataset}; see ${result_file}"
            exit 1
        fi
    done
done

# -----------------------------------------------------------------------
# Phase 8: Ablations (connectivity + initialization) on Texas
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Phase 8: Ablation Studies"
echo "=========================================="

if ! has_marker experiments/results/ablations/connectivity_texas.txt "CONNECTIVITY ABLATION"; then
    echo "RUN   connectivity ablation on texas..."
    python -u -m experiments.ablations.connectivity \
        --dataset texas \
        > experiments/results/ablations/connectivity_texas.txt 2>&1
    echo "DONE  connectivity ablation"
else
    echo "SKIP  connectivity ablation (already done)"
fi

if ! has_marker experiments/results/ablations/initialization_texas.txt "INITIALIZATION ABLATION"; then
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
SSL_DATASETS=("amazon-computers" "amazon-photos" "coauthor-cs")
SSL_OUTPUT_NAMES=("amazon_computers" "amazon_photo" "coauthor_cs")

for index in "${!SSL_DATASETS[@]}"; do
    dataset="${SSL_DATASETS[$index]}"
    output_name="${SSL_OUTPUT_NAMES[$index]}"
    for ssl_method in "bgrl" "grace"; do
        result_file="experiments/results/${output_name}_${ssl_method}.txt"
        if has_marker "$result_file" "FINAL SSL RESULTS"; then
            echo "SKIP  ${ssl_method} on ${dataset} (already done)"
            continue
        fi
        echo "RUN   ${ssl_method} on ${dataset}..."
        if python -u -m experiments.ssl.${ssl_method} \
            --dataset "$dataset" \
            --epochs "$SSL_EPOCHS" \
            | tee "$result_file"; then
            echo "DONE  ${ssl_method} on ${dataset}"
        else
            printf '\nRUN FAILED: %s on %s\n' "$ssl_method" "$dataset" >> "$result_file"
            echo "FAILED ${ssl_method} on ${dataset}; see ${result_file}"
            exit 1
        fi
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
