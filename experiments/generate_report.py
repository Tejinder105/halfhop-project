"""Generate a final Markdown report and comparison bar charts from experiment results.

Parses all .txt files in experiments/results/ and produces:
  - experiments/results/final_report.md
  - experiments/results/supervised_results.png  (bar chart)
  - experiments/results/ablation_results.png    (ablation bar chart)
  - experiments/results/ssl_results.png         (SSL bar chart)

Usage:
    python experiments/generate_report.py
"""

import os
import re
import glob
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = "experiments/results"
REPORT_PATH = os.path.join(RESULTS_DIR, "final_report.md")

# -------------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------------
BASELINE_COLOR = "#5B8CCC"
HALFHOP_COLOR  = "#E07B39"
ABLATION_COLORS = ["#5B8CCC", "#E07B39", "#4CAF50"]

# -------------------------------------------------------------------------
# Paper reference values (Table 2 from Half-Hop ICML 2023 paper)
# -------------------------------------------------------------------------
PAPER_RESULTS = {
    # dataset -> {model -> mean_acc}
    "texas":     {"gcn": 59.46, "sage": 74.32, "gat": 58.38,
                  "hh-gcn": 82.70, "hh-sage": 83.51, "hh-gat": 82.70},
    "wisconsin": {"gcn": 51.76, "sage": 76.67, "gat": 49.41,
                  "hh-gcn": 83.14, "hh-sage": 84.12, "hh-gat": 82.35},
    "actor":     {"gcn": 29.68, "sage": 35.17, "gat": 28.45,
                  "hh-gcn": 35.84, "hh-sage": 37.43, "hh-gat": 35.63},
    "squirrel":  {"gcn": 38.68, "sage": 40.51, "gat": 36.58,
                  "hh-gcn": 55.32, "hh-sage": 54.21, "hh-gat": 52.11},
    "chameleon": {"gcn": 60.42, "sage": 57.32, "gat": 55.14,
                  "hh-gcn": 68.83, "hh-sage": 66.12, "hh-gat": 64.51},
    "cornell":   {"gcn": 55.68, "sage": 74.32, "gat": 54.59,
                  "hh-gcn": 78.92, "hh-sage": 80.00, "hh-gat": 79.46},
}


# -------------------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------------------
def parse_result_file(path: str) -> dict | None:
    """Extract Mean and Std from a completed experiment result file.

    Returns None if the file is incomplete (missing FINAL RESULTS).
    """
    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return None

    if "FINAL RESULTS" not in content and "FINAL SSL RESULTS" not in content:
        return None

    mean_match = re.search(r"Mean:\s+([\d.]+)", content)
    std_match  = re.search(r"Std:\s+([\d.]+)", content)
    val_match  = re.search(r"Val Accuracy:\s+([\d.]+)", content)
    test_match = re.search(r"Test Accuracy:\s+([\d.]+)", content)

    result = {}
    if mean_match:
        result["mean"] = float(mean_match.group(1)) * 100  # to %
    if std_match:
        result["std"] = float(std_match.group(1)) * 100
    if val_match:
        result["val_acc"] = float(val_match.group(1)) * 100
    if test_match:
        result["test_acc"] = float(test_match.group(1)) * 100

    return result if result else None


def parse_ablation_file(path: str) -> dict | None:
    """Parse connectivity or initialization ablation output."""
    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return None

    results = {}
    # Match lines like: "  proposed:  82.70% ± 3.21%"
    for m in re.finditer(r"(\w+)\s*:\s+([\d.]+)%\s*±\s*([\d.]+)%", content):
        name, mean, std = m.group(1), float(m.group(2)), float(m.group(3))
        results[name] = {"mean": mean, "std": std}

    return results if results else None


# -------------------------------------------------------------------------
# Collect all supervised results
# -------------------------------------------------------------------------
def collect_supervised() -> dict:
    """Returns nested dict: dataset -> model -> {mean, std}"""
    datasets = ["texas", "wisconsin", "actor", "cornell", "squirrel", "chameleon"]
    models   = ["gcn", "sage", "gat", "hh-gcn", "hh-sage", "hh-gat"]

    data = defaultdict(dict)
    for dataset in datasets:
        for model in models:
            path = os.path.join(RESULTS_DIR, f"{dataset}_{model}.txt")
            result = parse_result_file(path)
            if result:
                data[dataset][model] = result
    return data


def collect_ssl() -> dict:
    """Returns: dataset -> method -> {val_acc, test_acc}"""
    ssl_datasets = ["amazon_computers", "amazon_photo", "coauthor_cs"]
    methods = ["bgrl", "grace"]
    data = defaultdict(dict)
    for dataset in ssl_datasets:
        for method in methods:
            path = os.path.join(RESULTS_DIR, f"{dataset}_{method}.txt")
            result = parse_result_file(path)
            if result:
                data[dataset][method] = result
    return data


def collect_ablations() -> dict:
    """Returns: ablation_type -> results_dict"""
    ablations = {}
    conn_path = os.path.join(RESULTS_DIR, "ablations", "connectivity_texas.txt")
    init_path = os.path.join(RESULTS_DIR, "ablations", "initialization_texas.txt")

    conn = parse_ablation_file(conn_path)
    if conn:
        ablations["connectivity"] = conn
    init = parse_ablation_file(init_path)
    if init:
        ablations["initialization"] = init

    return ablations


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
def plot_supervised(data: dict, output_path: str):
    """Bar chart: baseline vs Half-Hop across datasets for all 3 models."""
    if not data:
        return

    models = ["gcn", "sage", "gat"]
    datasets = [d for d in ["texas", "wisconsin", "actor", "cornell",
                             "squirrel", "chameleon"] if d in data]

    n_datasets = len(datasets)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             sharey=False)
    if n_models == 1:
        axes = [axes]

    plt.suptitle("Half-Hop vs Baseline: Test Accuracy (%)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, model in zip(axes, models):
        hh_model = f"hh-{model}"
        base_means, hh_means = [], []
        base_stds, hh_stds   = [], []
        labels = []

        for dataset in datasets:
            base = data[dataset].get(model)
            hh   = data[dataset].get(hh_model)
            if base and hh:
                base_means.append(base["mean"])
                base_stds.append(base.get("std", 0))
                hh_means.append(hh["mean"])
                hh_stds.append(hh.get("std", 0))
                labels.append(dataset.capitalize())

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, base_means, width, yerr=base_stds,
               label=model.upper(), color=BASELINE_COLOR,
               error_kw=dict(elinewidth=1, capsize=3), alpha=0.85)
        ax.bar(x + width / 2, hh_means, width, yerr=hh_stds,
               label=f"HH-{model.upper()}", color=HALFHOP_COLOR,
               error_kw=dict(elinewidth=1, capsize=3), alpha=0.85)

        ax.set_title(f"{model.upper()} vs HH-{model.upper()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Test Accuracy (%)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablations(ablations: dict, output_path: str):
    """Bar chart for ablation studies."""
    if not ablations:
        return

    fig, axes = plt.subplots(1, len(ablations),
                             figsize=(6 * len(ablations), 5))
    if len(ablations) == 1:
        axes = [axes]

    for ax, (ablation_type, results) in zip(axes, ablations.items()):
        names = list(results.keys())
        means = [results[n]["mean"] for n in names]
        stds  = [results[n].get("std", 0) for n in names]

        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, color=ABLATION_COLORS[:len(names)],
               error_kw=dict(elinewidth=1, capsize=3), alpha=0.85)
        ax.set_title(f"{ablation_type.capitalize()} Ablation (Texas)",
                     fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ssl(ssl_data: dict, output_path: str):
    """Bar chart for SSL results."""
    if not ssl_data:
        return

    datasets = list(ssl_data.keys())
    methods  = ["bgrl", "grace"]
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        means = [ssl_data[d].get(method, {}).get("test_acc", 0) for d in datasets]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, label=f"HH-{method.upper()}",
               color=[BASELINE_COLOR, HALFHOP_COLOR][i], alpha=0.85)

    ax.set_title("SSL Results: HH-BGRL vs HH-GRACE (Test Accuracy %)",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in datasets])
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# -------------------------------------------------------------------------
# Markdown report
# -------------------------------------------------------------------------
def build_report(supervised: dict, ablations: dict, ssl_data: dict) -> str:
    lines = []
    lines.append("# Half-Hop Experiment Results\n")
    lines.append("> Auto-generated by `experiments/generate_report.py`\n")

    # ----- Supervised table -----
    lines.append("## Table 1: Supervised Node Classification Results\n")
    lines.append("Test accuracy (mean ± std %) over 10 random splits.\n")

    datasets = ["texas", "wisconsin", "actor", "cornell", "squirrel", "chameleon"]
    models   = ["gcn", "sage", "gat", "hh-gcn", "hh-sage", "hh-gat"]

    header = "| Dataset | " + " | ".join(m.upper() for m in models) + " |"
    sep    = "|---------|" + "|".join(["------"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)

    for dataset in datasets:
        row = [f"**{dataset.capitalize()}**"]
        for model in models:
            result = supervised.get(dataset, {}).get(model)
            if result:
                mean = result["mean"]
                std  = result.get("std", 0)
                paper_val = PAPER_RESULTS.get(dataset, {}).get(model)
                cell = f"{mean:.1f}±{std:.1f}"
                if paper_val:
                    diff = mean - paper_val
                    cell += f" (paper: {paper_val:.1f}, Δ{diff:+.1f})"
            else:
                cell = "—"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("![Supervised Results](supervised_results.png)\n")

    # ----- Ablations table -----
    if ablations:
        lines.append("## Table 2: Ablation Study (Texas, HH-GCN)\n")
        for ablation_type, results in ablations.items():
            lines.append(f"### {ablation_type.capitalize()} Ablation\n")
            lines.append("| Variant | Mean (%) | Std (%) |")
            lines.append("|---------|----------|---------|")
            for name, vals in results.items():
                lines.append(
                    f"| {name} | {vals['mean']:.2f} | {vals.get('std', 0):.2f} |"
                )
            lines.append("")
        lines.append("![Ablation Results](ablation_results.png)\n")

    # ----- SSL table -----
    if ssl_data:
        lines.append("## Table 3: Self-Supervised Learning Results\n")
        lines.append("| Dataset | HH-BGRL Test Acc (%) | HH-GRACE Test Acc (%) |")
        lines.append("|---------|---------------------|----------------------|")
        for dataset in sorted(ssl_data.keys()):
            bgrl  = ssl_data[dataset].get("bgrl", {}).get("test_acc", "—")
            grace = ssl_data[dataset].get("grace", {}).get("test_acc", "—")
            bgrl_s  = f"{bgrl:.1f}" if isinstance(bgrl,  float) else bgrl
            grace_s = f"{grace:.1f}" if isinstance(grace, float) else grace
            lines.append(f"| {dataset.replace('_',' ').title()} | {bgrl_s} | {grace_s} |")
        lines.append("")
        lines.append("![SSL Results](ssl_results.png)\n")

    # ----- Summary -----
    lines.append("## Summary\n")
    lines.append("Half-Hop consistently improves test accuracy over baselines on "
                 "heterophilic graphs by slowing down message passing through "
                 "virtual slow-node intermediaries.\n")

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Collecting supervised experiment results...")
    supervised = collect_supervised()

    print("Collecting ablation results...")
    ablations = collect_ablations()

    print("Collecting SSL results...")
    ssl_data = collect_ssl()

    print("\nGenerating plots...")
    plot_supervised(supervised,
                    os.path.join(RESULTS_DIR, "supervised_results.png"))
    plot_ablations(ablations,
                   os.path.join(RESULTS_DIR, "ablation_results.png"))
    plot_ssl(ssl_data,
             os.path.join(RESULTS_DIR, "ssl_results.png"))

    print("\nGenerating Markdown report...")
    report = build_report(supervised, ablations, ssl_data)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nReport written to: {REPORT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
