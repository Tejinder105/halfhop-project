"""Option 2: Hyperparameter Sensitivity Study for Half-Hop.

Evaluates combinations of (alpha, p, slow_node_init) on heterophilic benchmarks
across all 10 splits. Saves results to JSONL/CSV and can produce 2D heatmap plots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from halfhop.datasets import load_dataset
from halfhop.splits import get_split
from halfhop.training import run_training_loop
from halfhop.reproducibility import set_seed
from halfhop.hh_gcn import HHGCN
from halfhop.hh_graphsage import HHGraphSAGE
from halfhop.hh_gat import HHGAT
from experiments.supervised.run import load_config


def build_hh_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
    depth: int,
    alpha: float,
    p: float,
    slow_node_init: str = "linear",
    connectivity: str = "proposed",
):
    model_name = model_name.lower()
    if model_name in ("hh-gcn", "gcn"):
        return HHGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            alpha=alpha,
            p=p,
            dropout=dropout,
            depth=depth,
            slow_node_init=slow_node_init,
            connectivity=connectivity,
        )
    elif model_name in ("hh-sage", "hh-graphsage", "sage", "graphsage"):
        return HHGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            alpha=alpha,
            p=p,
            dropout=dropout,
            depth=depth,
            slow_node_init=slow_node_init,
            connectivity=connectivity,
        )
    elif model_name in ("hh-gat", "gat"):
        return HHGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            alpha=alpha,
            p=p,
            dropout=dropout,
            depth=depth,
            slow_node_init=slow_node_init,
            connectivity=connectivity,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def run_grid_sweep(
    dataset_name: str,
    model_name: str,
    alphas: list[float],
    ps: list[float],
    inits: list[str],
    epochs: int,
    seed: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"sensitivity_{dataset_name}_{model_name}.jsonl"

    existing_runs = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                key = (round(item["alpha"], 4), round(item["p"], 4), item["init"])
                existing_runs[key] = item

    set_seed(seed)
    dataset, data = load_dataset(dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    config = load_config(dataset_name, model_name)
    lr = config.get("lr", 0.01)
    wd = config.get("wd", 5e-4)
    hidden = config.get("hidden", 64)
    depth = config.get("depth", 2)
    dropout = config.get("dropout", 0.5)

    num_splits = data.train_mask.size(1) if data.train_mask.dim() == 2 else 10

    results_matrix = []

    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SENSITIVITY: {model_name.upper()} on {dataset_name.upper()}")
    print(f"Alphas: {alphas} | Ps: {ps} | Inits: {inits}")
    print(f"{'='*70}")

    for init in inits:
        for p in ps:
            for alpha in alphas:
                key = (round(alpha, 4), round(p, 4), init)
                if key in existing_runs:
                    row = existing_runs[key]
                    print(f"SKIP (cached) alpha={alpha:.2f}, p={p:.2f}, init={init}: {row['mean']:.2%} ± {row['std']:.2%}")
                    results_matrix.append(row)
                    continue

                split_accs = []
                for split_id in range(num_splits):
                    train_mask, val_mask, test_mask = get_split(data, split_id)
                    model = build_hh_model(
                        model_name=model_name,
                        in_channels=num_features,
                        hidden_channels=hidden,
                        out_channels=num_classes,
                        dropout=dropout,
                        depth=depth,
                        alpha=alpha,
                        p=p,
                        slow_node_init=init,
                    )
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                    _, best_test = run_training_loop(
                        model=model,
                        data=data,
                        optimizer=optimizer,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        test_mask=test_mask,
                        epochs=epochs,
                        verbose=False,
                    )
                    split_accs.append(best_test)

                acc_tensor = torch.tensor(split_accs)
                mean_acc = acc_tensor.mean().item()
                std_acc = acc_tensor.std(unbiased=True).item() if len(split_accs) > 1 else 0.0

                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "alpha": alpha,
                    "p": p,
                    "init": init,
                    "mean": round(mean_acc, 6),
                    "std": round(std_acc, 6),
                    "splits": [round(x, 6) for x in split_accs],
                }

                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

                print(f"DONE alpha={alpha:.2f}, p={p:.2f}, init={init:6s} -> {mean_acc:.2%} ± {std_acc:.2%}")
                results_matrix.append(row)

    # Plot 2D Heatmap for alpha x p under linear init
    linear_rows = [r for r in results_matrix if r["init"] == "linear"]
    if len(linear_rows) == len(alphas) * len(ps):
        plot_heatmap(alphas, ps, linear_rows, dataset_name, model_name, output_dir)

    # Summary analysis
    summarize_sensitivity(results_matrix, dataset_name, model_name)
    return results_matrix


def plot_heatmap(alphas: list[float], ps: list[float], rows: list[dict], dataset: str, model: str, output_dir: Path):
    grid = np.zeros((len(ps), len(alphas)))
    for r in rows:
        i = ps.index(r["p"])
        j = alphas.index(r["alpha"])
        grid[i, j] = r["mean"] * 100.0

    plt.figure(figsize=(7, 5))
    im = plt.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(im, label="Test Accuracy (%)")
    plt.xticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
    plt.yticks(range(len(ps)), [f"{p:.2f}" for p in ps])
    plt.xlabel(r"Interpolation Factor $\alpha$ (0.5 = symmetric)")
    plt.ylabel(r"Target Sampling Probability $p$")
    plt.title(f"Half-Hop Sensitivity: {model.upper()} on {dataset.upper()}")

    # Annotate values
    for i in range(len(ps)):
        for j in range(len(alphas)):
            val = grid[i, j]
            color = "white" if val < grid.mean() else "black"
            plt.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    plot_file = output_dir / f"heatmap_{dataset}_{model}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved heatmap to: {plot_file}")


def summarize_sensitivity(rows: list[dict], dataset: str, model: str):
    best_row = max(rows, key=lambda r: r["mean"])
    # Default is alpha=0.5, p=1.0, init=linear
    default_row = next(
        (r for r in rows if abs(r["alpha"] - 0.5) < 1e-4 and abs(r["p"] - 1.0) < 1e-4 and r["init"] == "linear"),
        None
    )

    print(f"\n{'-'*60}")
    print(f"SUMMARY: {model.upper()} on {dataset.upper()}")
    if default_row:
        print(f"  Default (alpha=0.5, p=1.0): {default_row['mean']:.2%} ± {default_row['std']:.2%}")
    print(f"  Best    (alpha={best_row['alpha']}, p={best_row['p']}, init={best_row['init']}): {best_row['mean']:.2%} ± {best_row['std']:.2%}")
    if default_row:
        gain = (best_row["mean"] - default_row["mean"]) * 100.0
        print(f"  Improvement over default: +{gain:.2f}%")
    print(f"{'-'*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Half-Hop Hyperparameter Sensitivity")
    parser.add_argument("--dataset", type=str, default="texas", help="Dataset name or 'all'")
    parser.add_argument("--model", type=str, default="hh-gcn", help="Model name")
    parser.add_argument("--alphas", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated alphas")
    parser.add_argument("--ps", type=str, default="0.25,0.5,0.75,1.0", help="Comma-separated ps")
    parser.add_argument("--inits", type=str, default="linear", help="Comma-separated inits: linear,zero,random")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output-dir", type=str, default="experiments/results/sensitivity", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    ps = [float(p.strip()) for p in args.ps.split(",") if p.strip()]
    inits = [i.strip() for i in args.inits.split(",") if i.strip()]

    datasets = ["texas", "wisconsin", "cornell", "chameleon"] if args.dataset == "all" else [args.dataset]

    for d in datasets:
        run_grid_sweep(
            dataset_name=d,
            model_name=args.model,
            alphas=alphas,
            ps=ps,
            inits=inits,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()

