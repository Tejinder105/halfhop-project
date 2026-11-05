"""Option 3: Half-Hop + DropEdge Combination Experiment.

Systematically evaluates:
1. Baseline GNN
2. DropEdge alone
3. Half-Hop alone
4. Half-Hop + DropEdge combination

Across architectures (GCN, GraphSAGE), depths (2, 4, 8), and heterophilic datasets.
Saves detailed JSONL records and generates comparison plots and LaTeX tables.
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
from halfhop.gcn import GCN
from halfhop.graphsage import GraphSAGE
from halfhop.hh_gcn import HHGCN
from halfhop.hh_graphsage import HHGraphSAGE
from halfhop.hh_dropedge import DropEdgeGCN, DropEdgeGraphSAGE, HHDropEdgeGCN, HHDropEdgeGraphSAGE
from experiments.supervised.run import load_config


def build_experiment_model(
    arm: str,
    backbone: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    depth: int,
    dropout: float,
    p_drop: float = 0.2,
    alpha: float = 0.5,
    p_hh: float = 1.0,
):
    arm = arm.lower()
    backbone = backbone.lower()

    if arm == "baseline":
        if backbone == "gcn":
            return GCN(in_channels, hidden_channels, out_channels, dropout=dropout, depth=depth)
        elif backbone in ("sage", "graphsage"):
            return GraphSAGE(in_channels, hidden_channels, out_channels, dropout=dropout, depth=depth)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    elif arm == "dropedge":
        if backbone == "gcn":
            return DropEdgeGCN(in_channels, hidden_channels, out_channels, p_drop=p_drop, dropout=dropout, depth=depth)
        elif backbone in ("sage", "graphsage"):
            return DropEdgeGraphSAGE(in_channels, hidden_channels, out_channels, p_drop=p_drop, dropout=dropout, depth=depth)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    elif arm == "halfhop":
        if backbone == "gcn":
            return HHGCN(in_channels, hidden_channels, out_channels, alpha=alpha, p=p_hh, dropout=dropout, depth=depth)
        elif backbone in ("sage", "graphsage"):
            return HHGraphSAGE(in_channels, hidden_channels, out_channels, alpha=alpha, p=p_hh, dropout=dropout, depth=depth)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    elif arm in ("hh+dropedge", "combo"):
        if backbone == "gcn":
            return HHDropEdgeGCN(
                in_channels, hidden_channels, out_channels,
                p_drop=p_drop, alpha=alpha, p_hh=p_hh, dropout=dropout, depth=depth
            )
        elif backbone in ("sage", "graphsage"):
            return HHDropEdgeGraphSAGE(
                in_channels, hidden_channels, out_channels,
                p_drop=p_drop, alpha=alpha, p_hh=p_hh, dropout=dropout, depth=depth
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    else:
        raise ValueError(f"Unknown arm: {arm}")


def run_dropedge_sweep(
    dataset_name: str,
    backbone: str,
    depths: list[int],
    p_drop: float,
    alpha: float,
    p_hh: float,
    epochs: int,
    seed: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"dropedge_combo_{dataset_name}_{backbone}.jsonl"

    existing_runs = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                key = (item["arm"], item["depth"], round(item.get("p_drop", 0.0), 3))
                existing_runs[key] = item

    set_seed(seed)
    dataset, data = load_dataset(dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    config = load_config(dataset_name, backbone)
    lr = config.get("lr", 0.01)
    wd = config.get("wd", 5e-4)
    hidden = config.get("hidden", 64)
    dropout = config.get("dropout", 0.5)

    num_splits = data.train_mask.size(1) if data.train_mask.dim() == 2 else 10
    arms = ["baseline", "dropedge", "halfhop", "hh+dropedge"]

    print(f"\n{'='*75}")
    print(f"HALF-HOP + DROPEDGE STUDY: {backbone.upper()} on {dataset_name.upper()}")
    print(f"Depths: {depths} | p_drop: {p_drop} | alpha: {alpha} | p_hh: {p_hh}")
    print(f"{'='*75}")

    all_results = []

    for depth in depths:
        for arm in arms:
            cur_p_drop = p_drop if arm in ("dropedge", "hh+dropedge") else 0.0
            key = (arm, depth, round(cur_p_drop, 3))

            if key in existing_runs:
                row = existing_runs[key]
                print(f"SKIP (cached) {arm:12s} depth={depth}: {row['mean']:.2%} ± {row['std']:.2%}")
                all_results.append(row)
                continue

            split_accs = []
            for split_id in range(num_splits):
                train_mask, val_mask, test_mask = get_split(data, split_id)
                model = build_experiment_model(
                    arm=arm,
                    backbone=backbone,
                    in_channels=num_features,
                    hidden_channels=hidden,
                    out_channels=num_classes,
                    depth=depth,
                    dropout=dropout,
                    p_drop=cur_p_drop,
                    alpha=alpha,
                    p_hh=p_hh,
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

            acc_t = torch.tensor(split_accs)
            mean_acc = acc_t.mean().item()
            std_acc = acc_t.std(unbiased=True).item() if len(split_accs) > 1 else 0.0

            row = {
                "dataset": dataset_name,
                "backbone": backbone,
                "arm": arm,
                "depth": depth,
                "p_drop": cur_p_drop,
                "alpha": alpha,
                "p_hh": p_hh,
                "mean": round(mean_acc, 6),
                "std": round(std_acc, 6),
                "splits": [round(x, 6) for x in split_accs],
            }

            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            print(f"DONE {arm:12s} depth={depth} -> {mean_acc:.2%} ± {std_acc:.2%}")
            all_results.append(row)

    # Plot depth curve
    plot_depth_comparison(depths, arms, all_results, dataset_name, backbone, output_dir)
    # Generate summary table
    generate_summary_table(depths, arms, all_results, dataset_name, backbone, output_dir)
    return all_results


def plot_depth_comparison(depths: list[int], arms: list[str], results: list[dict], dataset: str, backbone: str, output_dir: Path):
    plt.figure(figsize=(7, 5))
    markers = {"baseline": "o", "dropedge": "s", "halfhop": "^", "hh+dropedge": "D"}
    colors = {"baseline": "#7f7f7f", "dropedge": "#2ca02c", "halfhop": "#1f77b4", "hh+dropedge": "#d62728"}
    labels = {
        "baseline": "Vanilla GNN",
        "dropedge": "DropEdge",
        "halfhop": "Half-Hop",
        "hh+dropedge": "Half-Hop + DropEdge (Ours)",
    }

    for arm in arms:
        arm_rows = [r for r in results if r["arm"] == arm]
        arm_rows.sort(key=lambda r: r["depth"])
        xs = [r["depth"] for r in arm_rows]
        ys = [r["mean"] * 100.0 for r in arm_rows]
        errs = [r["std"] * 100.0 for r in arm_rows]

        plt.errorbar(
            xs, ys, yerr=errs,
            label=labels[arm],
            marker=markers[arm],
            color=colors[arm],
            capsize=4,
            linewidth=2,
        )

    plt.xlabel("GNN Depth (Number of Layers)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Depth Robustness: {backbone.upper()} on {dataset.upper()}")
    plt.xticks(depths)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    plot_file = output_dir / f"depth_curve_{dataset}_{backbone}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved depth curve to: {plot_file}")


def generate_summary_table(depths: list[int], arms: list[str], results: list[dict], dataset: str, backbone: str, output_dir: Path):
    print(f"\n{'='*75}")
    print(f"SUMMARY TABLE: {backbone.upper()} on {dataset.upper()}")
    print(f"{'Depth':<8} | {'Baseline':<16} | {'DropEdge':<16} | {'Half-Hop':<16} | {'HH+DropEdge (Combo)':<20} | {'Synergy Delta'}")
    print("-" * 100)

    table_data = []
    for d in depths:
        by_arm = {r["arm"]: r for r in results if r["depth"] == d}
        b = by_arm.get("baseline", {}).get("mean", 0.0)
        de = by_arm.get("dropedge", {}).get("mean", 0.0)
        hh = by_arm.get("halfhop", {}).get("mean", 0.0)
        combo = by_arm.get("hh+dropedge", {}).get("mean", 0.0)

        synergy = combo - max(hh, de)
        syn_str = f"+{synergy*100:.2f}%" if synergy >= 0 else f"{synergy*100:.2f}%"

        b_str = f"{b:.2%}"
        de_str = f"{de:.2%}"
        hh_str = f"{hh:.2%}"
        combo_str = f"{combo:.2%}"

        print(f"{d:<8} | {b_str:<16} | {de_str:<16} | {hh_str:<16} | {combo_str:<20} | {syn_str}")
        table_data.append({"depth": d, "baseline": b, "dropedge": de, "halfhop": hh, "combo": combo, "synergy": synergy})

    print(f"{'='*75}\n")

    # Also save markdown summary
    md_file = output_dir / f"summary_{dataset}_{backbone}.md"
    with md_file.open("w", encoding="utf-8") as f:
        f.write(f"# Half-Hop + DropEdge Evaluation: {backbone.upper()} on {dataset.upper()}\n\n")
        f.write("| Depth | Baseline | DropEdge | Half-Hop | HH + DropEdge (Combo) | Synergy Delta |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in table_data:
            syn_str = f"+{row['synergy']*100:.2f}%" if row['synergy'] >= 0 else f"{row['synergy']*100:.2f}%"
            f.write(f"| {row['depth']} | {row['baseline']:.2%} | {row['dropedge']:.2%} | {row['halfhop']:.2%} | **{row['combo']:.2%}** | {syn_str} |\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Half-Hop + DropEdge Combination Experiments")
    parser.add_argument("--dataset", type=str, default="texas", help="Dataset name or 'all'")
    parser.add_argument("--backbone", type=str, default="gcn", help="Backbone: gcn or sage")
    parser.add_argument("--depths", type=str, default="2,4,8", help="Comma-separated depths")
    parser.add_argument("--p-drop", type=float, default=0.2, help="DropEdge drop probability")
    parser.add_argument("--alpha", type=float, default=0.5, help="Half-Hop alpha")
    parser.add_argument("--p-hh", type=float, default=1.0, help="Half-Hop p")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output-dir", type=str, default="experiments/results/dropedge_combo", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    datasets = ["texas", "cornell", "wisconsin", "chameleon"] if args.dataset == "all" else [args.dataset]

    for d in datasets:
        run_dropedge_sweep(
            dataset_name=d,
            backbone=args.backbone,
            depths=depths,
            p_drop=args.p_drop,
            alpha=args.alpha,
            p_hh=args.p_hh,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()

