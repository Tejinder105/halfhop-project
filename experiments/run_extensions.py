"""Master Runner for Extensions:
Option 2 (Hyperparameter Sensitivity) and Option 3 (Half-Hop + DropEdge Combination).

Executes the experiments, produces publication-grade figures, and formats tables
for direct inclusion in LaTeX / Markdown reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.hyperparameter_sensitivity import run_grid_sweep
from experiments.dropedge_combination import run_dropedge_sweep


def main():
    parser = argparse.ArgumentParser(description="Run Half-Hop Extensions (Options 2 and 3)")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "sensitivity", "dropedge"],
        help="Which extension to run",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="texas,cornell,wisconsin,chameleon",
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="hh-gcn",
        help="Comma-separated model list (e.g., hh-gcn,hh-sage)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs per split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run quick fast sweep with fewer grid points (for rapid validation)",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Sensitivity grid
    if args.fast:
        alphas = [0.3, 0.5, 0.7]
        ps = [0.5, 1.0]
        depths = [2, 4]
        epochs = min(args.epochs, 100)
    else:
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        ps = [0.25, 0.5, 0.75, 1.0]
        depths = [2, 4, 8]
        epochs = args.epochs

    inits = ["linear"]

    print("=" * 80)
    print("STARTING HALF-HOP EXTENSIONS RUNNER")
    print(f"Mode: {args.mode.upper()} | Datasets: {datasets} | Models: {models}")
    print(f"Epochs: {epochs} | Fast Mode: {args.fast}")
    print("=" * 80)

    # 1. OPTION 2: HYPERPARAMETER SENSITIVITY
    if args.mode in ("both", "sensitivity"):
        print("\n" + "#" * 80)
        print("PHASE 1: OPTION 2 — HYPERPARAMETER SENSITIVITY (alpha x p)")
        print("#" * 80)
        sens_dir = Path("experiments/results/sensitivity")
        for d in datasets:
            for m in models:
                run_grid_sweep(
                    dataset_name=d,
                    model_name=m,
                    alphas=alphas,
                    ps=ps,
                    inits=inits,
                    epochs=epochs,
                    seed=args.seed,
                    output_dir=sens_dir,
                )

    # 2. OPTION 3: HALF-HOP + DROPEDGE COMBINATION
    if args.mode in ("both", "dropedge"):
        print("\n" + "#" * 80)
        print("PHASE 2: OPTION 3 — HALF-HOP + DROPEDGE COMBINATION")
        print("#" * 80)
        drop_dir = Path("experiments/results/dropedge_combo")
        for d in datasets:
            backbone = "gcn" if "gcn" in models[0] else "sage"
            run_dropedge_sweep(
                dataset_name=d,
                backbone=backbone,
                depths=depths,
                p_drop=0.2,
                alpha=0.5,
                p_hh=1.0,
                epochs=epochs,
                seed=args.seed,
                output_dir=drop_dir,
            )

    print("\n" + "=" * 80)
    print("ALL EXTENSION RUNS COMPLETED SUCCESSFULLY!")
    print("Results and figures saved to:")
    print("  - experiments/results/sensitivity/")
    print("  - experiments/results/dropedge_combo/")
    print("=" * 80)


if __name__ == "__main__":
    main()

