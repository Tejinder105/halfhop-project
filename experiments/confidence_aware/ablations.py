"""CAHH feature ablations (§5.4 of the CAHH plan).

Trains the confidence MLP with different subsets of the 4-dimensional edge
feature vector, then evaluates edge-prediction quality.  This tells you
which geometric signals drive boundary detection.

Feature variants:
  A  normal disagreement only        → input_dim=1
  B  curvature difference only       → input_dim=1
  C  normal + curvature              → input_dim=2
  D  normal + curvature + length     → input_dim=3
  E  all four features (default)     → input_dim=4

Usage::

    python -m experiments.confidence_aware.ablations \\
        --category Chair --k 16 --epochs 50 \\
        --output experiments/results/cahh_ablations.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch_geometric.datasets import ShapeNet
from tqdm import tqdm

import experiments.segmentation.proxy  # noqa: F401
from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)
from halfhop.confidence.train_gate import (
    build_edge_dataset,
    train_confidence_mlp,
)
from experiments.pointcloud.run import knn_edge_index
from halfhop.reproducibility import set_seed

# Map variant name → column indices to keep from the 4-vector.
VARIANTS: dict[str, list[int]] = {
    "A_normal":           [0],
    "B_curvature":        [1],
    "C_normal+curv":      [0, 1],
    "D_normal+curv+len":  [0, 1, 2],
    "E_all":              [0, 1, 2, 3],
}


def run_ablation(
    variant: str,
    cols: list[int],
    tr_feats, tr_targets, tr_valid,
    va_feats, va_targets, va_valid,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    budget: float,
    device: torch.device,
    seed: int,
) -> dict:
    set_seed(seed)
    # Subset to selected columns.
    tr_x = tr_feats[:, cols]
    va_x = va_feats[:, cols]

    model, metrics = train_confidence_mlp(
        tr_x, tr_targets, tr_valid,
        va_x, va_targets, va_valid,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        budget=budget,
        device=device,
        verbose=False,
    )
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="Chair")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default="data/ShapeNet")
    p.add_argument(
        "--output", default="experiments/results/cahh_ablations.jsonl"
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    set_seed(args.seed)
    print(f"Device: {device}  Category: {args.category}")

    train_raw = ShapeNet(
        args.data_dir, categories=[args.category],
        include_normals=True, split="train",
    )
    val_raw = ShapeNet(
        args.data_dir, categories=[args.category],
        include_normals=True, split="val",
    )

    print("Extracting features...")
    tr_feats, tr_targets, tr_valid, normalizer = build_edge_dataset(
        train_raw, args.k, fit_normalizer=True
    )
    va_feats, va_targets, va_valid, _ = build_edge_dataset(
        val_raw, args.k, normalizer=normalizer, fit_normalizer=False
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Variant':<22}  AUPRC  AUROC  prec@bud")
    print("-" * 50)
    for variant, cols in VARIANTS.items():
        metrics = run_ablation(
            variant, cols,
            tr_feats, tr_targets, tr_valid,
            va_feats, va_targets, va_valid,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            budget=args.budget,
            device=device,
            seed=args.seed,
        )
        row = {
            "category": args.category,
            "variant": variant,
            "cols": cols,
            "seed": args.seed,
            **metrics,
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        auprc = metrics.get("auprc", float("nan"))
        auroc = metrics.get("auroc", float("nan"))
        prec = metrics.get(f"prec@{args.budget}", float("nan"))
        print(f"{variant:<22}  {auprc:.4f} {auroc:.4f} {prec:.4f}")

    print(f"\nResults appended → {out}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

