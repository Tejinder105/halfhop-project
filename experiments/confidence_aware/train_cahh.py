"""CAHH segmentation experiment — extends experiments/segmentation/run.py.

Adds ``hh-cahh`` as a fourth arm alongside gcn / hh-random / hh-geom.
The CAHH arm loads a pre-trained EdgeConfidenceMLP checkpoint and uses its
predicted scores to drive the existing HalfHop selector.

Usage::

    # 1. Train the confidence MLP first (or use a saved checkpoint):
    python -m halfhop.confidence.train_gate \\
        --category Chair --k 16 --epochs 50

    # 2. Run CAHH alongside other arms:
    python -m experiments.confidence_aware.train_cahh \\
        --category Chair --mlp-ckpt experiments/results/cahh_mlp_Chair.pt \\
        --sweep --seed 0

The script inherits the data-loading, kNN graph construction, GNN backbone,
and JSONL result-writing conventions from experiments/segmentation/run.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import experiments.segmentation.proxy  # noqa: F401
from experiments.pointcloud.run import knn_edge_index
from experiments.segmentation.run import (
    Segmenter,
    evaluate,
    instance_miou,
    train_one_epoch,
)
from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)
from halfhop.confidence.confidence_mlp import EdgeConfidenceMLP
from halfhop.confidence.adaptive_selector import AdaptiveSelector
from halfhop.halfhop import HalfHop
from halfhop.reproducibility import set_seed

ARMS = ("gcn", "hh-random", "hh-geom", "hh-cahh")


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_cahh_checkpoint(ckpt_path: str, device: torch.device):
    """Load a CAHH checkpoint saved by train_gate.py.

    Returns:
        mlp:        :class:`EdgeConfidenceMLP` in eval mode.
        normalizer: Fitted :class:`EdgeFeatureNormalizer`.
        config:     Dict with training configuration.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    mlp = EdgeConfidenceMLP(
        input_dim=4,
        dropout=config.get("dropout", 0.2),
    ).to(device)
    mlp.load_state_dict(ckpt["model_state"])
    mlp.eval()
    normalizer = EdgeFeatureNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    return mlp, normalizer, config


# ---------------------------------------------------------------------------
# Graph transform for CAHH arm
# ---------------------------------------------------------------------------

class GraphBuildCAHH:
    """kNN graph + coordinate features + CAHH Half-Hop.

    At training time the MLP is applied from the checkpoint.  Labels are NOT
    used for edge selection (only for computing targets during MLP training).

    Args:
        k:          kNN neighbourhood size.
        budget:     Fraction of undirected edges to half-hop.
        mlp:        Trained :class:`EdgeConfidenceMLP` (eval mode).
        normalizer: Fitted :class:`EdgeFeatureNormalizer`.
        device:     Device for MLP inference.
    """

    def __init__(
        self,
        k: int,
        budget: float,
        mlp: EdgeConfidenceMLP,
        normalizer: EdgeFeatureNormalizer,
        device: torch.device,
    ):
        self.k = k
        self.budget = budget
        self.mlp = mlp
        self.normalizer = normalizer
        self.device = device
        self.selector = AdaptiveSelector(budget=budget)

    @torch.no_grad()
    def __call__(self, data):
        data = data.clone()
        normals = data.x.float() if data.x is not None else None
        data.x = data.pos.clone()
        data.edge_index = knn_edge_index(data.pos, self.k)

        pos = data.pos.float()
        edge_index = data.edge_index

        # Compute edge features and normalise.
        density = compute_local_density(pos, edge_index)
        features, valid = build_edge_features(
            pos, edge_index, normals=normals, density=density
        )
        features_norm = self.normalizer.transform(features).to(self.device)

        # MLP inference → logits → scores.
        logits = self.mlp(features_norm)                # (E,)
        scores = torch.sigmoid(logits).cpu()

        # Map directed arc scores to undirected pair scores.
        pair_scores = self.selector.score_for_halfhop(scores, edge_index)

        # Attach to data for HalfHop 'score' selection.
        data.edge_score = pair_scores

        # Apply HalfHop with score-based selection.
        hh = HalfHop(
            selection="score",
            budget=self.budget,
            score_mode="high",
            inplace=True,
        )
        data = hh(data)
        return data


# ---------------------------------------------------------------------------
# Graph build for baseline arms (re-implements segmentation.run.GraphBuild)
# ---------------------------------------------------------------------------

class GraphBuildBaseline:
    """Equivalent to experiments.segmentation.run.GraphBuild."""

    def __init__(self, k: int, arm: str, budget: float):
        self.k = k
        self.arm = arm
        self.budget = budget

    def __call__(self, data):
        data = data.clone()
        normals = data.x
        data.x = data.pos.clone()
        data.edge_index = knn_edge_index(data.pos, self.k)
        n = F.normalize(normals.float(), dim=-1)
        n = torch.nan_to_num(n, nan=0.0)
        src, dst = data.edge_index
        data.edge_score = 1.0 - (n[src] * n[dst]).sum(dim=-1).abs().clamp(max=1.0)
        if self.arm == "hh-random":
            data = HalfHop(selection="edge", budget=self.budget, inplace=False)(data)
        elif self.arm == "hh-geom":
            data = HalfHop(selection="score", budget=self.budget, inplace=False)(data)
        return data


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_key(row: dict) -> tuple:
    return (
        row["stage"], row["model"], row["category"], row["seed"],
        row["budget"], row["k"], row["depth"], row["hidden"],
        row["dropout"], row["epochs"], row["lr"], row["batch_size"],
    )


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(run_key(json.loads(line)))
    return done


def train_arm(args, category: str, arm: str, seed: int, device: torch.device,
              cahh_mlp=None, cahh_norm=None):
    set_seed(seed)

    if arm == "hh-cahh":
        if cahh_mlp is None:
            raise ValueError("--mlp-ckpt is required for hh-cahh arm")
        transform = GraphBuildCAHH(args.k, args.budget, cahh_mlp, cahh_norm, device)
    else:
        transform = GraphBuildBaseline(args.k, arm, args.budget)

    raw_train = ShapeNet(args.data_dir, categories=[category],
                         include_normals=True, split="train")
    raw_val = ShapeNet(args.data_dir, categories=[category],
                       include_normals=True, split="val")
    raw_test = ShapeNet(args.data_dir, categories=[category],
                        include_normals=True, split="test")

    print(f"Building graphs: {category} [{arm}] k={args.k} budget={args.budget}")
    train_ds = [transform(d) for d in tqdm(raw_train, desc="train graphs", leave=False)]
    val_ds   = [transform(d) for d in tqdm(raw_val,   desc="val graphs",   leave=False)]
    test_ds  = [transform(d) for d in tqdm(raw_test,  desc="test graphs",  leave=False)]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = Segmenter(
        in_channels=3, hidden=args.hidden, num_classes=50,
        depth=args.depth, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_state = None
    pbar = tqdm(range(1, args.epochs + 1), desc=f"{category} {arm} s{seed}")
    for epoch in pbar:
        loss = train_one_epoch(train_loader, model, optimizer, device)
        val_acc, val_miou = evaluate(val_loader, model, device)
        if val_miou > best_val:
            best_val = val_miou
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        pbar.set_postfix({"loss": f"{loss:.3f}", "val_miou": f"{val_miou:.3f}"})
        if epoch == 1 or epoch % 10 == 0:
            print(f"{category} {arm} s{seed} ep{epoch:03d} "
                  f"loss={loss:.4f} val_miou={val_miou:.4f} best={best_val:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    test_acc, test_miou = evaluate(test_loader, model, device)
    return best_val, test_acc, test_miou


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="Chair")
    p.add_argument("--model", default="hh-cahh", choices=ARMS)
    p.add_argument("--sweep", action="store_true",
                   help="Run all four arms for this category and seed.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default="data/ShapeNet")
    p.add_argument("--mlp-ckpt", default=None,
                   help="Path to trained EdgeConfidenceMLP checkpoint (.pt). "
                        "Required for hh-cahh arm.")
    p.add_argument("--output", default="experiments/results/cahh_segmentation.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Device: {device}  Category: {args.category}")

    arms = ARMS if args.sweep else (args.model,)
    out = Path(args.output)
    done = load_done(out)
    proto = {
        "stage": "cahh_seg",
        "category": args.category,
        "seed": args.seed,
        "budget": args.budget,
        "k": args.k,
        "depth": args.depth,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    cahh_mlp = cahh_norm = None
    if "hh-cahh" in arms:
        if args.mlp_ckpt is None:
            # Try a default path.
            default = f"experiments/results/cahh_mlp_{args.category}.pt"
            if Path(default).exists():
                args.mlp_ckpt = default
                print(f"Auto-detected MLP checkpoint: {default}")
            else:
                raise ValueError(
                    "--mlp-ckpt is required for the hh-cahh arm. "
                    "Run halfhop.confidence.train_gate first."
                )
        cahh_mlp, cahh_norm, _ = load_cahh_checkpoint(args.mlp_ckpt, device)
        print(f"Loaded CAHH checkpoint from {args.mlp_ckpt}")

    for arm in arms:
        row = {**proto, "model": arm}
        if run_key(row) in done:
            print(f"skip {args.category} {arm} seed {args.seed}")
            continue
        best_val, test_acc, test_miou = train_arm(
            args, args.category, arm, args.seed, device,
            cahh_mlp=cahh_mlp, cahh_norm=cahh_norm,
        )
        row.update(
            best_val_miou=round(best_val, 6),
            test_acc=round(test_acc, 6),
            test_miou=round(test_miou, 6),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"FINAL {args.category} {arm} seed {args.seed} "
            f"val_miou={best_val:.4f} test_acc={test_acc:.4f} test_miou={test_miou:.4f}"
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

