"""CAHH evaluation: edge-prediction metrics + segmentation comparison.

Runs against a saved CAHH checkpoint and reports:
  - Edge-confidence: AUROC, AUPRC, precision@budget, recall@budget, Brier score
  - Segmentation: test accuracy and mIoU (boundary / interior breakdown)
  - Boundary/interior analysis (§5.5 of the CAHH plan)

Usage::

    python -m experiments.confidence_aware.evaluate \\
        --category Chair \\
        --mlp-ckpt experiments/results/cahh_mlp_Chair.pt \\
        --seg-results experiments/results/cahh_segmentation.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from torch_geometric.datasets import ShapeNet

import experiments.segmentation.proxy  # noqa: F401
from experiments.pointcloud.run import knn_edge_index
from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)
from halfhop.confidence.confidence_mlp import EdgeConfidenceMLP
from halfhop.confidence.train_gate import compute_metrics, load_cahh_checkpoint


# ---------------------------------------------------------------------------
# Boundary / interior analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def boundary_interior_analysis(
    test_shapes,
    mlp: EdgeConfidenceMLP,
    normalizer: EdgeFeatureNormalizer,
    k: int,
    budget: float,
    device: torch.device,
) -> dict:
    """Report precision@budget separately for boundary vs interior edges.

    A boundary point is one incident to at least one inter-label edge.

    Returns:
        dict with keys: boundary_prec, interior_prec, boundary_frac,
                        all_prec, all_rec, auprc, auroc.
    """
    all_logits = []
    all_targets = []
    all_is_boundary_node = []

    mlp.eval()
    for data in tqdm(test_shapes, desc="boundary analysis", leave=False):
        pos = data.pos.float()
        normals = data.x.float() if data.x is not None else None
        labels = data.y.view(-1)
        edge_index = knn_edge_index(pos, k)
        density = compute_local_density(pos, edge_index)
        feats, valid = build_edge_features(
            pos, edge_index, normals=normals, density=density
        )
        feats_norm = normalizer.transform(feats).to(device)
        logits = mlp(feats_norm).cpu()

        src, dst = edge_index
        targets = (labels[src] != labels[dst]).long()

        # Is each arc incident to a boundary node?
        is_boundary_node = torch.zeros(pos.size(0), dtype=torch.bool)
        boundary_arcs = targets.bool()
        is_boundary_node[src[boundary_arcs]] = True
        is_boundary_node[dst[boundary_arcs]] = True
        arc_in_boundary = is_boundary_node[src] | is_boundary_node[dst]

        all_logits.append(logits[valid])
        all_targets.append(targets[valid].float())
        all_is_boundary_node.append(arc_in_boundary[valid])

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_is_bnd = torch.cat(all_is_boundary_node)

    # Overall metrics
    overall = compute_metrics(
        all_logits,
        all_targets.long(),
        torch.ones(len(all_logits), dtype=torch.bool),
        budget=budget,
    )

    # Boundary edge precision@budget
    scores = torch.sigmoid(all_logits)
    n_total = len(scores)
    K = max(1, int(budget * n_total))
    topk_idx = scores.topk(K).indices
    topk_targets = all_targets[topk_idx]
    topk_is_bnd = all_is_bnd[topk_idx]

    bnd_tp = float((topk_targets * topk_is_bnd).sum())
    int_tp = float((topk_targets * ~topk_is_bnd).sum())
    bnd_sel = float(topk_is_bnd.sum())
    int_sel = float((~topk_is_bnd).sum())

    return {
        **overall,
        "boundary_prec@budget": round(bnd_tp / max(bnd_sel, 1), 4),
        "interior_prec@budget": round(int_tp / max(int_sel, 1), 4),
        "boundary_frac_selected": round(bnd_sel / max(K, 1), 4),
        "boundary_frac_all": round(float(all_is_bnd.float().mean()), 4),
    }


# ---------------------------------------------------------------------------
# Segmentation results reader
# ---------------------------------------------------------------------------

def print_seg_table(jsonl_path: str, category: str) -> None:
    """Pretty-print the segmentation comparison table from a JSONL file."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("category") == category:
                    rows.append(r)
    if not rows:
        print(f"No results for category={category} in {jsonl_path}")
        return

    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    print(f"\n=== Segmentation: {category} ===")
    print(f"{'Method':<15} {'mIoU (mean)':>12} {'mIoU (std)':>11} {'Seeds':>6}")
    print("-" * 50)
    for model, runs in sorted(by_model.items()):
        mious = [r["test_miou"] for r in runs]
        mean = sum(mious) / len(mious)
        std = (sum((m - mean) ** 2 for m in mious) / max(len(mious) - 1, 1)) ** 0.5
        print(f"{model:<15} {mean:>12.4f} {std:>11.4f} {len(mious):>6}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="Chair")
    p.add_argument("--mlp-ckpt", required=True)
    p.add_argument("--data-dir", default="data/ShapeNet")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seg-results", default=None,
                   help="Path to JSONL with segmentation results to print.")
    p.add_argument("--output", default=None,
                   help="Optional JSONL to append evaluation metrics.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Device: {device}  Category: {args.category}")

    mlp, normalizer, config = load_cahh_checkpoint(args.mlp_ckpt, device)
    print(f"Loaded checkpoint: {args.mlp_ckpt}")

    print("Loading test shapes...")
    test_shapes = ShapeNet(
        args.data_dir, categories=[args.category],
        include_normals=True, split="test",
    )

    print("Running boundary/interior analysis on test set...")
    metrics = boundary_interior_analysis(
        test_shapes, mlp, normalizer,
        k=args.k, budget=args.budget, device=device,
    )

    print("\n=== Edge-confidence metrics (test set) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.seg_results and Path(args.seg_results).exists():
        print_seg_table(args.seg_results, args.category)

    if args.output:
        row = {"category": args.category, "split": "test", **metrics}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"\nAppended metrics → {args.output}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

