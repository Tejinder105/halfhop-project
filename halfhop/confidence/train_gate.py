"""Train the edge-confidence MLP (CAHH §3.4).

Workflow
--------
1.  Build a shape-level train/val split (never mix shapes across splits).
2.  Extract edge features and binary targets (1 = boundary edge) for all
    training shapes.
3.  Fit :class:`~halfhop.confidence.edge_features.EdgeFeatureNormalizer` on
    training features only.
4.  Train :class:`~halfhop.confidence.confidence_mlp.EdgeConfidenceMLP` with
    weighted BCE.
5.  Report AUPRC, precision@budget, recall@budget, AUROC, Brier score on
    the validation split.

Usage::

    python -m halfhop.confidence.train_gate \\
        --data-dir data/ShapeNet \\
        --category Chair \\
        --epochs 50 \\
        --output experiments/results/cahh_mlp_Chair.pt

The saved checkpoint contains the model weights AND the fitted normaliser so
it can be loaded with a single call at inference time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import ShapeNet
from tqdm import tqdm

import experiments.segmentation.proxy  # noqa: F401  sets ShapeNet.url
from experiments.pointcloud.run import knn_edge_index
from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)
from halfhop.confidence.confidence_mlp import EdgeConfidenceMLP
from halfhop.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _extract_shape_edges(
    data,
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (features, targets, valid_mask) for one ShapeNet shape.

    Args:
        data: PyG Data with ``pos``, ``x`` (normals), ``y`` (part labels).
        k:    kNN neighbourhood size.

    Returns:
        features:   ``(E, 4)`` edge features (not yet normalised).
        targets:    ``(E,)`` int64, 1 if endpoint labels differ else 0.
        valid_mask: ``(E,)`` bool, False for self-loops.
    """
    pos = data.pos.float()
    normals = data.x.float() if data.x is not None else None
    labels = data.y.view(-1)                        # (N,)

    edge_index = knn_edge_index(pos, k)             # (2, E)
    density = compute_local_density(pos, edge_index)

    features, valid_mask = build_edge_features(
        pos, edge_index, normals=normals, density=density
    )

    src, dst = edge_index
    targets = (labels[src] != labels[dst]).long()

    return features, targets, valid_mask


def build_edge_dataset(
    shapes,
    k: int,
    normalizer: EdgeFeatureNormalizer | None = None,
    fit_normalizer: bool = False,
) -> tuple[Tensor, Tensor, Tensor, EdgeFeatureNormalizer]:
    """Collect all edges from a list of shapes into flat tensors.

    Args:
        shapes:          Iterable of PyG Data objects.
        k:               kNN k.
        normalizer:      Pre-fitted normaliser (None → create new one).
        fit_normalizer:  If True, fit the normaliser on this data.

    Returns:
        all_features:    ``(E_total, 4)`` normalised features.
        all_targets:     ``(E_total,)`` int64 boundary targets.
        all_valid:       ``(E_total,)`` bool — which edges are non-self-loops.
        normalizer:      Fitted :class:`EdgeFeatureNormalizer`.
    """
    if normalizer is None:
        normalizer = EdgeFeatureNormalizer()

    raw_feats, raw_targets, raw_valids = [], [], []
    for data in tqdm(shapes, desc="extracting edges", leave=False):
        f, t, v = _extract_shape_edges(data, k)
        raw_feats.append(f)
        raw_targets.append(t)
        raw_valids.append(v)

    all_feats = torch.cat(raw_feats, dim=0)      # (E_total, 4)
    all_targets = torch.cat(raw_targets, dim=0)  # (E_total,)
    all_valids = torch.cat(raw_valids, dim=0)    # (E_total,)

    if fit_normalizer:
        normalizer.update(all_feats[all_valids])
        normalizer.finalise()

    all_feats_norm = normalizer.transform(all_feats)
    # Keep -1.0 sentinel for the normal channel when normals were absent;
    # after normalisation, map it back (sentinel is preserved by clipping).

    return all_feats_norm, all_targets, all_valids, normalizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    budget: float = 0.1,
    n_thresholds: int = 100,
) -> dict:
    """Compute AUROC, AUPRC, precision@budget, recall@budget, Brier score.

    Only valid (non-self-loop) edges enter the metric computation.
    """
    logits_v = logits[valid].float().cpu()
    targets_v = targets[valid].float().cpu()
    scores = torch.sigmoid(logits_v)

    n_pos = int(targets_v.sum())
    n_neg = int((1 - targets_v).sum())
    n_total = len(targets_v)

    # Brier score
    brier = float(((scores - targets_v) ** 2).mean())

    # AUROC (trapezoidal)
    thresholds = torch.linspace(0.0, 1.0, n_thresholds + 1)
    tpr_list, fpr_list, prec_list, rec_list = [], [], [], []
    for t in thresholds:
        pred = (scores >= t).float()
        tp = float((pred * targets_v).sum())
        fp = float((pred * (1 - targets_v)).sum())
        tpr = tp / max(n_pos, 1)
        fpr = fp / max(n_neg, 1)
        prec = tp / max(float(pred.sum()), 1e-8)
        rec = tpr
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec_list.append(prec)
        rec_list.append(rec)

    # Area (sorted by fpr for AUROC, by rec for AUPRC)
    fpr_t = torch.tensor(fpr_list)
    tpr_t = torch.tensor(tpr_list)
    idx = torch.argsort(fpr_t)
    auroc = float(torch.trapezoid(tpr_t[idx], fpr_t[idx]).abs())

    rec_t = torch.tensor(rec_list)
    prec_t = torch.tensor(prec_list)
    idx2 = torch.argsort(rec_t)
    auprc = float(torch.trapezoid(prec_t[idx2], rec_t[idx2]).abs())

    # Precision/recall@budget
    K = max(1, int(budget * n_total))
    topk_idx = scores.topk(K).indices
    topk_targets = targets_v[topk_idx]
    prec_at_k = float(topk_targets.mean())
    rec_at_k = float(topk_targets.sum() / max(n_pos, 1))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        f"prec@{budget}": round(prec_at_k, 4),
        f"rec@{budget}": round(rec_at_k, 4),
        "brier": round(brier, 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_confidence_mlp(
    train_feats: Tensor,
    train_targets: Tensor,
    train_valid: Tensor,
    val_feats: Tensor,
    val_targets: Tensor,
    val_valid: Tensor,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 4096,
    dropout: float = 0.2,
    budget: float = 0.1,
    device: torch.device,
    verbose: bool = True,
) -> tuple[EdgeConfidenceMLP, dict]:
    """Train the MLP; return (model, best_val_metrics).

    Training uses only valid edges. Positive-class weight is computed from
    training label distribution.
    """
    # Only train on valid (non-self-loop) edges.
    tr_x = train_feats[train_valid].to(device)
    tr_y = train_targets[train_valid].float().to(device)
    va_x = val_feats[val_valid].to(device)
    va_y = val_targets[val_valid].float().to(device)

    n_pos = float(tr_y.sum())
    n_neg = float((1 - tr_y).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0)).to(device)
    if verbose:
        print(f"  Train edges: {len(tr_y)} | pos={int(n_pos)} neg={int(n_neg)} "
              f"pos_weight={pos_weight.item():.2f}")

    model = EdgeConfidenceMLP(input_dim=tr_x.size(-1), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_train = len(tr_x)
    best_auprc = -1.0
    best_state = None
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start: start + batch_size]
            xb, yb = tr_x[idx], tr_y[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(va_x)
            metrics = compute_metrics(
                val_logits.cpu(),
                val_targets[val_valid].cpu(),
                torch.ones(val_logits.size(0), dtype=torch.bool),
                budget=budget,
            )
            if metrics["auprc"] > best_auprc:
                best_auprc = metrics["auprc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = metrics
            if verbose:
                print(
                    f"  epoch {epoch:03d}  loss={total_loss/max(n_batches,1):.4f}  "
                    f"val_auprc={metrics['auprc']:.4f}  "
                    f"val_auroc={metrics['auroc']:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ShapeNet")
    parser.add_argument("--category", default="Chair")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save .pt checkpoint. Defaults to "
             "experiments/results/cahh_mlp_{category}.pt",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Path to append a JSONL row with validation metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Device: {device}  Category: {args.category}")

    print("Loading ShapeNet...")
    train_raw = ShapeNet(
        args.data_dir, categories=[args.category],
        include_normals=True, split="train",
    )
    val_raw = ShapeNet(
        args.data_dir, categories=[args.category],
        include_normals=True, split="val",
    )

    print("Extracting training edge features...")
    tr_feats, tr_targets, tr_valid, normalizer = build_edge_dataset(
        train_raw, args.k, fit_normalizer=True
    )
    print("Extracting validation edge features...")
    va_feats, va_targets, va_valid, _ = build_edge_dataset(
        val_raw, args.k, normalizer=normalizer, fit_normalizer=False
    )

    print("Training confidence MLP...")
    model, best_metrics = train_confidence_mlp(
        tr_feats, tr_targets, tr_valid,
        va_feats, va_targets, va_valid,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        budget=args.budget,
        device=device,
        verbose=True,
    )

    print("\nBest validation metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")

    # Save checkpoint.
    out_path = args.output or f"experiments/results/cahh_mlp_{args.category}.pt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "config": {
            "category": args.category,
            "k": args.k,
            "dropout": args.dropout,
            "seed": args.seed,
            "budget": args.budget,
        },
        "metrics": best_metrics,
    }
    torch.save(checkpoint, out_path)
    print(f"Saved checkpoint → {out_path}")

    if args.metrics_output:
        row = {"category": args.category, "seed": args.seed, **best_metrics}
        Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"Appended metrics → {args.metrics_output}")


# ---------------------------------------------------------------------------
# Checkpoint loader (also used by train_cahh.py and evaluate.py)
# ---------------------------------------------------------------------------

def load_cahh_checkpoint(ckpt_path: str, device=None):
    """Load a CAHH checkpoint saved by :func:`main`.

    Returns:
        mlp:        :class:`EdgeConfidenceMLP` in eval mode on ``device``.
        normalizer: Fitted :class:`EdgeFeatureNormalizer`.
        config:     Dict with training configuration.
    """
    if device is None:
        device = torch.device("cpu")
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


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
