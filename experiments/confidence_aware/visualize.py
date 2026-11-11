"""Visualise CAHH edge confidence on ShapeNet shapes.

Renders two figures per shape:
  1.  Point cloud coloured by predicted confidence score (high = red = likely boundary).
  2.  Selected top-K edges overlaid on the point cloud.

Requires matplotlib.  Save to PNG if a display is not available::

    python -m experiments.confidence_aware.visualize \\
        --category Chair --mlp-ckpt experiments/results/cahh_mlp_Chair.pt \\
        --n-shapes 3 --save-dir experiments/results/vis_Chair
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
from tqdm import tqdm

import experiments.segmentation.proxy  # noqa: F401
from experiments.pointcloud.run import knn_edge_index
from halfhop.confidence.edge_features import (
    build_edge_features,
    compute_local_density,
)
from halfhop.confidence.adaptive_selector import AdaptiveSelector
from halfhop.confidence.train_gate import load_cahh_checkpoint


def _plot_shape(
    pos,
    scores,
    edge_index,
    selected_mask,
    title: str,
    save_path: str | None = None,
):
    try:
        import matplotlib
        if save_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not available; skipping visualisation.")
        return

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    p = pos.numpy()
    s = scores.numpy()

    # Node confidence map.
    sc = ax1.scatter(p[:, 0], p[:, 1], p[:, 2], c=s, cmap="RdYlBu_r",
                     s=5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax1, fraction=0.03)
    ax1.set_title("Confidence score")
    ax1.axis("off")

    # Selected edges overlay.
    ax2.scatter(p[:, 0], p[:, 1], p[:, 2], c="lightgray", s=3, alpha=0.5)
    src, dst = edge_index
    sel_src = src[selected_mask].numpy()
    sel_dst = dst[selected_mask].numpy()
    for i in range(len(sel_src)):
        xs = [p[sel_src[i], 0], p[sel_dst[i], 0]]
        ys = [p[sel_src[i], 1], p[sel_dst[i], 1]]
        zs = [p[sel_src[i], 2], p[sel_dst[i], 2]]
        ax2.plot(xs, ys, zs, "r-", linewidth=0.6, alpha=0.8)
    ax2.set_title(f"Selected edges (top-K)")
    ax2.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="Chair")
    p.add_argument("--mlp-ckpt", required=True)
    p.add_argument("--data-dir", default="data/ShapeNet")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--n-shapes", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", default=None,
                   help="Directory to save PNGs (show interactively if None).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    mlp, normalizer, config = load_cahh_checkpoint(args.mlp_ckpt, device)
    mlp.eval()
    selector = AdaptiveSelector(budget=args.budget)

    test_shapes = ShapeNet(
        args.data_dir, categories=[args.category],
        include_normals=True, split="test",
    )
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for idx, data in enumerate(tqdm(test_shapes, total=args.n_shapes)):
        if idx >= args.n_shapes:
            break
        pos = data.pos.float()
        normals = data.x.float() if data.x is not None else None
        edge_index = knn_edge_index(pos, args.k)
        density = compute_local_density(pos, edge_index)
        feats, valid = build_edge_features(
            pos, edge_index, normals=normals, density=density
        )
        feats_norm = normalizer.transform(feats).to(device)
        with torch.no_grad():
            logits = mlp(feats_norm).cpu()
        arc_scores = torch.sigmoid(logits)                      # (E,)
        pair_scores = selector.score_for_halfhop(arc_scores, edge_index)
        selected = selector.select_mask(arc_scores, edge_index)

        # Per-node confidence = mean of incident arc scores.
        node_score = torch.zeros(pos.size(0))
        node_count = torch.zeros(pos.size(0))
        src, dst = edge_index
        node_score.index_add_(0, src, arc_scores)
        node_count.index_add_(0, src, torch.ones(arc_scores.size(0)))
        node_conf = node_score / node_count.clamp(min=1)

        title = f"{args.category} shape {idx}"
        save_path = (
            str(Path(args.save_dir) / f"{args.category}_{idx:03d}.png")
            if args.save_dir else None
        )
        _plot_shape(pos.cpu(), node_conf, edge_index.cpu(), selected.cpu(),
                    title=title, save_path=save_path)
        if save_path:
            print(f"Saved → {save_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

