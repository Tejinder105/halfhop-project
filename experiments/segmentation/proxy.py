"""Do high normal-disagreement edges land on part boundaries?

No training. For each shape, build a kNN graph, score every edge by
``1 - |n_i · n_j|``, and compare how often the two endpoints have different
part labels in the highest-scoring edges, the lowest-scoring edges, and
all edges. A random selection has the same rate as all edges, in expectation.

    python -m experiments.segmentation.proxy

ShapeNet-Part is about 1 GB the first time. Normals come with the dataset.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet

from experiments.pointcloud.run import knn_edge_index

# shapenet.cs.stanford.edu times out from Kaggle. Same zip, different host.
# The filename must stay unchanged: ShapeNet.download() renames the extracted
# folder using the last path component.
ShapeNet.url = (
    "https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/"
    "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
)


def edge_boundary_stats(score: torch.Tensor, disagree: torch.Tensor, fraction: float) -> dict:
    """Rates of label disagreement in the top and bottom ``fraction`` of edges."""
    if score.ndim != 1 or disagree.shape != score.shape:
        raise ValueError("score and disagree must be 1-D and the same length")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = score.numel()
    if n == 0:
        raise ValueError("no edges")
    k = max(1, int(round(fraction * n)))
    disagree = disagree.bool()
    top = disagree[torch.topk(score, k).indices].float().mean()
    bottom = disagree[torch.topk(score, k, largest=False).indices].float().mean()
    base = disagree.float().mean()
    centered_score = score - score.mean()
    centered_label = disagree.float() - base
    denom = centered_score.norm() * centered_label.norm()
    corr = (centered_score * centered_label).sum() / denom if float(denom) > 0 else score.new_zeros(())
    return {
        "k": k,
        "base": float(base),
        "top": float(top),
        "bottom": float(bottom),
        "corr": float(corr),
    }


def normal_disagreement(normals: torch.Tensor, edge_index: torch.Tensor):
    """``1 - |n_i · n_j|`` on edges whose endpoints both have a real normal."""
    n = F.normalize(normals, dim=-1)
    good = torch.isfinite(n).all(dim=-1)
    n = torch.nan_to_num(n, nan=0.0)
    src, dst = edge_index
    keep = good[src] & good[dst]
    src, dst = src[keep], dst[keep]
    dot = (n[src] * n[dst]).sum(dim=-1).abs().clamp(max=1.0)
    return 1.0 - dot, src, dst


def one_shape(data, k: int, fraction: float) -> dict:
    y = data.y.view(-1).long()
    score, src, dst = normal_disagreement(data.x, knn_edge_index(data.pos, k))
    return edge_boundary_stats(score, y[src] != y[dst], fraction)


def mean_field(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="Chair,Airplane")
    parser.add_argument("--shapes", type=int, default=40, help="Shapes per category.")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "trainval"])
    parser.add_argument("--data-dir", default="data/ShapeNet")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    print(
        f"fraction={args.fraction} k={args.k}  "
        "top = highest normal disagreement, bottom = lowest, "
        "base = all edges (what random selection hits)"
    )
    for category in categories:
        dataset = ShapeNet(
            args.data_dir,
            categories=[category],
            include_normals=True,
            split=args.split,
        )
        n = min(args.shapes, len(dataset))
        rows = [one_shape(dataset[i], args.k, args.fraction) for i in range(n)]
        base = mean_field(rows, "base")
        top = mean_field(rows, "top")
        bottom = mean_field(rows, "bottom")
        corr = mean_field(rows, "corr")
        top_x = top / base if base > 0 else float("nan")
        bot_x = bottom / base if base > 0 else float("nan")
        print(
            f"{category:12} n={n:<4} base={base:.3f}  "
            f"top={top:.3f} ({top_x:.1f}x)  "
            f"bottom={bottom:.3f} ({bot_x:.1f}x)  corr={corr:.3f}"
        )
    print(
        "Pass: top is clearly above base and bottom is clearly below it. "
        "Then geometry marks part boundaries, and Half-Hop on those edges is worth training."
    )


if __name__ == "__main__":
    main()
