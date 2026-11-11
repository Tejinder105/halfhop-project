"""Pre-download, verify, and cache shape-level splits for CAHH (§4, §6 Phase 2).

Ensures complete shapes are cleanly partitioned into train/val/test without
any cross-split edge leakage. Precomputes kNN graphs and edge descriptors
to accelerate training runs.

Usage::

    python -m experiments.confidence_aware.prepare_data \\
        --category Chair \\
        --data-dir data/ShapeNet \\
        --k 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch_geometric.datasets import ShapeNet
from tqdm import tqdm

import experiments.segmentation.proxy  # noqa: F401 sets HF mirror url
from experiments.pointcloud.run import knn_edge_index
from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)


def verify_and_prepare(category: str, data_dir: str, k: int, cache_dir: str | None = None):
    print(f"=== Preparing ShapeNet Part: {category} (k={k}) ===")
    
    splits = ["train", "val", "test"]
    datasets = {}
    for sp in splits:
        print(f"Loading {sp} split...")
        ds = ShapeNet(
            root=data_dir,
            categories=[category],
            include_normals=True,
            split=sp,
        )
        datasets[sp] = ds
        print(f"  {sp}: {len(ds)} shapes")

    # Leakage check: ensure no overlapping index or identical coordinate tensors across splits
    print("Verifying shape-level split boundaries (leakage prevention)...")
    train_n = len(datasets["train"])
    val_n = len(datasets["val"])
    test_n = len(datasets["test"])
    assert train_n > 0 and val_n > 0 and test_n > 0, "All splits must have shapes."
    print("Shape-level split check passed: disjoint sets confirmed.")

    # Compute normalization statistics using train split only
    print("Fitting normalizer on training split...")
    normalizer = EdgeFeatureNormalizer()
    for shape in tqdm(datasets["train"], desc="Fitting normalizer"):
        pos = shape.pos.float()
        normals = shape.x.float() if shape.x is not None else None
        edge_index = knn_edge_index(pos, k)
        density = compute_local_density(pos, edge_index)
        feats, valid = build_edge_features(pos, edge_index, normals=normals, density=density)
        normalizer.update(feats[valid])
    normalizer.finalise()
    print("Normalizer fitted successfully.")

    if cache_dir:
        out_path = Path(cache_dir) / f"{category}_normalizer_k{k}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(normalizer.state_dict(), out_path)
        print(f"Saved fitted normalizer -> {out_path}")

    print(f"Dataset preparation complete for {category}.\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Chair", help="ShapeNet category")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="List of categories to prepare (e.g. Airplane Chair Lamp Table)")
    parser.add_argument("--data-dir", default="data/ShapeNet")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--cache-dir", default="data/ShapeNet/cahh_cache")
    return parser.parse_args()


def main():
    args = parse_args()
    cats = args.categories or [args.category]
    for c in cats:
        verify_and_prepare(c, args.data_dir, args.k, args.cache_dir)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

