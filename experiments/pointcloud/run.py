"""G0 go/no-go: does a plain GCN on ModelNet point clouds over-smooth with depth?

Fixed 1024-point kNN graphs. No Half-Hop, no geometry. Dropout defaults to 0
so the curve measures over-smoothing, not repeated dropout.

    python -m experiments.pointcloud.run --sweep

Prepare once, then split seeds across two GPUs (they append to the same jsonl):

    python -m experiments.pointcloud.run --prepare-only
    CUDA_VISIBLE_DEVICES=0 python -m experiments.pointcloud.run --sweep --seeds 0,1
    CUDA_VISIBLE_DEVICES=1 python -m experiments.pointcloud.run --sweep --seeds 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, SamplePoints
from tqdm import tqdm

from experiments.modelnet.run import (
    accuracy,
    build_model,
    load_modelnet,
    resolve_device,
    train_one_epoch,
)
from halfhop.reproducibility import set_seed

G0_DEPTHS = (2, 4, 8, 16)
G0_SEEDS = (0, 1, 2)


def knn_edge_index(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Directed kNN. Each node gets edges from its k nearest other points.

    ponytail: O(N^2) cdist. Fine at N=1024 (~4MB). Past ~8k points use
    torch_cluster.knn_graph.
    """
    n = pos.size(0)
    if k >= n:
        raise ValueError(f"k={k} must be < num points ({n})")
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))
    neighbors = dist.topk(k, largest=False).indices.reshape(-1)
    target = torch.arange(n, device=pos.device).repeat_interleave(k)
    return torch.stack([neighbors, target], dim=0)


class KNNGraph:
    """pre_transform: write a kNN edge_index from data.pos. One graph, not a batch."""

    def __init__(self, k: int):
        self.k = k

    def __call__(self, data: Data) -> Data:
        data.edge_index = knn_edge_index(data.pos, self.k)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


class UsePosAsFeatures:
    """Copy coordinates into x. Clone so x is not an alias of pos."""

    def __call__(self, data: Data) -> Data:
        if data.x is None:
            data.x = data.pos.clone()
        return data


def pre_transform_for(num_points: int, k: int):
    return Compose(
        [
            NormalizeScale(),
            SamplePoints(num_points),
            KNNGraph(k),
            UsePosAsFeatures(),
        ]
    )


def split_train_val(n: int, seed: int) -> tuple[list[int], list[int]]:
    set_seed(seed)
    n_val = max(1, n // 10)
    perm = torch.randperm(n)
    return perm[n_val:].tolist(), perm[:n_val].tolist()


def run_key(row: dict) -> tuple:
    return (
        row["stage"],
        row["model"],
        row["dataset"],
        row["depth"],
        row["seed"],
        row["k"],
        row["num_points"],
        row["sample_seed"],
        row["dropout"],
        row["hidden"],
        row["epochs"],
        row["lr"],
        row["batch_size"],
    )


def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(run_key(json.loads(line)))
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def matching(rows: list[dict], proto: dict) -> list[dict]:
    ignore = {"depth", "seed", "best_val", "test"}
    return [
        r
        for r in rows
        if all(r.get(k) == proto[k] for k in proto if k not in ignore)
    ]


def summarize(path: Path, proto: dict) -> None:
    if not path.exists():
        return
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = matching(rows, proto)
    by_depth: dict[int, list[float]] = {}
    for row in rows:
        by_depth.setdefault(row["depth"], []).append(row["test"])
    print(f"--- {path} ---")
    for depth in sorted(by_depth):
        xs = by_depth[depth]
        mean = sum(xs) / len(xs)
        if len(xs) > 1:
            var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        print(f"depth {depth:>2}: test {mean:.4f} ± {std:.4f}  n={len(xs)}")
    if 2 in by_depth and 16 in by_depth:
        print(
            "Go/no-go: depth 16 clearly below depth 2 means over-smoothing "
            "is real here. Similar numbers mean it is not; move to segmentation."
        )


def train_depth(args, train_dataset, test_dataset, depth: int, seed: int, device):
    train_idx, val_idx = split_train_val(len(train_dataset), seed)
    set_seed(seed)
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        "gcn",
        train_dataset.num_features,
        args.hidden,
        train_dataset.num_classes,
        depth,
        args.dropout,
        alpha=0.5,
        p=1.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    best_state = None
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"depth {depth} seed {seed}"):
        train_one_epoch(train_loader, model, optimizer, device)
        val_acc = accuracy(val_loader, model, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            test_acc = accuracy(test_loader, model, device)
            print(
                f"depth {depth} seed {seed} epoch {epoch:03d} "
                f"val={val_acc:.4f} test={test_acc:.4f}"
            )

    model.load_state_dict(best_state)
    test_acc = accuracy(test_loader, model, device)
    return best_val, test_acc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="10", choices=["10", "40"])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds. Overrides --seed.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="G0 grid: depths 2,4,8,16. Seeds 0,1,2 unless --seeds is set.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Default 0 so a depth sweep is not confounded by dropout.",
    )
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ModelNetPC",
        help="k and num-points are appended. Separate from the mesh cache.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/pointcloud_g0.jsonl",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download and process the point clouds, then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = str(Path(args.data_dir) / f"k{args.k}_n{args.num_points}")
    transform = pre_transform_for(args.num_points, args.k)

    if args.prepare_only:
        set_seed(0)
        train_dataset, test_dataset = load_modelnet(root, args.dataset, transform)
        print(
            f"Ready: train={len(train_dataset)} test={len(test_dataset)} "
            f"feats={train_dataset.num_features} root={root}"
        )
        return

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    elif args.sweep:
        seeds = list(G0_SEEDS)
    else:
        seeds = [args.seed]
    depths = list(G0_DEPTHS) if args.sweep else [args.depth]

    out = Path(args.output)
    done = load_done(out)
    proto = {
        "stage": "G0",
        "dataset": f"ModelNet{args.dataset}",
        "model": "gcn",
        "k": args.k,
        "num_points": args.num_points,
        "dropout": args.dropout,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "sample_seed": 0,
    }

    pending = []
    for seed in seeds:
        for depth in depths:
            row = {**proto, "depth": depth, "seed": seed}
            if run_key(row) in done:
                print(f"skip depth={depth} seed={seed}")
            else:
                pending.append((depth, seed))

    if pending:
        # Seed only the one-time sampling. Later runs reload this cache.
        set_seed(0)
        device = resolve_device(args.device)
        print(f"Device: {device}  cache: {root}")
        train_dataset, test_dataset = load_modelnet(root, args.dataset, transform)
        print(
            f"ModelNet{args.dataset} points={args.num_points} k={args.k} "
            f"train={len(train_dataset)} test={len(test_dataset)} "
            f"feats={train_dataset.num_features}"
        )
        for depth, seed in pending:
            best_val, test_acc = train_depth(
                args, train_dataset, test_dataset, depth, seed, device
            )
            row = {**proto, "depth": depth, "seed": seed,
                   "best_val": round(best_val, 6), "test": round(test_acc, 6)}
            append_row(out, row)
            print(
                f"FINAL depth={depth} seed={seed} "
                f"best_val={best_val:.4f} test={test_acc:.4f}"
            )

    summarize(out, proto)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
