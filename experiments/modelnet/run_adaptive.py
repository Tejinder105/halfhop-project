"""ModelNet40 point-cloud experiments for adaptive multi-delay Half-Hop.

Run one configuration:
    python -m experiments.modelnet.run_adaptive --mode curvature

Run the complete ablation matrix on two GPUs:
    python -m experiments.modelnet.run_adaptive --run-all --gpus 0,1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.transforms import Compose, NormalizeScale, SamplePoints
from tqdm import tqdm

from experiments.modelnet.adaptive_halfhop import AdaptiveHalfHop
from halfhop.graphsage import GraphSAGE
from halfhop.reproducibility import set_seed


EXPERIMENTS = (
    ("none", (0, 1, 2), "none"),
    ("uniform", (1, 1, 1), "uniform"),
    ("curvature", (1, 2, 3), "curvature"),
    ("random", (1, 2, 3), "random"),
    ("inverse", (1, 2, 3), "inverse"),
    ("curvature_012", (0, 1, 2), "curvature"),
)


class PointCloudClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int, depth: int, classes: int, dropout: float):
        super().__init__()
        self.encoder = GraphSAGE(in_channels, hidden, hidden, dropout=dropout, depth=depth)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index)
        keep = ~data.slow_node_mask
        x = x[keep]
        batch = data.batch[keep]
        pooled = torch.cat((global_max_pool(x, batch), global_mean_pool(x, batch)), dim=1)
        return self.head(pooled)


def build_transform(args: argparse.Namespace):
    return Compose(
        [
            SamplePoints(args.points, include_normals=True),
            AdaptiveHalfHop(
                knn=args.knn,
                curv_k=args.curv_k,
                delays=tuple(args.delays),
                quantiles=tuple(args.quantiles),
                mode=args.mode,
                uniform_m=args.uniform_m,
                backward=args.backward,
            ),
        ]
    )


def load_datasets(args: argparse.Namespace):
    transform = build_transform(args)
    pre_transform = NormalizeScale()
    train = ModelNet(args.data_dir, name=args.dataset, train=True, pre_transform=pre_transform, transform=transform)
    test = ModelNet(args.data_dir, name=args.dataset, train=False, pre_transform=pre_transform, transform=transform)
    return train, test


def accuracy(loader, model, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            correct += int((model(data).argmax(dim=-1) == data.y).sum())
            total += data.num_graphs
    return correct / max(total, 1)


def train_epoch(loader, model, optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(data), data.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * data.num_graphs
    return loss_sum / max(len(loader.dataset), 1)


def run_one(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train_dataset, test_dataset = load_datasets(args)
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(train_dataset), generator=generator)
    val_size = max(1, len(train_dataset) // 10)
    from torch.utils.data import Subset

    val_dataset = Subset(train_dataset, permutation[:val_size].tolist())
    train_subset = Subset(train_dataset, permutation[val_size:].tolist())
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, **loader_args)
    test_loader = DataLoader(test_dataset, **loader_args)

    model = PointCloudClassifier(
        train_dataset.num_features,
        args.hidden,
        args.layers,
        train_dataset.num_classes,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_test = 0.0
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"ModelNet{args.dataset} {args.label}"):
        loss = train_epoch(train_loader, model, optimizer, device)
        val = accuracy(val_loader, model, device)
        test = accuracy(test_loader, model, device)
        if val >= best_val:
            best_val, best_test = val, test
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"epoch={epoch:03d} loss={loss:.4f} val={val:.4f} test={test:.4f}", flush=True)

    result = (
        f"FINAL RESULTS: {args.label} ModelNet{args.dataset} "
        f"best_val={best_val:.4f} test_at_best_val={best_test:.4f} "
        f"seed={args.seed} mode={args.mode} delays={tuple(args.delays)} "
        f"backward={args.backward} points={args.points} knn={args.knn}"
    )
    print(result, flush=True)
    output = Path(args.output or f"experiments/results/modelnet{args.dataset}_adaptive.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(result + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive multi-delay Half-Hop ModelNet40 runner")
    parser.add_argument("--dataset", default="40", choices=["10", "40"])
    parser.add_argument("--mode", default="curvature", choices=["none", "curvature", "inverse", "random", "uniform"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.85])
    parser.add_argument("--uniform-m", type=int, default=1)
    parser.add_argument("--backward", choices=["last", "none"], default="last")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--curv-k", type=int, default=24)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data/ModelNetPoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    return parser.parse_args()


def run_all(args: argparse.Namespace) -> None:
    if args.prepare_only:
        raise ValueError("--prepare-only cannot be combined with --run-all")
    prepare = [
        sys.executable,
        "-m",
        "experiments.modelnet.run_adaptive",
        "--dataset",
        args.dataset,
        "--data-dir",
        args.data_dir,
        "--prepare-only",
    ]
    subprocess.run(prepare, check=True)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")

    jobs = []
    for label, delays, mode in EXPERIMENTS:
        command = [
            sys.executable,
            "-m",
            "experiments.modelnet.run_adaptive",
            "--dataset",
            args.dataset,
            "--mode",
            mode,
            "--label",
            label,
            "--delays",
            *map(str, delays),
            "--quantiles",
            *map(str, args.quantiles),
            "--uniform-m",
            str(args.uniform_m),
            "--backward",
            args.backward,
            "--points",
            str(args.points),
            "--knn",
            str(args.knn),
            "--curv-k",
            str(args.curv_k),
            "--layers",
            str(args.layers),
            "--hidden",
            str(args.hidden),
            "--dropout",
            str(args.dropout),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--device",
            "cuda",
            "--data-dir",
            args.data_dir,
            "--output",
            f"experiments/results/modelnet{args.dataset}_{label}.txt",
        ]
        jobs.append((label, command))

    processes = []
    for index, (label, command) in enumerate(jobs):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
        print(f"Launching {label} on GPU {env['CUDA_VISIBLE_DEVICES']}", flush=True)
        processes.append((label, subprocess.Popen(command, env=env)))
        if len(processes) == len(gpu_ids):
            for running_label, process in processes:
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command, env=env)
                print(f"Completed {running_label}", flush=True)
            processes.clear()


def main() -> None:
    args = parse_args()
    if args.run_all:
        run_all(args)
        return
    if args.prepare_only:
        train, test = load_datasets(args)
        print(f"Prepared ModelNet{args.dataset}: train={len(train)} test={len(test)}", flush=True)
        return
    if args.label is None:
        args.label = args.mode
    if args.mode != "none" and args.layers <= max(args.delays) + 1:
        raise ValueError("layers must exceed max(delay)+1 so the longest chain can reach its target")
    run_one(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
