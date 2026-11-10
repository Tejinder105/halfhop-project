"""ModelNet point-cloud experiments for adaptive multi-delay Half-Hop.

Optimized for Kaggle Free Tier (13 GB RAM, 2x Tesla T4 GPUs).

Run one configuration:
    python -m experiments.modelnet.run_adaptive --mode curvature

Run the complete ablation matrix on two GPUs:
    python -m experiments.modelnet.run_adaptive --run-all --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
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
from tqdm import tqdm

from experiments.modelnet.adaptive_halfhop import AdaptiveHalfHop, PointCloudPreTransform
from halfhop.graphsage import GraphSAGE
from halfhop.reproducibility import set_seed


EXPERIMENT_CONFIGS = {
    "none": ((0, 1, 2), "none"),
    "uniform": ((1, 1, 1), "uniform"),
    "curvature": ((1, 2, 3), "curvature"),
    "random": ((1, 2, 3), "random"),
    "inverse": ((1, 2, 3), "inverse"),
    "curvature_012": ((0, 1, 2), "curvature"),
}


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


def load_datasets(args: argparse.Namespace):
    pre_transform = PointCloudPreTransform(points=args.points, knn=args.knn, curv_k=args.curv_k)
    train = ModelNet(args.data_dir, name=args.dataset, train=True, pre_transform=pre_transform)
    test = ModelNet(args.data_dir, name=args.dataset, train=False, pre_transform=pre_transform)
    return train, test


def accuracy(loader, model, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            correct += int((out.argmax(dim=-1) == data.y).sum())
            total += data.num_graphs
            del data, out
    return correct / max(total, 1)


def train_epoch(loader, model, optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * data.num_graphs
        del data, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_sum / max(len(loader.dataset), 1)


def run_one(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    train_dataset, test_dataset = load_datasets(args)

    transform = AdaptiveHalfHop(
        knn=args.knn,
        curv_k=args.curv_k,
        delays=tuple(args.delays),
        quantiles=tuple(args.quantiles),
        mode=args.mode,
        uniform_m=args.uniform_m,
        backward=args.backward,
    )

    if not args.no_cache:
        print(f"Caching in-memory transformed graphs for {args.label} (mode={args.mode})...", flush=True)
        train_items = [transform(data) for data in tqdm(train_dataset, desc="Transform train", leave=False)]
        test_items = [transform(data) for data in tqdm(test_dataset, desc="Transform test", leave=False)]
    else:
        class TransformedSubset:
            def __init__(self, ds, t):
                self.ds, self.t = ds, t

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                return self.t(self.ds[idx])

        train_items = TransformedSubset(train_dataset, transform)
        test_items = TransformedSubset(test_dataset, transform)

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(train_items), generator=generator)
    val_size = max(1, len(train_items) // 10)

    if not args.no_cache:
        val_dataset = [train_items[i] for i in permutation[:val_size]]
        train_subset = [train_items[i] for i in permutation[val_size:]]
    else:
        from torch.utils.data import Subset

        val_dataset = Subset(train_items, permutation[:val_size].tolist())
        train_subset = Subset(train_items, permutation[val_size:].tolist())

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_items, shuffle=False, **loader_args)

    in_channels = train_subset[0].x.size(1)
    num_classes = int(args.dataset) if args.dataset in ("10", "40") else getattr(train_dataset, "num_classes", 10)

    model = PointCloudClassifier(
        in_channels,
        args.hidden,
        args.layers,
        num_classes,
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
    output = Path(args.output or f"experiments/results/modelnet{args.dataset}_{args.label}.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(result + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive multi-delay Half-Hop ModelNet runner")
    parser.add_argument("--dataset", default="10", choices=["10", "40"])
    parser.add_argument("--mode", default="curvature", choices=["none", "curvature", "inverse", "random", "uniform"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.85])
    parser.add_argument("--uniform-m", type=int, default=1)
    parser.add_argument("--backward", choices=["last", "none"], default="last")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--curv-k", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data/ModelNet10")
    parser.add_argument("--output", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--experiments", nargs="+", default=["none", "uniform", "curvature", "random"])
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def run_all(args: argparse.Namespace) -> None:
    if args.prepare_only:
        raise ValueError("--prepare-only cannot be combined with --run-all")

    print(f"Preparing ModelNet{args.dataset} point clouds in {args.data_dir}...", flush=True)
    prepare = [
        sys.executable,
        "-m",
        "experiments.modelnet.run_adaptive",
        "--dataset",
        args.dataset,
        "--data-dir",
        args.data_dir,
        "--points",
        str(args.points),
        "--knn",
        str(args.knn),
        "--curv-k",
        str(args.curv_k),
        "--prepare-only",
    ]
    subprocess.run(prepare, check=True)

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if torch.cuda.is_available():
        avail = torch.cuda.device_count()
        gpu_ids = [gid for gid in gpu_ids if int(gid) < avail]
        if not gpu_ids:
            gpu_ids = ["0"]
    else:
        gpu_ids = ["cpu"]

    selected_experiments = []
    if "all" in args.experiments:
        selected_experiments = [(k, v[0], v[1]) for k, v in EXPERIMENT_CONFIGS.items()]
    else:
        for name in args.experiments:
            if name in EXPERIMENT_CONFIGS:
                delays, mode = EXPERIMENT_CONFIGS[name]
                selected_experiments.append((name, delays, mode))
            else:
                print(f"Warning: unknown experiment '{name}', skipping.", flush=True)

    if not selected_experiments:
        raise ValueError(f"No valid experiments selected in {args.experiments}")

    jobs = []
    for label, delays, mode in selected_experiments:
        target_device = "cuda:0" if gpu_ids[0] != "cpu" else "cpu"
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
            target_device,
            "--data-dir",
            args.data_dir,
            "--output",
            f"experiments/results/modelnet{args.dataset}_{label}.txt",
        ]
        if args.no_cache:
            command.append("--no-cache")
        jobs.append((label, command))

    print(f"Executing {len(jobs)} experiments across resources {gpu_ids}: {[j[0] for j in jobs]}", flush=True)

    processes = []
    for index, (label, command) in enumerate(jobs):
        env = os.environ.copy()
        if gpu_ids[0] != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
            dev_str = f"GPU {env['CUDA_VISIBLE_DEVICES']}"
        else:
            dev_str = "CPU"
        print(f"Launching {label} on {dev_str}", flush=True)
        processes.append((label, subprocess.Popen(command, env=env)))
        if len(processes) == len(gpu_ids):
            for running_label, process in processes:
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
                print(f"Completed {running_label}", flush=True)
            processes.clear()

    # Flush any remaining processes
    for running_label, process in processes:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        print(f"Completed {running_label}", flush=True)
    processes.clear()

    # Print summary table and generate JSON summary
    print("\n" + "=" * 65)
    print(f"ALL EXPERIMENTS COMPLETED FOR MODELNET{args.dataset}")
    print("=" * 65)
    summary_data = {}
    for label, _, _ in selected_experiments:
        res_file = Path(f"experiments/results/modelnet{args.dataset}_{label}.txt")
        if res_file.exists():
            lines = res_file.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                if line.startswith("FINAL RESULTS:"):
                    print(line)
                    # Parse metrics
                    parts = dict(token.split("=") for token in line.split() if "=" in token)
                    summary_data[label] = {
                        "label": label,
                        "best_val": float(parts.get("best_val", 0.0)),
                        "test_at_best_val": float(parts.get("test_at_best_val", 0.0)),
                        "raw": line,
                    }
    print("=" * 65 + "\n")

    summary_file = Path(f"experiments/results/modelnet{args.dataset}_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    print(f"Summary written to {summary_file}")


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
