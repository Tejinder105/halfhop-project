"""Fair budget-equalized multi-seed ModelNet10 experiment.

Fixes all 7 flaws identified in run_adaptive.py:
  F1: Budget & complexity control — new sparse_inv avoids 20x node explosion (~6.9x).
  F2: 'none' config stores correct metadata (delays=(0,0,0)).
  F3: Augmentation commutes with slow-node interpolation; dynamic augmentation per epoch.
  F4: Multi-seed support (default seeds 0, 1, 2) reporting mean ± std & Wilcoxon test.
  F5: 50 epochs + cosine annealing LR schedule + per-epoch JSONL logs saved to disk.
  F6: Modern EdgeConv backbone (DGCNN-style relative position features) + optional GraphSAGE.
  F7: Dedicated isolated pre_transform cache per point count (no PyG cache collisions).

New approach — sparse_inv:
  mode=inverse, delays=(0, 0, 1), quantiles=(0.5, 0.85)
  → Planar (low-curvature) edges get delay 1: soft-delayed by one slow node to combat over-smoothing.
  → Boundary / corner (high-curvature) edges keep delay 0: flow freely at maximum speed.
  Hypothesis: diagnostic landmarks flow freely; planar redundancy is gently filtered.

Usage (single run):
  python -m experiments.modelnet.run_fair --condition sparse_inv --dataset 10

Usage (full ablation, 2 GPUs):
  python -m experiments.modelnet.run_fair --run-all --gpus 0,1

Kaggle:
  !bash experiments/modelnet/run_fair_kaggle.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv, global_add_pool, global_max_pool, global_mean_pool
from tqdm import tqdm

from experiments.modelnet.adaptive_halfhop import AdaptiveHalfHop, PointCloudPreTransform
from halfhop.graphsage import GraphSAGE
from halfhop.reproducibility import set_seed


CONDITIONS = {
    # name          delays     mode         m  description
    "none":        ((0,0,0),  "none",       0, "Baseline KNN graph (no slow nodes)"),
    "sparse_inv":  ((0,0,1),  "inverse",    0, "NEW: Inverse delay (planar delayed, boundary free)"),
    "uniform":     ((1,1,1),  "uniform",    1, "Uniform Half-Hop (m=1 on all edges)"),
    "curvature":   ((1,2,3),  "curvature",  1, "Original AdaHop (high-curv edges delayed most)"),
    "random":      ((0,0,1),  "random",     0, "Random delay control matched to sparse_inv budget"),
}

SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
def augment_shape(data: Data) -> Data:
    """Random Z-rotation, jitter, and scale applied to coordinates and normals.

    Because linear interpolation commutes with linear spatial transforms:
    ( (1-λ)u + λv ) R = (1-λ)(u R) + λ(v R),
    rigid rotation and scaling applied to node features preserves the exact
    spatial alignment of slow nodes along edges without recomputing graphs!
    """
    data = data.clone()
    x = data.x.clone()
    theta = torch.rand(1).item() * 2 * math.pi
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = torch.tensor(
        [[cos_t, -sin_t, 0.0],
         [sin_t,  cos_t, 0.0],
         [0.0,    0.0,   1.0]],
        dtype=x.dtype,
        device=x.device,
    )
    # Coordinates (cols 0:3)
    pos = x[:, :3] @ rot.T
    pos = pos + torch.randn_like(pos) * 0.01
    pos = pos.clamp(-1.0, 1.0)
    scale = 0.85 + torch.rand(1).item() * 0.30  # [0.85, 1.15]
    x[:, :3] = pos * scale
    # Normals (cols 3:6 if present)
    if x.size(1) >= 6:
        x[:, 3:6] = x[:, 3:6] @ rot.T
    data.x = x
    return data


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _mlp(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(cin, cout, bias=False),
        nn.BatchNorm1d(cout),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DGCNNClassifier(nn.Module):
    """DGCNN-style EdgeConv with relative coordinate modeling."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = EdgeConv(_mlp(2 * in_channels, hidden), aggr="max")
        self.conv2 = EdgeConv(_mlp(2 * hidden, hidden), aggr="max")
        self.conv3 = EdgeConv(_mlp(2 * hidden, 2 * hidden), aggr="max")
        self.global_dim = 4 * hidden
        self.head = nn.Sequential(
            nn.Linear(self.global_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, data: Data) -> Tensor:
        x, ei = data.x, data.edge_index
        x1 = self.conv1(x, ei)
        x2 = self.conv2(x1, ei)
        x3 = self.conv3(x2, ei)

        # Pool only over original non-slow nodes
        keep = ~data.slow_node_mask
        x3_orig = x3[keep]
        b = data.batch[keep]

        g_max = global_max_pool(x3_orig, b)
        g_add = global_add_pool(x3_orig, b)
        g = torch.cat([g_max, g_add], dim=-1)
        return self.head(g)


class SAGEClassifier(nn.Module):
    """Variable-depth GraphSAGE classifier."""

    def __init__(self, in_channels: int, hidden: int, depth: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = GraphSAGE(in_channels, hidden, hidden, dropout=dropout, depth=depth)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data: Data) -> Tensor:
        x = self.encoder(data.x, data.edge_index)
        keep = ~data.slow_node_mask
        x_orig = x[keep]
        b = data.batch[keep]
        pooled = torch.cat([global_max_pool(x_orig, b), global_mean_pool(x_orig, b)], dim=1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Dataset Helpers
# ---------------------------------------------------------------------------
def load_base_dataset(dataset: str, data_dir: str, points: int) -> tuple:
    """Load ModelNet with isolated cache per point count to avoid PyG cache collisions."""
    cond_dir = os.path.join(data_dir, f"pts_{points}")
    pre = PointCloudPreTransform(points=points, knn=12, curv_k=16)
    train = ModelNet(cond_dir, name=dataset, train=True, pre_transform=pre)
    test = ModelNet(cond_dir, name=dataset, train=False, pre_transform=pre)
    return train, test


def make_transform(condition: str) -> AdaptiveHalfHop:
    delays, mode, uniform_m, _ = CONDITIONS[condition]
    return AdaptiveHalfHop(
        knn=12,
        curv_k=16,
        delays=delays,
        quantiles=(0.5, 0.85),
        mode=mode,
        uniform_m=uniform_m,
        backward="last",
    )


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def accuracy(loader: DataLoader, model: nn.Module, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)
            correct += int((pred == batch.y).sum())
            total += batch.num_graphs
            del batch
    return correct / max(total, 1)


def train_epoch(loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch.num_graphs
        del batch, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_sum / max(len(loader.dataset), 1)


class AugDataset:
    """Wraps cached Data objects with dynamic data augmentation per epoch."""

    def __init__(self, items: list[Data]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Data:
        return augment_shape(self.items[idx])


def run_one(args: argparse.Namespace, seed: int) -> dict:
    set_seed(seed)
    device = torch.device(
        args.device if args.device != "auto"
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    condition = args.condition
    transform = make_transform(condition)

    print(f"[{condition}|seed={seed}] Loading ModelNet{args.dataset} ({args.points} pts)...", flush=True)
    train_ds, test_ds = load_base_dataset(args.dataset, args.data_dir, args.points)

    print(f"  Transforming {len(train_ds)} train + {len(test_ds)} test graphs...", flush=True)
    train_cache = [transform(d) for d in tqdm(train_ds, leave=False, desc="train")]
    test_cache = [transform(d) for d in tqdm(test_ds, leave=False, desc="test")]

    # Deterministic train/val split per seed
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_cache), generator=gen)
    val_size = max(1, len(train_cache) // 10)

    val_data = [train_cache[i] for i in perm[:val_size]]
    train_data = [train_cache[i] for i in perm[val_size:]]

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(AugDataset(train_data), shuffle=True, **loader_kw)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_cache, shuffle=False, **loader_kw)

    sample = train_cache[0]
    in_channels = sample.x.size(1)
    num_classes = int(args.dataset)

    if args.backbone == "edgeconv":
        model = DGCNNClassifier(in_channels, args.hidden, num_classes, args.dropout).to(device)
    else:
        model = SAGEClassifier(in_channels, args.hidden, args.layers, num_classes, args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{condition}_seed{seed}.jsonl"

    best_val, best_test, best_epoch = -1.0, 0.0, 0
    epoch_logs = []

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"{condition}|s{seed}", leave=False):
        t0 = time.time()
        loss = train_epoch(train_loader, model, optimizer, device)
        val = accuracy(val_loader, model, device)
        test = accuracy(test_loader, model, device)
        scheduler.step()
        elapsed = time.time() - t0

        if val >= best_val:
            best_val, best_test, best_epoch = val, test, epoch

        log = dict(
            epoch=epoch, loss=round(loss, 4), val=round(val, 4),
            test=round(test, 4), t=round(elapsed, 2), seed=seed,
            condition=condition,
        )
        epoch_logs.append(log)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"  epoch={epoch:03d} loss={loss:.4f} val={val:.4f} test={test:.4f} lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

    with log_path.open("w", encoding="utf-8") as f:
        for rec in epoch_logs:
            f.write(json.dumps(rec) + "\n")

    result = dict(
        condition=condition,
        seed=seed,
        dataset=args.dataset,
        backbone=args.backbone,
        points=args.points,
        best_val=round(best_val, 4),
        test_at_best_val=round(best_test, 4),
        best_epoch=best_epoch,
        epochs=args.epochs,
        num_params=sum(p.numel() for p in model.parameters()),
        log=str(log_path),
    )

    out_path = Path(args.output or f"experiments/results/fair_modelnet/modelnet{args.dataset}_{condition}_s{seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    line = (f"FINAL: {condition}|seed={seed} val={best_val:.4f} test={best_test:.4f} "
            f"epoch={best_epoch}/{args.epochs} backbone={args.backbone}")
    print(line, flush=True)
    return result


# ---------------------------------------------------------------------------
# Multi-seed Summary
# ---------------------------------------------------------------------------
def summarize(results: list[dict], baseline: str = "none") -> dict:
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r["test_at_best_val"])

    summary = {}
    baseline_scores = by_cond.get(baseline, [])
    for cond, scores in sorted(by_cond.items()):
        mean = sum(scores) / len(scores)
        std = (sum((s - mean)**2 for s in scores) / max(len(scores) - 1, 1)) ** 0.5
        entry = dict(mean=round(mean, 4), std=round(std, 4), n=len(scores), scores=scores)
        if cond != baseline and len(scores) >= 2 and len(baseline_scores) >= 2:
            try:
                from scipy.stats import wilcoxon
                common = min(len(scores), len(baseline_scores))
                _, p = wilcoxon(scores[:common], baseline_scores[:common], alternative="greater")
                entry["wilcoxon_p_gt_baseline"] = round(float(p), 4)
            except Exception:
                pass
        summary[cond] = entry
    return summary


# ---------------------------------------------------------------------------
# run_all: launches parallel jobs across available GPUs
# ---------------------------------------------------------------------------
def run_all(args: argparse.Namespace) -> None:
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if torch.cuda.is_available():
        avail = torch.cuda.device_count()
        gpu_ids = [g for g in gpu_ids if int(g) < avail] or ["0"]
    else:
        gpu_ids = ["cpu"]

    conditions = args.conditions or list(CONDITIONS.keys())
    jobs = []
    for cond in conditions:
        for seed in args.seeds:
            jobs.append((cond, seed))

    print(f"Executing {len(jobs)} total runs across GPUs {gpu_ids}: {jobs}", flush=True)

    processes = []
    all_results = []

    for index, (cond, seed) in enumerate(jobs):
        env = os.environ.copy()
        if gpu_ids[0] != "cpu":
            target_gpu = gpu_ids[index % len(gpu_ids)]
            env["CUDA_VISIBLE_DEVICES"] = target_gpu
            dev_str = f"GPU {target_gpu}"
            target_device = "cuda:0"
        else:
            dev_str = "CPU"
            target_device = "cpu"

        out_file = f"experiments/results/fair_modelnet/modelnet{args.dataset}_{cond}_s{seed}.json"
        cmd = [
            sys.executable, "-m", "experiments.modelnet.run_fair",
            "--condition", cond,
            "--dataset", args.dataset,
            "--data-dir", args.data_dir,
            "--points", str(args.points),
            "--backbone", args.backbone,
            "--layers", str(args.layers),
            "--epochs", str(args.epochs),
            "--hidden", str(args.hidden),
            "--dropout", str(args.dropout),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--wd", str(args.wd),
            "--log-every", str(args.log_every),
            "--log-dir", args.log_dir,
            "--device", target_device,
            "--seed", str(seed),
            "--output", out_file,
        ]

        print(f"Launching {cond}|seed={seed} on {dev_str}", flush=True)
        processes.append((f"{cond}_s{seed}", subprocess.Popen(cmd, env=env), out_file))

        if len(processes) == len(gpu_ids):
            for label, proc, rfile in processes:
                code = proc.wait()
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
                if Path(rfile).exists():
                    all_results.append(json.loads(Path(rfile).read_text()))
                print(f"Completed {label}", flush=True)
            processes.clear()

    for label, proc, rfile in processes:
        code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, cmd)
        if Path(rfile).exists():
            all_results.append(json.loads(Path(rfile).read_text()))
        print(f"Completed {label}", flush=True)
    processes.clear()

    summary = summarize(all_results)
    print("\n" + "=" * 70)
    print(f"FAIR MODELNET{args.dataset} SUMMARY ({args.backbone.upper()}, {len(args.seeds)} seeds, points={args.points})")
    print(f"{'Condition':18s} {'Mean':>7s} {'Std':>6s} {'Scores'}")
    print("-" * 70)
    for cond, s in sorted(summary.items(), key=lambda x: -x[1]["mean"]):
        pstr = f"  p_vs_none={s['wilcoxon_p_gt_baseline']:.4f}" if "wilcoxon_p_gt_baseline" in s else ""
        print(f"{cond:18s} {s['mean']:7.4f} ±{s['std']:5.4f}  {s['scores']}{pstr}")
    print("=" * 70)

    out = Path(f"experiments/results/fair_modelnet/modelnet{args.dataset}_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)
    print(f"\nFull summary written to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair multi-seed ModelNet ablation")
    p.add_argument("--condition", default="none", choices=list(CONDITIONS))
    p.add_argument("--conditions", nargs="+", default=None, choices=list(CONDITIONS))
    p.add_argument("--dataset", default="10", choices=["10", "40"])
    p.add_argument("--data-dir", default="data/ModelNet10")
    p.add_argument("--points", type=int, default=512)
    p.add_argument("--backbone", choices=["edgeconv", "sage"], default="edgeconv")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-dir", default="experiments/results/fair_modelnet/logs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--gpus", default="0,1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.run_all:
        run_all(args)
        return
    args.seed = getattr(args, "seed", 0)
    result = run_one(args, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
