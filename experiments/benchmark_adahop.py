"""Benchmark Suite for AdaHop: Adaptive Path Dilation vs. Uniform Half-Hop & Baselines.

Evaluates whether selective / adaptive path dilation matches or outperforms uniform
Half-Hop while drastically reducing graph expansion and computational overhead.

Methods Evaluated:
  1. plain_gcn: Baseline GCN (0% extra nodes)
  2. lazywalk: Tuned Lazy Walk (0% extra nodes)
  3. uniform_halfhop: Standard Half-Hop (100% of edges dilated with k=1 slow node)
  4. uniform_multidelay_k2: Uniform MultiDelay (100% of edges dilated with k=2 slow nodes)
  5. adahop_b25: AdaHop (k=2, only top 25% of conflicting edges dilated - 75% savings)
  6. adahop_b50: AdaHop (k=2, top 50% of conflicting edges dilated - 50% savings)
  7. adahop_soft: Soft AdaHop (continuous soft edge routing, 0 extra nodes)

Usage:
    python -m experiments.benchmark_adahop --dataset wisconsin --depth 4 --device cuda:0
    python -m experiments.benchmark_adahop --dataset chameleon --depth 4 --device cuda:1
"""

import argparse
import copy
import json
import os
import sys
import time

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WebKB, WikipediaNetwork
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops

from halfhop.adaptive_delay import AdaHopGNN, SelectiveMultiDelay
from halfhop.soft_halfhop import SoftHalfHop


# -----------------------------------------------------------------------------
# 1. Dataset Loader
# -----------------------------------------------------------------------------
def load_dataset_data(name, root="data"):
    nl = name.lower()
    if nl in ("wisconsin", "texas", "cornell"):
        ds = WebKB(root=os.path.join(root, nl.capitalize()), name=nl.capitalize())
    elif nl in ("chameleon", "squirrel"):
        ds = WikipediaNetwork(root=os.path.join(root, nl), name=nl)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds[0], ds.num_classes


# -----------------------------------------------------------------------------
# 2. Lazy Walk Operator
# -----------------------------------------------------------------------------
def get_lazy_walk_matrix(ei, n, beta, device):
    ei_clean, _ = remove_self_loops(ei)
    row, col = ei_clean[0], ei_clean[1]
    deg = torch.zeros(n, device=device).scatter_add_(0, row, torch.ones(ei_clean.size(1), device=device))
    w = beta / deg.clamp(min=1.0)[row]
    idx = torch.arange(n, device=device)
    rows = torch.cat([row, idx])
    cols = torch.cat([col, idx])
    vals = torch.cat([w, torch.full((n,), 1.0 - beta, device=device)])
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n), device=device).coalesce()


# -----------------------------------------------------------------------------
# 3. Model Wrapper
# -----------------------------------------------------------------------------
class BenchmarkModel(nn.Module):
    def __init__(self, method, in_dim, hid_dim, out_dim, depth=4, dropout=0.3, beta=0.1, alpha=0.1):
        super().__init__()
        self.method = method
        self.depth = depth
        self.dropout = dropout

        dims = [in_dim] + [hid_dim] * (depth - 1) + [out_dim]

        if method in ("plain_gcn", "uniform_halfhop", "uniform_multidelay_k2"):
            self.convs = nn.ModuleList([GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:])])
        elif method == "lazywalk":
            self.convs = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        elif method.startswith("adahop_b"):
            budget = 0.25 if "b25" in method else 0.50
            self.adahop = AdaHopGNN(
                in_channels=in_dim,
                hidden_channels=hid_dim,
                out_channels=out_dim,
                depth=depth,
                k=2,
                budget=budget,
                alpha=alpha,
                mode="selective",
                dropout=dropout,
            )
        elif method == "adahop_soft":
            self.soft_hh = SoftHalfHop(in_dim, alpha=alpha, hidden=32)
            self.convs = nn.ModuleList([GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x, ei, P=None):
        if self.method.startswith("adahop_b"):
            return self.adahop(x, ei)

        if self.method == "adahop_soft":
            x_aug, ei_aug, ew_aug, slow_mask, _ = self.soft_hh(x, ei)
            h = x_aug
            for i, conv in enumerate(self.convs):
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = conv(h, ei_aug, edge_weight=ew_aug)
                if i < len(self.convs) - 1:
                    h = F.relu(h)
            return h[~slow_mask]

        h = x
        for i, c in enumerate(self.convs):
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.method == "lazywalk":
                h_next = torch.sparse.mm(P, c(h))
            else:
                h_next = c(h, ei)
            if i < len(self.convs) - 1:
                h = F.relu(h_next)
            else:
                h = h_next
        return h


# -----------------------------------------------------------------------------
# 4. Graph Construction for Uniform Baselines
# -----------------------------------------------------------------------------
def prepare_graph(data, method, alpha, device):
    x, ei = data.x.to(device), data.edge_index.to(device)
    P = None

    if method == "plain_gcn":
        return x, ei, P, x.size(0), ei.size(1)

    if method == "lazywalk":
        beta = 0.1033 if data.num_nodes < 500 else 0.9854
        P = get_lazy_walk_matrix(ei, data.num_nodes, beta, device)
        return x, ei, P, x.size(0), ei.size(1)

    if method == "uniform_halfhop":
        from experiments.depth_matched_delay import build_multidelay_graph
        x_aug, ei_aug, _ = build_multidelay_graph(data, k=1, alpha=alpha, p=1.0)
        return x_aug.to(device), ei_aug.to(device), P, x_aug.size(0), ei_aug.size(1)

    if method == "uniform_multidelay_k2":
        from experiments.depth_matched_delay import build_multidelay_graph
        x_aug, ei_aug, _ = build_multidelay_graph(data, k=2, alpha=alpha, p=1.0)
        return x_aug.to(device), ei_aug.to(device), P, x_aug.size(0), ei_aug.size(1)

    if method == "adahop_b25":
        ei_clean, _ = remove_self_loops(ei)
        dilated_e = int(round(0.25 * ei_clean.size(1)))
        nodes_est = data.num_nodes + 2 * dilated_e
        edges_est = (ei_clean.size(1) - dilated_e) + data.num_nodes + dilated_e * 4
        return x, ei, P, nodes_est, edges_est

    if method == "adahop_b50":
        ei_clean, _ = remove_self_loops(ei)
        dilated_e = int(round(0.50 * ei_clean.size(1)))
        nodes_est = data.num_nodes + 2 * dilated_e
        edges_est = (ei_clean.size(1) - dilated_e) + data.num_nodes + dilated_e * 4
        return x, ei, P, nodes_est, edges_est

    if method == "adahop_soft":
        ei_clean, _ = remove_self_loops(ei)
        return x, ei, P, data.num_nodes + ei_clean.size(1), ei_clean.size(1) * 4

    return x, ei, P, x.size(0), ei.size(1)


# -----------------------------------------------------------------------------
# 5. Training & Evaluation Runner
# -----------------------------------------------------------------------------
def run_split(model, x, ei, P, y, tr, va, te, N_orig, epochs, patience, lr, wd):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_va, best_te = -1.0, 0.0
    bad = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, ei, P)[:N_orig]
        loss = F.cross_entropy(out[tr], y[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(x, ei, P)[:N_orig]
            pred = out.argmax(dim=1)
            acc_va = float((pred[va] == y[va]).float().mean().item())
            acc_te = float((pred[te] == y[te]).float().mean().item())

        if acc_va > best_va:
            best_va = acc_va
            best_te = acc_te
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_va, best_te


def evaluate_benchmark(dataset_name, depth=4, methods=None, device="cuda:0",
                       epochs=200, patience=50, out_dir="experiments/results/adahop"):
    os.makedirs(out_dir, exist_ok=True)
    data, nc = load_dataset_data(dataset_name)
    N_orig = data.num_nodes
    y = data.y.to(device)

    # Defaults per dataset
    if dataset_name.lower() == "wisconsin":
        cfg = {"lr": 0.01, "hidden": 64, "dropout": 0.25, "wd": 1e-4, "alpha": 0.25}
    elif dataset_name.lower() == "chameleon":
        cfg = {"lr": 0.01, "hidden": 128, "dropout": 0.30, "wd": 1e-4, "alpha": 0.025}
    elif dataset_name.lower() == "texas":
        cfg = {"lr": 0.01, "hidden": 64, "dropout": 0.25, "wd": 1e-4, "alpha": 0.05}
    else:
        cfg = {"lr": 0.01, "hidden": 64, "dropout": 0.25, "wd": 1e-4, "alpha": 0.10}

    if methods is None:
        methods = [
            "plain_gcn",
            "lazywalk",
            "uniform_halfhop",
            "uniform_multidelay_k2",
            "adahop_b25",
            "adahop_b50",
            "adahop_soft",
        ]

    print(f"\n==================================================================")
    print(f"AdaHop Comprehensive Benchmark on {dataset_name.upper()} (Depth L={depth})")
    print(f"Device: {device} | Base Config: {cfg}")
    print(f"==================================================================\n")

    results = {}
    for method in methods:
        x, ei, P, total_nodes, total_edges = prepare_graph(data, method, cfg["alpha"], device)
        extra_nodes_pct = ((total_nodes - N_orig) / float(N_orig)) * 100.0

        val_scores, test_scores = [], []
        t0 = time.time()

        for s in range(data.train_mask.size(1)):
            torch.manual_seed(s)
            tr = data.train_mask[:, s].to(device)
            va = data.val_mask[:, s].to(device)
            te = data.test_mask[:, s].to(device)

            model = BenchmarkModel(
                method=method,
                in_dim=x.size(1),
                hid_dim=cfg["hidden"],
                out_dim=nc,
                depth=depth,
                dropout=cfg["dropout"],
                beta=0.1033,
                alpha=cfg["alpha"],
            ).to(device)

            va_acc, te_acc = run_split(
                model=model,
                x=x, ei=ei, P=P, y=y,
                tr=tr, va=va, te=te,
                N_orig=N_orig,
                epochs=epochs,
                patience=patience,
                lr=cfg["lr"],
                wd=cfg["wd"],
            )
            val_scores.append(va_acc)
            test_scores.append(te_acc)

        dt = time.time() - t0
        m_v = float(np.mean(val_scores))
        m_t = float(np.mean(test_scores))
        s_t = float(np.std(test_scores))

        results[method] = {
            "val_mean": m_v,
            "test_mean": m_t,
            "test_std": s_t,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "extra_nodes_pct": extra_nodes_pct,
            "runtime_sec": dt,
        }

        print(f"[{method:22s}] Test: {m_t*100:5.2f} ± {s_t*100:4.2f}% | "
              f"Val: {m_v*100:5.2f}% | Nodes: {total_nodes:6d} (+{extra_nodes_pct:5.1f}%) | "
              f"Time: {dt:.1f}s")

    # Save Results
    json_path = os.path.join(out_dir, f"adahop_benchmark_{dataset_name.lower()}_L{depth}.json")
    with open(json_path, "w") as f:
        json.dump({"dataset": dataset_name, "depth": depth, "results": results}, f, indent=2)
    print(f"\n[Saved Benchmark Summary] -> {json_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="AdaHop Benchmark Suite")
    parser.add_argument("--dataset", required=True, choices=["wisconsin", "chameleon", "texas", "cornell"])
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--out", default="experiments/results/adahop")
    args = parser.parse_args()

    evaluate_benchmark(
        dataset_name=args.dataset,
        depth=args.depth,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
