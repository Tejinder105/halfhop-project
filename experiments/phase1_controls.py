"""Phase 1 Controls Experiment: Disentangling Dilation vs Ego-Weighting.

Isolates whether Half-Hop's performance gains on heterophilic graphs
stem from path dilation (slow nodes) or pure ego-weighting / anti-smoothing.

Control Baselines:
1. MLP: Pure node features, zero graph propagation.
2. GCN (Standard): Symmetric normalized A + I (w=1).
3. GCN + Tuned Self-Loop (λ): A + λI, sweeping λ in {1, 2, 4, 8, 16}.
4. Lazy-Walk GCN (β): Operator (1 - β)I + β Â, sweeping β in {0.1, 0.2, 0.5, 0.8}.
5. APPNP: Teleport / initial residual anti-smoothing.
6. Half-Hop α -> 0: Slow node = target copy (maximal ego-weight).
7. Half-Hop α -> 1: Slow node = source copy (minimal ego-weight).
8. Half-Hop (Config): Dataset default HH-GCN.

All configurations are tuned strictly on validation splits, then reported on test.
"""

import argparse
import copy
import json
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, APPNP

from halfhop.datasets import load_dataset
from halfhop.splits import get_split
from halfhop.reproducibility import set_seed
from halfhop.evaluation import format_results
from halfhop.hh_gcn import HHGCN
from halfhop.training import train_step, evaluate


RESULTS_DIR = "experiments/results/phase1_controls"


# ---------------------------------------------------------------------------
# Models & Control Operators
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Pure node-feature baseline (no graph edges)."""
    def __init__(self, in_dim, hid, out_dim, dropout=0.5, depth=2):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hid] * (depth - 1) + [out_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(a, b))

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SelfLoopGCN(nn.Module):
    """GCN with custom self-loop weight λ (A + λI)."""
    def __init__(self, in_dim, hid, out_dim, loop_weight=1.0, dropout=0.5, depth=2):
        super().__init__()
        self.dropout = dropout
        self.loop_weight = loop_weight
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hid] * (depth - 1) + [out_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            # add_self_loops=False because we manually add weighted self loops
            self.convs.append(GCNConv(a, b, add_self_loops=False))

    def _prepare_adj(self, edge_index, num_nodes, device):
        # Filter existing self-loops
        mask = edge_index[0] != edge_index[1]
        ei = edge_index[:, mask]
        E = ei.size(1)
        ew = torch.ones(E, device=device)

        # Add self-loops with weight λ
        loop_idx = torch.arange(num_nodes, device=device)
        self_loops = torch.stack([loop_idx, loop_idx], dim=0)
        self_weights = torch.full((num_nodes,), self.loop_weight, device=device)

        ei_all = torch.cat([ei, self_loops], dim=1)
        ew_all = torch.cat([ew, self_weights], dim=0)
        return ei_all, ew_all

    def forward(self, x, edge_index):
        ei, ew = self._prepare_adj(edge_index, x.size(0), x.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, ei, edge_weight=ew)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LazyWalkGCN(nn.Module):
    """GCN using Lazy Random Walk operator: (1 - β)I + β Â."""
    def __init__(self, in_dim, hid, out_dim, beta=0.5, dropout=0.5, depth=2):
        super().__init__()
        self.dropout = dropout
        self.beta = beta
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hid] * (depth - 1) + [out_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            self.convs.append(GCNConv(a, b, add_self_loops=False))

    def _prepare_adj(self, edge_index, num_nodes, device):
        # 1. Symmetric normalize graph edges Â = D^{-1/2} A D^{-1/2}
        mask = edge_index[0] != edge_index[1]
        ei = edge_index[:, mask]
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, ei[0], torch.ones(ei.size(1), device=device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt.isinf()] = 0.0

        # Â edge weights
        ew_norm = deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]
        # Scale by β
        ew_scaled = self.beta * ew_norm

        # Add (1 - β) self-loops
        loop_idx = torch.arange(num_nodes, device=device)
        self_loops = torch.stack([loop_idx, loop_idx], dim=0)
        self_weights = torch.full((num_nodes,), 1.0 - self.beta, device=device)

        ei_all = torch.cat([ei, self_loops], dim=1)
        ew_all = torch.cat([ew_scaled, self_weights], dim=0)
        return ei_all, ew_all

    def forward(self, x, edge_index):
        ei, ew = self._prepare_adj(edge_index, x.size(0), x.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, ei, edge_weight=ew)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class APPNPModel(nn.Module):
    """APPNP (Initial residual anti-smoothing control)."""
    def __init__(self, in_dim, hid, out_dim, alpha=0.1, K=10, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )
        self.prop = APPNP(K=K, alpha=alpha, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        return self.prop(x, edge_index)


# ---------------------------------------------------------------------------
# Training & Split Runner
# ---------------------------------------------------------------------------

def run_single_model(model, data, optimizer, train_mask, val_mask, test_mask, epochs, device):
    model = model.to(device)
    data_x = data.x.to(device)
    data_ei = data.edge_index.to(device)
    data_y = data.y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    is_hh = hasattr(model, "halfhop")
    if is_hh:
        data_d = copy.deepcopy(data).to(device)

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if is_hh:
            out = model(data_d)
        else:
            out = model(data_x, data_ei)
        loss = F.cross_entropy(out[train_mask], data_y[train_mask])
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            if is_hh:
                out = model(data_d)
            else:
                out = model(data_x, data_ei)
            pred = out.argmax(dim=-1)
            val_acc = (pred[val_mask] == data_y[val_mask]).sum().item() / int(val_mask.sum())
            test_acc = (pred[test_mask] == data_y[test_mask]).sum().item() / int(test_mask.sum())

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

    return best_val, best_test


def run_phase1_controls(dataset_name: str, epochs: int = 200, seed: int = 42, device: str = None):
    dev = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"\n=======================================================")
    print(f"Phase 1 Controls Evaluation on: {dataset_name.upper()} (device={dev})")
    print(f"=======================================================")

    dataset, data = load_dataset(dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Load base config
    cfg_path = f"configs/{dataset_name}.yaml"
    cfg = {"lr": 0.01, "wd": 5e-4, "hidden": 64, "depth": 2, "dropout": 0.5, "alpha": 0.01, "p": 1.0}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        hh = raw.get("hh-gcn", {})
        cfg.update(hh)

    lr = cfg["lr"]
    wd = cfg["wd"]
    h = cfg["hidden"]
    d = cfg["depth"]
    dr = cfg["dropout"]
    alpha_hh = cfg.get("alpha", 0.01)
    p_hh = cfg.get("p", 1.0)

    num_splits = data.train_mask.size(1) if data.train_mask.dim() == 2 else 10

    # Controls to evaluate:
    # 1. MLP
    # 2. GCN (λ=1)
    # 3. GCN + SelfLoop (sweeping λ in {2, 4, 8, 16}) -> picked by val
    # 4. LazyWalk (sweeping β in {0.1, 0.2, 0.5, 0.8}) -> picked by val
    # 5. APPNP (sweeping α in {0.1, 0.2, 0.5}) -> picked by val
    # 6. HH α->0 (0.001)
    # 7. HH α->1 (0.999)
    # 8. HH (Config default)

    per_split_results = {
        "mlp": [],
        "gcn_standard": [],
        "gcn_selfloop_tuned": [],
        "lazy_walk_tuned": [],
        "appnp_tuned": [],
        "hh_alpha_0": [],
        "hh_alpha_1": [],
        "hh_config": [],
    }

    for split_id in range(num_splits):
        set_seed(seed + split_id)
        train_mask, val_mask, test_mask = get_split(data, split_id)

        # 1. MLP
        m = MLP(num_features, h, num_classes, dropout=dr, depth=d)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        _, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
        per_split_results["mlp"].append(t)

        # 2. Standard GCN (loop_weight=1.0)
        m = SelfLoopGCN(num_features, h, num_classes, loop_weight=1.0, dropout=dr, depth=d)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        _, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
        per_split_results["gcn_standard"].append(t)

        # 3. Tuned Self-Loop GCN (pick best λ on val)
        best_v, best_t = -1, 0
        for loop_w in [2.0, 4.0, 8.0, 16.0]:
            set_seed(seed + split_id)
            m = SelfLoopGCN(num_features, h, num_classes, loop_weight=loop_w, dropout=dr, depth=d)
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
            v, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
            if v > best_v:
                best_v, best_t = v, t
        per_split_results["gcn_selfloop_tuned"].append(best_t)

        # 4. Tuned Lazy Walk (pick best β on val)
        best_v, best_t = -1, 0
        for beta in [0.1, 0.2, 0.5, 0.8]:
            set_seed(seed + split_id)
            m = LazyWalkGCN(num_features, h, num_classes, beta=beta, dropout=dr, depth=d)
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
            v, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
            if v > best_v:
                best_v, best_t = v, t
        per_split_results["lazy_walk_tuned"].append(best_t)

        # 5. Tuned APPNP (pick best α_teleport on val)
        best_v, best_t = -1, 0
        for alpha_tele in [0.1, 0.2, 0.5]:
            set_seed(seed + split_id)
            m = APPNPModel(num_features, h, num_classes, alpha=alpha_tele, dropout=dr)
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
            v, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
            if v > best_v:
                best_v, best_t = v, t
        per_split_results["appnp_tuned"].append(best_t)

        # 6. Half-Hop α -> 0 (0.001)
        set_seed(seed + split_id)
        m = HHGCN(num_features, h, num_classes, alpha=0.001, p=p_hh, dropout=dr, depth=d)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        _, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
        per_split_results["hh_alpha_0"].append(t)

        # 7. Half-Hop α -> 1 (0.999)
        set_seed(seed + split_id)
        m = HHGCN(num_features, h, num_classes, alpha=0.999, p=p_hh, dropout=dr, depth=d)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        _, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
        per_split_results["hh_alpha_1"].append(t)

        # 8. Half-Hop (Config default)
        set_seed(seed + split_id)
        m = HHGCN(num_features, h, num_classes, alpha=alpha_hh, p=p_hh, dropout=dr, depth=d)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        _, t = run_single_model(m, data, opt, train_mask, val_mask, test_mask, epochs, dev)
        per_split_results["hh_config"].append(t)

        print(f"  Split {split_id:02d} complete. HH: {per_split_results['hh_config'][-1]*100:.2f}% | "
              f"SelfLoop: {per_split_results['gcn_selfloop_tuned'][-1]*100:.2f}% | "
              f"LazyWalk: {per_split_results['lazy_walk_tuned'][-1]*100:.2f}% | "
              f"MLP: {per_split_results['mlp'][-1]*100:.2f}%")

    # Summary table
    print(f"\n{'='*70}")
    print(f"PHASE 1 CONTROLS SUMMARY: {dataset_name.upper()}")
    print(f"{'='*70}")
    summary = {}
    hh_mean = np.mean(per_split_results["hh_config"]) * 100
    for name, vals in per_split_results.items():
        mean = float(np.mean(vals) * 100)
        std = float(np.std(vals) * 100)
        delta_hh = mean - hh_mean
        summary[name] = {
            "mean": mean,
            "std": std,
            "delta_vs_hh": delta_hh,
            "formatted": f"{mean:.2f} ± {std:.2f}",
            "per_split": [float(v * 100) for v in vals],
        }
        marker = " [CONTROL MATCHES HH]" if abs(delta_hh) < 1.5 and name != "hh_config" else ""
        print(f"  {name:25s}: {mean:6.2f}% ± {std:4.2f}% (Δ vs HH: {delta_hh:+5.2f}%){marker}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, f"{dataset_name}_phase1_controls.json")
    with open(out_file, "w") as f:
        json.dump({"dataset": dataset_name, "config": cfg, "results": summary}, f, indent=2)
    print(f"\nSaved results to {out_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 controls")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset name")
    parser.add_argument("--datasets", type=str, default="texas,cornell,wisconsin,chameleon",
                        help="Comma-separated datasets to run")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    ds_list = [args.dataset] if args.dataset else [d.strip() for d in args.datasets.split(",") if d.strip()]
    for ds in ds_list:
        run_phase1_controls(ds, args.epochs, args.seed, args.device)


if __name__ == "__main__":
    main()

