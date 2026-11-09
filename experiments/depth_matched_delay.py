"""Step 3: Depth-Matched Dilation & Multi-Delay Study.

Investigates whether Half-Hop's physical path dilation (slow nodes) or multi-delay (k > 1)
provides any structural advantage over simpler ego-weighting (Lazy Walk) or deep baselines
at depths L in {2, 3, 4, 6, 8, 16}.

Protocol Designed per Reviewer Guidance:
  1. Weightless / Linear Propagation Diagnostic (Figure 3-style):
     - Unparameterized propagation over steps m in {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}.
     - Tracks:
       a) Normalized Dirichlet Energy: E(X) = Tr(X^T L X) / ||X||_F^2
       b) Mean Average Distance (MAD): 1 - mean cosine similarity (diversity)
       c) Weightless Linear Probe Accuracy (RidgeClassifier on X^(m) across 10 splits)
     - Operators: Plain GCN, LazyWalk (beta), HalfHop (k=1), MultiDelay (k=2, k=3), APPNP.
  2. Full Supervised Depth Scaling Experiment:
     - Methods: plain, lazywalk, halfhop_k1, halfhop_k2, halfhop_k3, appnp, gcn_res.
     - Depths: L in {2, 3, 4, 6, 8, 16} (reporting effective hops L / (k+1)).
     - Tracks: Best Val, Test at Best Val (mean +- std over 10 splits), final Dirichlet Energy, final MAD.

Usage:
    python -m experiments.depth_matched_delay --dataset wisconsin --device cuda:0
    python -m experiments.depth_matched_delay --dataset chameleon --device cuda:1
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, WebKB, WikipediaNetwork
from torch_geometric.nn import APPNP, GCNConv
from torch_geometric.utils import add_self_loops, remove_self_loops


# -----------------------------------------------------------------------------
# 1. Dataset Loader
# -----------------------------------------------------------------------------
def load_data(name, root="data"):
    nl = name.lower()
    if nl in ("wisconsin", "texas", "cornell"):
        ds = WebKB(root=os.path.join(root, nl.capitalize()), name=nl.capitalize())
    elif nl == "actor":
        ds = Actor(root=os.path.join(root, "Actor"))
    elif nl in ("chameleon", "squirrel"):
        ds = WikipediaNetwork(root=os.path.join(root, nl), name=nl)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    d = ds[0]
    return d, ds.num_classes


# -----------------------------------------------------------------------------
# 2. Dirichlet Energy & MAD Metrics (Oversmoothing Quantifiers)
# -----------------------------------------------------------------------------
def compute_dirichlet_energy(x, edge_index):
    """Normalized Dirichlet energy: Tr(X^T L X) / ||X||_F^2."""
    if x.numel() == 0:
        return 0.0
    src, dst = edge_index[0], edge_index[1]
    diff = x[src] - x[dst]
    sq_dist = (diff * diff).sum()
    norm_sq = (x * x).sum()
    if norm_sq.item() <= 1e-12:
        return 0.0
    return float((sq_dist / (2.0 * norm_sq)).item())


def compute_mad(x, sample_size=1000):
    """Mean Average Distance (MAD) measuring cosine diversity."""
    n = x.size(0)
    if n <= 1:
        return 0.0
    if n > sample_size:
        idx = torch.randperm(n, device=x.device)[:sample_size]
        x_sub = x[idx]
    else:
        x_sub = x
    x_norm = F.normalize(x_sub.float(), p=2, dim=1)
    cos_sim = torch.mm(x_norm, x_norm.t())
    n_sub = x_sub.size(0)
    mask = ~torch.eye(n_sub, dtype=torch.bool, device=x.device)
    mean_cos = float(cos_sim[mask].mean().item())
    return max(0.0, 1.0 - mean_cos)


# -----------------------------------------------------------------------------
# 3. Multi-Delay Graph Construction (k slow nodes in series per edge)
# -----------------------------------------------------------------------------
def build_multidelay_graph(data, k=1, alpha=0.5, p=1.0, seed=0):
    """Builds a Multi-Delay Half-Hop graph with k slow nodes along each edge.

    For edge u -> v:
      - k=0: untouched u -> v
      - k=1: u -> nu_1 <-> v (standard Half-Hop)
      - k>=2: u -> nu_1 <-> nu_2 <-> ... <-> nu_k <-> v
    Effective hop distance from u to v becomes (k + 1) hops.
    """
    if k == 0:
        return data.x.clone(), data.edge_index.clone(), None

    torch.manual_seed(seed)
    x, edge_index = data.x, data.edge_index
    device = x.device
    num_orig = x.size(0)

    # 1. Isolate self loops
    self_loop_mask = edge_index[0] == edge_index[1]
    edge_index_self_loop = edge_index[:, self_loop_mask]
    edge_index_non_self = edge_index[:, ~self_loop_mask]

    # 2. Node-level sampling with probability p
    if p >= 1.0:
        edge_index_to_delay = edge_index_non_self
        edge_index_to_keep = torch.empty((2, 0), dtype=edge_index.dtype, device=device)
    else:
        node_mask = torch.rand(num_orig, device=device) < p
        # Check target node
        edge_mask = node_mask[edge_index_non_self[1]]
        edge_index_to_delay = edge_index_non_self[:, edge_mask]
        edge_index_to_keep = edge_index_non_self[:, ~edge_mask]

    num_edges_to_delay = edge_index_to_delay.size(1)
    if num_edges_to_delay == 0:
        return x.clone(), edge_index.clone(), None

    src = edge_index_to_delay[0]
    dst = edge_index_to_delay[1]

    # 3. Allocate slow nodes: k slow nodes per sampled edge
    # Total slow nodes = num_edges_to_delay * k
    slow_node_features = []
    edges_list = []

    # Keep unmodified edges & self-loops
    if edge_index_to_keep.size(1) > 0:
        edges_list.append(edge_index_to_keep)
    if edge_index_self_loop.size(1) > 0:
        edges_list.append(edge_index_self_loop)

    # IDs for slow nodes:
    # slow_ids[s] has shape [num_edges_to_delay] for stage s in {0, ..., k-1}
    slow_ids = [
        torch.arange(num_edges_to_delay, device=device, dtype=edge_index.dtype)
        + (num_orig + s * num_edges_to_delay)
        for s in range(k)
    ]

    # 4. Feature interpolation for each stage s in {0, ..., k-1}
    for s in range(k):
        # Position along the edge t_s in (0, 1)
        t_s = float(s + 1) / float(k + 1)
        # Blend source and target with alpha
        # x_slow = (1 - t_s) * (alpha * x_u + (1 - alpha) * x_v) + t_s * x_v
        # Note: when alpha=0, x_slow = x_v (ego buffering)
        # Note: when alpha=1, x_slow = (1-t_s)*x_u + t_s*x_v (linear interpolation)
        x_src = x[src]
        x_dst = x[dst]
        x_u_blend = alpha * x_src + (1.0 - alpha) * x_dst
        x_s = (1.0 - t_s) * x_u_blend + t_s * x_dst
        slow_node_features.append(x_s)

    # 5. Wire connectivity
    # Edge: src -> slow_ids[0]
    edges_list.append(torch.stack([src, slow_ids[0]], dim=0))

    # Intermediate chain connections
    for s in range(k - 1):
        # Forward: slow_ids[s] -> slow_ids[s+1]
        edges_list.append(torch.stack([slow_ids[s], slow_ids[s + 1]], dim=0))
        # Backward: slow_ids[s+1] -> slow_ids[s]
        edges_list.append(torch.stack([slow_ids[s + 1], slow_ids[s]], dim=0))

    # Connection to target:
    # Forward: slow_ids[k-1] -> dst
    edges_list.append(torch.stack([slow_ids[k - 1], dst], dim=0))
    # Backward: dst -> slow_ids[k-1] (Half-Hop feedback motif)
    edges_list.append(torch.stack([dst, slow_ids[k - 1]], dim=0))

    new_x = torch.cat([x] + slow_node_features, dim=0)
    new_edge_index = torch.cat(edges_list, dim=1)
    total_slow = num_edges_to_delay * k
    slow_mask = torch.cat([
        torch.zeros(num_orig, dtype=torch.bool, device=device),
        torch.ones(total_slow, dtype=torch.bool, device=device),
    ], dim=0)

    return new_x, new_edge_index, slow_mask


# -----------------------------------------------------------------------------
# 4. Sparse Propagation Operators
# -----------------------------------------------------------------------------
def make_gcn_operator(edge_index, n, device):
    """Symmetric GCN operator: P = D~^(-1/2) A~ D~^(-1/2)."""
    ei, _ = remove_self_loops(edge_index)
    ei, _ = add_self_loops(ei, num_nodes=n)
    row, col = ei[0], ei[1]
    deg = torch.zeros(n, device=device).scatter_add_(0, row, torch.ones(ei.size(1), device=device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt.isinf()] = 0.0
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(ei, val, (n, n), device=device).coalesce()


def make_lazy_operator(edge_index, n, beta, device):
    """Row-stochastic Lazy Walk operator: P = (1-beta) I + beta D^-1 A."""
    ei, _ = remove_self_loops(edge_index)
    row, col = ei[0], ei[1]
    deg = torch.zeros(n, device=device).scatter_add_(0, row, torch.ones(ei.size(1), device=device))
    w = beta / deg.clamp(min=1.0)[row]
    idx = torch.arange(n, device=device)
    rows = torch.cat([row, idx])
    cols = torch.cat([col, idx])
    vals = torch.cat([w, torch.full((n,), 1.0 - beta, device=device)])
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n), device=device).coalesce()


# -----------------------------------------------------------------------------
# 5. Weightless / Linear Propagation Diagnostic
# -----------------------------------------------------------------------------
def run_weightless_diagnostic(data, steps=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32), device="cpu"):
    """Evaluates oversmoothing in the weightless linear regime across step counts.

    Produces Dirichlet energy, MAD, and 10-split linear probe accuracy for:
      - Plain GCN
      - LazyWalk (beta=0.1, 0.5, 0.9)
      - HalfHop k=1, k=2, k=3
      - APPNP
    """
    x = data.x.to(device).float()
    ei = data.edge_index.to(device)
    y = data.y.cpu().numpy()
    N = x.size(0)

    results = {}
    variants = [
        ("Plain GCN", "plain", {}),
        ("LazyWalk (beta=0.1)", "lazy", {"beta": 0.1}),
        ("LazyWalk (beta=0.5)", "lazy", {"beta": 0.5}),
        ("Half-Hop (k=1, a=0.05)", "hh", {"k": 1, "alpha": 0.05}),
        ("Half-Hop (k=1, a=0.5)", "hh", {"k": 1, "alpha": 0.5}),
        ("MultiDelay (k=2, a=0.05)", "hh", {"k": 2, "alpha": 0.05}),
        ("MultiDelay (k=3, a=0.05)", "hh", {"k": 3, "alpha": 0.05}),
        ("APPNP (tele=0.2)", "appnp", {"tele": 0.2}),
    ]

    for label, kind, cfg in variants:
        print(f"  [Diagnostic] Running {label}...")
        dirichlet_list = []
        mad_list = []
        acc_list = []

        if kind == "plain":
            P = make_gcn_operator(ei, N, device)
            h = x.clone()
            for m in range(1, max(steps) + 1):
                h = torch.sparse.mm(P, h)
                if m in steps:
                    de = compute_dirichlet_energy(h, ei)
                    mad = compute_mad(h)
                    # Linear probe on original nodes
                    h_np = h.cpu().numpy()
                    split_accs = []
                    for s in range(data.train_mask.size(1)):
                        tr = data.train_mask[:, s].cpu().numpy()
                        te = data.test_mask[:, s].cpu().numpy()
                        clf = RidgeClassifier(alpha=1.0)
                        clf.fit(h_np[tr], y[tr])
                        split_accs.append(float((clf.predict(h_np[te]) == y[te]).mean()))
                    dirichlet_list.append(de)
                    mad_list.append(mad)
                    acc_list.append(float(np.mean(split_accs)))

        elif kind == "lazy":
            P = make_lazy_operator(ei, N, cfg["beta"], device)
            h = x.clone()
            for m in range(1, max(steps) + 1):
                h = torch.sparse.mm(P, h)
                if m in steps:
                    de = compute_dirichlet_energy(h, ei)
                    mad = compute_mad(h)
                    h_np = h.cpu().numpy()
                    split_accs = []
                    for s in range(data.train_mask.size(1)):
                        tr = data.train_mask[:, s].cpu().numpy()
                        te = data.test_mask[:, s].cpu().numpy()
                        clf = RidgeClassifier(alpha=1.0)
                        clf.fit(h_np[tr], y[tr])
                        split_accs.append(float((clf.predict(h_np[te]) == y[te]).mean()))
                    dirichlet_list.append(de)
                    mad_list.append(mad)
                    acc_list.append(float(np.mean(split_accs)))

        elif kind == "hh":
            # Build dilated graph
            x_hh, ei_hh, _ = build_multidelay_graph(data, k=cfg["k"], alpha=cfg["alpha"], p=1.0)
            x_hh = x_hh.to(device)
            ei_hh = ei_hh.to(device)
            N_hh = x_hh.size(0)
            P_hh = make_gcn_operator(ei_hh, N_hh, device)
            h = x_hh.clone()
            for m in range(1, max(steps) + 1):
                h = torch.sparse.mm(P_hh, h)
                if m in steps:
                    # Metric evaluated on original nodes
                    h_orig = h[:N]
                    de = compute_dirichlet_energy(h_orig, ei)
                    mad = compute_mad(h_orig)
                    h_np = h_orig.cpu().numpy()
                    split_accs = []
                    for s in range(data.train_mask.size(1)):
                        tr = data.train_mask[:, s].cpu().numpy()
                        te = data.test_mask[:, s].cpu().numpy()
                        clf = RidgeClassifier(alpha=1.0)
                        clf.fit(h_np[tr], y[tr])
                        split_accs.append(float((clf.predict(h_np[te]) == y[te]).mean()))
                    dirichlet_list.append(de)
                    mad_list.append(mad)
                    acc_list.append(float(np.mean(split_accs)))

        elif kind == "appnp":
            # APPNP: h^(m) = (1 - alpha_tele) * P_gcn * h^(m-1) + alpha_tele * x_0
            P = make_gcn_operator(ei, N, device)
            tele = cfg["tele"]
            h = x.clone()
            for m in range(1, max(steps) + 1):
                h = (1.0 - tele) * torch.sparse.mm(P, h) + tele * x
                if m in steps:
                    de = compute_dirichlet_energy(h, ei)
                    mad = compute_mad(h)
                    h_np = h.cpu().numpy()
                    split_accs = []
                    for s in range(data.train_mask.size(1)):
                        tr = data.train_mask[:, s].cpu().numpy()
                        te = data.test_mask[:, s].cpu().numpy()
                        clf = RidgeClassifier(alpha=1.0)
                        clf.fit(h_np[tr], y[tr])
                        split_accs.append(float((clf.predict(h_np[te]) == y[te]).mean()))
                    dirichlet_list.append(de)
                    mad_list.append(mad)
                    acc_list.append(float(np.mean(split_accs)))

        results[label] = {
            "steps": list(steps),
            "dirichlet": dirichlet_list,
            "mad": mad_list,
            "acc": acc_list,
        }

    return results


# -----------------------------------------------------------------------------
# 6. Deep Supervised Models
# -----------------------------------------------------------------------------
class DeepGNN(nn.Module):
    def __init__(self, kind, din, hid, dout, depth, drop=0.5, beta=0.1, tele=0.2):
        super().__init__()
        self.kind = kind
        self.depth = depth
        self.drop = drop
        self.beta = beta

        dims = [din] + [hid] * (depth - 1) + [dout]
        if kind in ("plain", "halfhop", "gcn_res"):
            self.convs = nn.ModuleList([GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:])])
        elif kind == "lazywalk":
            self.convs = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        elif kind == "appnp":
            self.convs = nn.ModuleList([nn.Linear(din, hid), nn.Linear(hid, dout)])
            self.prop = APPNP(K=depth, alpha=tele)

    def forward(self, x, ei, P=None):
        if self.kind == "appnp":
            x = F.dropout(x, self.drop, self.training)
            x = F.relu(self.convs[0](x))
            x = F.dropout(x, self.drop, self.training)
            x = self.convs[1](x)
            return self.prop(x, ei), x

        h_last = x
        for i, c in enumerate(self.convs):
            x = F.dropout(x, self.drop, self.training)
            if self.kind == "lazywalk":
                x_next = torch.sparse.mm(P, c(x))
            elif self.kind == "gcn_res":
                x_next = c(x, ei)
                # Skip connection if dimensions match
                if x.size(-1) == x_next.size(-1):
                    x_next = x_next + x
            else:
                x_next = c(x, ei)

            if i < len(self.convs) - 1:
                x = F.relu(x_next)
                h_last = x
            else:
                x = x_next
        return x, h_last


def train_eval_split(model, x, ei, P, y, tr, va, te, N_orig, ei_orig, epochs, patience, lr, wd):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_va, best_te = -1.0, 0.0
    bad = 0
    best_weights = None

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out, _ = model(x, ei, P)
        loss = F.cross_entropy(out[:N_orig][tr], y[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out, _ = model(x, ei, P)
            pred = out[:N_orig].argmax(dim=1)
            acc_va = float((pred[va] == y[va]).float().mean().item())
            acc_te = float((pred[te] == y[te]).float().mean().item())

        if acc_va > best_va:
            best_va = acc_va
            best_te = acc_te
            bad = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        _, h_penult = model(x, ei, P)
        de = compute_dirichlet_energy(h_penult[:N_orig], ei_orig)
        mad = compute_mad(h_penult[:N_orig])

    return best_va, best_te, de, mad


# -----------------------------------------------------------------------------
# 7. Experiment Driver
# -----------------------------------------------------------------------------
def run_depth_study(dataset_name, depths=(2, 3, 4, 6, 8, 16), methods=None,
                    device="cuda:0", epochs=200, patience=50, out_dir="experiments/results/depth"):
    os.makedirs(out_dir, exist_ok=True)
    data, nc = load_data(dataset_name)
    N_orig = data.num_nodes
    y = data.y.to(device)

    # Defaults per dataset from Step 2
    if dataset_name.lower() == "wisconsin":
        cfg = {"lr": 0.01, "hidden": 32, "dropout": 0.25, "wd": 1e-4, "beta": 0.1033, "alpha": 0.25}
    elif dataset_name.lower() == "chameleon":
        cfg = {"lr": 0.01, "hidden": 128, "dropout": 0.30, "wd": 1e-4, "beta": 0.9854, "alpha": 0.025}
    else:
        cfg = {"lr": 0.01, "hidden": 64, "dropout": 0.30, "wd": 1e-4, "beta": 0.2, "alpha": 0.1}

    if methods is None:
        methods = ["plain", "lazywalk", "halfhop_k1", "halfhop_k2", "halfhop_k3", "appnp", "gcn_res"]

    print(f"\n==================================================================")
    print(f"Step 3: Depth-Matched Dilation Study on {dataset_name.upper()}")
    print(f"Depths: {depths} | Methods: {methods}")
    print(f"Device: {device} | Base Config: {cfg}")
    print(f"==================================================================\n")

    # 1. Weightless Diagnostic
    print("--- 1. Weightless / Linear Propagation Diagnostic ---")
    diag_results = run_weightless_diagnostic(data, device=device)

    # 2. Supervised Depth Sweep
    print("\n--- 2. Supervised Depth Scaling Sweep ---")
    supervised_results = {}

    for method in methods:
        supervised_results[method] = {
            "depths": list(depths),
            "effective_hops": [],
            "val_mean": [],
            "test_mean": [],
            "test_std": [],
            "dirichlet_mean": [],
            "mad_mean": [],
        }

        # Prepare graph data
        if method == "plain":
            x, ei, P = data.x.to(device), data.edge_index.to(device), None
            kind = "plain"
            eff_fn = lambda L: L
        elif method == "lazywalk":
            x, ei = data.x.to(device), data.edge_index.to(device)
            P = make_lazy_operator(ei, N_orig, cfg["beta"], device)
            kind = "lazywalk"
            eff_fn = lambda L: L
        elif method == "gcn_res":
            x, ei, P = data.x.to(device), data.edge_index.to(device), None
            kind = "gcn_res"
            eff_fn = lambda L: L
        elif method == "appnp":
            x, ei, P = data.x.to(device), data.edge_index.to(device), None
            kind = "appnp"
            eff_fn = lambda L: L
        elif method == "halfhop_k1":
            x, ei, _ = build_multidelay_graph(data, k=1, alpha=cfg["alpha"], p=1.0)
            x, ei, P = x.to(device), ei.to(device), None
            kind = "halfhop"
            eff_fn = lambda L: L / 2.0
        elif method == "halfhop_k2":
            x, ei, _ = build_multidelay_graph(data, k=2, alpha=cfg["alpha"], p=1.0)
            x, ei, P = x.to(device), ei.to(device), None
            kind = "halfhop"
            eff_fn = lambda L: L / 3.0
        elif method == "halfhop_k3":
            x, ei, _ = build_multidelay_graph(data, k=3, alpha=cfg["alpha"], p=1.0)
            x, ei, P = x.to(device), ei.to(device), None
            kind = "halfhop"
            eff_fn = lambda L: L / 4.0
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"\nEvaluating Method: {method.upper()} (Nodes: {x.size(0)}, Edges: {ei.size(1)})")
        for depth in depths:
            # MultiDelay k=2 requires depth >= 3; k=3 requires depth >= 4
            if method == "halfhop_k2" and depth < 3:
                continue
            if method == "halfhop_k3" and depth < 4:
                continue

            eff_hop = eff_fn(depth)
            val_scores, test_scores, de_scores, mad_scores = [], [], [], []

            t0 = time.time()
            for s in range(data.train_mask.size(1)):
                torch.manual_seed(s)
                tr = data.train_mask[:, s].to(device)
                va = data.val_mask[:, s].to(device)
                te = data.test_mask[:, s].to(device)

                model = DeepGNN(
                    kind=kind,
                    din=x.size(1),
                    hid=cfg["hidden"],
                    dout=nc,
                    depth=depth,
                    drop=cfg["dropout"],
                    beta=cfg["beta"],
                    tele=0.2,
                ).to(device)

                v_acc, t_acc, de, mad = train_eval_split(
                    model=model,
                    x=x, ei=ei, P=P, y=y,
                    tr=tr, va=va, te=te,
                    N_orig=N_orig,
                    ei_orig=data.edge_index.to(device),
                    epochs=epochs,
                    patience=patience,
                    lr=cfg["lr"],
                    wd=cfg["wd"],
                )
                val_scores.append(v_acc)
                test_scores.append(t_acc)
                de_scores.append(de)
                mad_scores.append(mad)

            dt = time.time() - t0
            m_v = float(np.mean(val_scores))
            m_t = float(np.mean(test_scores))
            s_t = float(np.std(test_scores))
            m_de = float(np.mean(de_scores))
            m_mad = float(np.mean(mad_scores))

            supervised_results[method]["effective_hops"].append(eff_hop)
            supervised_results[method]["val_mean"].append(m_v)
            supervised_results[method]["test_mean"].append(m_t)
            supervised_results[method]["test_std"].append(s_t)
            supervised_results[method]["dirichlet_mean"].append(m_de)
            supervised_results[method]["mad_mean"].append(m_mad)

            print(f"  [L={depth:2d} (eff={eff_hop:4.1f})] Val: {m_v*100:5.2f}% | "
                  f"Test: {m_t*100:5.2f} ± {s_t*100:4.2f}% | "
                  f"Dirichlet: {m_de:7.4f} | MAD: {m_mad:6.4f} ({dt:.1f}s)")

    # 3. Save Summary JSON
    full_output = {
        "dataset": dataset_name,
        "config": cfg,
        "diagnostic": diag_results,
        "supervised": supervised_results,
    }
    json_path = os.path.join(out_dir, f"depth_study_{dataset_name.lower()}.json")
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[Saved JSON] -> {json_path}")

    # 4. Generate Visualization Plots
    plot_depth_study(full_output, out_dir)
    return full_output


# -----------------------------------------------------------------------------
# 8. Publication Plotting Functions
# -----------------------------------------------------------------------------
def plot_depth_study(results, out_dir):
    dataset = results["dataset"].capitalize()
    diag = results["diagnostic"]
    sup = results["supervised"]

    # Plot 1: Weightless Diagnostic (Accuracy & Dirichlet vs Step)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, metrics in diag.items():
        steps = metrics["steps"]
        ax1.plot(steps, [a * 100 for a in metrics["acc"]], marker="o", label=label)
        ax2.plot(steps, metrics["dirichlet"], marker="s", label=label)

    ax1.set_title(f"{dataset}: Linear Weightless Accuracy vs. Steps")
    ax1.set_xlabel("Propagation Steps (m)")
    ax1.set_ylabel("Linear Probe Test Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title(f"{dataset}: Dirichlet Energy vs. Steps")
    ax2.set_xlabel("Propagation Steps (m)")
    ax2.set_ylabel("Normalized Dirichlet Energy")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    diag_path = os.path.join(out_dir, f"weightless_diagnostic_{dataset.lower()}.png")
    fig.savefig(diag_path, dpi=200)
    plt.close(fig)
    print(f"[Saved Plot] -> {diag_path}")

    # Plot 2: Supervised Depth vs. Test Accuracy & Dirichlet Energy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "plain": "#1f77b4", "lazywalk": "#2ca02c", "halfhop_k1": "#d62728",
        "halfhop_k2": "#ff7f0e", "halfhop_k3": "#9467bd", "appnp": "#8c564b", "gcn_res": "#17becf"
    }

    for method, metrics in sup.items():
        d_vals = metrics["depths"]
        t_means = [m * 100 for m in metrics["test_mean"]]
        t_stds = [s * 100 for s in metrics["test_std"]]
        de_means = metrics["dirichlet_mean"]

        if len(t_means) == 0:
            continue
        # Slice d_vals if method skipped depths
        d_active = d_vals[-len(t_means):]
        c = colors.get(method, None)
        ax1.errorbar(d_active, t_means, yerr=t_stds, marker="o", capsize=3, label=method, color=c)
        ax2.plot(d_active, de_means, marker="s", label=method, color=c)

    ax1.set_title(f"{dataset}: Supervised Test Accuracy vs. Depth")
    ax1.set_xlabel("Network Depth (L)")
    ax1.set_ylabel("10-Split Test Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title(f"{dataset}: Representation Dirichlet Energy vs. Depth")
    ax2.set_xlabel("Network Depth (L)")
    ax2.set_ylabel("Final Layer Dirichlet Energy")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    sup_path = os.path.join(out_dir, f"depth_scaling_{dataset.lower()}.png")
    fig.savefig(sup_path, dpi=200)
    plt.close(fig)
    print(f"[Saved Plot] -> {sup_path}")


# -----------------------------------------------------------------------------
# 9. CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 3: Depth-Matched Dilation Study")
    parser.add_argument("--dataset", required=True, choices=["wisconsin", "chameleon", "squirrel", "texas", "cornell"])
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 6, 8, 16])
    parser.add_argument("--methods", nargs="+", default=["plain", "lazywalk", "halfhop_k1", "halfhop_k2", "halfhop_k3", "appnp", "gcn_res"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="experiments/results/depth")
    args = parser.parse_args()

    run_depth_study(
        dataset_name=args.dataset,
        depths=args.depths,
        methods=args.methods,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
