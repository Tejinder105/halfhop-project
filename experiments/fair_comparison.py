"""experiments/fair_comparison.py  --  Step 2: equal-budget fair comparison.

Protocol
  * Every method gets the SAME number of random-search trials (default 100).
  * Shared hyperparameters (lr, hidden, dropout, wd, layers) come from one seeded RNG,
    so trial i has identical shared values for every method. Method-specific knobs
    (Half-Hop alpha/p, lazy-walk beta, APPNP K/teleport) come from a separate RNG.
  * alpha / beta are searched over the FULL [0,1] range, so the alpha-convention
    question (0.9937 vs 0.0063) cannot bias the result.
  * Model selection = mean VALIDATION accuracy across splits. Test acc is only read
    for the winning trial (all trials are logged, so nothing is hidden).
  * If search_splits < 10 (e.g. 3 splits used during search for speed), the winning
    configuration for each method is automatically re-evaluated across all 10 splits.
  * Output: per-trial JSONL (resumable), best-val-vs-trials curves, paired Wilcoxon.

Usage
  python -m experiments.fair_comparison --dataset wisconsin --backbone gcn --trials 100
  python -m experiments.fair_comparison --dataset chameleon --backbone gcn --trials 100 --search_splits 3
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import zlib

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import WebKB, WikipediaNetwork
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

# ----------------------------------------------------------------------------- data
def load_dataset(name, root="data"):
    if name == "wisconsin":
        ds = WebKB(root, "Wisconsin")
    elif name == "chameleon":
        ds = WikipediaNetwork(root, "chameleon", geom_gcn_preprocess=True)
    else:
        raise ValueError(name)
    d = ds[0]
    return d, ds.num_classes

# ----------------------------------------------------------------------------- graph variants
def lazy_walk_matrix(ei, n, beta, device):
    """P = (1-beta) I + beta D^-1 A  (row-stochastic), as a sparse tensor."""
    ei, _ = remove_self_loops(ei)
    deg = torch.zeros(n, device=device).scatter_add_(0, ei[0], torch.ones(ei.size(1), device=device))
    w = beta / deg.clamp(min=1)[ei[0]]
    idx = torch.arange(n, device=device)
    rows = torch.cat([ei[0], idx]); cols = torch.cat([ei[1], idx])
    vals = torch.cat([w, torch.full((n,), 1 - beta, device=device)])
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()

def make_graph(data, method, hp, device, seed):
    """Returns (x, edge_index, P). Original nodes are always rows [0:N]."""
    x, ei, P = data.x.to(device), data.edge_index.to(device), None
    if method == "halfhop":
        torch.manual_seed(seed)
        g = T.HalfHop(alpha=hp["alpha"], p=hp["p"])(Data(x=data.x.cpu(), edge_index=data.edge_index.cpu()))
        assert g.x.size(0) >= data.num_nodes and torch.equal(g.x[:data.num_nodes], data.x.cpu()), \
            "HalfHop must keep original nodes first; adapt make_graph to your implementation"
        x, ei = g.x.to(device), g.edge_index.to(device)
    elif method == "lazywalk":
        P = lazy_walk_matrix(ei, data.num_nodes, hp["beta"], device)
    return x, ei, P

# ----------------------------------------------------------------------------- models
class Net(nn.Module):
    def __init__(self, kind, din, hid, dout, layers, drop, hp):
        super().__init__()
        self.kind, self.drop = kind, drop
        dims = [din] + [hid] * (layers - 1) + [dout]
        if kind in ("mlp", "lazywalk"):
            self.convs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        elif kind == "gcn":
            self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        elif kind == "sage":
            self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        elif kind == "gat":
            heads, cs = 4, []
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
                last = i == layers - 1
                cs.append(GATConv(a, b if last else b // heads,
                                  heads=1 if last else heads, dropout=drop))
            self.convs = nn.ModuleList(cs)  # hidden width = heads*(hid//heads) = hid
        elif kind == "appnp":
            self.convs = nn.ModuleList([nn.Linear(din, hid), nn.Linear(hid, dout)])
            self.prop = APPNP(K=hp["K"], alpha=hp["tele"])

    def forward(self, x, ei, P=None):
        for i, c in enumerate(self.convs):
            x = F.dropout(x, self.drop, self.training)
            if self.kind == "mlp" or self.kind == "appnp":
                x = c(x)
            elif self.kind == "lazywalk":
                x = torch.sparse.mm(P, c(x))
            else:
                x = c(x, ei)
            if i < len(self.convs) - 1:
                x = F.elu(x) if self.kind == "gat" else F.relu(x)
        return self.prop(x, ei) if self.kind == "appnp" else x

# ----------------------------------------------------------------------------- search space
SHARED_KEYS = ("lr", "hidden", "dropout", "wd", "layers")

def sample_shared(rng):
    return dict(lr=float(10 ** rng.uniform(-3, -1.3)), hidden=int(rng.choice([32, 64, 128, 256])),
                dropout=float(rng.uniform(0, 0.8)), wd=float(10 ** rng.uniform(-6, -2)),
                layers=int(rng.choice([2, 3])))

def sample_specific(method, rng):
    if method == "halfhop":
        return dict(alpha=float(rng.uniform(0, 1)), p=float(rng.uniform(0.1, 1.0)))
    if method == "lazywalk":
        return dict(beta=float(rng.uniform(0, 1)))
    if method == "appnp":
        return dict(K=int(rng.choice([2, 5, 10])), tele=float(rng.uniform(0.05, 0.5)))
    return {}

# ----------------------------------------------------------------------------- training
def run_split(model_kind, x, ei, P, data, nc, split, hp, device, epochs, patience, seed):
    torch.manual_seed(seed + split)
    y = data.y.to(device); N = data.num_nodes
    tr, va, te = (m[:, split].to(device) for m in (data.train_mask, data.val_mask, data.test_mask))
    model = Net(model_kind, x.size(1), hp["hidden"], nc, hp["layers"], hp["dropout"], hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best_va, best_te, bad = -1, 0, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(x, ei, P)[:N]
        F.cross_entropy(out[tr], y[tr]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei, P)[:N].argmax(1)
            a_va = (pred[va] == y[va]).float().mean().item()
            a_te = (pred[te] == y[te]).float().mean().item()
        if a_va > best_va:
            best_va, best_te, bad = a_va, a_te, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_va, best_te

def run_trial(method, backbone, data, nc, hp, device, n_splits, epochs, patience, seed):
    kind = {"plain": backbone, "halfhop": backbone, "lazywalk": "lazywalk",
            "mlp": "mlp", "appnp": "appnp"}[method]
    x, ei, P = make_graph(data, method, hp, device, seed)
    vals, tests = [], []
    for s in range(n_splits):
        v, t = run_split(kind, x, ei, P, data, nc, s, hp, device, epochs, patience, seed)
        vals.append(v); tests.append(t)
    return vals, tests

# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["wisconsin", "chameleon"])
    ap.add_argument("--backbone", default="gcn", choices=["gcn", "sage", "gat"])
    ap.add_argument("--methods", nargs="+", default=["plain", "halfhop", "lazywalk", "mlp", "appnp"])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--search_splits", type=int, default=10, help="splits used per trial (lower = faster)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/fair")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    data, nc = load_dataset(a.dataset)
    tag = f"{a.dataset}_{a.backbone}"
    log_path = os.path.join(a.out, f"{tag}_trials.jsonl")

    done = {}
    if os.path.exists(log_path):                       # resume after Kaggle timeouts
        for line in open(log_path):
            r = json.loads(line); done[(r["method"], r["trial"])] = r

    shared_rng = np.random.RandomState(a.seed)          # identical shared draws for all methods
    shared = [sample_shared(shared_rng) for _ in range(a.trials)]
    t0 = time.time()
    with open(log_path, "a") as log:
        for m in a.methods:
            spec_rng = np.random.RandomState(a.seed + 1000 + zlib.crc32(m.encode()) % 997)
            for i in range(a.trials):
                hp = {**shared[i], **sample_specific(m, spec_rng)}   # always draw -> deterministic
                if (m, i) in done:
                    continue
                vals, tests = run_trial(m, a.backbone, data, nc, hp, a.device,
                                        a.search_splits, a.epochs, a.patience, a.seed + i)
                rec = dict(method=m, trial=i, hp=hp, val=vals, test=tests,
                           mean_val=float(np.mean(vals)))
                log.write(json.dumps(rec) + "\n"); log.flush(); done[(m, i)] = rec
                print(f"[{tag}] {m:9s} trial {i+1:3d}/{a.trials} val={rec['mean_val']*100:.2f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- summarise: best trial by mean validation, curves, paired tests
    total_splits = data.train_mask.size(1) if data.train_mask.dim() == 2 else 10
    summary, curves = {}, {}
    for m in a.methods:
        recs = [done[(m, i)] for i in range(a.trials)]
        best = max(recs, key=lambda r: r["mean_val"])
        curves[m] = np.maximum.accumulate([r["mean_val"] for r in recs]).tolist()

        # If search used fewer splits than total, re-evaluate the winning trial on ALL 10 splits
        if a.search_splits < total_splits:
            print(f"\nRe-evaluating best {m} (trial {best['trial']}) across all {total_splits} splits...")
            full_vals, full_tests = run_trial(
                m, a.backbone, data, nc, best["hp"], a.device,
                total_splits, a.epochs, a.patience, a.seed + best["trial"]
            )
            test_scores = full_tests
            mean_val = float(np.mean(full_vals))
        else:
            test_scores = best["test"]
            mean_val = best["mean_val"]

        summary[m] = dict(
            best_trial=best["trial"],
            hp=best["hp"],
            val=mean_val,
            test_mean=float(np.mean(test_scores)),
            test_std=float(np.std(test_scores)),
            test_per_split=test_scores,
        )

    from scipy.stats import wilcoxon
    tests = {}
    pairs = [("halfhop", "lazywalk"), ("halfhop", "plain"), ("lazywalk", "plain")]
    for p, q in pairs:
        if p in summary and q in summary:
            d = np.array(summary[p]["test_per_split"]) - np.array(summary[q]["test_per_split"])
            try:
                pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                pv = 1.0
            tests[f"{p}_vs_{q}"] = dict(mean_delta=float(d.mean()), p=pv)

    summary_file = os.path.join(a.out, f"{tag}_summary.json")
    json.dump(dict(summary=summary, curves=curves, paired_wilcoxon=tests, args=vars(a)),
              open(summary_file, "w"), indent=2)

    print(f"\n=== {tag}: best-by-val, equal budget = {a.trials} trials ===")
    for m, s in summary.items():
        print(f"  {m:9s} val {s['val']*100:5.2f} | test {s['test_mean']*100:5.2f} ± {s['test_std']*100:4.2f}  {s['hp']}")
    for k, v in tests.items():
        print(f"  {k}: delta {v['mean_delta']*100:+.2f}%  p={v['p']:.4f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for m, c in curves.items():
            plt.plot(range(1, len(c) + 1), np.array(c) * 100, label=m)
        plt.xlabel("Search Trials")
        plt.ylabel("Best Mean Val Acc (%)")
        plt.legend()
        plt.title(f"{tag.upper()}: Best Validation Accuracy vs Search Budget")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = os.path.join(a.out, f"{tag}_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Curves plot saved to {plot_path}")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()

