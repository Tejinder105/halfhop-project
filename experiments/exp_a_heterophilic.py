#!/usr/bin/env python
"""Option A: 'Where should message passing slow down?' on heterophilic node classification.

Arms (all 2-layer, same optimiser / epochs; model + config chosen on VALIDATION accuracy later):
  gcn            plain GCN
  hh_all         Half-Hop on every edge                          (the paper's method)
  hh_<selector>  Half-Hop on a budgeted subset chosen by:
      random         uniform random                               (control)
      feat           1 - cosine(x_i, x_j)                         (label-free)
      mlp            1 - <p_i, p_j>, p from an MLP trained on TRAIN labels only  (label-light)
      oracle_train   true label disagreement, only where BOTH ends are train nodes (else 0.5)
      oracle_full    true disagreement on ALL nodes -> uses test labels, UPPER BOUND ONLY
      inverted       1 - oracle_full (slows homophilic edges)     (should hurt)
  gcn_selfloop   GCN with self-loop weight w                      (ego-preservation control)
  appnp          APPNP (initial residual / teleport)              (ego-preservation control)

Usage:
  python -m experiments.exp_a_heterophilic --datasets synthetic --quick          # offline smoke test
  python -m experiments.exp_a_heterophilic --datasets texas,cornell,wisconsin,chameleon --root data
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from halfhop.selective_halfhop import half_hop, select_edges, norm_adj


# ----------------------------------------------------------------------------- data
def make_synthetic(seed=0, n=300, c=5, d=50, p_hetero=0.8, deg=4, n_splits=10):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    means = rng.normal(size=(c, d))
    x = np.clip(means[y] + 2.0 * rng.normal(size=(n, d)), 0, None)
    src, dst = [], []
    for _ in range(n * deg // 2):
        u = int(rng.integers(0, n))
        same = rng.random() >= p_hetero
        cand = np.where((y == y[u]) if same else (y != y[u]))[0]
        cand = cand[cand != u]
        v = int(rng.choice(cand))
        src += [u, v]; dst += [v, u]
    tr = np.zeros((n, n_splits), bool); va = tr.copy(); te = tr.copy()
    for s in range(n_splits):
        p = rng.permutation(n)
        tr[p[: int(.6 * n)], s] = True
        va[p[int(.6 * n): int(.8 * n)], s] = True
        te[p[int(.8 * n):], s] = True
    return dict(x=torch.tensor(x, dtype=torch.float32), y=torch.tensor(y),
                edge_index=torch.tensor([src, dst]), train=torch.tensor(tr),
                val=torch.tensor(va), test=torch.tensor(te))


def load_dataset(name, root):
    name = name.lower()
    if name == "synthetic":
        return make_synthetic()
    from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor
    if name in ("texas", "cornell", "wisconsin"):
        ds = WebKB(root, name=name.capitalize())
    elif name in ("chameleon", "squirrel"):
        ds = WikipediaNetwork(root, name=name, geom_gcn_preprocess=True)
    elif name == "actor":
        ds = Actor(os.path.join(root, "actor"))
    else:
        raise ValueError(name)
    d = ds[0]
    ei = torch.cat([d.edge_index, d.edge_index.flip(0)], 1)       # symmetrise
    ei = ei[:, ei[0] != ei[1]]                                       # drop self-loops
    ei = torch.unique(ei, dim=1)
    g = lambda m: m if m.dim() == 2 else m.unsqueeze(1)
    return dict(x=d.x.float(), y=d.y, edge_index=ei,
                train=g(d.train_mask), val=g(d.val_mask), test=g(d.test_mask))


# ----------------------------------------------------------------------------- models
class GCN(nn.Module):
    def __init__(self, i, h, o, layers=2, dropout=0.5):
        super().__init__()
        dims = [i] + [h] * (layers - 1) + [o]
        self.lins = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.bias = nn.ParameterList(nn.Parameter(torch.zeros(b)) for b in dims[1:])
        self.p = dropout

    def forward(self, x, A):
        for l, (lin, b) in enumerate(zip(self.lins, self.bias)):
            x = F.dropout(x, self.p, self.training)
            x = torch.sparse.mm(A, lin(x)) + b
            if l < len(self.lins) - 1:
                x = F.relu(x)
        return x


class APPNPNet(nn.Module):
    def __init__(self, i, h, o, K=10, teleport=0.1, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.K, self.a, self.p = nn.Linear(i, h), nn.Linear(h, o), K, teleport, dropout

    def forward(self, x, A):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.l1(x))
        x = F.dropout(x, self.p, self.training)
        h0 = self.l2(x)
        h = h0
        for _ in range(self.K):
            h = (1 - self.a) * torch.sparse.mm(A, h) + self.a * h0
        return h


# ----------------------------------------------------------------------------- selectors
def mlp_probs(x, y, train, seed, hid=64, epochs=100):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(x.size(1), hid), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hid, int(y.max()) + 1))
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(x[train]), y[train]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return F.softmax(m(x), -1)


def edge_scores(sel, x, ei, y, train, probs):
    src, dst = ei
    if sel == "random":
        return torch.zeros(ei.size(1))                      # random tie-breaking does the work
    if sel == "feat":
        xn = F.normalize(x, dim=1)
        return 1 - (xn[src] * xn[dst]).sum(1)
    if sel == "mlp":
        return 1 - (probs[src] * probs[dst]).sum(1)
    dis = (y[src] != y[dst]).float()
    if sel == "oracle_full":
        return dis
    if sel == "inverted":
        return 1 - dis
    if sel == "oracle_train":
        known = train[src] & train[dst]
        return torch.where(known, dis, torch.full_like(dis, 0.5))
    raise ValueError(sel)


# ----------------------------------------------------------------------------- one run
def build_graph(cfg, D, split, seed, probs, device):
    x, ei, y = D["x"], D["edge_index"], D["y"]
    n = x.size(0)
    arm = cfg["arm"]
    xd, eid = x.to(device), ei.to(device)
    frac, n_slow = 0.0, 0
    if arm in ("gcn", "appnp"):
        A = norm_adj(eid, n, 1.0)
    elif arm == "gcn_selfloop":
        A = norm_adj(eid, n, cfg["sl_w"])
    else:
        sel = arm[3:]
        if sel == "all":
            mask = torch.ones(ei.size(1), dtype=torch.bool)
        else:
            gen = torch.Generator().manual_seed(seed * 100003 + split * 17 + 1)
            sc = edge_scores(sel, x, ei, y, D["train"][:, split], probs)
            mask = select_edges(ei, n, sc, cfg["budget"], cfg["level"], gen)
        xd, eid2, slow = half_hop(xd, eid, mask.to(device), cfg["alpha"])
        A = norm_adj(eid2, xd.size(0), 1.0)
        frac, n_slow = float(mask.float().mean()), int(slow.sum())
    return xd, A, frac, n_slow


def run_one(cfg, D, split, seed, probs, args, device):
    n, c, d = D["x"].size(0), int(D["y"].max()) + 1, D["x"].size(1)
    torch.manual_seed(seed * 1000 + split)
    xd, A, frac, n_slow = build_graph(cfg, D, split, seed, probs, device)
    y = D["y"].to(device)
    tr, va, te = (D[k][:, split].to(device) for k in ("train", "val", "test"))
    if cfg["arm"] == "appnp":
        model = APPNPNet(d, args.hidden, c, teleport=cfg["appnp_alpha"], dropout=args.dropout)
    else:
        model = GCN(d, args.hidden, c, args.layers, args.dropout)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best = (-1.0, 1e9, 0.0)
    for _ in range(args.epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(xd, A)[:n][tr], y[tr]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(xd, A)[:n]
            vl = F.cross_entropy(out[va], y[va]).item()
            pred = out.argmax(1)
            v = (pred[va] == y[va]).float().mean().item()
            t = (pred[te] == y[te]).float().mean().item()
        if v > best[0] or (v == best[0] and vl < best[1]):
            best = (v, vl, t)
    return dict(val_acc=best[0], test_acc=best[2], frac_edges_slowed=frac, n_slow=n_slow)


# ----------------------------------------------------------------------------- configs
def build_configs(a):
    cf = [dict(arm="gcn")]
    cf += [dict(arm="hh_all", alpha=al, budget=1.0, level="edge") for al in a.alphas]
    for sel in a.selectors:
        for lv in a.levels:
            for b in a.budgets:
                for al in a.alphas:
                    cf.append(dict(arm=f"hh_{sel}", level=lv, budget=b, alpha=al))
    cf += [dict(arm="gcn_selfloop", sl_w=w) for w in a.selfloop_ws]
    cf += [dict(arm="appnp", appnp_alpha=t) for t in a.appnp_alphas]
    for c in cf:
        for k in ("level", "budget", "alpha", "sl_w", "appnp_alpha"):
            c.setdefault(k, None)
    return cf


def main():
    ap = argparse.ArgumentParser()
    f = lambda s: [float(v) for v in s.split(",")]
    ap.add_argument("--datasets", default="texas,cornell,wisconsin,chameleon")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/a/runs.jsonl")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alphas", type=f, default=[0.1, 0.5])
    ap.add_argument("--budgets", type=f, default=[0.25, 0.5])
    ap.add_argument("--levels", default="edge")
    ap.add_argument("--selectors", default="random,feat,mlp,oracle_train,oracle_full,inverted")
    ap.add_argument("--selfloop_ws", type=f, default=[2, 4, 8, 16])
    ap.add_argument("--appnp_alphas", type=f, default=[0.1, 0.5])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.levels, a.selectors = a.levels.split(","), a.selectors.split(",")
    if a.quick:
        a.splits, a.seeds, a.alphas, a.budgets, a.epochs = 2, 1, [0.1], [0.5], 100
        a.selfloop_ws, a.appnp_alphas = [4.0], [0.1]
    device = torch.device(a.device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add(json.dumps([r[k] for k in ("dataset", "split", "seed", "arm", "level", "budget", "alpha", "sl_w", "appnp_alpha")]))
    cfgs = build_configs(a)
    need_mlp = any(c["arm"] == "hh_mlp" for c in cfgs)
    fh = open(a.out, "a")
    t0 = time.time(); count = 0
    for ds in a.datasets.split(","):
        D = load_dataset(ds, a.root)
        D["x"] = D["x"] / D["x"].sum(1, keepdim=True).clamp(min=1e-12)   # row-normalise features
        for sp in range(min(a.splits, D["train"].size(1))):
            probs = mlp_probs(D["x"], D["y"], D["train"][:, sp], sp) if need_mlp else None
            for sd in range(a.seeds):
                for c in cfgs:
                    key = json.dumps([ds, sp, sd, c["arm"], c["level"], c["budget"], c["alpha"], c["sl_w"], c["appnp_alpha"]])
                    if key in done:
                        continue
                    r = run_one(c, D, sp, sd, probs, a, device)
                    rec = dict(dataset=ds, split=sp, seed=sd, **c, **r)
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); count += 1
            print(f"[{ds}] split {sp} done | {count} runs | {time.time() - t0:.0f}s", flush=True)
    fh.close()


if __name__ == "__main__":
    main()

