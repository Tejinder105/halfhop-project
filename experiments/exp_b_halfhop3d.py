#!/usr/bin/env python
"""Option B: Half-Hop-3D on ShapeNet-Part segmentation (per-category, per-point).

Pipeline
  1. Sample N points / shape, build a static kNN graph (k) on xyz, estimate normals + curvature
     from local covariance (no dataset normals needed).
  2. Choose WHICH kNN edges to subdivide ("selector") at NODE level (paper's scheme: top-p boundary
     points, all incoming edges) or EDGE level, with a matched budget p per shape.
  3. Slow nodes are real points: p_k = alpha*p_i + (1-alpha)*p_j, and ALL input features are
     interpolated the same way (small alpha => sits near the target, stays near the surface).
  4. Backbone: GCN, or EdgeConv (static-graph DGCNN-style, max aggregation, relative features),
     optionally with dense skip connections ("strong") or none ("plain" -> oversmoothing test).
  5. Metrics per TEST SHAPE are saved (mIoU, acc, boundary acc, interior acc, boundary mIoU) so
     analyze_b.py can do paired per-shape bootstrap CIs.

Selectors (arm = <selector>_<level>, level in {node, edge}; plus 'none' and 'all'):
  rand     random (control)           geom   1-|n_i.n_j| normal-angle (Stage-0 best feature)
  curv     |curvature_i - curvature_j|   edgelen  ||p_i-p_j|| (cheap control)
  oracle   GT label disagreement (UPPER BOUND, uses test labels)   inv   homophilic edges/nodes (should hurt)

Smoke test with no data:   python -m experiments.exp_b_halfhop3d --synthetic --epochs 3 --seeds 1
Real:                      python -m experiments.exp_b_halfhop3d --root data/ShapeNet --categories Airplane,Table
"""
import argparse, hashlib, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from halfhop.selective_halfhop import half_hop, norm_adj


# ----------------------------------------------------------------------------- data
def _normalise(pos):
    pos = pos - pos.mean(0, keepdim=True)
    return pos / pos.norm(dim=1).max().clamp(min=1e-8)


def load_shapenet(root, category, split, num_points, max_shapes, seed=1234):
    from torch_geometric.datasets import ShapeNet
    ds = ShapeNet(root, categories=[category], split=split)
    part = ShapeNet.seg_classes[category]
    ymap = torch.full((50,), -1, dtype=torch.long)
    ymap[torch.tensor(part)] = torch.arange(len(part))
    g = torch.Generator().manual_seed(seed)
    P, Y = [], []
    for i in range(min(len(ds), max_shapes)):
        d = ds[i]
        n = d.pos.size(0)
        idx = torch.randperm(n, generator=g)[:num_points] if n >= num_points else torch.randint(n, (num_points,), generator=g)
        P.append(_normalise(d.pos[idx])); Y.append(ymap[d.y[idx]])
    return torch.stack(P), torch.stack(Y), len(part)


def make_synthetic(n_shapes, num_points, seed):
    """3 parts: sphere / disk / cylinder with random size and offset (offline smoke test only)."""
    rng = np.random.default_rng(seed)
    P, Y = [], []
    for _ in range(n_shapes):
        frac = rng.dirichlet([4, 3, 3]); cnt = np.maximum((frac * num_points).astype(int), 1)
        cnt[0] = num_points - cnt[1:].sum()
        v = rng.normal(size=(cnt[0], 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        sph = v * rng.uniform(.3, .6) + np.array([0, 0, rng.uniform(0, .4)])
        r = np.sqrt(rng.random(cnt[1])) * rng.uniform(.8, 1.2); th = rng.random(cnt[1]) * 6.2832
        disk = np.stack([r * np.cos(th), r * np.sin(th), np.full(cnt[1], -.5)], 1)
        th = rng.random(cnt[2]) * 6.2832; rad = rng.uniform(.1, .2)
        cyl = np.stack([rad * np.cos(th), rad * np.sin(th), rng.uniform(-.5, 0, cnt[2])], 1)
        pts = np.concatenate([sph, disk, cyl]); lab = np.concatenate([np.zeros(cnt[0]), np.ones(cnt[1]), 2 * np.ones(cnt[2])])
        pts = pts + rng.normal(0, .005, pts.shape)
        P.append(_normalise(torch.tensor(pts, dtype=torch.float32))); Y.append(torch.tensor(lab, dtype=torch.long))
    return torch.stack(P), torch.stack(Y), 3


@torch.no_grad()
def prep_geometry(pos, k, device, chunk=32):
    """kNN indices (excluding self), PCA normals (unoriented), curvature = l0 / (l0+l1+l2)."""
    S, N, _ = pos.shape
    idx_all = torch.empty(S, N, k, dtype=torch.int16 if N < 32000 else torch.int32)
    nrm_all, curv_all = torch.empty(S, N, 3), torch.empty(S, N)
    for s in range(0, S, chunk):
        p = pos[s:s + chunk].to(device)
        B = p.size(0)
        d = torch.cdist(p, p)
        d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        idx = d.topk(k, dim=-1, largest=False).indices
        nb = p[torch.arange(B, device=device).view(B, 1, 1), idx]                  # B,N,k,3
        nb = torch.cat([p.unsqueeze(2), nb], 2)
        nb = nb - nb.mean(2, keepdim=True)
        cov = nb.transpose(-1, -2) @ nb / nb.size(2)
        ev, evec = torch.linalg.eigh(cov)                                          # ascending
        idx_all[s:s + B] = idx.to(idx_all.dtype).cpu()
        nrm_all[s:s + B] = evec[..., :, 0].cpu()
        curv_all[s:s + B] = (ev[..., 0] / ev.sum(-1).clamp(min=1e-12)).cpu()
    return idx_all, nrm_all, curv_all


def make_split(P, Y, n_cls, k, device):
    idx, nrm, curv = prep_geometry(P, k, device)
    return dict(pos=P, y=Y, idx=idx, nrm=nrm, curv=curv, n_cls=n_cls)


def get_batch(D, ids, device):
    b = {k: D[k][ids].to(device) for k in ("pos", "y", "idx", "nrm", "curv")}
    b["idx"] = b["idx"].long()
    return b


def gather_nb(t, idx):
    B = idx.size(0)
    return t[torch.arange(B, device=idx.device).view(B, 1, 1), idx]


# ----------------------------------------------------------------------------- selection
def edge_score(kind, b):
    idx = b["idx"]
    if kind == "rand":
        return torch.zeros(idx.shape, device=idx.device)
    if kind == "geom":
        return 1 - (b["nrm"].unsqueeze(2) * gather_nb(b["nrm"], idx)).sum(-1).abs()
    if kind == "curv":
        return (b["curv"].unsqueeze(-1) - gather_nb(b["curv"], idx)).abs()
    if kind == "edgelen":
        return (b["pos"].unsqueeze(2) - gather_nb(b["pos"], idx)).norm(dim=-1)
    dis = (gather_nb(b["y"], idx) != b["y"].unsqueeze(-1)).float()
    if kind == "oracle":
        return dis
    if kind == "inv":
        return 1 - dis
    raise ValueError(kind)


def select_mask(score, budget, level, gen):
    """score (B,N,k) -> bool mask (B,N,k); matched budget PER SHAPE; random tie-break."""
    B, N, k = score.shape
    if level == "edge":
        flat = (score + 1e-6 * torch.rand(score.shape, generator=gen, device=score.device)).reshape(B, N * k)
        kk = int(round(budget * N * k))
        m = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, flat.topk(kk, dim=1).indices, True)
        return m.reshape(B, N, k)
    node = score.mean(-1) + 1e-6 * torch.rand((B, N), generator=gen, device=score.device)
    kn = int(round(budget * N))
    nm = torch.zeros(B, N, dtype=torch.bool, device=score.device).scatter_(1, node.topk(kn, dim=1).indices, True)
    return nm.unsqueeze(-1).expand(B, N, k)


def build_graph(b, cfg, gen):
    pos, idx = b["pos"], b["idx"]
    B, N, k = idx.shape
    dev = pos.device
    base = (torch.arange(B, device=dev) * N).view(B, 1, 1)
    src = (idx + base).reshape(-1)
    dst = (torch.arange(N, device=dev).view(1, N, 1) + base).expand(B, N, k).reshape(-1)
    ei = torch.stack([src, dst])
    x = pos.reshape(B * N, 3)
    if cfg["feat"] == "pos_normal":
        x = torch.cat([x, b["nrm"].reshape(B * N, 3)], 1)
    frac = 0.0
    arm = cfg["arm"]
    if arm != "none":
        if arm == "all":
            mask = torch.ones(B * N * k, dtype=torch.bool, device=dev)
        else:
            kind, level = arm.split("_")
            mask = select_mask(edge_score(kind, b), cfg["budget"], level, gen).reshape(-1)
        x, ei, _ = half_hop(x, ei, mask, cfg["alpha"])
        frac = float(mask.float().mean())
    A = norm_adj(ei, x.size(0), 1.0) if cfg["backbone"] == "gcn" else None
    return dict(x=x, ei=ei, A=A, n0=B * N, frac=frac)


# ----------------------------------------------------------------------------- model
class EdgeConvLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * i, o, bias=False), nn.BatchNorm1d(o), nn.LeakyReLU(0.2))

    def forward(self, x, ei):
        src, dst = ei
        m = self.mlp(torch.cat([x[dst], x[src] - x[dst]], 1))
        out = torch.zeros(x.size(0), m.size(1), device=x.device, dtype=m.dtype)
        return out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(m), m, reduce="amax", include_self=False)


class SegNet(nn.Module):
    def __init__(self, backbone, in_dim, hid, depth, n_cls, dropout, dense, residual):
        super().__init__()
        self.backbone, self.dense, self.residual = backbone, dense, residual
        self.layers, self.norms = nn.ModuleList(), nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            if backbone == "gcn":
                self.layers.append(nn.Linear(d, hid, bias=False)); self.norms.append(nn.BatchNorm1d(hid))
            else:
                self.layers.append(EdgeConvLayer(d, hid))
            d = hid
        f = hid * depth if dense else hid
        self.head = nn.Sequential(nn.Linear(f, 128, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                                  nn.Dropout(dropout), nn.Linear(128, n_cls))

    def forward(self, g):
        h, outs = g["x"], []
        for i, layer in enumerate(self.layers):
            prev = h
            if self.backbone == "gcn":
                h = F.leaky_relu(self.norms[i](torch.sparse.mm(g["A"], layer(h))), 0.2)
            else:
                h = layer(h, g["ei"])
            if self.residual and i > 0:
                h = h + prev
            outs.append(h)
        return self.head(torch.cat(outs, 1) if self.dense else h)


# ----------------------------------------------------------------------------- metrics
def shape_metrics(pred, y, bmask, C):
    ph, yh = F.one_hot(pred, C).bool(), F.one_hot(y, C).bool()
    inter, union = (ph & yh).sum(1).float(), (ph | yh).sum(1).float()
    miou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).mean(1)  # absent parts -> 1 (standard)
    corr = (pred == y).float()
    bm = bmask.float(); nb = bm.sum(1); ni = (1 - bm).sum(1)
    nan = torch.full_like(nb, float("nan"))
    b_acc = torch.where(nb > 0, (corr * bm).sum(1) / nb.clamp(min=1), nan)
    i_acc = torch.where(ni > 0, (corr * (1 - bm)).sum(1) / ni.clamp(min=1), nan)
    bk = bmask.unsqueeze(-1)
    ib, ub = (ph & yh & bk).sum(1).float(), ((ph | yh) & bk).sum(1).float()
    pres = ub > 0
    b_miou = torch.where(pres, ib / ub.clamp(min=1), torch.zeros_like(ub)).sum(1) / pres.sum(1).clamp(min=1)  # present parts only
    b_miou = torch.where(nb > 0, b_miou, nan)
    return dict(miou=miou, acc=corr.mean(1), b_acc=b_acc, i_acc=i_acc, b_miou=b_miou, frac_boundary=bm.mean(1))


@torch.no_grad()
def evaluate(model, D, cfg, device, bs):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(777)            # fixed selection for random arms at eval
    acc, fr = {}, []
    for i in range(0, D["pos"].size(0), bs):
        b = get_batch(D, torch.arange(i, min(i + bs, D["pos"].size(0))), device)
        g = build_graph(b, cfg, gen)
        B, N = b["y"].shape
        pred = model(g)[:g["n0"]].view(B, N, -1).argmax(-1)
        bmask = (gather_nb(b["y"], b["idx"]) != b["y"].unsqueeze(-1)).any(-1)
        for k, v in shape_metrics(pred, b["y"], bmask, D["n_cls"]).items():
            acc.setdefault(k, []).append(v.cpu())
        fr.append(g["frac"])
    return {k: torch.cat(v).numpy() for k, v in acc.items()}, float(np.mean(fr))


def train_run(cfg, Dtr, Dva, Dte, a, device):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    in_dim = 6 if cfg["feat"] == "pos_normal" else 3
    model = SegNet(cfg["backbone"], in_dim, a.hidden, cfg["depth"], Dtr["n_cls"], a.dropout, a.dense, a.residual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    cpu_gen = torch.Generator().manual_seed(cfg["seed"])
    bs = a.bs_all if (cfg["arm"] == "all" and a.bs_all) else a.bs
    S = Dtr["pos"].size(0)
    best_v, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(S, generator=cpu_gen)
        for i in range(0, S, bs):
            ids = perm[i:i + bs]
            if ids.numel() < 2:
                continue
            b = get_batch(Dtr, ids, device)
            g = build_graph(b, cfg, gen)
            loss = F.cross_entropy(model(g)[:g["n0"]], b["y"].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            m, _ = evaluate(model, Dva, cfg, device, bs)
            v = float(np.nanmean(m["miou"]))
            if v > best_v:
                best_v, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    test, frac = evaluate(model, Dte, cfg, device, bs)
    return best_v, test, frac


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    L = lambda s: s.split(",")
    ap.add_argument("--root", default="data/ShapeNet")
    ap.add_argument("--out_dir", default="results/b")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--categories", type=L, default=["Airplane", "Table"])
    ap.add_argument("--backbones", type=L, default=["edgeconv"])
    ap.add_argument("--depths", type=lambda s: [int(v) for v in s.split(",")], default=[4])
    ap.add_argument("--arms", type=L, default=["none", "all", "rand_node", "geom_node", "edgelen_node", "oracle_node",
                                               "inv_node", "rand_edge", "geom_edge", "oracle_edge"])
    ap.add_argument("--budgets", type=lambda s: [float(v) for v in s.split(",")], default=[0.1])
    ap.add_argument("--alphas", type=lambda s: [float(v) for v in s.split(",")], default=[0.1])
    ap.add_argument("--feat", default="pos", choices=["pos", "pos_normal"])
    ap.add_argument("--dense", action="store_true", help="DGCNN-style dense skip connections")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bs_all", type=int, default=0, help="smaller batch for the 'all' arm (memory)")
    ap.add_argument("--max_shapes", type=int, default=100000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(os.path.join(a.out_dir, "per_shape"), exist_ok=True)
    runs_path = os.path.join(a.out_dir, "runs.jsonl")
    done = set(json.loads(l)["run_id"] for l in open(runs_path)) if os.path.exists(runs_path) else set()
    fh = open(runs_path, "a")
    t0 = time.time()
    for cat in (["Synthetic"] if a.synthetic else a.categories):
        if a.synthetic:
            sp = {"train": make_synthetic(48, a.num_points, 0), "val": make_synthetic(16, a.num_points, 1),
                  "test": make_synthetic(24, a.num_points, 2)}
        else:
            sp = {s: load_shapenet(a.root, cat, s, a.num_points, a.max_shapes) for s in ("train", "val", "test")}
        D = {s: make_split(*v, a.k, device) for s, v in sp.items()}
        print(f"[{cat}] shapes train/val/test = {[D[s]['pos'].size(0) for s in ('train', 'val', 'test')]}", flush=True)
        for bb in a.backbones:
            for depth in a.depths:
                for seed in range(a.seeds):
                    for arm in a.arms:
                        sweep = [(None, None)] if arm in ("none", "all") else [(b, al) for b in a.budgets for al in a.alphas]
                        if arm == "all":
                            sweep = [(1.0, al) for al in a.alphas]
                        for budget, alpha in sweep:
                            cfg = dict(category=cat, backbone=bb, depth=depth, seed=seed, arm=arm, budget=budget, alpha=alpha,
                                       feat=a.feat, k=a.k, num_points=a.num_points, dense=a.dense, residual=a.residual, hidden=a.hidden)
                            rid = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
                            if rid in done:
                                continue
                            v, test, frac = train_run(cfg, D["train"], D["val"], D["test"], a, device)
                            np.savez_compressed(os.path.join(a.out_dir, "per_shape", f"{rid}.npz"), **test)
                            summ = {k: float(np.nanmean(t)) for k, t in test.items()}
                            rec = dict(run_id=rid, **cfg, val_miou=v, frac_edges_slowed=frac, **summ)
                            fh.write(json.dumps(rec) + "\n"); fh.flush()
                            print(f"[{cat}|{bb}|L{depth}|s{seed}|{arm}|b={budget}|a={alpha}] "
                                  f"val {v:.4f} test mIoU {summ['miou']:.4f} bMIoU {summ['b_miou']:.4f} "
                                  f"slowed {frac:.3f} | {time.time() - t0:.0f}s", flush=True)
    fh.close()


if __name__ == "__main__":
    main()

