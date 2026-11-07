"""Selective Half-Hop: pure-PyTorch core (no PyG dependency).

half_hop()      : subdivide an arbitrary SUBSET of directed edges with slow nodes.
select_edges()  : pick that subset from per-edge scores (edge-level top-budget, or the
                  paper's node-level scheme: top-budget target nodes, all incoming edges).
norm_adj()      : sparse GCN-normalised adjacency with a tunable self-loop weight.

Conventions (same as the paper): directed edge e_ij = (src=i -> dst=j).
    v_i -> nu_k <-> v_j          (E' = E minus {e_ij}, plus {i->k, j->k, k->j})
    x_k = alpha * x_i + (1 - alpha) * x_j      (small alpha => slow node sits near the target)
"""
import torch


def half_hop(x, edge_index, edge_mask, alpha):
    """Half-hop the edges where `edge_mask` is True (self-loops are never half-hopped).

    Returns (x_all, edge_index_all, slow_mask). Original nodes keep ids 0..N-1, slow nodes
    are appended, so `out[:N]` are the original-node embeddings.
    """
    n = x.size(0)
    src, dst = edge_index
    sel = edge_mask.bool() & (src != dst)
    s, d = src[sel], dst[sel]
    m = s.numel()
    new = torch.arange(n, n + m, device=x.device)
    x_new = alpha * x[s] + (1.0 - alpha) * x[d]
    keep = edge_index[:, ~sel]
    added = torch.cat([torch.stack([s, new]), torch.stack([d, new]), torch.stack([new, d])], dim=1)
    ei = torch.cat([keep, added], dim=1)
    x_all = torch.cat([x, x_new], dim=0)
    slow = torch.zeros(n + m, dtype=torch.bool, device=x.device)
    slow[n:] = True
    return x_all, ei, slow


def topk_mask(score, k, generator=None):
    """Boolean mask of the k largest scores; ties are broken uniformly at random.

    `generator` must live on the same device as `score` (use a CPU generator with CPU scores).
    """
    e = score.numel()
    k = max(0, min(int(k), e))
    mask = torch.zeros(e, dtype=torch.bool, device=score.device)
    if k == 0:
        return mask
    if k == e:
        mask[:] = True
        return mask
    perm = torch.randperm(e, generator=generator, device=score.device)
    order = torch.sort(score[perm], descending=True, stable=True).indices
    mask[perm[order[:k]]] = True
    return mask


def select_edges(edge_index, num_nodes, edge_score, budget, level="edge", generator=None):
    """Return a bool mask over edges.

    level='edge': the top `budget` fraction of edges by score.
    level='node': node score = mean score of incoming edges; the top `budget` fraction of
                  nodes are selected and ALL their incoming edges are half-hopped (paper's scheme).
    Higher score => more likely to be slowed. Random tie-breaking, so a constant score gives
    uniform random selection.
    """
    src, dst = edge_index
    valid = src != dst
    score = edge_score.to(torch.float32)
    if level == "edge":
        sc = score.clone()
        sc[~valid] = float("-inf")
        k = int(round(budget * int(valid.sum())))
        return topk_mask(sc, k, generator) & valid
    if level == "node":
        ones = torch.ones(int(valid.sum()), device=score.device)
        deg = torch.zeros(num_nodes, device=score.device).index_add_(0, dst[valid], ones)
        tot = torch.zeros(num_nodes, device=score.device).index_add_(0, dst[valid], score[valid])
        node_score = tot / deg.clamp(min=1)
        has = deg > 0
        node_score[~has] = float("-inf")
        kn = int(round(budget * int(has.sum())))
        nmask = topk_mask(node_score, kn, generator)
        return nmask[dst] & valid
    raise ValueError(f"unknown level {level!r}")


def norm_adj(edge_index, n, self_loop_weight=1.0):
    """Sparse D^-1/2 (A + w I) D^-1/2, indexed [dst, src], so that (A @ H)[dst] aggregates sources."""
    dev = edge_index.device
    loop = torch.arange(n, device=dev)
    src = torch.cat([edge_index[0], loop])
    dst = torch.cat([edge_index[1], loop])
    ew = torch.cat([torch.ones(edge_index.size(1), device=dev),
                    torch.full((n,), float(self_loop_weight), device=dev)])
    deg = torch.zeros(n, device=dev).index_add_(0, dst, ew)
    dinv = deg.pow(-0.5)
    norm = dinv[src] * ew * dinv[dst]
    return torch.sparse_coo_tensor(torch.stack([dst, src]), norm, (n, n)).coalesce()

