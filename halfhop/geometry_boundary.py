import torch
from torch import Tensor
from typing import Optional

_EPS = 1e-8


def _knn(pos: Tensor, k: int) -> Tensor:
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float('inf'))
    nb = dist.topk(k, largest=False, dim=1).indices
    dst = torch.arange(pos.size(0), device=pos.device).repeat_interleave(k)
    return torch.stack([nb.reshape(-1), dst], dim=0).long()


def _local_covariance(pos: Tensor, k: int):
    """Returns (N, 3, 3) local covariance matrices and neighbour positions."""
    edge = _knn(pos, k)          # [2, N*k]
    src, dst = edge[0], edge[1]  # src = neighbour, dst = centre
    # Group neighbours: shape [N, k, 3]
    N = pos.size(0)
    nb_pos = pos[src].view(N, k, 3)
    centre = pos.unsqueeze(1)    # [N, 1, 3]
    diff = nb_pos - centre       # [N, k, 3]
    # Covariance = (1/k) * diff^T diff  → [N, 3, 3]
    cov = torch.bmm(diff.transpose(1, 2), diff) / k
    return cov


def compute_normals(pos: Tensor, knn: int = 16) -> Tensor:
    """PCA-based unoriented normal estimation. Returns unit normals [N,3]."""
    cov = _local_covariance(pos, knn)          # [N,3,3]
    _, vecs = torch.linalg.eigh(cov)           # eigenvalues ascending; vecs[:,i] is i-th
    normals = vecs[:, :, 0]                    # smallest eigenvalue → normal direction
    normals = normals / (normals.norm(dim=1, keepdim=True).clamp_min(_EPS))
    return normals


def surface_variation(pos: Tensor, knn: int = 16) -> Tensor:
    """Pauly curvature: lambda_min / (lambda_0 + lambda_1 + lambda_2). Returns [N]."""
    cov = _local_covariance(pos, knn)          # [N,3,3]
    evals = torch.linalg.eigvalsh(cov)         # [N,3] ascending
    return evals[:, 0] / evals.sum(dim=1).clamp_min(_EPS)


def normal_disagreement(normals: Tensor, edge_index: Tensor) -> Tensor:
    """Per-edge score = 1 - |n_src · n_dst|. Returns [E] in [0,1]."""
    n = normals / (normals.norm(dim=1, keepdim=True).clamp_min(_EPS))
    src, dst = edge_index[0], edge_index[1]
    dot = (n[src] * n[dst]).sum(dim=1)
    return 1.0 - dot.abs()


def inter_part_boundary_edges(edge_index: Tensor, part_labels: Tensor) -> Tensor:
    """Bool [E]: True where edge crosses a part boundary."""
    src, dst = edge_index[0], edge_index[1]
    return part_labels[src] != part_labels[dst]


def geometry_boundary_score(
    pos: Tensor,
    edge_index: Tensor,
    normals: Optional[Tensor] = None,
    knn: int = 16,
) -> Tensor:
    """Combined score = 0.5 * max(curv[src], curv[dst]) + 0.5 * normal_disagreement. Returns [E]."""
    curv = surface_variation(pos, knn)         # [N]
    if normals is None:
        normals = compute_normals(pos, knn)
    src, dst = edge_index[0], edge_index[1]
    edge_curv = torch.maximum(curv[src], curv[dst])
    nd = normal_disagreement(normals, edge_index)
    return 0.5 * edge_curv + 0.5 * nd


def boundary_edge_mask(
    pos: Tensor,
    edge_index: Tensor,
    normals: Optional[Tensor] = None,
    knn: int = 16,
    quantile: float = 0.85,
) -> Tensor:
    """Bool [E]: True for edges with geometry_boundary_score >= quantile threshold."""
    score = geometry_boundary_score(pos, edge_index, normals=normals, knn=knn)
    threshold = torch.quantile(score, quantile)
    return score >= threshold
