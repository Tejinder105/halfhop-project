"""PyTorch point-cloud graph construction for adaptive multi-delay Half-Hop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AdaptiveGraph:
    x: Tensor
    edge_index: Tensor
    slow_node_mask: Tensor
    curvature: Tensor
    delays: Tensor


def knn_edge_index(pos: Tensor, k: int) -> Tensor:
    """Return directed edges ``neighbour -> centre`` for a point cloud."""
    if pos.ndim != 2 or pos.size(1) != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}")
    if k < 1 or k >= pos.size(0):
        raise ValueError(f"k must be in [1, N-1], got k={k}, N={pos.size(0)}")

    distances = torch.cdist(pos, pos)
    distances.fill_diagonal_(torch.inf)
    neighbours = distances.topk(k, largest=False, dim=1).indices
    dst = torch.arange(pos.size(0), device=pos.device).repeat_interleave(k)
    return torch.stack((neighbours.reshape(-1), dst), dim=0).long()


def surface_variation(pos: Tensor, k: int) -> Tensor:
    """Estimate Pauly surface variation from local covariance eigenvalues."""
    neighbours = knn_edge_index(pos, k)[0].view(pos.size(0), k)
    local = pos[neighbours]
    centred = local - local.mean(dim=1, keepdim=True)
    covariance = centred.transpose(1, 2).matmul(centred) / float(k)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    return eigenvalues[:, 0] / eigenvalues.sum(dim=1).clamp_min(torch.finfo(pos.dtype).eps)


def assign_delays(
    curvature: Tensor,
    edge_index: Tensor,
    delays: tuple[int, ...],
    quantiles: tuple[float, ...],
    mode: str = "curvature",
    uniform_m: int = 1,
) -> Tensor:
    """Assign a non-negative delay to each edge."""
    if len(delays) != len(quantiles) + 1:
        raise ValueError("delays must contain one more value than quantiles")
    if any(delay < 0 for delay in delays):
        raise ValueError("delays must be non-negative")
    if mode == "uniform":
        if uniform_m < 0:
            raise ValueError("uniform_m must be non-negative")
        return torch.full(
            (edge_index.size(1),), uniform_m, dtype=torch.long, device=edge_index.device
        )
    if mode not in {"curvature", "inverse", "random"}:
        raise ValueError(f"unknown mode {mode}")

    src, dst = edge_index
    edge_curvature = torch.maximum(curvature[src], curvature[dst])
    thresholds = torch.quantile(
        edge_curvature, torch.tensor(quantiles, device=edge_curvature.device)
    )
    level = torch.bucketize(edge_curvature, thresholds, right=True)
    if mode == "inverse":
        level = len(quantiles) - level
    elif mode == "random":
        level = level[torch.randperm(level.numel(), device=level.device)]

    delay_values = torch.tensor(delays, dtype=torch.long, device=edge_index.device)
    return delay_values[level]


def insert_slow_chains(
    x: Tensor, edge_index: Tensor, delays: Tensor, backward: str = "last"
) -> tuple[Tensor, Tensor, Tensor]:
    """Insert linearly interpolated slow nodes along selected edges."""
    if backward not in {"last", "none"}:
        raise ValueError("backward must be 'last' or 'none'")
    if delays.ndim != 1 or delays.numel() != edge_index.size(1):
        raise ValueError("delays must have one value per edge")

    num_original = x.size(0)
    src, dst = edge_index
    slow_edges = torch.nonzero(delays > 0, as_tuple=False).flatten()
    if slow_edges.numel() == 0:
        return x, edge_index, torch.zeros(num_original, dtype=torch.bool, device=x.device)

    slow_delays = delays[slow_edges].long()
    starts = torch.cumsum(slow_delays, dim=0) - slow_delays
    owner = torch.repeat_interleave(torch.arange(slow_edges.numel(), device=x.device), slow_delays)
    positions = torch.arange(owner.numel(), device=x.device) - starts[owner] + 1
    fractions = positions.to(x.dtype) / (slow_delays[owner].to(x.dtype) + 1.0)
    slow_src = src[slow_edges][owner]
    slow_dst = dst[slow_edges][owner]
    slow_x = (1.0 - fractions[:, None]) * x[slow_src] + fractions[:, None] * x[slow_dst]

    slow_ids = num_original + torch.arange(slow_x.size(0), device=x.device)
    first = num_original + starts
    last = first + slow_delays - 1
    inner = positions < slow_delays[owner]

    edge_parts = [
        edge_index[:, delays == 0],
        torch.stack((src[slow_edges], first), dim=0),
        torch.stack((slow_ids[inner], slow_ids[inner] + 1), dim=0),
        torch.stack((last, dst[slow_edges]), dim=0),
    ]
    if backward == "last":
        edge_parts.append(torch.stack((dst[slow_edges], last), dim=0))

    new_edges = torch.cat([part for part in edge_parts if part.numel() > 0], dim=1)
    new_x = torch.cat((x, slow_x), dim=0)
    slow_mask = torch.cat(
        (
            torch.zeros(num_original, dtype=torch.bool, device=x.device),
            torch.ones(slow_x.size(0), dtype=torch.bool, device=x.device),
        )
    )
    return new_x, new_edges.long(), slow_mask


def build_graph(
    pos: Tensor,
    x: Tensor,
    knn: int,
    curv_k: int,
    delays: tuple[int, ...],
    quantiles: tuple[float, ...],
    mode: str,
    uniform_m: int,
    backward: str,
    edge_index: Tensor | None = None,
    curvature: Tensor | None = None,
) -> AdaptiveGraph:
    if edge_index is None:
        edge_index = knn_edge_index(pos, knn)
    if mode == "none":
        slow_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        edge_delays = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        curv_out = curvature if curvature is not None else torch.zeros(x.size(0), device=x.device)
        return AdaptiveGraph(x, edge_index, slow_mask, curv_out, edge_delays)

    if curvature is None:
        curvature = surface_variation(pos, curv_k)
    edge_delays = assign_delays(
        curvature, edge_index, delays, quantiles, mode=mode, uniform_m=uniform_m
    )
    new_x, new_edges, slow_mask = insert_slow_chains(x, edge_index, edge_delays, backward)
    return AdaptiveGraph(new_x, new_edges, slow_mask, curvature, edge_delays)


class PointCloudPreTransform:
    """Pre-transform to sample points, compute curvature, and base KNN graph once."""

    def __init__(self, points: int = 512, knn: int = 12, curv_k: int = 16):
        from torch_geometric.transforms import NormalizeScale, SamplePoints

        self.points = points
        self.knn = knn
        self.curv_k = curv_k
        self.norm = NormalizeScale()
        self.sampler = SamplePoints(points, include_normals=True)

    def __call__(self, data):
        data = self.norm(data)
        data = self.sampler(data)
        pos = data.pos.float()
        data.curvature = surface_variation(pos, self.curv_k)
        data.base_edge_index = knn_edge_index(pos, self.knn)
        if hasattr(data, "face"):
            data.face = None
        return data


class AdaptiveHalfHop:
    """PyG transform that builds an adaptive point-cloud graph per shape."""

    def __init__(
        self,
        knn: int = 16,
        curv_k: int = 24,
        delays: tuple[int, ...] = (1, 2, 3),
        quantiles: tuple[float, ...] = (0.5, 0.85),
        mode: str = "curvature",
        uniform_m: int = 1,
        backward: str = "last",
    ):
        self.knn = knn
        self.curv_k = curv_k
        self.delays = tuple(delays)
        self.quantiles = tuple(quantiles)
        self.mode = mode
        self.uniform_m = uniform_m
        self.backward = backward

    def __call__(self, data):
        data = data.clone()
        if data.pos is None:
            raise ValueError("AdaptiveHalfHop requires data.pos")
        pos = data.pos.float()
        if pos.size(0) <= max(self.knn, self.curv_k):
            raise ValueError("point cloud must contain more points than knn and curv_k")
        normals = getattr(data, "normal", None)
        features = pos if normals is None else torch.cat((pos, normals.float()), dim=1)

        base_edge_index = getattr(data, "base_edge_index", None)
        if base_edge_index is None and hasattr(data, "edge_index") and data.edge_index is not None:
            base_edge_index = data.edge_index
        curvature = getattr(data, "curvature", None)

        graph = build_graph(
            pos,
            features,
            self.knn,
            self.curv_k,
            self.delays,
            self.quantiles,
            self.mode,
            self.uniform_m,
            self.backward,
            edge_index=base_edge_index,
            curvature=curvature,
        )
        data.x = graph.x
        data.edge_index = graph.edge_index
        data.slow_node_mask = graph.slow_node_mask
        data.num_nodes = graph.x.size(0)
        data.pos = None
        if hasattr(data, "normal"):
            data.normal = None
        if hasattr(data, "curvature"):
            data.curvature = None
        if hasattr(data, "base_edge_index"):
            data.base_edge_index = None
        return data
