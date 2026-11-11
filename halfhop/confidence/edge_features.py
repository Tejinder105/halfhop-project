"""Edge-level geometric descriptor extraction for CAHH.

For each undirected edge (i, j) the descriptor is a 4-vector::

    z_ij = [normal_disagreement, curvature_diff, edge_length, density_diff]

where
  normal_disagreement = 1 - |n_i · n_j|  (∈ [0,1]; 0=parallel, 1=orthogonal)
  curvature_diff      = |κ_i - κ_j|
  edge_length         = ‖p_i - p_j‖      (normalised to scene scale)
  density_diff        = |d_i - d_j|      (d_i = inverse mean kNN distance)

All features are floats. If normals are unavailable the normal channel is set
to -1.0 and a validity flag is set on the returned tensor attribute
``has_normals``.

Normalisation
-------------
Call :class:`EdgeFeatureNormalizer` to fit mean/std on training shapes only.
Clip at ±3σ before standardising to reduce outlier influence.

Degenerate edges
----------------
Self-loops and edges where both endpoints are identical are assigned
all-zero features and excluded from MLP training (but not from the graph).
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Local density
# ---------------------------------------------------------------------------

def compute_local_density(pos: Tensor, edge_index: Tensor) -> Tensor:
    """Per-node local density = 1 / (mean kNN edge length + ε).

    Args:
        pos: ``(N, 3)`` point coordinates.
        edge_index: ``(2, E)`` directed edges (kNN graph).

    Returns:
        ``(N,)`` density estimate (higher = denser neighbourhood).
    """
    src, dst = edge_index
    valid = src != dst  # exclude self-loops
    if valid.sum() == 0:
        return torch.zeros(pos.size(0), device=pos.device)
    lengths = (pos[src[valid]] - pos[dst[valid]]).norm(dim=-1)  # (E_valid,)
    deg = torch.zeros(pos.size(0), device=pos.device)
    sum_len = torch.zeros(pos.size(0), device=pos.device)
    deg.index_add_(0, src[valid], torch.ones_like(lengths))
    sum_len.index_add_(0, src[valid], lengths)
    mean_len = sum_len / deg.clamp(min=1.0)
    return 1.0 / (mean_len + 1e-8)


# ---------------------------------------------------------------------------
# Edge feature extraction
# ---------------------------------------------------------------------------

def build_edge_features(
    pos: Tensor,
    edge_index: Tensor,
    normals: Tensor | None = None,
    curvature: Tensor | None = None,
    density: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the 4-dimensional edge descriptor for every edge.

    Args:
        pos:        ``(N, 3)`` point coordinates.
        edge_index: ``(2, E)`` directed edges.  Directed pairs (i→j and j→i)
                    are both processed; deduplication is the caller's concern.
        normals:    ``(N, 3)`` unit surface normals, or ``None``.
        curvature:  ``(N,)`` per-point curvature estimate, or ``None``.
        density:    ``(N,)`` per-point local density, or ``None``.
                    If ``None`` it is computed from ``pos`` and ``edge_index``.

    Returns:
        features:   ``(E, 4)`` float32 edge features.
        valid_mask: ``(E,)`` bool — False for self-loops and degenerate edges.

    Feature layout (column index):
        0  normal disagreement  (or -1.0 when ``normals`` is None)
        1  curvature difference (or  0.0 when ``curvature`` is None)
        2  edge length
        3  density difference
    """
    src, dst = edge_index
    E = edge_index.size(1)
    device = pos.device

    valid_mask = (src != dst)

    # --- feature 0: normal disagreement ---
    if normals is not None:
        n = torch.nn.functional.normalize(normals.float(), dim=-1)
        n = torch.nan_to_num(n, nan=0.0)
        cos_sim = (n[src] * n[dst]).sum(dim=-1).abs().clamp(max=1.0)
        f0 = 1.0 - cos_sim                              # (E,)
        has_normals = True
    else:
        f0 = torch.full((E,), -1.0, device=device)
        has_normals = False

    # --- feature 1: curvature difference ---
    if curvature is not None:
        kappa = curvature.float()
        f1 = (kappa[src] - kappa[dst]).abs()
    else:
        f1 = torch.zeros(E, device=device)

    # --- feature 2: edge length ---
    diff = pos[src].float() - pos[dst].float()
    f2 = diff.norm(dim=-1)                              # (E,)

    # --- feature 3: density difference ---
    if density is None:
        density = compute_local_density(pos, edge_index)
    d = density.float()
    f3 = (d[src] - d[dst]).abs()

    features = torch.stack([f0, f1, f2, f3], dim=1)    # (E, 4)

    # Zero out self-loops so they can safely enter normalisation without bias.
    features[~valid_mask] = 0.0

    # Store metadata for downstream callers.
    features.has_normals_ = has_normals  # type: ignore[attr-defined]

    return features, valid_mask


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class EdgeFeatureNormalizer:
    """Fit mean/std on training data; transform at inference.

    Only features from *valid* (non-self-loop) edges are used for fitting.
    The normal-disagreement channel is fitted only when normals were available;
    otherwise it passes through unchanged (keeps the -1.0 sentinel).

    Usage::

        norm = EdgeFeatureNormalizer()
        for data in train_shapes:
            feats, valid = build_edge_features(data.pos, data.edge_index,
                                               data.normal)
            norm.update(feats[valid])
        norm.finalise()

        # At inference / validation:
        feats_norm = norm.transform(feats)
    """

    def __init__(self, clip_sigma: float = 3.0):
        self.clip_sigma = clip_sigma
        self._sum: Tensor | None = None
        self._sum_sq: Tensor | None = None
        self._n: int = 0
        self.mean_: Tensor | None = None
        self.std_: Tensor | None = None
        self._fitted = False

    def update(self, features: Tensor) -> None:
        """Accumulate statistics.  ``features`` shape: ``(E_valid, 4)``."""
        if features.numel() == 0:
            return
        x = features.float()
        if self._sum is None:
            self._sum = x.sum(0)
            self._sum_sq = (x ** 2).sum(0)
        else:
            self._sum = self._sum + x.sum(0)
            self._sum_sq = self._sum_sq + (x ** 2).sum(0)
        self._n += x.size(0)

    def finalise(self) -> None:
        """Compute mean and std from accumulated statistics."""
        if self._n == 0:
            raise RuntimeError("No data accumulated; call update() first.")
        mean = self._sum / self._n
        var = self._sum_sq / self._n - mean ** 2
        std = var.clamp(min=0.0).sqrt()
        std = std.clamp(min=1e-8)  # avoid divide-by-zero
        self.mean_ = mean
        self.std_ = std
        self._fitted = True

    def transform(self, features: Tensor) -> Tensor:
        """Standardise and clip.  Operates on ``(E, 4)`` tensors."""
        if not self._fitted:
            raise RuntimeError("Call finalise() before transform().")
        x = features.float()
        mean = self.mean_.to(x.device)
        std = self.std_.to(x.device)
        x = (x - mean) / std
        x = x.clamp(-self.clip_sigma, self.clip_sigma)
        return x

    def state_dict(self) -> dict:
        return {
            "mean": self.mean_,
            "std": self.std_,
            "clip_sigma": self.clip_sigma,
            "fitted": self._fitted,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mean_ = d["mean"]
        self.std_ = d["std"]
        self.clip_sigma = d["clip_sigma"]
        self._fitted = d["fitted"]

