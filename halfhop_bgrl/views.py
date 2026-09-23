"""View construction for official BGRL, with optional Half-Hop.

Augmentation order (implementation assumption; see module docstring in train.py):

    deepcopy → Half-Hop (if p>0) → EdgeDrop (if pe>0) → FeatDrop (if pf>0)

Official BGRL (`bgrl/transforms.py`) is EdgeDrop then FeatDrop after a deepcopy.
Half-Hop is inserted before those drops so slow-node edges can also be dropped.

Original nodes stay at indices ``0 .. N-1``. Slow nodes are excluded from the
BGRL loss and from linear evaluation via ``slow_node_mask``.
"""

from __future__ import annotations

import copy

import torch
from torch_geometric.data import Data

from halfhop.halfhop import HalfHop

from .compat import patch_dropout_adj


class DropFeatures:
    """Official BGRL column-wise feature drop. Drops whole feature dims with prob p."""

    def __init__(self, p: float):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty(
            (data.x.size(1),), dtype=torch.float32, device=data.x.device
        ).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data


class DropEdges:
    """Official BGRL edge drop via PyG dropout_adj / dropout_edge."""

    def __init__(self, p: float, force_undirected: bool = False):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if "edge_attr" in data else None
        edge_index, edge_attr = _dropout_adj(
            edge_index,
            edge_attr,
            p=self.p,
            force_undirected=self.force_undirected,
        )
        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data


def _dropout_adj(edge_index, edge_attr, p, force_undirected):
    patch_dropout_adj()
    try:
        from torch_geometric.utils.dropout import dropout_adj

        return dropout_adj(
            edge_index, edge_attr, p=p, force_undirected=force_undirected
        )
    except ImportError:
        from torch_geometric.utils import dropout_edge

        edge_index, edge_mask = dropout_edge(
            edge_index, p=p, force_undirected=force_undirected
        )
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask]
        return edge_index, edge_attr


class ViewTransform:
    def __init__(
        self,
        p_hh: float,
        alpha: float,
        drop_edge_p: float,
        drop_feat_p: float,
        log_counts: bool = False,
        name: str = "view",
    ):
        self.p_hh = p_hh
        self.alpha = alpha
        self.drop_edge_p = drop_edge_p
        self.drop_feat_p = drop_feat_p
        self.log_counts = log_counts
        self.name = name
        self._logged = False
        self._halfhop = (
            HalfHop(alpha=alpha, p=p_hh, inplace=True) if p_hh > 0.0 else None
        )
        self._drop_edges = DropEdges(drop_edge_p) if drop_edge_p > 0.0 else None
        self._drop_feats = DropFeatures(drop_feat_p) if drop_feat_p > 0.0 else None

    def __call__(self, data: Data):
        num_original = data.num_nodes
        aug = copy.deepcopy(data)

        if self._halfhop is not None:
            aug = self._halfhop(aug)

        if self._drop_edges is not None:
            aug = self._drop_edges(aug)

        if self._drop_feats is not None:
            aug = self._drop_feats(aug)

        if hasattr(aug, "slow_node_mask") and aug.slow_node_mask is not None:
            original_mask = ~aug.slow_node_mask
        else:
            original_mask = torch.ones(
                aug.num_nodes, dtype=torch.bool, device=aug.x.device
            )

        # Guard: Half-Hop must keep original nodes as the first N rows.
        if original_mask.sum().item() != num_original:
            raise RuntimeError(
                f"{self.name}: expected {num_original} original nodes, "
                f"got {int(original_mask.sum().item())}"
            )
        if not torch.all(original_mask[:num_original]) or torch.any(
            original_mask[num_original:]
        ):
            raise RuntimeError(
                f"{self.name}: original nodes are not a prefix of the node list"
            )

        if self.log_counts and not self._logged:
            n_slow = int(aug.num_nodes - num_original)
            print(
                f"[debug] {self.name}: nodes {num_original}->{aug.num_nodes} "
                f"(slow={n_slow}), edges {data.edge_index.size(1)}->"
                f"{aug.edge_index.size(1)}"
            )
            self._logged = True

        return aug, original_mask


def build_view_transform(
    p_hh: float,
    alpha: float,
    drop_edge_p: float,
    drop_feat_p: float,
    log_counts: bool = False,
    name: str = "view",
) -> ViewTransform:
    return ViewTransform(
        p_hh=p_hh,
        alpha=alpha,
        drop_edge_p=drop_edge_p,
        drop_feat_p=drop_feat_p,
        log_counts=log_counts,
        name=name,
    )
