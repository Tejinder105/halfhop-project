import copy
import torch
from torch import nn


def drop_edge(edge_index: torch.Tensor, p: float = 0.2, force_undirected: bool = False, training: bool = True):
    r"""Randomly drops edges from the adjacency matrix with probability :obj:`p`.

    Args:
        edge_index (LongTensor): The edge indices of shape :obj:`[2, num_edges]`.
        p (float): Edge drop probability in [0, 1]. Default: 0.2.
        force_undirected (bool): If True, ensures that if (u, v) is dropped,
            (v, u) is also dropped. Default: False.
        training (bool): If False, no edges are dropped. Default: True.

    Returns:
        torch.Tensor: Filtered edge_index.
        torch.Tensor: Boolean mask indicating retained edges.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Drop probability p must be in [0, 1], got {p}")

    if not training or p == 0.0:
        mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        return edge_index, mask

    num_edges = edge_index.size(1)
    device = edge_index.device

    if force_undirected:
        # Symmetrize drop: consider canonical edges (min(u, v), max(u, v))
        row, col = edge_index[0], edge_index[1]
        pair_min = torch.minimum(row, col)
        pair_max = torch.maximum(row, col)
        # Pair hash
        pairs = torch.stack([pair_min, pair_max], dim=0)
        unique_pairs, inverse_indices = torch.unique(pairs, dim=1, return_inverse=True)
        num_unique = unique_pairs.size(1)
        pair_mask = torch.rand(num_unique, device=device) >= p
        mask = pair_mask[inverse_indices]
    else:
        mask = torch.rand(num_edges, device=device) >= p

    # Ensure self-loops are preserved if present so nodes retain ego-information
    self_loop_mask = edge_index[0] == edge_index[1]
    mask = mask | self_loop_mask

    return edge_index[:, mask], mask


class DropEdge(nn.Module):
    r"""DropEdge data transformation and training module.

    Randomly removes edges from the graph during training with probability :obj:`p`.

    Args:
        p (float): Drop probability. Default: 0.2.
        force_undirected (bool): Whether to drop undirected edge pairs together.
            Default: False.
    """

    def __init__(self, p: float = 0.2, force_undirected: bool = False):
        super().__init__()
        self.p = p
        self.force_undirected = force_undirected

    def forward(self, edge_index: torch.Tensor, training: bool = True):
        return drop_edge(
            edge_index,
            p=self.p,
            force_undirected=self.force_undirected,
            training=training and self.training,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, force_undirected={self.force_undirected})"

