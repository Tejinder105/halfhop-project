"""Stateful slow nodes via GRU cell (Component C).

Replaces the one-shot linear interpolation used by Half-Hop with a lightweight
GRU cell that maintains a recurrent state per slow node. The slow node acts as
an edge memory, blending source messages over time.
"""

import copy
import torch
import torch.nn as nn


class StatefulHalfHop(nn.Module):
    r"""Half-Hop with a GRU-cell slow-node state (Component C).

    For each selected edge (i → j):
    - Initialises ``h_slow`` as the standard linear interpolation.
    - Applies a GRU cell: ``h_slow = GRUCell(x_src, h_slow)``
      where ``x_src`` is the source message and ``h_slow`` is the hidden state.

    After one GRU step the resulting ``h_slow`` is used as the slow-node
    feature, and the standard Half-Hop connectivity (proposed) is applied.

    Args:
        in_dim (int): Input node feature dimension.
        alpha (float): Initialisation mixing coefficient. Default: ``0.5``.
        p (float): Node-level Half-Hop sampling probability. Default: ``1.0``.
        inplace (bool): Modify data in-place. Default: ``False``.
    """

    def __init__(
        self,
        in_dim: int,
        alpha: float = 0.5,
        p: float = 1.0,
        inplace: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.p = p
        self.inplace = inplace
        # GRU cell: input = source features, hidden = slow-node state
        self.gru = nn.GRUCell(input_size=in_dim, hidden_size=in_dim)
        # Lightweight projection to keep parameter count low
        self.proj = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, data):
        """Apply stateful Half-Hop augmentation.

        Args:
            data: PyG Data with ``data.x`` and ``data.edge_index``.

        Returns:
            data: Augmented Data with slow nodes and ``data.slow_node_mask``.
        """
        if not self.inplace:
            data = copy.deepcopy(data)

        x, edge_index = data.x, data.edge_index
        device = x.device
        N = x.size(0)

        # --- select edges (same logic as HalfHop node-level sampling) ---
        self_loop_mask = edge_index[0] == edge_index[1]
        ei_self = edge_index[:, self_loop_mask]
        ei = edge_index[:, ~self_loop_mask]

        if self.p == 1.0:
            ei_hh = ei
            ei_keep = None
        else:
            node_mask = torch.rand(N, device=device) < self.p
            ei_hh = ei[:, node_mask[ei[1]]]
            ei_keep = ei[:, ~node_mask[ei[1]]]

        src, dst = ei_hh[0], ei_hh[1]
        K = src.numel()
        slow_ids = torch.arange(N, N + K, device=device, dtype=edge_index.dtype)

        # --- GRU-based slow-node feature ---
        h_init = (1.0 - self.alpha) * x[dst] + self.alpha * x[src]
        x_slow = self.gru(self.proj(x[src]), h_init)   # [K, D]

        new_x = torch.cat([x, x_slow], dim=0)

        # --- proposed connectivity: vi→ν, vj→ν, ν→vj ---
        ei_s2n = torch.stack([src, slow_ids])
        ei_d2n = torch.stack([dst, slow_ids])
        ei_n2d = torch.stack([slow_ids, dst])

        parts = [p for p in [ei_keep, ei_self, ei_s2n, ei_d2n, ei_n2d]
                 if p is not None and p.numel() > 0]
        new_ei = torch.cat(parts, dim=1)

        slow_mask = torch.zeros(N + K, dtype=torch.bool, device=device)
        slow_mask[N:] = True

        data.x = new_x
        data.edge_index = new_ei
        data.slow_node_mask = slow_mask
        return data
