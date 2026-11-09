"""Variable-delay Half-Hop with k ≥ 1 slow nodes per edge (Component D).

Subdivides each selected edge into a chain of k slow nodes:
    vi → ν1 → ν2 → ... → νk → vj
each with its own features, plus the backward edges from vj to each slow node
(as in the proposed connectivity).

k=1 reduces to standard Half-Hop.
"""

import copy
import torch


class MultiDelayHalfHop:
    r"""Half-Hop with k slow nodes per edge (Component D).

    Args:
        k (int): Number of slow nodes to insert per edge. Default: ``1``
            (identical to standard Half-Hop).
        alpha (float): Interpolation coefficient for slow-node features.
            Node ℓ along the chain uses features interpolated between
            x_src and x_dst weighted by position. Default: ``0.5``.
        p (float): Node-level Half-Hop sampling probability. Default: ``1.0``.
        inplace (bool): Modify data in-place. Default: ``False``.
    """

    def __init__(
        self,
        k: int = 1,
        alpha: float = 0.5,
        p: float = 1.0,
        inplace: bool = False,
    ):
        assert k >= 1, f"k must be >= 1, got {k}"
        self.k = k
        self.alpha = alpha
        self.p = p
        self.inplace = inplace

    def __call__(self, data):
        if not self.inplace:
            data = copy.deepcopy(data)

        x, edge_index = data.x, data.edge_index
        device = x.device
        N = x.size(0)

        # Strip self-loops
        self_loop_mask = edge_index[0] == edge_index[1]
        ei_self = edge_index[:, self_loop_mask]
        ei = edge_index[:, ~self_loop_mask]

        # Node-level sampling
        if self.p == 1.0:
            ei_hh = ei
            ei_keep = None
        else:
            node_mask = torch.rand(N, device=device) < self.p
            ei_hh = ei[:, node_mask[ei[1]]]
            ei_keep = ei[:, ~node_mask[ei[1]]]

        src, dst = ei_hh[0], ei_hh[1]
        E = src.numel()

        # Build k slow nodes per selected edge
        all_slow_x = []
        new_edges = []
        offset = N

        for step in range(self.k):
            t = (step + 1) / (self.k + 1)          # position fraction [0,1]
            mix = self.alpha * (1.0 - t) + (1.0 - self.alpha) * t
            x_slow_i = mix * x[src] + (1.0 - mix) * x[dst]
            all_slow_x.append(x_slow_i)

        # Node IDs for each step: [offset .. offset+E), [offset+E .. offset+2E), ...
        slow_ids_per_step = [
            torch.arange(offset + s * E, offset + s * E + E,
                         device=device, dtype=edge_index.dtype)
            for s in range(self.k)
        ]

        # Build chain: vi → ν1 → ν2 → ... → νk → vj
        # Plus backward edges from vj to each slow node (proposed)
        for s, slow_ids in enumerate(slow_ids_per_step):
            prev = src if s == 0 else slow_ids_per_step[s - 1]
            new_edges.append(torch.stack([prev, slow_ids]))      # forward
            new_edges.append(torch.stack([dst, slow_ids]))       # target → slow

        # Last slow node → target
        new_edges.append(torch.stack([slow_ids_per_step[-1], dst]))

        # Assemble
        total_slow = self.k * E
        x_slow_all = torch.cat(all_slow_x, dim=0)              # [k*E, D]
        new_x = torch.cat([x, x_slow_all], dim=0)

        parts = [p_ for p_ in [ei_keep, ei_self] + new_edges
                 if p_ is not None and p_.numel() > 0]
        new_ei = torch.cat(parts, dim=1)

        slow_mask = torch.zeros(N + total_slow, dtype=torch.bool, device=device)
        slow_mask[N:] = True

        data.x = new_x
        data.edge_index = new_ei
        data.slow_node_mask = slow_mask
        return data

    def __repr__(self):
        return (f"{self.__class__.__name__}(k={self.k}, alpha={self.alpha}, "
                f"p={self.p})")
