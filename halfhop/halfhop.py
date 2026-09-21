import copy

import torch
from torch_geometric.utils import subgraph


class HalfHop:
    r"""Graph upsampling augmentation. Adds artificial slow nodes between
    neighbors to slow down message propagation.

    For a directed edge ``vi → vj``, the proposed Half-Hop transformation
    introduces a new slow node ``νk`` as::

        vi → νk ↔ vj   (proposed connectivity)

    That is: ``vi → νk``, ``vj → νk``, ``νk → vj``.

    After message passing, use the :attr:`slow_node_mask` to discard slow
    node embeddings and keep only original-node outputs.

    Args:
        alpha (float): Interpolation factor for slow-node feature initialization.
            ``x_k = alpha * x_i + (1 - alpha) * x_j``. Default: ``0.5``.
        p (float): Probability of half-hopping each target node's incoming
            edges (node-level sampling). ``p=1`` applies Half-Hop to all
            non-self-loop edges deterministically. Default: ``1.0``.
        inplace (bool): If ``False``, operates on a deep copy of the input
            :class:`~torch_geometric.data.Data` object. Default: ``True``.
        slow_node_init (str): Feature initialization scheme for slow nodes.
            One of ``'linear'`` (default), ``'zero'``, or ``'random'``.
            ``'linear'`` uses ``alpha``-interpolation. ``'zero'`` sets all
            slow-node features to zero. ``'random'`` samples from
            ``Uniform(0, 1)``.
        connectivity (str): Edge connectivity scheme for ablations.
            One of:

            * ``'proposed'`` (default): ``vi → νk ↔ vj``
              (edges: ``vi→νk``, ``vj→νk``, ``νk→vj``)
            * ``'hh1'``: ``vi → νk → vj``
              (edges: ``vi→νk``, ``νk→vj``)
            * ``'hh2'``: ``vi ↔ νk ↔ vj``
              (edges: ``vi→νk``, ``νk→vi``, ``vj→νk``, ``νk→vj``)

    .. note::
        Use the :attr:`slow_node_mask` attribute to mask out the slow nodes
        after message passing::

            data = halfhop(data)
            out = model(data)
            out = out[~data.slow_node_mask]  # original nodes only
    """

    VALID_INIT = ('linear', 'zero', 'random')
    VALID_CONNECTIVITY = ('proposed', 'hh1', 'hh2')

    def __init__(
        self,
        alpha: float = 0.5,
        p: float = 1.0,
        inplace: bool = True,
        slow_node_init: str = 'linear',
        connectivity: str = 'proposed',
    ):
        assert 0.0 <= p <= 1.0, f"p must be in [0, 1], got {p}"
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
        assert slow_node_init in self.VALID_INIT, (
            f"slow_node_init must be one of {self.VALID_INIT}, "
            f"got '{slow_node_init}'"
        )
        assert connectivity in self.VALID_CONNECTIVITY, (
            f"connectivity must be one of {self.VALID_CONNECTIVITY}, "
            f"got '{connectivity}'"
        )

        self.alpha = alpha
        self.p = p
        self.inplace = inplace
        self.slow_node_init = slow_node_init
        self.connectivity = connectivity

    def __call__(self, data):
        if not self.inplace:
            data = copy.deepcopy(data)

        x, edge_index = data.x, data.edge_index
        device = x.device

        # ------------------------------------------------------------------
        # 1. Isolate self-loops — these are never half-hopped
        # ------------------------------------------------------------------
        self_loop_mask = edge_index[0] == edge_index[1]
        edge_index_self_loop = edge_index[:, self_loop_mask]
        edge_index = edge_index[:, ~self_loop_mask]

        # ------------------------------------------------------------------
        # 2. Decide which edges to half-hop (node-level target sampling)
        # ------------------------------------------------------------------
        if self.p == 1.0:
            # All non-self-loop edges are half-hopped (deterministic)
            edge_index_to_halfhop = edge_index
            edge_index_to_keep = None
        else:
            # Randomly select target nodes; half-hop their incoming edges.
            # Uses torch_geometric.utils.subgraph for efficient node-level
            # masking (matches official repository implementation exactly).
            node_mask = torch.rand(data.num_nodes, device=device) < self.p
            _, _, edge_mask = subgraph(
                node_mask,
                torch.stack([edge_index[1], edge_index[1]], dim=0),
                return_edge_mask=True,
            )
            edge_index_to_halfhop = edge_index[:, edge_mask]
            edge_index_to_keep = edge_index[:, ~edge_mask]

        # ------------------------------------------------------------------
        # 3. Assign slow-node IDs (consecutive, starting after original nodes)
        # ------------------------------------------------------------------
        num_slow = edge_index_to_halfhop.size(1)
        slow_node_ids = (
            torch.arange(num_slow, device=device, dtype=edge_index.dtype)
            + data.num_nodes
        )

        source = edge_index_to_halfhop[0]  # vi
        target = edge_index_to_halfhop[1]  # vj

        # ------------------------------------------------------------------
        # 4. Initialize slow-node features
        # ------------------------------------------------------------------
        if self.slow_node_init == 'linear':
            # x_k = alpha * x_i + (1 - alpha) * x_j  (matches paper & repo)
            x_slow = x[source].clone()
            x_slow.mul_(self.alpha).add_(x[target], alpha=1.0 - self.alpha)
        elif self.slow_node_init == 'zero':
            x_slow = torch.zeros(
                num_slow, x.size(1), dtype=x.dtype, device=device
            )
        else:  # 'random'
            x_slow = torch.rand(
                num_slow, x.size(1), dtype=x.dtype, device=device
            )

        new_x = torch.cat([x, x_slow], dim=0)

        # ------------------------------------------------------------------
        # 5. Build new edge set according to chosen connectivity scheme
        # ------------------------------------------------------------------
        # Shared edges (both proposed and ablation variants)
        edge_src_to_slow = torch.stack([source, slow_node_ids], dim=0)
        edge_slow_to_tgt = torch.stack([slow_node_ids, target], dim=0)

        if self.connectivity == 'proposed':
            # vi → νk, vj → νk, νk → vj
            edge_tgt_to_slow = torch.stack([target, slow_node_ids], dim=0)
            hh_edges = [edge_src_to_slow, edge_tgt_to_slow, edge_slow_to_tgt]

        elif self.connectivity == 'hh1':
            # vi → νk → vj  (no backward edge from target to slow)
            hh_edges = [edge_src_to_slow, edge_slow_to_tgt]

        else:  # 'hh2'
            # vi ↔ νk ↔ vj  (add extra slow → source edge)
            edge_tgt_to_slow = torch.stack([target, slow_node_ids], dim=0)
            edge_slow_to_src = torch.stack([slow_node_ids, source], dim=0)
            hh_edges = [
                edge_src_to_slow,
                edge_tgt_to_slow,
                edge_slow_to_tgt,
                edge_slow_to_src,
            ]

        # Combine: non-halfhopped edges + self-loops + new half-hop edges
        parts = [
            p for p in [edge_index_to_keep, edge_index_self_loop] + hh_edges
            if p is not None and p.numel() > 0
        ]
        new_edge_index = torch.cat(parts, dim=1)

        # ------------------------------------------------------------------
        # 6. Build slow-node mask
        # ------------------------------------------------------------------
        slow_node_mask = torch.cat([
            torch.zeros(x.size(0), dtype=torch.bool, device=device),
            torch.ones(num_slow, dtype=torch.bool, device=device),
        ], dim=0)

        # ------------------------------------------------------------------
        # 7. Update data object
        # ------------------------------------------------------------------
        data.x = new_x
        data.edge_index = new_edge_index
        data.slow_node_mask = slow_node_mask

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, p={self.p}, "
            f"init='{self.slow_node_init}', "
            f"connectivity='{self.connectivity}')"
        )