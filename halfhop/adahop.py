"""AdaHop: Adaptive learnable Half-Hop GNN models.

Provides four model classes, one per component, plus a unified AdaHopGCN
that combines Components A and B together:

    AdaHopGCN     — Component A (learned α_e) + GCN
    AdaHopSAGE    — Component A (learned α_e) + GraphSAGE
    AdaHopGAT     — Component A (learned α_e) + GAT
    SoftAdaHopGCN — Component B (soft gate γ_e) + GCN  (no extra nodes)
    StatefulHHGCN — Component C (GRU slow nodes) + GCN
    MultiDelayGCN — Component D (k slow nodes)   + GCN
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from halfhop.gcn import GCN
from halfhop.graphsage import GraphSAGE
from halfhop.gat import GAT
from halfhop.edge_alpha import EdgeAlpha
from halfhop.soft_halfhop import SoftHalfHop
from halfhop.stateful_halfhop import StatefulHalfHop
from halfhop.multidelay_halfhop import MultiDelayHalfHop


# ---------------------------------------------------------------------------
# Helper: apply learned α_e to build a Half-Hop augmented graph
# ---------------------------------------------------------------------------

def _apply_learned_alpha(
    data,
    alpha_net: EdgeAlpha,
    p: float = 1.0,
    inplace: bool = False,
):
    """Apply Half-Hop with learned per-edge α_e from alpha_net.

    Args:
        data: PyG Data object.
        alpha_net: Trained EdgeAlpha module.
        p: Node-level sampling probability.
        inplace: Modify data in-place.

    Returns:
        Augmented data, entropy_loss scalar.
    """
    if not inplace:
        data = copy.deepcopy(data)

    x, edge_index = data.x, data.edge_index
    device = x.device
    N = x.size(0)

    # Strip self-loops
    self_loop_mask = edge_index[0] == edge_index[1]
    ei_self = edge_index[:, self_loop_mask]
    ei = edge_index[:, ~self_loop_mask]

    # Node-level sampling
    if p == 1.0:
        ei_hh = ei
        ei_keep = None
    else:
        node_mask = torch.rand(N, device=device) < p
        ei_hh = ei[:, node_mask[ei[1]]]
        ei_keep = ei[:, ~node_mask[ei[1]]]

    src, dst = ei_hh[0], ei_hh[1]
    K = src.numel()
    slow_ids = torch.arange(N, N + K, device=device, dtype=edge_index.dtype)

    # Learned α_e
    alpha_e = alpha_net(x[src], x[dst])              # [K, 1]
    x_slow = (1.0 - alpha_e) * x[dst] + alpha_e * x[src]  # [K, D]
    entropy_loss = alpha_net.entropy_loss(alpha_e)

    new_x = torch.cat([x, x_slow], dim=0)

    # Proposed connectivity: vi→ν, vj→ν, ν→vj
    ei_s2n = torch.stack([src, slow_ids])
    ei_d2n = torch.stack([dst, slow_ids])
    ei_n2d = torch.stack([slow_ids, dst])

    parts = [p_ for p_ in [ei_keep, ei_self, ei_s2n, ei_d2n, ei_n2d]
             if p_ is not None and p_.numel() > 0]
    new_ei = torch.cat(parts, dim=1)

    slow_mask = torch.zeros(N + K, dtype=torch.bool, device=device)
    slow_mask[N:] = True

    data.x = new_x
    data.edge_index = new_ei
    data.slow_node_mask = slow_mask
    return data, entropy_loss


# ---------------------------------------------------------------------------
# Component A models: Learned per-edge α_e
# ---------------------------------------------------------------------------

class AdaHopGCN(nn.Module):
    r"""AdaHop with learned α_e + GCN encoder (Component A).

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): GCN hidden size.
        out_channels (int): Number of output classes.
        alpha_hidden (int): Hidden size of the EdgeAlpha MLP. Default: ``32``.
        entropy_reg (float): Entropy regulariser on α_e. Default: ``0.0``.
        p (float): Node-level Half-Hop sampling probability. Default: ``1.0``.
        dropout (float): GCN dropout. Default: ``0.5``.
        depth (int): GCN depth. Default: ``2``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        alpha_hidden: int = 32,
        entropy_reg: float = 0.0,
        p: float = 1.0,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        self.alpha_net = EdgeAlpha(in_channels, hidden=alpha_hidden,
                                   entropy_reg=entropy_reg)
        self.gnn = GCN(in_channels, hidden_channels, out_channels,
                       dropout=dropout, depth=depth)
        self.p = p
        # Sentinel so training.py calls model(data) not model(x, ei)
        self.halfhop = True

    def forward(self, data):
        data, ent_loss = _apply_learned_alpha(
            data, self.alpha_net, p=self.p, inplace=False
        )
        x = self.gnn(data.x, data.edge_index)
        self._entropy_loss = ent_loss          # store for caller to add to loss
        return x[~data.slow_node_mask]


class AdaHopSAGE(nn.Module):
    r"""AdaHop with learned α_e + GraphSAGE encoder (Component A).

    Same interface as :class:`AdaHopGCN` but uses SAGEConv layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        alpha_hidden: int = 32,
        entropy_reg: float = 0.0,
        p: float = 1.0,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        self.alpha_net = EdgeAlpha(in_channels, hidden=alpha_hidden,
                                   entropy_reg=entropy_reg)
        self.gnn = GraphSAGE(in_channels, hidden_channels, out_channels,
                             dropout=dropout, depth=depth)
        self.p = p
        self.halfhop = True

    def forward(self, data):
        data, ent_loss = _apply_learned_alpha(
            data, self.alpha_net, p=self.p, inplace=False
        )
        x = self.gnn(data.x, data.edge_index)
        self._entropy_loss = ent_loss
        return x[~data.slow_node_mask]


class AdaHopGAT(nn.Module):
    r"""AdaHop with learned α_e + GAT encoder (Component A).

    Same interface as :class:`AdaHopGCN` but uses GATConv layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        alpha_hidden: int = 32,
        entropy_reg: float = 0.0,
        p: float = 1.0,
        dropout: float = 0.5,
        depth: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.alpha_net = EdgeAlpha(in_channels, hidden=alpha_hidden,
                                   entropy_reg=entropy_reg)
        self.gnn = GAT(in_channels, hidden_channels, out_channels,
                       dropout=dropout, depth=depth, num_heads=num_heads)
        self.p = p
        self.halfhop = True

    def forward(self, data):
        data, ent_loss = _apply_learned_alpha(
            data, self.alpha_net, p=self.p, inplace=False
        )
        x = self.gnn(data.x, data.edge_index)
        self._entropy_loss = ent_loss
        return x[~data.slow_node_mask]


# ---------------------------------------------------------------------------
# Component B model: Soft selective subdivision
# ---------------------------------------------------------------------------

class SoftAdaHopGCN(nn.Module):
    r"""AdaHop with soft gate γ_e + GCN encoder (Component B, variant B1).

    Uses :class:`~halfhop.soft_halfhop.SoftHalfHop`. No extra nodes are
    inserted — the soft weighting is applied via edge weights. The GCN must
    therefore support ``edge_weight`` (PyG's GCNConv does).

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): GCN hidden size.
        out_channels (int): Number of output classes.
        alpha (float): Fixed interpolation coefficient for slow-node features.
        gate_hidden (int): SoftGate MLP hidden size. Default: ``32``.
        gamma_init (float, optional): Initial gate value.
        dropout (float): GCN dropout. Default: ``0.5``.
        depth (int): GCN depth. Default: ``2``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        alpha: float = 0.5,
        gate_hidden: int = 32,
        gamma_init: float | None = None,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.soft_hh = SoftHalfHop(in_channels, alpha=alpha,
                                    hidden=gate_hidden, gamma_init=gamma_init)
        self.dropout = dropout
        if depth == 1:
            dims = [in_channels, out_channels]
        else:
            dims = [in_channels] + [hidden_channels] * (depth - 1) + [out_channels]
        self.convs = nn.ModuleList(
            GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.halfhop = True

    def forward(self, data):
        x, ei, ew, slow_mask, gamma = self.soft_hh(data.x, data.edge_index)
        self._gamma = gamma   # store for logging
        for i, conv in enumerate(self.convs):
            x = conv(x, ei, edge_weight=ew)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x[~slow_mask]


# ---------------------------------------------------------------------------
# Component C model: Stateful slow nodes
# ---------------------------------------------------------------------------

class StatefulHHGCN(nn.Module):
    r"""Half-Hop with GRU-based slow nodes + GCN encoder (Component C).

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): GCN hidden size.
        out_channels (int): Number of output classes.
        alpha (float): Initial slow-node interpolation. Default: ``0.5``.
        p (float): Node-level sampling probability. Default: ``1.0``.
        dropout (float): GCN dropout. Default: ``0.5``.
        depth (int): GCN depth. Default: ``2``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        alpha: float = 0.5,
        p: float = 1.0,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        self.stateful_hh = StatefulHalfHop(in_channels, alpha=alpha,
                                            p=p, inplace=False)
        self.gnn = GCN(in_channels, hidden_channels, out_channels,
                       dropout=dropout, depth=depth)
        self.halfhop = True

    def forward(self, data):
        data = self.stateful_hh(data)
        x = self.gnn(data.x, data.edge_index)
        return x[~data.slow_node_mask]


# ---------------------------------------------------------------------------
# Component D model: Variable delay (k slow nodes)
# ---------------------------------------------------------------------------

class MultiDelayGCN(nn.Module):
    r"""Half-Hop with k slow nodes per edge + GCN encoder (Component D).

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): GCN hidden size.
        out_channels (int): Number of output classes.
        k (int): Number of slow nodes per edge. Default: ``1``.
        alpha (float): Interpolation coefficient. Default: ``0.5``.
        p (float): Node-level sampling probability. Default: ``1.0``.
        dropout (float): GCN dropout. Default: ``0.5``.
        depth (int): GCN depth. Default: ``2``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        k: int = 1,
        alpha: float = 0.5,
        p: float = 1.0,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        self.multi_hh = MultiDelayHalfHop(k=k, alpha=alpha, p=p, inplace=False)
        self.gnn = GCN(in_channels, hidden_channels, out_channels,
                       dropout=dropout, depth=depth)
        self.halfhop = True

    def forward(self, data):
        data = self.multi_hh(data)
        x = self.gnn(data.x, data.edge_index)
        return x[~data.slow_node_mask]
