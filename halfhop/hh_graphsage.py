import torch
from torch import nn

from halfhop.halfhop import HalfHop
from halfhop.graphsage import GraphSAGE


class HHGraphSAGE(nn.Module):
    r"""Half-Hop augmented GraphSAGE.

    Applies :class:`~halfhop.halfhop.HalfHop` before running a
    :class:`~halfhop.graphsage.GraphSAGE` encoder. Slow-node embeddings
    are discarded after message passing.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): GraphSAGE hidden layer size.
        out_channels (int): Number of output classes/dimensions.
        alpha (float): Half-Hop interpolation factor. Default: ``0.5``.
        p (float): Half-Hop node-sampling probability. Default: ``1.0``.
        dropout (float): GraphSAGE dropout probability. Default: ``0.5``.
        depth (int): Number of SAGE layers. Default: ``2``.
        inplace (bool): Whether HalfHop modifies input in-place.
            Default: ``False``.
        slow_node_init (str): Slow-node feature initialization.
            One of ``'linear'``, ``'zero'``, ``'random'``. Default: ``'linear'``.
        connectivity (str): Edge connectivity scheme.
            One of ``'proposed'``, ``'hh1'``, ``'hh2'``. Default: ``'proposed'``.
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
        inplace: bool = False,
        slow_node_init: str = 'linear',
        connectivity: str = 'proposed',
    ):
        super().__init__()
        self.halfhop = HalfHop(
            alpha=alpha,
            p=p,
            inplace=inplace,
            slow_node_init=slow_node_init,
            connectivity=connectivity,
        )
        self.gnn = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            depth=depth,
        )

    def forward(self, data):
        r"""Forward pass: HalfHop → GraphSAGE → remove slow nodes."""
        data = self.halfhop(data)
        x = self.gnn(data.x, data.edge_index)
        return x[~data.slow_node_mask]
