import torch
import torch.nn.functional as F
from torch import nn

from halfhop.halfhop import HalfHop
from halfhop.gcn import GCN


class HHGCN(nn.Module):
    r"""Half-Hop augmented GCN.

    Applies the :class:`~halfhop.halfhop.HalfHop` transformation to the input
    graph before running a :class:`~halfhop.gcn.GCN` encoder. Slow-node
    embeddings are discarded after message passing so outputs have exactly
    one row per original node.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): GCN hidden layer size.
        out_channels (int): Number of output classes/dimensions.
        alpha (float): Half-Hop interpolation factor. Default: ``0.5``.
        p (float): Half-Hop node-sampling probability. Default: ``1.0``.
        dropout (float): GCN dropout probability. Default: ``0.5``.
        depth (int): Number of GCN layers. Default: ``2``.
        inplace (bool): Whether HalfHop modifies the input data in-place.
            **Should be** ``False`` **when the same data object is reused
            across training steps.** Default: ``False``.
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
        self.gnn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            depth=depth,
        )

    def forward(self, data):
        r"""Forward pass: HalfHop → GCN → remove slow nodes.

        Args:
            data (:class:`~torch_geometric.data.Data`): Input graph with
                ``data.x`` and ``data.edge_index``.

        Returns:
            torch.Tensor: Node logits of shape ``[N_original, out_channels]``.
        """
        data = self.halfhop(data)
        x = self.gnn(data.x, data.edge_index)
        return x[~data.slow_node_mask]