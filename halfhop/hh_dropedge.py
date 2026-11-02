import copy
import torch
from torch import nn

from halfhop.dropedge import drop_edge
from halfhop.halfhop import HalfHop
from halfhop.gcn import GCN
from halfhop.graphsage import GraphSAGE
from halfhop.gat import GAT


class DropEdgeGCN(nn.Module):
    r"""GCN model with DropEdge augmentation applied during training."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        p_drop: float = 0.2,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        self.p_drop = p_drop
        self.gnn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            depth=depth,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if self.training and self.p_drop > 0.0:
            edge_index, _ = drop_edge(edge_index, p=self.p_drop, training=True)
        return self.gnn(x, edge_index)


class DropEdgeGraphSAGE(nn.Module):
    r"""GraphSAGE model with DropEdge augmentation applied during training."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        p_drop: float = 0.2,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        self.p_drop = p_drop
        self.gnn = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            depth=depth,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if self.training and self.p_drop > 0.0:
            edge_index, _ = drop_edge(edge_index, p=self.p_drop, training=True)
        return self.gnn(x, edge_index)


class HHDropEdgeGCN(nn.Module):
    r"""Combined Half-Hop + DropEdge GCN.

    Applies Half-Hop upsampling to the graph and applies DropEdge regularization
    during message passing in training.

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden dimension.
        out_channels (int): Output classes.
        p_drop (float): Probability of dropping edges during training.
        alpha (float): Half-Hop interpolation parameter. Default: 0.5.
        p_hh (float): Probability of half-hopping edges. Default: 1.0.
        dropout (float): Dropout probability. Default: 0.5.
        depth (int): Number of GCN layers. Default: 2.
        drop_mode (str): 'post_hh' (drop edges on Half-Hop graph) or
            'pre_hh' (drop original edges before Half-Hop). Default: 'post_hh'.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        p_drop: float = 0.2,
        alpha: float = 0.5,
        p_hh: float = 1.0,
        dropout: float = 0.5,
        depth: int = 2,
        inplace: bool = False,
        slow_node_init: str = 'linear',
        connectivity: str = 'proposed',
        drop_mode: str = 'post_hh',
    ):
        super().__init__()
        self.p_drop = p_drop
        self.drop_mode = drop_mode
        self.halfhop = HalfHop(
            alpha=alpha,
            p=p_hh,
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
        if self.drop_mode == 'pre_hh' and self.training and self.p_drop > 0.0:
            data = copy.copy(data)
            data.edge_index, _ = drop_edge(data.edge_index, p=self.p_drop, training=True)
            data = self.halfhop(data)
            x = self.gnn(data.x, data.edge_index)
            return x[~data.slow_node_mask]

        # Default post_hh: Half-Hop graph with edge dropout during training
        data = self.halfhop(data)
        edge_index = data.edge_index
        if self.training and self.p_drop > 0.0:
            edge_index, _ = drop_edge(edge_index, p=self.p_drop, training=True)

        x = self.gnn(data.x, edge_index)
        return x[~data.slow_node_mask]


class HHDropEdgeGraphSAGE(nn.Module):
    r"""Combined Half-Hop + DropEdge GraphSAGE."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        p_drop: float = 0.2,
        alpha: float = 0.5,
        p_hh: float = 1.0,
        dropout: float = 0.5,
        depth: int = 2,
        inplace: bool = False,
        slow_node_init: str = 'linear',
        connectivity: str = 'proposed',
        drop_mode: str = 'post_hh',
    ):
        super().__init__()
        self.p_drop = p_drop
        self.drop_mode = drop_mode
        self.halfhop = HalfHop(
            alpha=alpha,
            p=p_hh,
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
        if self.drop_mode == 'pre_hh' and self.training and self.p_drop > 0.0:
            data = copy.copy(data)
            data.edge_index, _ = drop_edge(data.edge_index, p=self.p_drop, training=True)
            data = self.halfhop(data)
            x = self.gnn(data.x, data.edge_index)
            return x[~data.slow_node_mask]

        data = self.halfhop(data)
        edge_index = data.edge_index
        if self.training and self.p_drop > 0.0:
            edge_index, _ = drop_edge(edge_index, p=self.p_drop, training=True)

        x = self.gnn(data.x, edge_index)
        return x[~data.slow_node_mask]

