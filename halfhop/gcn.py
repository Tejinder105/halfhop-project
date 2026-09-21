import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    r"""Graph Convolutional Network (Kipf & Welling, 2016).

    Implements a variable-depth GCN using PyG's :class:`~torch_geometric.nn.GCNConv`.
    Architecture: [GCNConv + ReLU + Dropout] × (depth-1) → GCNConv.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Size of hidden layer(s).
        out_channels (int): Number of output classes/dimensions.
        dropout (float): Dropout probability applied after each hidden layer.
            Default: ``0.5``.
        depth (int): Total number of GCN layers (must be ≥ 1). Default: ``2``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        depth: int = 2,
    ):
        super().__init__()
        assert depth >= 1, f"depth must be at least 1, got {depth}"

        self.dropout = dropout

        self.convs = nn.ModuleList()

        if depth == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(depth - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature matrix of shape ``[N, in_channels]``.
            edge_index (torch.Tensor): Graph connectivity of shape ``[2, E]``.

        Returns:
            torch.Tensor: Node logits of shape ``[N, out_channels]``.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x