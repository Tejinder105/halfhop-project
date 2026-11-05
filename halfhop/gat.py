import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    r"""Graph Attention Network (Veličković et al., 2017).

    Implements a variable-depth GAT using PyG's
    :class:`~torch_geometric.nn.GATConv`.
    Architecture: [GATConv + ELU + Dropout] × (depth-1) → GATConv.

    Intermediate layers use ``num_heads`` attention heads with concatenation,
    so the hidden dimension is multiplied by ``num_heads`` between layers.
    The final layer uses a single head for classification.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Size of hidden representation per head.
        out_channels (int): Number of output classes/dimensions.
        dropout (float): Dropout probability applied after each hidden layer
            and inside attention. Default: ``0.5``.
        depth (int): Total number of GAT layers (must be ≥ 1). Default: ``2``.
        num_heads (int): Number of attention heads in hidden layers.
            Default: ``8``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        depth: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        assert depth >= 1, f"depth must be at least 1, got {depth}"

        self.dropout = dropout

        self.convs = nn.ModuleList()

        if depth == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, dropout=dropout)
            )
        else:
            # First layer: in_channels → hidden_channels * num_heads (concat)
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=num_heads,
                        dropout=dropout, concat=True)
            )
            # Middle layers: hidden_channels*num_heads → hidden_channels*num_heads
            for _ in range(depth - 2):
                self.convs.append(
                    GATConv(hidden_channels * num_heads, hidden_channels,
                            heads=num_heads, dropout=dropout, concat=True)
                )
            # Last layer: hidden_channels*num_heads → out_channels (mean)
            self.convs.append(
                GATConv(hidden_channels * num_heads, out_channels,
                        heads=1, dropout=dropout, concat=False)
            )

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
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
