"""Learned per-edge mixing coefficient for AdaHop (Component A)."""

import torch
import torch.nn as nn


class EdgeAlpha(nn.Module):
    r"""Learned per-edge mixing coefficient α_e ∈ (0, 1).

    For each edge (src → dst), computes a scalar α_e from the concatenation
    of source and target node features:

        z = [x_src ‖ x_dst ‖ |x_src − x_dst| ‖ x_src ⊙ x_dst]
        α_e = σ(MLP(z))

    The slow-node feature is then:
        x_slow = (1 − α_e) · x_dst + α_e · x_src

    Args:
        in_dim (int): Input node feature dimension.
        hidden (int): Hidden size of the MLP. Default: ``32``.
        entropy_reg (float): Coefficient for entropy regularisation that
            pushes α_e toward 0 or 1 (binary gate). Set to 0 to disable.
            Default: ``0.0``.
    """

    def __init__(self, in_dim: int, hidden: int = 32, entropy_reg: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.entropy_reg = entropy_reg

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-edge α values.

        Args:
            x_src: Source node features, shape ``[E, in_dim]``.
            x_dst: Target node features, shape ``[E, in_dim]``.

        Returns:
            α_e: Per-edge mixing coefficients, shape ``[E, 1]``.
        """
        z = torch.cat(
            [x_src, x_dst, (x_src - x_dst).abs(), x_src * x_dst], dim=-1
        )
        return torch.sigmoid(self.mlp(z))

    def entropy_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """Binary-entropy regulariser to encourage α → 0 or 1.

        Args:
            alpha: Tensor of shape ``[E, 1]`` with values in (0, 1).

        Returns:
            Scalar loss (zero when ``entropy_reg == 0``).
        """
        if self.entropy_reg == 0.0:
            return alpha.new_zeros(1)
        eps = 1e-6
        a = alpha.clamp(eps, 1 - eps)
        h = -(a * a.log() + (1 - a) * (1 - a).log())
        return self.entropy_reg * h.mean()
