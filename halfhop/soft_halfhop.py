"""Soft selective Half-Hop transform (Component B — variant B1).

Instead of inserting slow nodes into the graph, keeps both the direct edge
(weight 1 − γ_e) and the slow route through a virtual node (weight γ_e)
using PyG's edge_weight support. When γ_e → 1 this approaches standard
Half-Hop; when γ_e → 0 this reduces to the original graph.

This variant is fully differentiable and adds no extra nodes.
"""

import math
import torch
import torch.nn as nn


class SoftGate(nn.Module):
    r"""Learned per-edge subdivision gate γ_e ∈ (0, 1).

    Args:
        in_dim (int): Input node feature dimension.
        hidden (int): Hidden MLP size. Default: ``32``.
    """

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        """Compute per-edge gate values γ_e.

        Args:
            x_src: ``[E, in_dim]``
            x_dst: ``[E, in_dim]``

        Returns:
            γ_e: ``[E, 1]`` values in (0, 1).
        """
        z = torch.cat([x_src, x_dst, (x_src - x_dst).abs()], dim=-1)
        return torch.sigmoid(self.mlp(z))


class SoftHalfHop(nn.Module):
    r"""Soft selective Half-Hop graph augmentation (Component B, variant B1).

    For each edge (i → j):
    - Keeps the direct edge with weight ``(1 − γ_e)``
    - Adds edges through a slow interpolation node with weight ``γ_e``

    When ``gamma_init`` is provided all gates are initialised to that value
    (useful to start near plain Half-Hop with gamma=1 or plain GCN with gamma=0).

    Args:
        in_dim (int): Input node feature dimension.
        alpha (float): Fixed interpolation coefficient for slow-node features.
            ``x_slow = (1 − alpha) · x_dst + alpha · x_src``. Default: ``0.5``.
        hidden (int): Hidden size of the gate MLP. Default: ``32``.
        gamma_init (float, optional): If given, bias the gate output toward
            this value at initialisation by shifting the MLP's final bias.
    """

    def __init__(
        self,
        in_dim: int,
        alpha: float = 0.5,
        hidden: int = 32,
        gamma_init: float | None = None,
    ):
        super().__init__()
        self.gate = SoftGate(in_dim, hidden)
        self.alpha = alpha

        if gamma_init is not None:
            bias_val = math.log(gamma_init / (1 - gamma_init + 1e-8))
            with torch.no_grad():
                self.gate.mlp[-1].bias.fill_(bias_val)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Compute soft-weighted augmented graph.

        Args:
            x: Node features ``[N, in_dim]``.
            edge_index: Graph connectivity ``[2, E]``.

        Returns:
            Tuple:
                - ``x_aug``: Augmented node features ``[N + E_sel, in_dim]``
                  (slow nodes appended).
                - ``ei_aug``: Augmented edge index ``[2, E_aug]``.
                - ``ew_aug``: Edge weights ``[E_aug]``.
                - ``slow_mask``: Bool mask ``[N + E_sel]``, True for slow nodes.
                - ``gamma``: Gate values ``[E, 1]`` (for regularisation or logging).
        """
        N = x.size(0)
        src, dst = edge_index

        # --- gate ---
        gamma = self.gate(x[src], x[dst])          # [E, 1]
        gamma_flat = gamma.squeeze(1)               # [E]

        # --- slow-node features ---
        x_slow = (1.0 - self.alpha) * x[dst] + self.alpha * x[src]  # [E, D]

        # Append slow nodes after original nodes
        x_aug = torch.cat([x, x_slow], dim=0)      # [N+E, D]
        slow_ids = torch.arange(N, N + src.size(0), device=x.device)

        slow_mask = torch.zeros(N + src.size(0), dtype=torch.bool, device=x.device)
        slow_mask[N:] = True

        # --- edges ---
        # Direct edge:  i → j   weight (1 - γ_e)
        ei_direct = edge_index                      # [2, E]
        ew_direct = 1.0 - gamma_flat               # [E]

        # Slow route:   i → ν_e  weight γ_e
        ei_src2slow = torch.stack([src, slow_ids])  # [2, E]
        ew_src2slow = gamma_flat                    # [E]

        # Slow route:   ν_e → j  weight γ_e
        ei_slow2dst = torch.stack([slow_ids, dst])  # [2, E]
        ew_slow2dst = gamma_flat                    # [E]

        # Also keep ν_e ← j (target → slow) as in proposed connectivity
        ei_dst2slow = torch.stack([dst, slow_ids])  # [2, E]
        ew_dst2slow = gamma_flat                    # [E]

        ei_aug = torch.cat(
            [ei_direct, ei_src2slow, ei_slow2dst, ei_dst2slow], dim=1
        )
        ew_aug = torch.cat(
            [ew_direct, ew_src2slow, ew_slow2dst, ew_dst2slow], dim=0
        )

        return x_aug, ei_aug, ew_aug, slow_mask, gamma
