"""Edge-confidence MLP for CAHH.

Architecture (from the CAHH plan §3.3)::

    4 → Linear(32) → ReLU → Dropout(0.2) → Linear(16) → ReLU → Linear(1)

Output is a *logit* (unbounded scalar per edge).  At selection time apply
``torch.sigmoid`` to get a probability.  During training use
``BCEWithLogitsLoss`` directly on the logit — do **not** chain sigmoid +
BCELoss (numerical instability).

Example::

    model = EdgeConfidenceMLP(input_dim=4, dropout=0.2)
    logits = model(feats)                        # (E,)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets.float(), pos_weight=pos_w
    )
    scores = torch.sigmoid(logits)               # (E,) ∈ [0,1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EdgeConfidenceMLP(nn.Module):
    """Compact MLP that outputs one boundary-confidence logit per edge.

    Args:
        input_dim:  Number of edge features (default 4).
        hidden1:    Width of the first hidden layer (default 32).
        hidden2:    Width of the second hidden layer (default 16).
        dropout:    Dropout rate applied after the first hidden layer
                    (default 0.2).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden1: int = 32,
        hidden2: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, edge_features: Tensor) -> Tensor:
        """Return per-edge logits.

        Args:
            edge_features: ``(E, input_dim)`` float tensor.

        Returns:
            ``(E,)`` logit tensor (unbounded).
        """
        return self.network(edge_features).squeeze(-1)

    def predict_scores(self, edge_features: Tensor) -> Tensor:
        """Return sigmoid-transformed confidence scores ∈ [0, 1].

        Convenience wrapper for inference; not used during training.
        """
        with torch.no_grad():
            return torch.sigmoid(self(edge_features))

    def __repr__(self) -> str:
        sizes = [m for m in self.network if isinstance(m, nn.Linear)]
        dims = "→".join(
            str(m.in_features) for m in sizes
        ) + f"→{sizes[-1].out_features}"
        return f"{self.__class__.__name__}({dims})"

