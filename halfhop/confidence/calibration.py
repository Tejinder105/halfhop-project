"""Optional Platt-scaling calibration for CAHH confidence scores.

Calibration is fit on **validation shapes only**, using an existing trained
:class:`~halfhop.confidence.confidence_mlp.EdgeConfidenceMLP`.  The calibrated
model outputs probability estimates that better match empirical boundary
frequency, which slightly improves the precision@budget metric.

Calibration is optional (the CAHH plan §3.3 notes it).  If skipped, raw
sigmoid scores are used for edge selection — the ranking is unaffected by a
monotone calibration, so selection quality is unchanged.

Usage::

    calibrator = PlattCalibrator()
    calibrator.fit(val_logits, val_targets)
    calibrated_scores = calibrator.transform(test_logits)
"""

from __future__ import annotations

import torch
from torch import Tensor


class PlattCalibrator:
    """Temperature + bias scaling fitted on validation logits.

    Finds scalar (temperature T, bias b) that minimises NLL on the
    validation set::

        P_calibrated = sigmoid((logit + b) / T)

    Only valid (non-self-loop) edges should enter ``fit``.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.bias: float = 0.0
        self._fitted = False

    def fit(
        self,
        logits: Tensor,
        targets: Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> "PlattCalibrator":
        """Fit temperature and bias using LBFGS on the validation set.

        Args:
            logits:  ``(E,)`` raw MLP logits.
            targets: ``(E,)`` float binary targets.

        Returns:
            self (for chaining).
        """
        logits = logits.float().detach()
        targets = targets.float().detach()

        T = torch.nn.Parameter(torch.ones(1))
        b = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([T, b], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = (logits + b) / T.clamp(min=1e-4)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled, targets
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(T.item())
        self.bias = float(b.item())
        self._fitted = True
        return self

    def transform(self, logits: Tensor) -> Tensor:
        """Return calibrated probability scores ∈ [0, 1].

        Args:
            logits: ``(E,)`` raw MLP logits.

        Returns:
            ``(E,)`` calibrated probabilities.
        """
        if not self._fitted:
            # If not fitted, fall back to raw sigmoid.
            return torch.sigmoid(logits.float())
        scaled = (logits.float() + self.bias) / max(self.temperature, 1e-4)
        return torch.sigmoid(scaled)

    def state_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "bias": self.bias,
            "fitted": self._fitted,
        }

    def load_state_dict(self, d: dict) -> None:
        self.temperature = d["temperature"]
        self.bias = d["bias"]
        self._fitted = d["fitted"]

    def __repr__(self) -> str:
        if self._fitted:
            return (
                f"PlattCalibrator(T={self.temperature:.4f}, "
                f"b={self.bias:.4f})"
            )
        return "PlattCalibrator(unfitted)"

