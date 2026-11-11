"""Tests for halfhop/confidence/confidence_mlp.py."""

import pytest
import torch

from halfhop.confidence.confidence_mlp import EdgeConfidenceMLP


class TestEdgeConfidenceMLP:
    def test_output_shape(self):
        model = EdgeConfidenceMLP()
        x = torch.randn(100, 4)
        out = model(x)
        assert out.shape == (100,)

    def test_output_unbounded(self):
        """Logits should not be squashed to [0,1]."""
        model = EdgeConfidenceMLP()
        x = torch.randn(200, 4) * 10  # large inputs
        out = model(x)
        # At least some logits should exceed 1 or be below 0.
        assert (out.abs() > 0.0).any()

    def test_predict_scores_range(self):
        model = EdgeConfidenceMLP()
        x = torch.randn(50, 4)
        s = model.predict_scores(x)
        assert s.shape == (50,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_no_gradient_in_predict_scores(self):
        model = EdgeConfidenceMLP()
        x = torch.randn(10, 4, requires_grad=True)
        s = model.predict_scores(x)
        # predict_scores uses no_grad, so no grad_fn on output
        assert s.grad_fn is None

    def test_custom_dims(self):
        model = EdgeConfidenceMLP(input_dim=3, hidden1=16, hidden2=8)
        x = torch.randn(5, 3)
        out = model(x)
        assert out.shape == (5,)

    def test_bce_loss_with_logits(self):
        """Training step should not raise."""
        model = EdgeConfidenceMLP()
        x = torch.randn(64, 4)
        y = (torch.rand(64) > 0.7).float()
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        assert loss.item() > 0

    def test_dropout_train_vs_eval(self):
        """Output should differ (stochastically) in train vs eval mode."""
        torch.manual_seed(42)
        model = EdgeConfidenceMLP(dropout=0.5)
        x = torch.randn(200, 4)
        model.train()
        out_train_1 = model(x).detach()
        out_train_2 = model(x).detach()
        # With dropout=0.5 on 200 samples, expect some variation.
        assert not torch.allclose(out_train_1, out_train_2)

        model.eval()
        out_eval_1 = model(x).detach()
        out_eval_2 = model(x).detach()
        assert torch.allclose(out_eval_1, out_eval_2)

    def test_repr(self):
        model = EdgeConfidenceMLP()
        r = repr(model)
        assert "EdgeConfidenceMLP" in r

