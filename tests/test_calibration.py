"""Tests for halfhop/confidence/calibration.py."""

import pytest
import torch

from halfhop.confidence.calibration import PlattCalibrator


class TestPlattCalibrator:
    def test_unfitted_behavior(self):
        calibrator = PlattCalibrator()
        logits = torch.randn(20)
        # Unfitted should fallback directly to raw sigmoid
        probs = calibrator.transform(logits)
        assert torch.allclose(probs, torch.sigmoid(logits))

    def test_fit_and_transform(self):
        calibrator = PlattCalibrator()
        logits = torch.randn(200)
        # Synthetic targets correlated with logits
        targets = (torch.sigmoid(logits) > 0.5).float()
        calibrator.fit(logits, targets)
        assert calibrator._fitted
        probs = calibrator.transform(logits)
        assert probs.shape == logits.shape
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_state_dict_roundtrip(self):
        calibrator = PlattCalibrator()
        logits = torch.randn(100)
        targets = (torch.rand(100) > 0.5).float()
        calibrator.fit(logits, targets)
        
        state = calibrator.state_dict()
        calibrator2 = PlattCalibrator()
        calibrator2.load_state_dict(state)
        
        assert calibrator2._fitted
        assert calibrator2.temperature == calibrator.temperature
        assert calibrator2.bias == calibrator.bias
        
        p1 = calibrator.transform(logits)
        p2 = calibrator2.transform(logits)
        assert torch.allclose(p1, p2)

