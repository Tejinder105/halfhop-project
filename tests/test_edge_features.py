"""Tests for halfhop/confidence/edge_features.py."""

import pytest
import torch
from torch_geometric.data import Data

from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ring(n: int):
    """n points on a unit circle in the xy-plane, ring kNN graph."""
    theta = torch.linspace(0, 2 * torch.pi, n + 1)[:n]
    pos = torch.stack([theta.cos(), theta.sin(), torch.zeros(n)], dim=1)
    # fully connected for simplicity
    src = torch.arange(n).repeat_interleave(n)
    dst = torch.arange(n).repeat(n)
    keep = src != dst
    edge_index = torch.stack([src[keep], dst[keep]], dim=0)
    return pos, edge_index


# ---------------------------------------------------------------------------
# compute_local_density
# ---------------------------------------------------------------------------

class TestLocalDensity:
    def test_shape(self):
        pos, ei = _ring(10)
        d = compute_local_density(pos, ei)
        assert d.shape == (10,)

    def test_non_negative(self):
        pos, ei = _ring(20)
        d = compute_local_density(pos, ei)
        assert (d >= 0).all()

    def test_no_self_loops_influence(self):
        """Adding self-loops should not change density."""
        pos, ei = _ring(8)
        self_loops = torch.arange(8).unsqueeze(0).repeat(2, 1)
        ei_with_sl = torch.cat([ei, self_loops], dim=1)
        d1 = compute_local_density(pos, ei)
        d2 = compute_local_density(pos, ei_with_sl)
        assert torch.allclose(d1, d2, atol=1e-5)

    def test_empty_valid_edges(self):
        """Only self-loops → density should be all zeros."""
        pos = torch.randn(5, 3)
        ei = torch.arange(5).unsqueeze(0).repeat(2, 1)
        d = compute_local_density(pos, ei)
        assert d.shape == (5,)
        # all zeros because no valid (non-self-loop) edges exist
        assert (d == 0).all()


# ---------------------------------------------------------------------------
# build_edge_features
# ---------------------------------------------------------------------------

class TestBuildEdgeFeatures:
    def test_output_shape(self):
        pos, ei = _ring(16)
        E = ei.size(1)
        feats, valid = build_edge_features(pos, ei)
        assert feats.shape == (E, 4)
        assert valid.shape == (E,)
        assert valid.dtype == torch.bool

    def test_self_loops_invalid(self):
        pos, ei = _ring(8)
        self_loops = torch.arange(8).unsqueeze(0).repeat(2, 1)
        ei_sl = torch.cat([ei, self_loops], dim=1)
        feats, valid = build_edge_features(pos, ei_sl)
        # self-loop positions
        sl_start = ei.size(1)
        assert (~valid[sl_start:]).all(), "Self-loops should be invalid"
        assert valid[:sl_start].all(), "Non-self-loop edges should be valid"

    def test_self_loop_features_zero(self):
        pos, ei = _ring(8)
        self_loops = torch.arange(8).unsqueeze(0).repeat(2, 1)
        ei_sl = torch.cat([ei, self_loops], dim=1)
        feats, valid = build_edge_features(pos, ei_sl)
        sl_feats = feats[ei_sl.size(1) - 8:]
        assert (sl_feats == 0).all()

    def test_no_normals_sentinel(self):
        pos, ei = _ring(10)
        feats, valid = build_edge_features(pos, ei, normals=None)
        assert (feats[valid, 0] == -1.0).all(), "Normal channel should be -1 sentinel"

    def test_with_normals_range(self):
        pos, ei = _ring(10)
        normals = torch.randn(10, 3)
        feats, valid = build_edge_features(pos, ei, normals=normals)
        # normal disagreement ∈ [0, 1]
        nd = feats[valid, 0]
        assert (nd >= -1e-5).all()
        assert (nd <= 1.0 + 1e-5).all()

    def test_edge_length_non_negative(self):
        pos, ei = _ring(12)
        feats, valid = build_edge_features(pos, ei)
        assert (feats[valid, 2] >= 0).all()

    def test_no_nan(self):
        pos, ei = _ring(16)
        normals = torch.randn(16, 3)
        feats, valid = build_edge_features(pos, ei, normals=normals)
        assert not torch.isnan(feats).any()

    def test_curvature_provided(self):
        pos, ei = _ring(10)
        kappa = torch.rand(10)
        feats, valid = build_edge_features(pos, ei, curvature=kappa)
        # curvature diff should be non-negative for valid edges
        assert (feats[valid, 1] >= 0).all()

    def test_symmetry_undirected(self):
        """For undirected features (length, density_diff) both arcs should agree."""
        pos = torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # both arcs
        feats, valid = build_edge_features(pos, ei)
        # Edge length should be identical for both arcs.
        assert abs(feats[0, 2].item() - feats[1, 2].item()) < 1e-5


# ---------------------------------------------------------------------------
# EdgeFeatureNormalizer
# ---------------------------------------------------------------------------

class TestEdgeFeatureNormalizer:
    def test_fit_transform(self):
        torch.manual_seed(0)
        x = torch.randn(1000, 4)
        norm = EdgeFeatureNormalizer()
        norm.update(x)
        norm.finalise()
        x_t = norm.transform(x)
        # After normalising, should be roughly standard normal (clipped at 3σ)
        assert x_t.abs().max() <= 3.0 + 1e-4

    def test_not_fitted_raises(self):
        norm = EdgeFeatureNormalizer()
        with pytest.raises(RuntimeError):
            norm.transform(torch.zeros(10, 4))

    def test_no_data_raises(self):
        norm = EdgeFeatureNormalizer()
        with pytest.raises(RuntimeError):
            norm.finalise()

    def test_state_dict_roundtrip(self):
        torch.manual_seed(1)
        x = torch.randn(200, 4)
        norm = EdgeFeatureNormalizer()
        norm.update(x)
        norm.finalise()
        sd = norm.state_dict()
        norm2 = EdgeFeatureNormalizer()
        norm2.load_state_dict(sd)
        x_t1 = norm.transform(x)
        x_t2 = norm2.transform(x)
        assert torch.allclose(x_t1, x_t2)

    def test_multiple_updates(self):
        """Accumulating in chunks should equal one-shot fit."""
        torch.manual_seed(2)
        x = torch.randn(300, 4)
        norm_full = EdgeFeatureNormalizer()
        norm_full.update(x)
        norm_full.finalise()

        norm_chunk = EdgeFeatureNormalizer()
        for i in range(3):
            norm_chunk.update(x[i * 100:(i + 1) * 100])
        norm_chunk.finalise()

        assert torch.allclose(norm_full.mean_, norm_chunk.mean_, atol=1e-5)
        assert torch.allclose(norm_full.std_, norm_chunk.std_, atol=1e-5)

    def test_empty_update_ignored(self):
        norm = EdgeFeatureNormalizer()
        norm.update(torch.zeros(0, 4))  # should not crash
        norm.update(torch.randn(50, 4))
        norm.finalise()
        assert norm._fitted

