"""Integration test: CAHH end-to-end without ShapeNet data.

Verifies that the full pipeline (edge features → normaliser → MLP → selector
→ HalfHop) runs without errors on synthetic point clouds.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)
from halfhop.confidence.confidence_mlp import EdgeConfidenceMLP
from halfhop.confidence.adaptive_selector import AdaptiveSelector
from halfhop.halfhop import HalfHop


def _make_cloud(n: int = 128, k: int = 16, seed: int = 0) -> Data:
    """Synthetic point cloud with kNN graph and random part labels."""
    torch.manual_seed(seed)
    pos = torch.randn(n, 3)
    normals = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    y = torch.randint(0, 4, (n,))

    # Naive kNN
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))
    nbrs = dist.topk(k, largest=False).indices.reshape(-1)
    tgt = torch.arange(n).repeat_interleave(k)
    edge_index = torch.stack([nbrs, tgt], dim=0)

    return Data(pos=pos, x=normals, y=y, edge_index=edge_index)


class TestCAHHIntegration:
    def test_pipeline_no_crash(self):
        """Full pipeline runs without errors."""
        data = _make_cloud(64, k=8)

        # 1. Edge features.
        density = compute_local_density(data.pos, data.edge_index)
        feats, valid = build_edge_features(
            data.pos, data.edge_index, normals=data.x, density=density
        )
        assert feats.shape[1] == 4

        # 2. Normaliser (fit + transform on the same data for simplicity).
        norm = EdgeFeatureNormalizer()
        norm.update(feats[valid])
        norm.finalise()
        feats_norm = norm.transform(feats)

        # 3. MLP inference.
        mlp = EdgeConfidenceMLP(input_dim=4)
        mlp.eval()
        with torch.no_grad():
            logits = mlp(feats_norm)
        arc_scores = torch.sigmoid(logits)

        # 4. Selector → pair scores.
        sel = AdaptiveSelector(budget=0.1)
        pair_scores = sel.score_for_halfhop(arc_scores, data.edge_index)
        assert pair_scores.shape == (data.edge_index.size(1),)

        # 5. HalfHop with score-based selection.
        data2 = data.clone()
        data2.x = data2.pos.clone()
        data2.edge_score = pair_scores
        hh = HalfHop(selection="score", budget=0.1, score_mode="high", inplace=False)
        data_hh = hh(data2)

        # Original nodes plus slow nodes.
        assert data_hh.x.size(0) > data2.x.size(0)
        assert hasattr(data_hh, "slow_node_mask")
        assert data_hh.slow_node_mask.sum() > 0

    def test_pipeline_zero_budget(self):
        """Budget=0 → no slow nodes added."""
        data = _make_cloud(32, k=4)
        data_x = data.clone()
        data_x.x = data_x.pos.clone()
        # Provide dummy edge_score.
        data_x.edge_score = torch.rand(data_x.edge_index.size(1))
        hh = HalfHop(selection="score", budget=0.0, inplace=False)
        data_hh = hh(data_x)
        # No slow nodes at budget=0.
        assert data_hh.slow_node_mask.sum() == 0

    def test_pipeline_full_budget(self):
        """Budget=1.0 → all non-self-loop edges half-hopped."""
        data = _make_cloud(32, k=4)
        data_x = data.clone()
        data_x.x = data_x.pos.clone()
        data_x.edge_score = torch.rand(data_x.edge_index.size(1))
        hh = HalfHop(selection="score", budget=1.0, inplace=False)
        data_hh = hh(data_x)
        n_valid = int((data_x.edge_index[0] != data_x.edge_index[1]).sum())
        assert data_hh.slow_node_mask.sum() == n_valid

    def test_mlp_training_step(self):
        """Single gradient step should not raise."""
        data = _make_cloud(64, k=8)
        density = compute_local_density(data.pos, data.edge_index)
        feats, valid = build_edge_features(
            data.pos, data.edge_index, normals=data.x, density=density
        )
        norm = EdgeFeatureNormalizer()
        norm.update(feats[valid])
        norm.finalise()
        feats_norm = norm.transform(feats)

        # Binary targets
        src, dst = data.edge_index
        targets = (data.y[src] != data.y[dst]).float()

        mlp = EdgeConfidenceMLP(input_dim=4)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        mlp.train()
        logits = mlp(feats_norm[valid])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets[valid]
        )
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

