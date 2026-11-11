"""Tests for geometry_boundary and BoundaryHalfHop modules."""
import torch
import pytest
from torch_geometric.data import Data

from halfhop.geometry_boundary import (
    inter_part_boundary_edges,
    surface_variation,
    normal_disagreement,
    geometry_boundary_score,
    boundary_edge_mask,
    compute_normals,
)
from halfhop.boundary_halfhop import BoundaryHalfHop


# ── geometry_boundary ────────────────────────────────────────────────────────

def test_inter_part_boundary_edges_known():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    labels = torch.tensor([0, 0, 1, 1])
    mask = inter_part_boundary_edges(edge_index, labels)
    # edge 1->2 and 3->0 cross parts; 0->1 and 2->3 don't
    assert mask.tolist() == [False, True, False, True]


def test_inter_part_boundary_edges_all_same():
    edge_index = torch.randint(0, 10, (2, 40))
    labels = torch.zeros(10, dtype=torch.long)
    mask = inter_part_boundary_edges(edge_index, labels)
    assert not mask.any()


def test_surface_variation_range():
    pos = torch.randn(30, 3)
    curv = surface_variation(pos, knn=8)
    assert curv.shape == (30,)
    assert (curv >= 0).all() and (curv <= 1).all()


def test_normal_disagreement_parallel_and_perp():
    normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]])
    score = normal_disagreement(normals, edge_index)
    assert torch.allclose(score, torch.tensor([0.0, 1.0]), atol=1e-5)


def test_geometry_boundary_score_shape():
    pos = torch.randn(20, 3)
    edge_index = torch.randint(0, 20, (2, 60))
    score = geometry_boundary_score(pos, edge_index)
    assert score.shape == (60,)
    assert (score >= 0).all() and (score <= 1).all()


def test_boundary_edge_mask_fraction():
    pos = torch.randn(30, 3)
    edge_index = torch.randint(0, 30, (2, 100))
    mask = boundary_edge_mask(pos, edge_index, quantile=0.85, min_score=0.0)
    assert mask.shape == (100,)
    # roughly 15% should be True
    assert 5 <= mask.sum().item() <= 30


def test_boundary_edge_mask_min_score_suppresses_flat():
    # Pure flat plane with identical normals
    pos = torch.zeros(30, 3)
    pos[:, 0] = torch.linspace(-1, 1, 30)
    nrm = torch.zeros(30, 3)
    nrm[:, 2] = 1.0
    ei = torch.stack([torch.arange(29), torch.arange(1, 30)])
    # min_score=0.15 should produce 0 boundary edges even at low quantile
    mask = boundary_edge_mask(pos, ei, normals=nrm, quantile=0.5, min_score=0.15)
    assert mask.sum().item() == 0


def test_compute_normals_unit_length():
    pos = torch.randn(20, 3)
    normals = compute_normals(pos, knn=8)
    assert normals.shape == (20, 3)
    norms = normals.norm(dim=1)
    assert torch.allclose(norms, torch.ones(20), atol=1e-5)


# ── BoundaryHalfHop ──────────────────────────────────────────────────────────

def test_no_slow_nodes_when_all_same_part():
    pos = torch.randn(30, 3)
    y = torch.zeros(30, dtype=torch.long)
    data = Data(pos=pos, y=y, num_nodes=30)
    out = BoundaryHalfHop(knn=5, use_labels=True)(data)
    assert out.slow_node_mask.sum().item() == 0


def test_slow_nodes_only_on_boundaries_gt():
    pos = torch.randn(50, 3)
    y = torch.randint(0, 4, (50,))
    data = Data(pos=pos, y=y, num_nodes=50)
    out = BoundaryHalfHop(knn=5, use_labels=True)(data)
    assert out.slow_node_mask.sum().item() > 0
    assert out.x.size(0) == out.num_nodes
    assert out.slow_node_mask.size(0) == out.num_nodes


def test_slow_nodes_proxy_inference():
    pos = torch.randn(40, 3)
    data = Data(pos=pos, num_nodes=40)
    out = BoundaryHalfHop(knn=5, use_labels=False, curv_quantile=0.85)(data)
    assert out.slow_node_mask.size(0) == out.num_nodes


def test_original_node_count_preserved():
    n = 60
    pos = torch.randn(n, 3)
    y = torch.randint(0, 3, (n,))
    data = Data(pos=pos, y=y, num_nodes=n)
    out = BoundaryHalfHop(knn=8, use_labels=True)(data)
    n_orig = (~out.slow_node_mask).sum().item()
    assert n_orig == n


def test_feature_dim_preserved():
    n = 50
    pos = torch.randn(n, 3)
    normal = torch.randn(n, 3)
    y = torch.randint(0, 4, (n,))
    data = Data(pos=pos, normal=normal, y=y, num_nodes=n)
    out = BoundaryHalfHop(knn=5, use_labels=True)(data)
    assert out.x.size(1) == 6  # pos + normal

