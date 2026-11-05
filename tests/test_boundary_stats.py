import torch

from experiments.segmentation.proxy import edge_boundary_stats, normal_disagreement


def test_top_edges_are_the_boundaries_and_bottom_edges_are_not():
    score = torch.tensor([0.0, 0.1, 0.2, 0.9, 0.8])
    disagree = torch.tensor([False, False, False, True, True])
    stats = edge_boundary_stats(score, disagree, fraction=0.4)
    assert stats["k"] == 2
    assert stats["top"] == 1.0
    assert stats["bottom"] == 0.0
    assert stats["base"] == 0.4
    assert stats["corr"] > 0.9


def test_flat_score_has_no_correlation():
    score = torch.ones(6)
    disagree = torch.tensor([True, False, True, False, True, False])
    stats = edge_boundary_stats(score, disagree, fraction=0.5)
    assert stats["corr"] == 0.0
    assert stats["top"] == stats["base"]


def test_normal_disagreement_is_zero_for_parallel_and_one_for_perpendicular():
    normals = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 0], [1, 2]])
    score, src, dst = normal_disagreement(normals, edge_index)
    assert src.tolist() == [0, 0]
    assert dst.tolist() == [1, 2]
    assert torch.allclose(score, torch.tensor([0.0, 1.0]))
