import torch

from experiments.modelnet.adaptive_halfhop import (
    AdaptiveHalfHop,
    build_graph,
    insert_slow_chains,
)


def test_multi_delay_chain_invariants():
    pos = torch.rand(12, 3)
    x = torch.cat((pos, torch.randn(12, 3)), dim=1)
    graph = build_graph(
        pos,
        x,
        knn=3,
        curv_k=4,
        delays=(0, 1, 2),
        quantiles=(0.5, 0.85),
        mode="curvature",
        uniform_m=1,
        backward="last",
    )

    assert graph.x.size(0) == x.size(0) + int(graph.delays.sum())
    assert graph.slow_node_mask[: x.size(0)].eq(False).all()
    assert graph.slow_node_mask[x.size(0) :].eq(True).all()
    assert graph.edge_index.dtype == torch.long


def test_interpolation_and_zero_delay_edges():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    edges = torch.tensor([[0, 1], [1, 2]])
    delays = torch.tensor([1, 0])
    new_x, new_edges, slow_mask = insert_slow_chains(x, edges, delays)

    assert torch.allclose(new_x[3], torch.tensor([0.5, 0.5]))
    assert (new_edges == torch.tensor([[1], [2]])).all(dim=0).any()
    assert slow_mask.tolist() == [False, False, False, True]


def test_transform_builds_features_without_normals():
    data = type("DataLike", (), {})()
    data.pos = torch.rand(10, 3)
    transform = AdaptiveHalfHop(knn=3, curv_k=4, delays=(0, 1, 2))
    result = transform(data)

    assert result.x.shape == (result.num_nodes, 3)
    assert result.edge_index.size(0) == 2
    assert result.slow_node_mask.size(0) == result.num_nodes
