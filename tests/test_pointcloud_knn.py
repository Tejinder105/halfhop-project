import pytest
import torch

from experiments.pointcloud.run import knn_edge_index


def test_endpoints_of_a_line_connect_to_their_only_neighbour():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    edge_index = knn_edge_index(pos, k=1)
    incoming = {
        int(edge_index[1, i]): int(edge_index[0, i]) for i in range(edge_index.size(1))
    }
    assert incoming[0] == 1
    assert incoming[3] == 2
    assert (edge_index[0] == edge_index[1]).sum() == 0


def test_each_node_has_exactly_k_incoming_edges():
    pos = torch.tensor(
        [[float(i), 0.0, 0.0] for i in range(10)]
    )
    k = 3
    edge_index = knn_edge_index(pos, k)
    counts = torch.bincount(edge_index[1], minlength=10)
    assert torch.equal(counts, torch.full((10,), k))
    # Node 0's three nearest are 1, 2, 3.
    sources = edge_index[0, edge_index[1] == 0].tolist()
    assert sorted(sources) == [1, 2, 3]


def test_k_must_be_smaller_than_the_cloud():
    pos = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        knn_edge_index(pos, 4)
