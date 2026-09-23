import torch
from torch_geometric.data import Data
import pytest

from halfhop.halfhop import HalfHop

@pytest.fixture
def basic_data():
    x = torch.tensor([
        [1.0, 0.0],  # node 0
        [0.0, 1.0],  # node 1
    ])
    edge_index = torch.tensor([
        [0],
        [1],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def data_with_self_loops():
    x = torch.tensor([
        [1.0, 0.0],  # node 0
        [0.0, 1.0],  # node 1
    ])
    edge_index = torch.tensor([
        [0, 0, 1],
        [1, 0, 1],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_single_edge(basic_data):
    transform = HalfHop(alpha=0.5, p=1.0)
    data = transform(basic_data)
    
    assert data.x.size(0) == 3
    assert data.x[2, 0] == 0.5
    assert data.x[2, 1] == 0.5
    
    assert data.edge_index.size(1) == 3

    src, dst = data.edge_index
    edges = set(zip(src.tolist(), dst.tolist()))
    assert (0, 2) in edges
    assert (1, 2) in edges
    assert (2, 1) in edges
    
    assert data.slow_node_mask.tolist() == [False, False, True]


def test_self_loops(data_with_self_loops):
    transform = HalfHop(alpha=0.5, p=1.0)
    data = transform(data_with_self_loops)
    
    assert data.x.size(0) == 3
    
    src, dst = data.edge_index
    edges = set(zip(src.tolist(), dst.tolist()))
    
    assert (0, 0) in edges
    assert (1, 1) in edges
    
    # 0->1 is half hopped (0->2, 1->2, 2->1)
    assert (0, 2) in edges
    assert (1, 2) in edges
    assert (2, 1) in edges


def test_inplace_false(basic_data):
    transform = HalfHop(alpha=0.5, p=1.0, inplace=False)
    data = transform(basic_data)
    
    assert data is not basic_data
    assert data.x.size(0) == 3
    assert basic_data.x.size(0) == 2


def test_p_sampling():
    torch.manual_seed(42)
    # create a fully connected 10-node graph without self loops
    nodes = 10
    x = torch.ones(nodes, 2)
    src = []
    dst = []
    for i in range(nodes):
        for j in range(nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    transform = HalfHop(alpha=0.5, p=0.5, inplace=False)
    out = transform(data)
    
    # p=0.5 means roughly half the nodes will have their incoming edges halfhopped.
    # We just ensure it's not all edges and not 0 edges.
    assert out.x.size(0) > nodes
    assert out.x.size(0) < nodes + len(src)


def test_connectivity_hh1(basic_data):
    transform = HalfHop(alpha=0.5, p=1.0, connectivity='hh1')
    data = transform(basic_data)
    
    src, dst = data.edge_index
    edges = set(zip(src.tolist(), dst.tolist()))
    # For hh1: 0->2, 2->1 (no 1->2)
    assert (0, 2) in edges
    assert (2, 1) in edges
    assert (1, 2) not in edges


def test_connectivity_hh2(basic_data):
    transform = HalfHop(alpha=0.5, p=1.0, connectivity='hh2')
    data = transform(basic_data)
    
    src, dst = data.edge_index
    edges = set(zip(src.tolist(), dst.tolist()))
    # For hh2: 0->2, 2->0, 1->2, 2->1
    assert (0, 2) in edges
    assert (2, 0) in edges
    assert (1, 2) in edges
    assert (2, 1) in edges


def test_init_zero(basic_data):
    transform = HalfHop(alpha=0.5, p=1.0, slow_node_init='zero')
    data = transform(basic_data)
    
    assert torch.all(data.x[2] == 0)


def test_init_random(basic_data):
    transform = HalfHop(alpha=0.5, p=1.0, slow_node_init='random')
    data = transform(basic_data)
    
    assert data.x.size(0) == 3
    # Check it's not the linear init
    assert data.x[2, 0] != 0.5 or data.x[2, 1] != 0.5


def test_invalid_args():
    with pytest.raises(AssertionError):
        HalfHop(p=-0.1)
    with pytest.raises(AssertionError):
        HalfHop(alpha=2.0)
    with pytest.raises(AssertionError):
        HalfHop(slow_node_init='invalid')
    with pytest.raises(AssertionError):
        HalfHop(connectivity='invalid')