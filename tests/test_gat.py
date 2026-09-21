import torch
import pytest
from torch_geometric.data import Data

from halfhop.gat import GAT

@pytest.fixture
def basic_data():
    x = torch.randn(5, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3],
        [1, 0, 2, 1, 4],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def test_gat_depth_1(basic_data):
    model = GAT(in_channels=4, hidden_channels=8, out_channels=3, depth=1)
    out = model(basic_data.x, basic_data.edge_index)
    assert out.shape == (5, 3)
    assert len(model.convs) == 1
    assert model.convs[0].heads == 1

def test_gat_depth_2(basic_data):
    model = GAT(in_channels=4, hidden_channels=8, out_channels=3, depth=2, num_heads=4)
    out = model(basic_data.x, basic_data.edge_index)
    assert out.shape == (5, 3)
    assert len(model.convs) == 2
    assert model.convs[0].heads == 4
    assert model.convs[1].heads == 1

def test_gat_depth_3(basic_data):
    model = GAT(in_channels=4, hidden_channels=8, out_channels=3, depth=3, num_heads=4)
    out = model(basic_data.x, basic_data.edge_index)
    assert out.shape == (5, 3)
    assert len(model.convs) == 3
    assert model.convs[0].heads == 4
    assert model.convs[1].heads == 4
    assert model.convs[2].heads == 1

def test_invalid_depth():
    with pytest.raises(AssertionError):
        GAT(in_channels=4, hidden_channels=8, out_channels=3, depth=0)
