import torch
import pytest
from torch_geometric.data import Data

from halfhop.gcn import GCN

@pytest.fixture
def basic_data():
    x = torch.randn(5, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3],
        [1, 0, 2, 1, 4],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def test_gcn_depth_1(basic_data):
    model = GCN(in_channels=4, hidden_channels=8, out_channels=3, depth=1)
    out = model(basic_data.x, basic_data.edge_index)
    assert out.shape == (5, 3)
    assert len(model.convs) == 1

def test_gcn_depth_2(basic_data):
    model = GCN(in_channels=4, hidden_channels=8, out_channels=3, depth=2)
    out = model(basic_data.x, basic_data.edge_index)
    assert out.shape == (5, 3)
    assert len(model.convs) == 2

def test_gcn_depth_3(basic_data):
    model = GCN(in_channels=4, hidden_channels=8, out_channels=3, depth=3)
    out = model(basic_data.x, basic_data.edge_index)
    assert out.shape == (5, 3)
    assert len(model.convs) == 3

def test_invalid_depth():
    with pytest.raises(AssertionError):
        GCN(in_channels=4, hidden_channels=8, out_channels=3, depth=0)