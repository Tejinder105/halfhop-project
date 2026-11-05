import torch
import pytest
from torch_geometric.data import Data

from halfhop.hh_gcn import HHGCN

@pytest.fixture
def basic_data():
    x = torch.randn(5, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3],
        [1, 0, 2, 1, 4],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def test_hh_gcn_output_shape(basic_data):
    original_num_nodes = basic_data.num_nodes
    model = HHGCN(in_channels=4, hidden_channels=8, out_channels=3, alpha=0.5, p=1.0)
    out = model(basic_data)
    
    # Check that output size matches original nodes, despite slow nodes added internally
    assert out.shape == (original_num_nodes, 3)

def test_hh_gcn_inplace_false_default(basic_data):
    original_num_nodes = basic_data.num_nodes
    model = HHGCN(in_channels=4, hidden_channels=8, out_channels=3, inplace=False)
    out = model(basic_data)
    
    # Original data object should remain unmodified
    assert basic_data.num_nodes == original_num_nodes
    assert not hasattr(basic_data, 'slow_node_mask')

def test_hh_gcn_variable_depth(basic_data):
    model = HHGCN(in_channels=4, hidden_channels=8, out_channels=3, depth=3)
    out = model(basic_data)
    assert out.shape == (5, 3)
    assert len(model.gnn.convs) == 3

def test_hh_gcn_ablations(basic_data):
    model = HHGCN(
        in_channels=4, hidden_channels=8, out_channels=3, 
        slow_node_init='zero', connectivity='hh2'
    )
    out = model(basic_data)
    assert out.shape == (5, 3)