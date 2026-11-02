import pytest
import torch
from torch_geometric.data import Data

from halfhop.dropedge import drop_edge, DropEdge
from halfhop.hh_dropedge import DropEdgeGCN, DropEdgeGraphSAGE, HHDropEdgeGCN, HHDropEdgeGraphSAGE


def test_drop_edge_basic():
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 0]], dtype=torch.long)
    # p=0 should keep all edges
    out, mask = drop_edge(edge_index, p=0.0, training=True)
    assert out.size(1) == edge_index.size(1)
    assert mask.all()

    # training=False should keep all edges
    out, mask = drop_edge(edge_index, p=0.5, training=False)
    assert out.size(1) == edge_index.size(1)
    assert mask.all()

    # Self-loops must always be preserved
    # Edge at index 4 is (0, 0), which is a self-loop
    out, mask = drop_edge(edge_index, p=1.0, training=True)
    assert (out[0] == out[1]).all()
    assert 4 in mask.nonzero(as_tuple=True)[0]


def test_drop_edge_module():
    drop = DropEdge(p=0.4)
    drop.eval()
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out, mask = drop(edge_index)
    assert out.size(1) == 3


def test_dropedge_models_forward():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    model_gcn = DropEdgeGCN(in_channels=16, hidden_channels=32, out_channels=4, p_drop=0.2, depth=2)
    model_sage = DropEdgeGraphSAGE(in_channels=16, hidden_channels=32, out_channels=4, p_drop=0.2, depth=2)

    out1 = model_gcn(data.x, data.edge_index)
    assert out1.shape == (10, 4)

    out2 = model_sage(data.x, data.edge_index)
    assert out2.shape == (10, 4)


def test_hh_dropedge_forward():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    model_hh_de_gcn = HHDropEdgeGCN(
        in_channels=16, hidden_channels=32, out_channels=4,
        p_drop=0.2, alpha=0.5, p_hh=1.0, depth=2, drop_mode='post_hh'
    )
    model_hh_de_sage = HHDropEdgeGraphSAGE(
        in_channels=16, hidden_channels=32, out_channels=4,
        p_drop=0.2, alpha=0.5, p_hh=1.0, depth=2, drop_mode='post_hh'
    )

    model_hh_de_gcn.train()
    out1 = model_hh_de_gcn(data)
    assert out1.shape == (10, 4)

    model_hh_de_gcn.eval()
    out1_eval = model_hh_de_gcn(data)
    assert out1_eval.shape == (10, 4)

    model_hh_de_sage.train()
    out2 = model_hh_de_sage(data)
    assert out2.shape == (10, 4)

