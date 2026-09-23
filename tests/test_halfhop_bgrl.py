"""Checks that Half-Hop views keep original nodes as a prefix."""

import torch
from torch_geometric.data import Data

from halfhop_bgrl.views import build_view_transform


def _toy_graph():
    x = torch.arange(4, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3],
         [1, 2, 3, 0]],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index, num_nodes=4)


def test_feat_edge_view_keeps_node_count():
    data = _toy_graph()
    transform = build_view_transform(
        p_hh=0.0, alpha=0.5, drop_edge_p=0.5, drop_feat_p=0.2, name="feat_edge"
    )
    aug, mask = transform(data)
    assert int(mask.sum()) == data.num_nodes
    assert aug.x.size(0) == data.num_nodes


def test_halfhop_view_masks_slow_nodes():
    data = _toy_graph()
    torch.manual_seed(0)
    transform = build_view_transform(
        p_hh=1.0, alpha=0.5, drop_edge_p=0.0, drop_feat_p=0.0, name="hh"
    )
    aug, mask = transform(data)
    n = data.num_nodes
    assert int(mask.sum()) == n
    assert torch.all(mask[:n])
    assert not torch.any(mask[n:])
    assert aug.num_nodes > n
    assert hasattr(aug, "slow_node_mask")
    assert torch.equal(mask, ~aug.slow_node_mask)
