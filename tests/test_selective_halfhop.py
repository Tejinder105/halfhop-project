import torch
from halfhop.selective_halfhop import half_hop, select_edges, topk_mask, norm_adj


def _ring(n=6):
    a = torch.arange(n)
    b = (a + 1) % n
    return torch.stack([torch.cat([a, b]), torch.cat([b, a])])  # undirected ring, 2n directed edges


def test_counts_and_replacement():
    ei = _ring(6); x = torch.randn(6, 4)
    mask = torch.zeros(ei.size(1), dtype=torch.bool); mask[:3] = True
    x2, ei2, slow = half_hop(x, ei, mask, 0.3)
    assert x2.size(0) == 9 and slow.sum() == 3 and not slow[:6].any()
    assert ei2.size(1) == ei.size(1) - 3 + 9
    for (i, j) in ei[:, :3].t().tolist():  # original edge removed
        assert not ((ei2[0] == i) & (ei2[1] == j)).any()


def test_feature_interpolation_and_direction():
    ei = torch.tensor([[0], [1]]); x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x2, ei2, _ = half_hop(x, ei, torch.tensor([True]), 0.1)
    assert torch.allclose(x2[2], 0.1 * x[0] + 0.9 * x[1])
    edges = set(map(tuple, ei2.t().tolist()))
    assert edges == {(0, 2), (1, 2), (2, 1)}


def test_self_loops_untouched():
    ei = torch.tensor([[0, 1], [0, 0]])  # (0->0) self loop, (1->0)
    _, ei2, slow = half_hop(torch.randn(2, 3), ei, torch.ones(2, dtype=torch.bool), 0.5)
    assert slow.sum() == 1 and ((ei2[0] == 0) & (ei2[1] == 0)).any()


def test_budget_edge_level():
    ei = _ring(50); score = torch.rand(ei.size(1))
    m = select_edges(ei, 50, score, 0.25, "edge", torch.Generator().manual_seed(0))
    assert int(m.sum()) == round(0.25 * ei.size(1))
    thr = score[m].min()
    assert (score[~m] <= thr + 1e-9).all()  # picked the top scores


def test_node_level_selects_all_incoming():
    ei = _ring(20); score = torch.rand(ei.size(1))
    m = select_edges(ei, 20, score, 0.5, "node", torch.Generator().manual_seed(0))
    tgt = ei[1][m].unique()
    assert tgt.numel() == 10
    for t in tgt.tolist():
        assert m[ei[1] == t].all()


def test_oracle_prefers_disagreeing_edges():
    ei = _ring(40); y = torch.arange(40) // 10  # 4 blocks -> few boundary edges
    dis = (y[ei[0]] != y[ei[1]]).float()
    m = select_edges(ei, 40, dis, float(dis.mean()), "edge", torch.Generator().manual_seed(0))
    assert (dis[m] == 1).all()


def test_constant_score_is_random():
    ei = _ring(200)
    a = select_edges(ei, 200, torch.zeros(ei.size(1)), 0.1, "edge", torch.Generator().manual_seed(1))
    b = select_edges(ei, 200, torch.zeros(ei.size(1)), 0.1, "edge", torch.Generator().manual_seed(2))
    assert (a != b).any() and int(a.sum()) == int(b.sum())


def test_norm_adj_symmetric_weights():
    ei = _ring(5); A = norm_adj(ei, 5, 2.0).to_dense()
    assert torch.allclose(A, A.t(), atol=1e-6)
    assert (A.diagonal() > 0).all()

