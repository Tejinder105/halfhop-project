"""Tests for halfhop/confidence/adaptive_selector.py."""

import pytest
import torch

from halfhop.confidence.adaptive_selector import AdaptiveSelector, _canonical


# ---------------------------------------------------------------------------
# _canonical helper
# ---------------------------------------------------------------------------

class TestCanonical:
    def test_self_loops_get_minus_one(self):
        ei = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
        _, pair_id = _canonical(ei)
        assert pair_id[0] == -1   # arc 0→0 is a self-loop
        assert pair_id[1] == -1   # arc 1→1

    def test_both_arcs_same_pair(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        _, pair_id = _canonical(ei)
        assert pair_id[0] == pair_id[1]  # i→j and j→i belong to the same pair

    def test_distinct_pairs(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        _, pair_id = _canonical(ei)
        assert pair_id[0] != pair_id[1]


# ---------------------------------------------------------------------------
# AdaptiveSelector
# ---------------------------------------------------------------------------

class TestAdaptiveSelector:
    def _simple_graph(self, n=6):
        """Star graph: node 0 connected to nodes 1..n-1 (both directions)."""
        srcs = list(range(1, n)) + [0] * (n - 1)
        dsts = [0] * (n - 1) + list(range(1, n))
        ei = torch.tensor([srcs, dsts], dtype=torch.long)
        scores = torch.rand(ei.size(1))
        return ei, scores

    def test_budget_zero(self):
        sel = AdaptiveSelector(budget=0.0)
        ei, scores = self._simple_graph()
        mask = sel.select_mask(scores, ei)
        assert not mask.any()

    def test_budget_one(self):
        sel = AdaptiveSelector(budget=1.0)
        ei, scores = self._simple_graph()
        mask = sel.select_mask(scores, ei)
        # All non-self-loop arcs should be selected.
        assert mask.all()

    def test_exact_count(self):
        torch.manual_seed(0)
        sel = AdaptiveSelector(budget=0.5)
        n = 10
        ei, scores = self._simple_graph(n)
        # Number of unique undirected pairs = n-1 (star)
        mask = sel.select_mask(scores, ei)
        # Count selected undirected pairs.
        _, pair_id = _canonical(ei)
        selected_pair_ids = pair_id[mask]
        unique_selected = selected_pair_ids[selected_pair_ids >= 0].unique()
        expected_K = int(0.5 * (n - 1))
        assert len(unique_selected) == expected_K

    def test_both_arcs_selected_together(self):
        """If pair (i,j) is selected, both arcs i→j and j→i must be selected."""
        torch.manual_seed(1)
        sel = AdaptiveSelector(budget=0.3)
        ei = torch.tensor([[0, 1, 2, 1, 0, 2],
                            [1, 0, 0, 2, 2, 1]], dtype=torch.long)
        scores = torch.rand(ei.size(1))
        mask = sel.select_mask(scores, ei)
        _, pair_id = _canonical(ei)
        for arc_i, selected in enumerate(mask):
            if selected:
                # Find the sibling arc (same pair_id, different arc).
                pid = pair_id[arc_i].item()
                if pid == -1:
                    continue
                siblings = (pair_id == pid).nonzero(as_tuple=True)[0]
                for sib in siblings:
                    assert mask[sib], f"Arc {sib} should be selected (sibling of {arc_i})"

    def test_self_loops_never_selected(self):
        ei_arcs = torch.tensor([[0, 1, 2, 2], [1, 0, 2, 0]], dtype=torch.long)
        sel = AdaptiveSelector(budget=1.0)
        scores = torch.rand(ei_arcs.size(1))
        mask = sel.select_mask(scores, ei_arcs)
        # Arc 2→2 (index 2) should never be selected.
        assert not mask[2]

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            AdaptiveSelector(budget=1.5)
        with pytest.raises(ValueError):
            AdaptiveSelector(budget=-0.1)

    def test_score_for_halfhop_shape(self):
        sel = AdaptiveSelector(budget=0.5)
        ei, scores = self._simple_graph(8)
        pair_scores = sel.score_for_halfhop(scores, ei)
        assert pair_scores.shape == scores.shape

    def test_score_for_halfhop_symmetric(self):
        """Both arcs of an undirected pair should get the same pair score."""
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        scores = torch.tensor([0.3, 0.7])  # different arc scores
        sel = AdaptiveSelector(budget=0.5)
        pair_scores = sel.score_for_halfhop(scores, ei)
        # Both arcs → same mean score
        assert abs(pair_scores[0].item() - pair_scores[1].item()) < 1e-5

    def test_deterministic_with_same_seed(self):
        ei, _ = self._simple_graph(10)
        scores = torch.rand(ei.size(1))
        torch.manual_seed(99)
        sel = AdaptiveSelector(budget=0.4)
        mask1 = sel.select_mask(scores, ei)
        torch.manual_seed(99)
        mask2 = sel.select_mask(scores, ei)
        assert (mask1 == mask2).all()

    def test_empty_graph(self):
        """Graph with only self-loops → nothing to select."""
        ei = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        scores = torch.rand(2)
        sel = AdaptiveSelector(budget=0.5)
        mask = sel.select_mask(scores, ei)
        assert not mask.any()

    def test_repr(self):
        sel = AdaptiveSelector(budget=0.25)
        assert "0.25" in repr(sel)

