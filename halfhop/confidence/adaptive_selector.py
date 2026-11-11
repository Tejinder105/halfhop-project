"""Fixed-budget undirected top-K edge selector for CAHH.

Design rules (from the CAHH plan §3.5):
  - Treat directed pairs (i→j, j→i) as a single undirected edge.
  - Select K = floor(budget × M) canonical pairs, where M is the number of
    unique undirected pairs (self-loops excluded).
  - Half-hop **both** arcs of every selected pair.
  - Ties are broken with ε-noise (uniform, tiny) for reproducibility.

The selector returns a boolean mask over the *directed* edge_index (shape E),
where True marks every arc (both directions) of the selected undirected pairs.
This mask can be fed directly to the existing HalfHop transform via::

    data.edge_score = selector.score_to_directed(scores, edge_index)
    hh = HalfHop(selection='score', budget=budget, score_mode='high')
    data = hh(data)

or used manually::

    mask = selector.select_mask(scores, edge_index, budget)
    # mask[i] is True iff arc i belongs to a selected pair
"""

from __future__ import annotations

import torch
from torch import Tensor


def _canonical(edge_index: Tensor) -> tuple[Tensor, Tensor]:
    """Return (canonical_index, pair_id) for each directed arc.

    canonical_index: (2, M) with M = #unique undirected pairs,
                     each pair stored as (min_node, max_node).
    pair_id:         (E,)  — which canonical pair each arc belongs to
                     (-1 for self-loops).
    """
    src, dst = edge_index
    self_loop = src == dst
    mn = torch.minimum(src, dst)
    mx = torch.maximum(src, dst)
    pairs = torch.stack([mn, mx], dim=0)         # (2, E)

    # Assign an integer id to each unique pair.
    # torch.unique on dim=1 is not in all PyTorch versions; use a hash trick.
    N = edge_index.max().item() + 1
    keys = mn * N + mx                            # unique integer per pair
    keys[self_loop] = -1                          # sentinel for self-loops

    unique_keys, pair_id = torch.unique(keys, return_inverse=True)

    # Remap self-loops to id=-1 (they won't be selected).
    # torch.unique maps -1 to some id; we need to find it.
    # Because keys=-1 is a negative value and node ids are ≥0, -1 maps to
    # the first bucket (sorted order).  We just override self-loop positions.
    pair_id = pair_id.clone()
    pair_id[self_loop] = -1

    return unique_keys, pair_id


class AdaptiveSelector:
    """Undirected top-K edge selector driven by per-arc confidence scores.

    Args:
        budget: Fraction of undirected pairs to select (∈ [0, 1]).
        eps:    Magnitude of uniform tie-breaking noise (default 1e-7).
    """

    def __init__(self, budget: float = 0.1, eps: float = 1e-7) -> None:
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"budget must be in [0, 1], got {budget}")
        self.budget = budget
        self.eps = eps

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select_mask(
        self,
        arc_scores: Tensor,
        edge_index: Tensor,
        budget: float | None = None,
    ) -> Tensor:
        """Return a bool mask (shape E) marking arcs of the selected pairs.

        ``arc_scores`` must have length E (one value per directed arc).
        For undirected pairs the score is the mean of the two arc scores.

        Args:
            arc_scores: ``(E,)`` scores (higher → more likely selected).
            edge_index: ``(2, E)`` directed edge indices.
            budget:     Override ``self.budget`` for this call.

        Returns:
            ``(E,)`` bool mask — True for every arc of a selected pair.
        """
        rho = budget if budget is not None else self.budget
        device = arc_scores.device
        E = edge_index.size(1)

        unique_keys, pair_id = _canonical(edge_index)
        # Number of valid (non-self-loop) unique pairs.
        # pair_id == -1 marks self-loops; unique_keys maps them to key=-1.
        # We need count of real pairs only.
        valid_pair_mask = pair_id >= 0
        num_valid_pairs = int((unique_keys >= 0).sum().item())

        K = min(num_valid_pairs, max(0, int(rho * num_valid_pairs)))

        if K == 0:
            return torch.zeros(E, dtype=torch.bool, device=device)
        if K == num_valid_pairs:
            return pair_id >= 0  # all valid arcs

        # Aggregate arc scores to pair scores (mean of two arcs).
        num_pairs = int(unique_keys.size(0))
        pair_score = torch.full((num_pairs,), float('-inf'), device=device)
        pair_count = torch.zeros(num_pairs, device=device)

        valid_ids = pair_id[valid_pair_mask]
        valid_scores = arc_scores[valid_pair_mask].float()
        pair_score.index_add_(0, valid_ids, valid_scores)
        pair_count.index_add_(0, valid_ids, torch.ones_like(valid_scores))
        has_data = pair_count > 0
        pair_score[has_data] = pair_score[has_data] / pair_count[has_data]

        # Tie-breaking noise.
        noise = torch.rand(num_pairs, device=device) * self.eps
        pair_score = pair_score + noise

        # Select top-K pairs (by score).
        topk_vals, topk_idx = pair_score.topk(K, largest=True, sorted=False)
        selected_pairs = torch.zeros(num_pairs, dtype=torch.bool, device=device)
        selected_pairs[topk_idx] = True

        # Map back to directed arcs.
        arc_selected = torch.zeros(E, dtype=torch.bool, device=device)
        valid_arc_indices = valid_pair_mask.nonzero(as_tuple=True)[0]
        arc_selected[valid_arc_indices] = selected_pairs[valid_ids]

        return arc_selected

    def score_for_halfhop(
        self,
        arc_scores: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Return per-arc undirected-pair scores, suitable for HalfHop
        ``selection='score'``.

        The returned tensor has the same shape as ``arc_scores`` but each
        arc's value is replaced by its pair's mean score (so both arcs of
        every undirected pair receive the same score, ensuring they are
        selected or rejected together by HalfHop's top-K).

        Self-loops get score ``-inf``.
        """
        device = arc_scores.device
        E = edge_index.size(1)
        unique_keys, pair_id = _canonical(edge_index)
        num_pairs = int(unique_keys.size(0))

        pair_score = torch.full((num_pairs,), float('-inf'), device=device)
        pair_count = torch.zeros(num_pairs, device=device)
        valid = pair_id >= 0
        valid_ids = pair_id[valid]
        valid_scores = arc_scores[valid].float()
        pair_score.index_add_(0, valid_ids, valid_scores)
        pair_count.index_add_(0, valid_ids, torch.ones_like(valid_scores))
        has_data = pair_count > 0
        pair_score[has_data] = pair_score[has_data] / pair_count[has_data]

        out = torch.full((E,), float('-inf'), device=device)
        out[valid] = pair_score[valid_ids]
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(budget={self.budget})"

