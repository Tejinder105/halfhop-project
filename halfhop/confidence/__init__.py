"""Confidence-Aware Adaptive Half-Hop (CAHH) — core modules.

Provides:
  edge_features       — geometric edge descriptors (z_ij ∈ R^4)
  confidence_mlp      — EdgeConfidenceMLP (4→32→16→1)
  adaptive_selector   — fixed-budget undirected top-K selector
  train_gate          — shape-level MLP training + evaluation

Usage at a glance::

    from halfhop.confidence import EdgeConfidenceMLP, build_edge_features
    from halfhop.confidence import AdaptiveSelector
"""

from halfhop.confidence.edge_features import (
    EdgeFeatureNormalizer,
    build_edge_features,
    compute_local_density,
)
from halfhop.confidence.confidence_mlp import EdgeConfidenceMLP
from halfhop.confidence.adaptive_selector import AdaptiveSelector

__all__ = [
    "build_edge_features",
    "EdgeFeatureNormalizer",
    "compute_local_density",
    "EdgeConfidenceMLP",
    "AdaptiveSelector",
]

