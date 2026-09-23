"""Half-Hop BGRL on top of the official Thakoor et al. implementation.

Official training, architecture, optimizer, cosine/warmup schedules, EMA,
FeatDrop/EdgeDrop, and liblinear evaluation live in `bgrl_original/`.

This package only:
    - composes Half-Hop as an extra view augmentation
    - keeps original-node embeddings for the BGRL loss and linear eval
    - writes the experiment runner outputs
"""

from .configs import DATASET_ALIASES, get_run_config

__all__ = [
    "DATASET_ALIASES",
    "get_run_config",
]
