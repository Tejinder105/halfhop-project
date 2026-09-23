"""Import shims so official BGRL code runs on current PyG/sklearn.

These do not change the BGRL algorithm. Official `dropout_adj` was removed
from newer PyTorch Geometric; sklearn replaced `sparse=` and `np.bool`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ORIGINAL_ROOT = Path(__file__).resolve().parents[1] / "bgrl_original"


def add_official_bgrl_to_path() -> None:
    root = str(ORIGINAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def patch_dropout_adj() -> None:
    try:
        import torch_geometric.utils.dropout as dropout_mod
    except ImportError:
        return
    if hasattr(dropout_mod, "dropout_adj"):
        return

    from torch_geometric.utils import dropout_edge

    def dropout_adj(
        edge_index,
        edge_attr=None,
        p=0.5,
        force_undirected=False,
        num_nodes=None,
        training=True,
    ):
        # Same contract as official BGRL's DropEdges → pyg dropout_adj.
        edge_index, edge_mask = dropout_edge(
            edge_index,
            p=p,
            force_undirected=force_undirected,
            training=training,
        )
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask]
        return edge_index, edge_attr

    dropout_mod.dropout_adj = dropout_adj


def one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(categories="auto", sparse_output=False)
    except TypeError:
        return OneHotEncoder(categories="auto", sparse=False)


def prepare_official_imports() -> None:
    add_official_bgrl_to_path()
    patch_dropout_adj()
