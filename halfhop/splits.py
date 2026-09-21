"""Dataset split utilities for the Half-Hop paper experiments.

Heterophilic datasets (Texas, Wisconsin, Cornell, Actor, Squirrel, Chameleon):
    - 10 pre-set train/val/test splits from Pei et al. 2020
    - Masks stored as 2D tensors: shape [N, 10]
    - Access via: data.train_mask[:, split_id]

Homophilic datasets (Amazon, Coauthor):
    - 20 random splits, 60:20:20 train/val/test
    - Masks may be 1D (single split) or need to be constructed

WikiCS:
    - 20 pre-set train masks, single val mask, single test mask
    - data.train_mask shape: [N, 20]
    - data.val_mask shape: [N]
    - data.stopping_mask shape: [N]  (used as validation in some protocols)
    - data.test_mask shape: [N]
"""

import torch
from torch import Tensor
from torch_geometric.data import Data


def get_split(data: Data, split_id: int = 0) -> tuple:
    """Extract a single train/val/test split from a data object.

    Handles both 1D masks (single split) and 2D masks (multiple pre-set splits).

    For heterophilic datasets (Texas, Wisconsin, Cornell, Actor, Squirrel,
    Chameleon), masks are 2D: shape ``[N, 10]``.
    For homophilic datasets with a single split, masks are 1D: shape ``[N]``.

    Args:
        data (:class:`~torch_geometric.data.Data`): Graph data object.
        split_id (int): Index of the split to extract (0-based). Only used
            when masks are 2D. Default: ``0``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Boolean tensors
        ``(train_mask, val_mask, test_mask)``, each of shape ``[N]``.
    """
    if data.train_mask.dim() == 1:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        train_mask = data.train_mask[:, split_id]
        val_mask = data.val_mask[:, split_id]
        test_mask = data.test_mask[:, split_id]

    return train_mask, val_mask, test_mask


def make_random_splits(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_splits: int = 20,
    seed: int = 42,
) -> tuple:
    """Generate reproducible random train/val/test splits.

    Used for homophilic datasets (Amazon Photos/Computers, Coauthor CS/Physics)
    which follow a 60:20:20 split protocol over 20 random splits.

    Args:
        num_nodes (int): Total number of nodes.
        train_ratio (float): Fraction of nodes for training. Default: ``0.6``.
        val_ratio (float): Fraction of nodes for validation. Default: ``0.2``.
        num_splits (int): Number of random splits to generate. Default: ``20``.
        seed (int): Base random seed. Split ``i`` uses seed ``seed + i``.
            Default: ``42``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Boolean mask tensors
        ``(train_masks, val_masks, test_masks)`` each of shape
        ``[num_nodes, num_splits]``.
    """
    train_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
    val_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
    test_masks = torch.zeros(num_nodes, num_splits, dtype=torch.bool)

    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    for i in range(num_splits):
        g = torch.Generator()
        g.manual_seed(seed + i)
        perm = torch.randperm(num_nodes, generator=g)
        train_masks[perm[:n_train], i] = True
        val_masks[perm[n_train:n_train + n_val], i] = True
        test_masks[perm[n_train + n_val:], i] = True

    return train_masks, val_masks, test_masks


def generate_homophilic_splits(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_splits: int = 20,
    seed: int = 42,
) -> Data:
    """Generate and attach reproducible random splits to a data object.

    Args:
        data (Data): Graph data object.
        train_ratio (float): Fraction of nodes for training. Default: 0.6.
        val_ratio (float): Fraction of nodes for validation. Default: 0.2.
        num_splits (int): Number of splits. Default: 20.
        seed (int): Base random seed. Default: 42.

    Returns:
        Data: The graph data object with 2D masks attached.
    """
    train_masks, val_masks, test_masks = make_random_splits(
        data.num_nodes, train_ratio, val_ratio, num_splits, seed
    )
    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks
    return data


def get_wikics_split(data: Data, split_id: int = 0) -> tuple:
    """Extract a single split from WikiCS data.

    WikiCS has 20 pre-set train masks, 1 val mask, and 1 test mask.
    The ``stopping_mask`` is sometimes used as the val mask in BGRL.
    Here we use ``val_mask`` as provided by PyG.

    Args:
        data (:class:`~torch_geometric.data.Data`): WikiCS graph data.
        split_id (int): Index of the training split (0-19). Default: ``0``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Boolean tensors
        ``(train_mask, val_mask, test_mask)``, each of shape ``[N]``.
    """
    if data.train_mask.dim() == 2:
        train_mask = data.train_mask[:, split_id]
    else:
        train_mask = data.train_mask

    # WikiCS: val_mask and test_mask are 1D
    val_mask = data.val_mask
    test_mask = data.test_mask

    return train_mask, val_mask, test_mask


def num_splits_for(data: Data) -> int:
    """Return the number of pre-set splits in a data object.

    Args:
        data (:class:`~torch_geometric.data.Data`): Graph data.

    Returns:
        int: Number of available splits (1 if masks are 1D).
    """
    if data.train_mask.dim() == 1:
        return 1
    return data.train_mask.size(1)