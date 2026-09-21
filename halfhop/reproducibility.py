"""Reproducibility utilities for rigorous experimental setups."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries.

    Note that some PyTorch operations (especially sparse operations or 
    scatter operations) might still be non-deterministic on some hardware,
    but this sets the seed for CPU and basic operations.

    Args:
        seed (int): The random seed. Default: ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
