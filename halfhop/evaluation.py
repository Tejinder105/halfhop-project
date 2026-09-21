"""Evaluation metrics and formatting utilities."""

import numpy as np


def format_results(results: list) -> str:
    """Format a list of accuracy results into 'mean ± std'.

    Args:
        results (list or np.ndarray): List of accuracy values (0 to 1).

    Returns:
        str: Formatted string 'mean ± std' in percentage format (e.g., '72.88 ± 7.17').
    """
    results = np.array(results) * 100
    mean = results.mean()
    std = results.std()
    return f"{mean:.2f} ± {std:.2f}"
