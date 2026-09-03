"""Decoder-geometry diagnostics for pre-audit SAE configuration screens."""

from __future__ import annotations

import numpy as np


def pairwise_decoder_cosine_similarity(decoder: np.ndarray, block_size: int = 1024) -> float:
    """Mean absolute cosine over all distinct decoder-row pairs.

    This is the c_dec proxy from Chanin and Garriga-Alonso (2026).  It is a
    configuration diagnostic, not evidence that individual latents are
    monosemantic.
    """
    matrix = np.asarray(decoder)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("decoder must have shape [latents >= 2, hidden >= 1]")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.all(np.isfinite(matrix)):
        raise ValueError("decoder must contain finite floating-point values")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    matrix = matrix.astype(np.float64, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("decoder rows must be nonzero")
    normalized = matrix / norms[:, None]
    total = 0.0
    count = 0
    rows = normalized.shape[0]
    for left_start in range(0, rows, block_size):
        left = normalized[left_start:left_start + block_size]
        for right_start in range(left_start, rows, block_size):
            right = normalized[right_start:right_start + block_size]
            similarities = np.abs(left @ right.T)
            if left_start == right_start:
                values = similarities[np.triu_indices(similarities.shape[0], k=1)]
            else:
                values = similarities.reshape(-1)
            total += float(values.sum(dtype=np.float64))
            count += int(values.size)
    expected = rows * (rows - 1) // 2
    if count != expected:
        raise RuntimeError(f"pair accounting mismatch: {count} != {expected}")
    return total / count
