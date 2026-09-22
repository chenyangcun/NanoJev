#!/usr/bin/env python3
"""Linear Scorer Head matching Dohnuts / TypeSafe architecture in Apple MLX.

Scores each candidate option via: score = w^T h (Linear with bias=False).
"""

import mlx.core as mx
import mlx.nn as nn


class LinearScorerHead(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def __call__(self, leaves: mx.array, examples: list = None, kmax: int = None):
        """Forward pass scoring [K, hidden_size] representations."""
        scores = self.proj(leaves).squeeze(-1)
        if scores.ndim == 1:
            return scores[None, :], None
        return scores, None
