#!/usr/bin/env python3
"""MLX-Native PointerHead Architecture for NanoJev Large Candidate Selection.

Inspired by Pointer Networks and Kev's Cross-Attention pointer mechanism:
Instead of absolute score MLP on candidates independently, PointerHead computes:
    logits = (K(h_opts) @ Q(h_decide)) * scale

Key Advantages for Large Skill Catalogs (K = 10 ~ 256):
1. Permutation Invariant: Candidate order does not affect similarity score.
2. High Discrimination on Large K: Performs cross-attention dot product matching in a joint projection space.
3. Ultra Lightweight: 1024 -> 256 projection matrices (<0.5 MB total weights).
"""
import math
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int = 1024, pointer_dim: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.pointer_dim = pointer_dim
        self.scale = 1.0 / math.sqrt(pointer_dim)

        self.norm = nn.LayerNorm(hidden_size)
        # Q projection: maps the task/instruction summary state (h_decide) to pointer space
        self.q_proj = nn.Linear(hidden_size, pointer_dim)
        # K projection: maps the candidate representations (h_opts) to pointer space
        self.k_proj = nn.Linear(hidden_size, pointer_dim)

    def compute_logits(self, h_decide: mx.array, h_opts: mx.array) -> mx.array:
        """Compute matching logits between decision query and candidates.

        Args:
            h_decide: [hidden_size] or [B, hidden_size] representation of task prompt/decide token.
            h_opts: [K, hidden_size] or [B, K, hidden_size] representations of candidates.

        Returns:
            logits: [K] or [B, K] matching scores.
        """
        # Norm
        h_decide_norm = self.norm(h_decide)
        h_opts_norm = self.norm(h_opts)

        # Project
        q = self.q_proj(h_decide_norm)  # [..., pointer_dim]
        k = self.k_proj(h_opts_norm)    # [..., K, pointer_dim]

        if q.ndim == 1:
            # q: [dp], k: [K, dp] -> [K]
            logits = mx.matmul(k, q) * self.scale
        elif q.ndim == 2 and k.ndim == 2:
            # q: [B, dp], k: [K, dp]
            logits = (k @ q.T).T * self.scale
        elif q.ndim == 2 and k.ndim == 3:
            # q: [B, 1, dp], k: [B, K, dp] -> [B, K]
            logits = mx.matmul(k, q[:, :, None]).squeeze(-1) * self.scale
        else:
            logits = (k @ q[..., None]).squeeze(-1) * self.scale

        return logits.astype(mx.float32)

    def __call__(
        self,
        leaves: mx.array,
        examples: List[dict],
        kmax: int,
        h_decide: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Drop-in interface compatible with DeepDecisionHeads.

        Handles both Choice (K options dot product) and Boolean (2 options: [0, s0]).
        """
        batch_size = len(examples)
        offset = 0
        logits_list = []
        valid_list = []

        for i, ex in enumerate(examples):
            n = len(ex["leaf_tokens"])
            ex_leaves = leaves[offset : offset + n]
            k_valid = len(ex["candidate_ids"])

            # Determine query: either passed h_decide or anchor from first candidate prefix
            if h_decide is not None:
                q_repr = h_decide[i] if h_decide.ndim > 1 else h_decide
            else:
                q_repr = ex_leaves[0]

            if ex["type"] == "boolean":
                raw_score = self.compute_logits(q_repr, ex_leaves[:1])
                s0 = raw_score[0]
                b_scores = mx.stack([0.0 * s0, s0], axis=0)
                if kmax > 2:
                    pad_val = mx.full((kmax - 2,), -1e9, dtype=b_scores.dtype)
                    padded_scores = mx.concatenate([b_scores, pad_val], axis=0)
                    ex_valid = mx.concatenate([mx.ones(2, dtype=mx.bool_), mx.zeros(kmax - 2, dtype=mx.bool_)], axis=0)
                else:
                    padded_scores = b_scores
                    ex_valid = mx.ones(2, dtype=mx.bool_)
            else:
                scores = self.compute_logits(q_repr, ex_leaves[:k_valid])
                if k_valid < kmax:
                    pad_val = mx.full((kmax - k_valid,), -1e9, dtype=scores.dtype)
                    padded_scores = mx.concatenate([scores, pad_val], axis=0)
                    ex_valid = mx.concatenate([mx.ones(k_valid, dtype=mx.bool_), mx.zeros(kmax - k_valid, dtype=mx.bool_)], axis=0)
                else:
                    padded_scores = scores
                    ex_valid = mx.ones(k_valid, dtype=mx.bool_)

            logits_list.append(padded_scores)
            valid_list.append(ex_valid)
            offset += n

        logits = mx.stack(logits_list, axis=0)
        valid = mx.stack(valid_list, axis=0)
        return logits, valid
