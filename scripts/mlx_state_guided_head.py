#!/usr/bin/env python3
"""State-Guided Residual Candidate Set Head in Apple MLX.

Mathematical Guarantees:
1. Permutation Equivariant over candidates:
   [h_state, c_1, c_2, ..., c_K] attends jointly via TransformerEncoder.
   Since h_state is fixed at pos 0 and no positional embeddings are used,
   any permutation of [c_1...c_K] permutes the outputs identically.
2. Step 0 Zero-Delta Guarantee:
   fc2.weight and fc2.bias are initialized to EXACT zero.
   -> Step 0 JevBench score is bit-for-bit identical to the 65.37% baseline.
3. K-Adaptive Routing:
   For K <= 2, strictly bypasses the Transformer branch to eliminate noise on Boolean/Noul.
"""

from typing import Any, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


class StateGuidedCandidateSetHead(nn.Module):
    def __init__(
        self,
        base_weight: Optional[mx.array] = None,
        in_dim: int = 1024,
        set_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.set_dim = set_dim

        if base_weight is not None:
            self.base_w = base_weight.astype(mx.float32)
        else:
            self.base_w = mx.zeros((1, in_dim), dtype=mx.float32)

        self.proj_in = nn.Linear(in_dim, set_dim)
        self.set_encoder = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=set_dim,
            num_heads=num_heads,
            mlp_dims=4 * set_dim,
            norm_first=True,
        )
        self.proj_out = nn.Linear(set_dim, in_dim)

        self.norm = nn.RMSNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, set_dim)
        self.act = nn.silu
        self.fc2 = nn.Linear(set_dim, 1)

        # Zero-initialize the final projection layer
        self.fc2.weight = mx.zeros((1, set_dim), dtype=mx.float32)
        self.fc2.bias = mx.zeros((1,), dtype=mx.float32)

    def __call__(
        self,
        leaves: mx.array,
        examples: Optional[List[dict]] = None,
        kmax: Optional[int] = None,
    ) -> Tuple[mx.array, Optional[Any]]:
        if leaves.ndim == 3 and leaves.shape[0] == 1:
            leaves = leaves[0]

        num_vecs = leaves.shape[0]
        cand_count = len(examples[0]["candidate_ids"]) if examples and "candidate_ids" in examples[0] else num_vecs

        if num_vecs == cand_count + 1:
            # State token is present at index 0!
            h_state = leaves[0]
            cands = leaves[1:]
            K = cands.shape[0]
            base_score = mx.matmul(cands, self.base_w.T).squeeze(-1)
            if K <= 2:
                if base_score.ndim == 1:
                    return base_score[None, :], None
                return base_score, None

            # Joint Transformer attention: [h_state, c_1, ..., c_K]
            tokens = leaves  # [1 + K, in_dim]
            x = self.proj_in(tokens)
            x_enc = self.set_encoder(x[None, :], None)[0]
            cand_enc = x_enc[1:]
            rep = cands + self.proj_out(cand_enc)
            delta = self.fc2(self.act(self.fc1(self.norm(rep)))).squeeze(-1)
            score = base_score + delta
            if score.ndim == 1:
                return score[None, :], None
            return score, None
        else:
            # Fallback without state token
            K = num_vecs
            base_score = mx.matmul(leaves, self.base_w.T).squeeze(-1)
            if K <= 2:
                if base_score.ndim == 1:
                    return base_score[None, :], None
                return base_score, None
            x = self.proj_in(leaves)
            x_enc = self.set_encoder(x[None, :], None)[0]
            rep = leaves + self.proj_out(x_enc)
            delta = self.fc2(self.act(self.fc1(self.norm(rep)))).squeeze(-1)
            score = base_score + delta
            if score.ndim == 1:
                return score[None, :], None
            return score, None
