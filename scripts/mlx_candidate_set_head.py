#!/usr/bin/env python3
"""CandidateSetHead: Permutation-Equivariant Cross-Candidate Attention Head in Apple MLX.

Architecture:
  candidate vectors [K, 1024]
    -> Linear(1024, 256)
    -> CandidateSetEncoder: 2-layer TransformerEncoder (no positional embeddings!)
    -> Linear(256, 1024) + Residual Connection (leaves + proj_out(enc))
    -> ScalarScorer: RMSNorm(1024) -> Linear(1024, 256) -> SiLU -> Linear(256, 1)

Guarantees:
  - Exact permutation equivariance (option order permutation = identical output permutation)
  - Inter-candidate mutual comparison & suppression (sharp margins)
  - Reusable across choice, boolean, and score primitives
"""

import mlx.core as mx
import mlx.nn as nn


class CandidateSetHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        set_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.set_dim = set_dim

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

    def __call__(self, leaves: mx.array, examples: list = None, kmax: int = None):
        """Forward pass scoring [K, in_dim] representations.

        Args:
            leaves: [K, in_dim] candidate representations at delimiter markers.
        Returns:
            scores: [1, K] or [K] unnormalized logits.
            None: auxiliary cache placeholder.
        """
        if leaves.ndim == 3 and leaves.shape[0] == 1:
            leaves = leaves[0]

        # 1. Project to candidate set hidden space
        x = self.proj_in(leaves)  # [K, set_dim]

        # 2. Permutation-equivariant Self-Attention across all K candidates
        x_enc = self.set_encoder(x[None, :], None)[0]  # [K, set_dim]

        # 3. Residual connection with original candidate representation
        rep = leaves + self.proj_out(x_enc)  # [K, in_dim]

        # 4. Scalar scoring
        scores = self.fc2(self.act(self.fc1(self.norm(rep)))).squeeze(-1)  # [K]
        if scores.ndim == 1:
            return scores[None, :], None
        return scores, None
