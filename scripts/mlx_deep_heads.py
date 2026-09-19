#!/usr/bin/env python3
"""Multi-layer MLP and Head Architecture for NanoJev in Apple MLX.

Replaces shallow 1-layer Linear with a 2-layer Non-linear Projection Head
(hidden_size -> 512 -> GELU -> 1), dramatically increasing capacity
without touching the Qwen3 backbone!
"""
from typing import Dict, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


class DeepDecisionHeads(nn.Module):
    def __init__(self, hidden_size: int = 1024, set_head: str = "none"):
        super().__init__()
        self.hidden_size = hidden_size
        self.set_head = set_head
        self.norm = nn.LayerNorm(hidden_size)
        
        # 2-layer MLP for classification
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 1)

        if set_head == "attention":
            self.set_project = nn.Linear(hidden_size + 1, 128)
            self.set_attention = nn.MultiHeadAttention(dims=128, num_heads=4, bias=True)
            self.set_output = nn.Linear(128, 1)

    def scalar_forward(self, x):
        h = nn.gelu(self.fc1(x))
        return self.fc2(h)

    def __call__(
        self,
        leaves: mx.array,
        examples: List[dict],
        kmax: int,
    ) -> Tuple[mx.array, mx.array]:
        batch_size = len(examples)
        offset = 0
        h_list = []
        valid_list = []

        for i, ex in enumerate(examples):
            n = len(ex["leaf_tokens"])
            ex_leaves = leaves[offset : offset + n]
            if n < kmax:
                pad_h = mx.zeros((kmax - n, self.hidden_size), dtype=leaves.dtype)
                ex_h = mx.concatenate([ex_leaves, pad_h], axis=0)
            else:
                ex_h = ex_leaves

            k_valid = len(ex["candidate_ids"])
            if k_valid < kmax:
                ex_valid = mx.concatenate([mx.ones(k_valid, dtype=mx.bool_), mx.zeros(kmax - k_valid, dtype=mx.bool_)], axis=0)
            else:
                ex_valid = mx.ones(k_valid, dtype=mx.bool_)

            h_list.append(ex_h)
            valid_list.append(ex_valid)
            offset += n

        h = mx.stack(h_list, axis=0)
        valid = mx.stack(valid_list, axis=0)

        # Norm + 2-layer MLP
        h_norm = self.norm(h)
        z = mx.squeeze(self.scalar_forward(h_norm), axis=-1).astype(mx.float32)

        # Choice set_attention
        choice_indices = [i for i, ex in enumerate(examples) if ex["type"] == "choice"]
        if self.set_head == "attention" and len(choice_indices) > 0:
            choice_idx = mx.array(choice_indices)
            h_choice = h_norm[choice_idx]
            valid_choice = valid[choice_idx]

            counts = mx.sum(valid_choice.astype(mx.float32), axis=-1, keepdims=True)
            log_k = mx.log(counts)[:, :, None]
            log_k_expanded = mx.broadcast_to(log_k, (len(choice_indices), kmax, 1))

            u_input = mx.concatenate([h_choice, log_k_expanded.astype(h_choice.dtype)], axis=-1)
            u = self.set_project(u_input)

            attn_mask = mx.where(valid_choice[:, None, None, :], 0.0, -1e9)
            mixed = self.set_attention(u, u, u, mask=attn_mask)
            delta = mx.squeeze(self.set_output(mx.tanh(u + mixed)), axis=-1).astype(mx.float32)

            z_list = []
            c_pos = 0
            for i in range(batch_size):
                if i in choice_indices:
                    z_list.append(z[i] + delta[c_pos])
                    c_pos += 1
                else:
                    z_list.append(z[i])
            z = mx.stack(z_list, axis=0)

        out = []
        for i, ex in enumerate(examples):
            if ex["type"] == "boolean":
                b_logits = mx.stack([z[i, 0] * 0.0, z[i, 0]], axis=0)
                if kmax > 2:
                    pad = mx.zeros((kmax - 2,), dtype=b_logits.dtype)
                    b_logits = mx.concatenate([b_logits, pad], axis=0)
                out.append(b_logits)
            else:
                out.append(z[i])

        logits = mx.stack(out, axis=0)
        logits = mx.where(valid, logits, -1e9)
        return logits, valid
