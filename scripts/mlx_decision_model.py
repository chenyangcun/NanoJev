"""NanoJev DecisionModel implementation in Apple MLX.

Includes:
- DecisionHeads: Norm, Scalar head, and optional Set-Attention head
- MLXDecisionModel: Qwen backbone + DecisionHeads
- Forward pass with candidate packing and multi-path pooling
"""
import math
from typing import Dict, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


class DecisionHeads(nn.Module):
    def __init__(self, hidden_size: int, set_head: str = "none"):
        super().__init__()
        self.hidden_size = hidden_size
        self.set_head = set_head
        self.norm = nn.LayerNorm(hidden_size)
        self.scalar = nn.Linear(hidden_size, 1)

        if set_head == "attention":
            self.set_project = nn.Linear(hidden_size + 1, 128)
            # bias=True to match PyTorch nn.MultiheadAttention default
            self.set_attention = nn.MultiHeadAttention(dims=128, num_heads=4, bias=True)
            self.set_output = nn.Linear(128, 1)

    def __call__(
        self,
        leaves: mx.array,
        examples: List[dict],
        kmax: int,
    ) -> Tuple[mx.array, mx.array]:
        """Compute decision logits from candidate leaf embeddings."""
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

        # Norm + Linear
        h_norm = self.norm(h)
        z = mx.squeeze(self.scalar(h_norm), axis=-1).astype(mx.float32)  # [batch_size, kmax]

        # Handle Choice set_attention
        choice_indices = [i for i, ex in enumerate(examples) if ex["type"] == "choice"]
        if self.set_head == "attention" and len(choice_indices) > 0:
            choice_idx = mx.array(choice_indices)
            h_choice = h_norm[choice_idx]  # [num_choice, kmax, hidden_size]
            valid_choice = valid[choice_idx]  # [num_choice, kmax]

            counts = mx.sum(valid_choice.astype(mx.float32), axis=-1, keepdims=True)  # [num_choice, 1]
            log_k = mx.log(counts)[:, :, None]  # [num_choice, 1, 1]
            log_k_expanded = mx.broadcast_to(log_k, (len(choice_indices), kmax, 1))

            u_input = mx.concatenate([h_choice, log_k_expanded.astype(h_choice.dtype)], axis=-1)
            u = self.set_project(u_input)  # [num_choice, kmax, 128]

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


class MLXDecisionModel(nn.Module):
    def __init__(self, backbone, set_head: str = "none"):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.model.embed_tokens.weight.shape[1]
        self.heads = DecisionHeads(hidden_size=hidden_size, set_head=set_head)

    def __call__(self, examples: List[dict], pad_token_id: int):
        paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
        lengths = [len(ids) for ids in paths]
        width = max(lengths)
        num_paths = len(paths)

        tokens_mat = []
        for ids in paths:
            pad_len = width - len(ids)
            tokens_mat.append(ids + [pad_token_id] * pad_len)
        input_ids = mx.array(tokens_mat, dtype=mx.int32)

        hidden = self.backbone.model(input_ids)
        if hasattr(hidden, "last_hidden_state"):
            hidden = hidden.last_hidden_state

        leaf_indices = mx.array([l - 1 for l in lengths], dtype=mx.int32)
        row_indices = mx.arange(num_paths, dtype=mx.int32)
        leaves = hidden[row_indices, leaf_indices]

        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        return self.heads(leaves, examples, kmax)
