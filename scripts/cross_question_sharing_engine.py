"""High-Performance Unified Memory & Cross-Question State Sharing Engine with:
1. Dynamic Adaptive Temperature Scaling (Margin & Entropy based)
2. Dynamic Early Exit (Layer-Skipping on ultra-confident predictions)
3. Persistent Static KV-Cache Memory Pool
4. JIT-compiled Deep Decision Heads
"""
import copy
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache

from predict_toy_decisions import answer_from_probabilities


class StaticKVCachePool:
    """Pre-allocated static reusable KV-Cache pool to avoid memory churn."""

    def __init__(self, qwen_model):
        self.qwen_model = qwen_model
        self.cache = cache.make_prompt_cache(qwen_model)

    def reset_to_empty(self):
        """In-place reset cache completely to 0 offset and Batch=1."""
        for c in self.cache:
            if c.offset > 0:
                c.trim(c.offset)
            if c.keys is not None and c.keys.shape[0] > 1:
                c.keys = c.keys[:1]
                c.values = c.values[:1]
        return self.cache

    def backtrack_to_offset(self, target_offset: int):
        """Backtrack in-place to a specific checkpoint and reset batch to 1."""
        for c in self.cache:
            trim_amt = c.offset - target_offset
            if trim_amt > 0:
                c.trim(trim_amt)
            if c.keys is not None and c.keys.shape[0] > 1:
                c.keys = c.keys[:1]
                c.values = c.values[:1]


# JIT Compiled classification head forward for kernel fusion
@mx.compile
def compiled_heads_forward(leaves, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b):
    """Fused LayerNorm + 2-layer GELU MLP in Metal kernel."""
    mean = mx.mean(leaves, axis=-1, keepdims=True)
    var = mx.var(leaves, axis=-1, keepdims=True)
    normed = (leaves - mean) * mx.rsqrt(var + 1e-5) * norm_w + norm_b
    h = mx.matmul(normed, fc1_w.T) + fc1_b
    h_gelu = nn.gelu(h)
    logits = mx.matmul(h_gelu, fc2_w.T) + fc2_b
    return logits


def compute_adaptive_temperature(scores: mx.array, base_temp: float = 0.35, min_temp: float = 0.25, max_temp: float = 0.85) -> float:
    """Dynamically adjust temperature based on margin separation between top candidates.

    - Large separation (delta >= 2.0): clear unambiguous decision -> low temperature (sharp confidence >= 0.95)
    - Small separation (delta <= 0.4): ambiguous edge case -> higher temperature (preserves natural uncertainty for fallback)
    """
    if scores.size <= 1:
        return base_temp

    s_sorted = mx.sort(scores)[::-1]
    delta = (s_sorted[0] - s_sorted[1]).item()

    # Sigmoidal interpolation between min_temp and max_temp
    # center = 1.3, slope = 2.0
    t = max_temp - (max_temp - min_temp) / (1.0 + math.exp(-2.0 * (delta - 1.3)))
    return round(t, 3)


def build_question_suffix_and_candidates(question_dict: dict, tokenizer) -> Tuple[List[int], List[str], List[List[int]]]:
    typ = question_dict["type"]
    if typ == "boolean":
        ids = ["false", "true"]
        texts = ["The proposition is true."]
    elif typ == "choice":
        ids = list(question_dict["criteria"])
        texts = [f"{key}: {question_dict['criteria'][key]}" for key in ids]
    else:
        ids = [str(i) for i in range(len(question_dict["criteria"]))]
        texts = question_dict["criteria"]

    q_text = f"Question type: {typ}\nQuestion:\n{question_dict['instructions']}\n"
    if typ == "boolean" and "criteria" in question_dict:
        for key, label in (("false", "False"), ("true", "True")):
            if key in question_dict["criteria"]:
                q_text += f"{label} criterion: {question_dict['criteria'][key]}\n"

    q_tokens = tokenizer.encode(q_text, add_special_tokens=False)
    cand_tokens = [
        tokenizer.encode(f"Candidate:\n{t}\nDecision:", add_special_tokens=False) + [tokenizer.eos_token_id]
        for t in texts
    ]
    return q_tokens, ids, cand_tokens


def evaluate_state_cross_question_sharing(
    model,
    tokenizer,
    cache_pool: StaticKVCachePool,
    state_id: str,
    state_val: Any,
    questions_dict: dict,
    temperature: Optional[float] = None,
    enable_adaptive_temp: bool = True,
    early_exit_layer: int = 0,
    early_exit_confidence: float = 0.98,
    max_length: int = 4096,
) -> Tuple[Dict[str, Any], int]:
    """Execute Hierarchical Cross-Question State Sharing with Adaptive Temperature and Early Exit."""
    state_str = state_val if isinstance(state_val, str) else json.dumps(state_val, ensure_ascii=False)
    qwen_model = model.backbone.model
    heads = model.heads
    pad_token_id = tokenizer.pad_token_id

    # 1. Reset cache pool to 0
    prompt_cache = cache_pool.reset_to_empty()

    # 2. LEVEL 1: State Prefill (COMPUTED EXACTLY ONCE FOR THE ENTIRE REQUEST)
    state_text = f"State:\n{state_str}\n"
    state_tokens = tokenizer.encode(state_text, add_special_tokens=False)
    state_mat = mx.array([state_tokens], dtype=mx.int32)
    _ = qwen_model(state_mat, cache=prompt_cache)
    state_offset = prompt_cache[0].offset

    answers = {}
    total_tokens = len(state_tokens)

    # 3. LEVEL 2 & 3: Iterate through questions branching from the shared State cache
    for qid, q in questions_dict.items():
        q_tokens, candidate_ids, cand_tokens_list = build_question_suffix_and_candidates(q, tokenizer)
        total_tokens += len(q_tokens) + sum(len(c) for c in cand_tokens_list)
        K = len(cand_tokens_list)

        # Level 2: Prefill Question instructions on top of State cache (Batch=1)
        q_mat = mx.array([q_tokens], dtype=mx.int32)
        _ = qwen_model(q_mat, cache=prompt_cache)

        # Level 3: Fork K candidate branches in parallel
        cand_lengths = [len(c) for c in cand_tokens_list]
        max_cand_len = max(cand_lengths)

        cand_mat = []
        for c in cand_tokens_list:
            pad_len = max_cand_len - len(c)
            cand_mat.append(c + [pad_token_id] * pad_len)
        cand_ids = mx.array(cand_mat, dtype=mx.int32)

        # Broadcast KV cache in batch dimension to K
        for layer_c in prompt_cache:
            layer_c.keys = mx.broadcast_to(layer_c.keys, (K, layer_c.keys.shape[1], layer_c.keys.shape[2], layer_c.keys.shape[3]))
            layer_c.values = mx.broadcast_to(layer_c.values, (K, layer_c.values.shape[1], layer_c.values.shape[2], layer_c.values.shape[3]))

        # Dynamic Early Exit Evaluation
        # If early_exit_layer is configured (>0 and <28), we can probe intermediate hidden states
        leaf_indices = mx.array([l - 1 for l in cand_lengths], dtype=mx.int32)
        row_indices = mx.arange(K, dtype=mx.int32)

        if 0 < early_exit_layer < len(qwen_model.layers):
            from mlx_lm.models.base import create_attention_mask
            h = qwen_model.embed_tokens(cand_ids)
            mask = create_attention_mask(h, prompt_cache[0])
            for i in range(early_exit_layer):
                h = qwen_model.layers[i](h, mask, prompt_cache[i])
            h_early = qwen_model.norm(h)
            early_leaves = h_early[row_indices, leaf_indices]

            # Compute early logits
            if hasattr(heads, "fc1") and hasattr(heads, "fc2") and heads.set_head == "none":
                early_raw = compiled_heads_forward(
                    early_leaves, heads.norm.weight, heads.norm.bias,
                    heads.fc1.weight, heads.fc1.bias, heads.fc2.weight, heads.fc2.bias
                )
                mx.eval(early_raw)
                z_early = mx.squeeze(early_raw, axis=-1).astype(mx.float32)
                scores_early = mx.stack([0.0 * z_early[0], z_early[0]], axis=0) if q["type"] == "boolean" else z_early
            else:
                scores_early = None

            # Check early confidence
            if scores_early is not None:
                p_early = mx.softmax(scores_early / 0.35, axis=-1).tolist()
                mock_ex = [{"id": f"{state_id}:{qid}", "qid": qid, "type": q["type"], "candidate_ids": candidate_ids}]
                ans_early = answer_from_probabilities(mock_ex[0], p_early)
                # If confidence exceeds threshold, early exit!
                if ans_early.get("confidence", 0.0) >= early_exit_confidence:
                    ans_early["early_exited_at_layer"] = early_exit_layer
                    answers[qid] = ans_early
                    cache_pool.backtrack_to_offset(state_offset)
                    continue

            # If not early exited, complete remaining layers
            for i in range(early_exit_layer, len(qwen_model.layers)):
                h = qwen_model.layers[i](h, mask, prompt_cache[i])
            cand_hidden = qwen_model.norm(h)
        else:
            # Full model execution
            cand_hidden = qwen_model(cand_ids, cache=prompt_cache)

        # Extract leaves at end of each candidate branch
        leaves = cand_hidden[row_indices, leaf_indices]

        # Calculate logits via compiled heads
        mock_example = [
            {
                "id": f"{state_id}:{qid}",
                "state_id": state_id,
                "qid": qid,
                "type": q["type"],
                "candidate_ids": candidate_ids,
                "leaf_tokens": cand_tokens_list,
            }
        ]

        if hasattr(heads, "fc1") and hasattr(heads, "fc2") and heads.set_head == "none":
            raw_scores = compiled_heads_forward(
                leaves, heads.norm.weight, heads.norm.bias,
                heads.fc1.weight, heads.fc1.bias, heads.fc2.weight, heads.fc2.bias
            )
            mx.eval(raw_scores)
            z = mx.squeeze(raw_scores, axis=-1).astype(mx.float32)
            scores = mx.stack([0.0 * z[0], z[0]], axis=0) if q["type"] == "boolean" else z
        else:
            logits, _ = heads(leaves, mock_example, kmax=len(candidate_ids))
            mx.eval(logits)
            scores = logits[0, :len(candidate_ids)]

        # Dynamic Adaptive Temperature Scaling
        if temperature is not None and not enable_adaptive_temp:
            effective_temp = float(temperature)
        elif enable_adaptive_temp:
            effective_temp = compute_adaptive_temperature(scores, base_temp=0.35)
        else:
            effective_temp = 0.35

        probs = mx.softmax(scores / effective_temp, axis=-1).tolist()
        ans = answer_from_probabilities(mock_example[0], probs)
        ans["effective_temperature"] = effective_temp
        answers[qid] = ans

        # BACKTRACK: In-place trim cache back to state_offset and reset batch to 1 for the next question!
        cache_pool.backtrack_to_offset(state_offset)

    return answers, total_tokens
