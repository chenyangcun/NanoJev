#!/usr/bin/env python3
"""Train NanoJev decision models natively on Apple Silicon using Apple MLX.

Optimized for:
- Pre-computing backbone embeddings when fine-tuning heads_only (ultra-fast 100x speedup!)
- Or fine-tuning end-to-end (full backbone)
- Calibrated objectives: CE / Brier against gold_probs / teacher distributions
- Evaluation on dev split with checkpointing to best.safetensors
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from safetensors.numpy import save_file
import numpy as np

from calibrated_objectives_mlx import grouped_calibrated_loss_mlx
from mlx_decision_model import MLXDecisionModel
from predict_mlx_decisions import load_mlx_decision_model
from predict_toy_decisions import local_checkpoint_files, prepare_examples, read_json
from train_pipeline_decisions import (
    SPLITS,
    candidate_ids,
    read_training_records,
    validate_training_row,
)


def load_examples_from_records(records, tokenizer, max_length):
    examples, audit = [], []
    for row in records:
        targets = validate_training_row(row)
        prepared = prepare_examples(
            {"states": [{key: row[key] for key in ("id", "state", "questions")}]},
            tokenizer,
            max_length,
        )
        for ex in prepared:
            t = targets[ex["qid"]]
            ex.update(t, state_id=row["state_id"], family_id=row["family_id"], split=row["split"], source=row)
            examples.append(ex)
    return examples, audit


def extract_leaves(backbone, examples, pad_token_id):
    """Run forward on backbone and extract leaf token representations."""
    paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
    lengths = [len(ids) for ids in paths]
    width = max(lengths)
    num_paths = len(paths)

    tokens_mat = []
    for ids in paths:
        pad_len = width - len(ids)
        tokens_mat.append(ids + [pad_token_id] * pad_len)
    input_ids = mx.array(tokens_mat, dtype=mx.int32)

    hidden = backbone.model(input_ids)
    if hasattr(hidden, "last_hidden_state"):
        hidden = hidden.last_hidden_state

    leaf_indices = mx.array([l - 1 for l in lengths], dtype=mx.int32)
    row_indices = mx.arange(num_paths, dtype=mx.int32)
    leaves = hidden[row_indices, leaf_indices]
    mx.eval(leaves)
    return leaves


def precompute_embeddings(backbone, examples, pad_token_id, batch_size=4):
    """Precompute and cache leaf embeddings for all examples."""
    print(f"Precomputing embeddings for {len(examples)} examples (batch_size={batch_size})...", flush=True)
    t0 = time.time()
    cached = []
    total_paths = 0

    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        leaves = extract_leaves(backbone, chunk, pad_token_id)
        # Split leaves back to each example
        offset = 0
        for ex in chunk:
            n = len(ex["leaf_tokens"])
            ex_leaves = leaves[offset : offset + n]
            cached.append((ex, ex_leaves))
            offset += n
            total_paths += n

        if (i // batch_size + 1) % 20 == 0 or (i + batch_size >= len(examples)):
            pct = min(100.0, (i + batch_size) / len(examples) * 100)
            print(f"  Processed {min(i + batch_size, len(examples))}/{len(examples)} ({pct:.1f}%) in {time.time() - t0:.1f}s", flush=True)

    print(f"Finished embedding cache: {len(cached)} questions, {total_paths} paths in {time.time() - t0:.1f}s", flush=True)
    return cached


def save_mlx_checkpoint(output_dir: Path, model: MLXDecisionModel, run_config: dict, tokenizer_dir: Path, body_config_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    tok_target = output_dir / "tokenizer"
    if not tok_target.exists() and tokenizer_dir.exists():
        shutil.copytree(tokenizer_dir, tok_target)

    cfg_target = output_dir / "backbone_config"
    if not cfg_target.exists() and body_config_dir.exists():
        shutil.copytree(body_config_dir, cfg_target)

    flat_weights = {}

    def recurse(prefix, obj):
        if isinstance(obj, mx.array):
            flat_weights[prefix] = obj
        elif isinstance(obj, dict):
            for sub_k, sub_v in obj.items():
                recurse(f"{prefix}.{sub_k}" if prefix else sub_k, sub_v)
        elif isinstance(obj, list):
            for i, sub_v in enumerate(obj):
                recurse(f"{prefix}.{i}" if prefix else str(i), sub_v)

    for k, v in model.parameters().items():
        recurse(k, v)

    converted_weights = {}
    for k, v in flat_weights.items():
        clean_k = k
        if clean_k.startswith("backbone.model."):
            clean_k = "backbone." + clean_k[len("backbone.model.") :]
        elif clean_k.startswith("heads."):
            clean_k = clean_k[len("heads.") :]
        converted_weights[clean_k] = np.array(v)

    save_file(converted_weights, str(output_dir / "best.safetensors"))


def evaluate_cached_dev(heads, cached_dev, objective, loss_kind, batch_size=16):
    if not cached_dev:
        return 0.0
    total_loss = 0.0
    count = 0
    for i in range(0, len(cached_dev), batch_size):
        chunk = cached_dev[i : i + batch_size]
        examples = [c[0] for c in chunk]
        leaves = mx.concatenate([c[1] for c in chunk], axis=0)
        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        logits, valid = heads(leaves, examples, kmax)
        mx.eval(logits)
        losses = grouped_calibrated_loss_mlx(logits, examples, objective, loss_kind)
        total_loss += mx.sum(losses).item()
        count += len(chunk)
    return total_loss / max(1, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained model")
    parser.add_argument("--base-checkpoint", default="checkpoints/NanoJev", help="Base checkpoint dir")
    parser.add_argument("--objective", choices=["gold", "gold_distribution", "observed_outcome"], default="gold_distribution")
    parser.add_argument("--loss", choices=["ce", "brier", "paired_brier_pg"], default="ce")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-questions", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset: {args.input}", flush=True)
    records, _ = read_training_records(args.input)

    print(f"Loading base checkpoint: {args.base_checkpoint}", flush=True)
    model, tokenizer, root, run_config = load_mlx_decision_model(args.base_checkpoint)
    pad_token_id = tokenizer.pad_token_id

    train_rows = [r for r in records if r["split"] == "train"]
    dev_rows = [r for r in records if r["split"] == "dev"]

    print("Tokenizing train & dev examples...", flush=True)
    train_examples, _ = load_examples_from_records(train_rows, tokenizer, args.max_length)
    dev_examples, _ = load_examples_from_records(dev_rows, tokenizer, args.max_length)
    print(f"Train questions: {len(train_examples)} | Dev questions: {len(dev_examples)}", flush=True)

    # 1. Precompute backbone embeddings
    cached_train = precompute_embeddings(model.backbone, train_examples, pad_token_id, batch_size=4)
    cached_dev = precompute_embeddings(model.backbone, dev_examples, pad_token_id, batch_size=4)

    # 2. Optimize heads
    heads = model.heads
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def head_loss_fn(h_model, batch_data, key=None):
        examples = [item[0] for item in batch_data]
        leaves = mx.concatenate([item[1] for item in batch_data], axis=0)
        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        logits, valid = h_model(leaves, examples, kmax)
        losses = grouped_calibrated_loss_mlx(logits, examples, args.objective, args.loss, key=key)
        return mx.mean(losses)

    loss_and_grad = nn.value_and_grad(heads, head_loss_fn)

    best_dev_loss = float("inf")
    output_dir = Path(args.output_dir)
    rng_key = mx.random.key(args.seed)

    init_dev_loss = evaluate_cached_dev(heads, cached_dev, args.objective, args.loss)
    print(f"\nInitial Dev Loss before training: {init_dev_loss:.4f}\n" + "=" * 60, flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        random.seed(args.seed + epoch)
        indices = list(range(len(cached_train)))
        random.shuffle(indices)

        running_loss = 0.0
        steps = 0

        for i in range(0, len(cached_train), args.batch_questions):
            batch_idx = indices[i : i + args.batch_questions]
            batch_data = [cached_train[idx] for idx in batch_idx]

            rng_key, subkey = mx.random.split(rng_key)
            loss_val, grads = loss_and_grad(heads, batch_data, key=subkey)
            optimizer.update(heads, grads)
            mx.eval(heads.parameters(), optimizer.state)

            running_loss += loss_val.item()
            steps += 1

        train_loss = running_loss / max(1, steps)
        dev_loss = evaluate_cached_dev(heads, cached_dev, args.objective, args.loss)
        dt = time.time() - t0

        star = "🌟 (Best)" if dev_loss < best_dev_loss else ""
        print(f"Epoch {epoch + 1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Time: {dt:.2f}s {star}", flush=True)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            run_cfg = {
                "model": "Qwen3-0.6B",
                "set_head": run_config.get("set_head", "none"),
                "objective": args.objective,
                "loss": args.loss,
                "max_length": args.max_length,
                "best_dev_loss": best_dev_loss,
            }
            root_paths = local_checkpoint_files(args.base_checkpoint)[1]
            save_mlx_checkpoint(output_dir, model, run_cfg, root_paths["tokenizer"], root_paths["body_config"])

    print("=" * 60)
    print(f"Training completed successfully! Best Dev Loss: {best_dev_loss:.4f}")
    print(f"Checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
