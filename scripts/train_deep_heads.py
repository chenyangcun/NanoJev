#!/usr/bin/env python3
"""Train calibrated deep-MLP decision heads for router classification in MLX."""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from safetensors.numpy import save_file
import numpy as np

from calibrated_objectives_mlx import grouped_calibrated_loss_mlx
from mlx_decision_model import MLXDecisionModel
from mlx_deep_heads import DeepDecisionHeads
from predict_mlx_decisions import load_mlx_decision_model
from predict_toy_decisions import local_checkpoint_files, prepare_examples
from train_pipeline_decisions import (
    SPLITS,
    candidate_ids,
    read_training_records,
    validate_training_row,
)


def load_examples_from_records(records, tokenizer, max_length):
    examples = []
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
    return examples


def extract_leaves(backbone, examples, pad_token_id):
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


def precompute_embeddings(backbone, examples, pad_token_id, batch_size=8):
    print(f"Precomputing embeddings for {len(examples)} examples...", flush=True)
    t0 = time.time()
    cached = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        leaves = extract_leaves(backbone, chunk, pad_token_id)
        offset = 0
        for ex in chunk:
            n = len(ex["leaf_tokens"])
            ex_leaves = leaves[offset : offset + n]
            cached.append((ex, ex_leaves))
            offset += n
    print(f"Embedding cache ready in {time.time() - t0:.1f}s", flush=True)
    return cached


def evaluate_cached_dev(heads, cached_dev, batch_size=16):
    if not cached_dev:
        return 0.0, 0
    total_loss = 0.0
    correct_comp = 0
    total_comp = 0

    for i in range(0, len(cached_dev), batch_size):
        chunk = cached_dev[i : i + batch_size]
        examples = [c[0] for c in chunk]
        leaves = mx.concatenate([c[1] for c in chunk], axis=0)
        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        logits, valid = heads(leaves, examples, kmax)
        mx.eval(logits)
        losses = grouped_calibrated_loss_mlx(logits, examples, "gold_distribution", "ce")
        total_loss += mx.sum(losses).item()

        # Measure accuracy
        for ex, l in zip(examples, logits):
            pred_idx = int(mx.argmax(l[:len(ex["candidate_ids"])]).item())
            pred_val = ex["candidate_ids"][pred_idx]
            gold_val = ex.get("gold")
            if isinstance(gold_val, dict):
                gold_val = gold_val.get(ex["qid"])
            if gold_val is not None:
                if ex["type"] == "boolean":
                    if bool(pred_idx) == bool(gold_val):
                        correct_comp += 1
                elif str(pred_val) == str(gold_val):
                    correct_comp += 1
            total_comp += 1

    acc = correct_comp / max(1, total_comp)
    return total_loss / len(cached_dev), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-checkpoint", default="checkpoints/NanoJev")
    parser.add_argument("--head-name", default="router", help="Name of the head domain to train (e.g. router, skill, agent, news)")
    parser.add_argument("--export-standalone-head", action="store_true", help="Also export lightweight standalone head to <output_dir>/heads/<head_name>.safetensors (~2MB)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-questions", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset: {args.input}")
    records, _ = read_training_records(args.input)

    print(f"Loading base checkpoint: {args.base_checkpoint}")
    model, tokenizer, root, run_config = load_mlx_decision_model(args.base_checkpoint)
    pad_token_id = tokenizer.pad_token_id

    # Upgrade model.heads to DeepDecisionHeads (2-layer MLP)
    deep_heads = DeepDecisionHeads(hidden_size=1024, set_head=run_config.get("set_head", "none"))
    model.heads = deep_heads

    train_rows = [r for r in records if r["split"] == "train"]
    dev_rows = [r for r in records if r["split"] == "dev"]

    train_examples = load_examples_from_records(train_rows, tokenizer, args.max_length)
    dev_examples = load_examples_from_records(dev_rows, tokenizer, args.max_length)
    print(f"Train questions: {len(train_examples)} | Dev questions: {len(dev_examples)}")

    cached_train = precompute_embeddings(model.backbone, train_examples, pad_token_id, batch_size=8)
    cached_dev = precompute_embeddings(model.backbone, dev_examples, pad_token_id, batch_size=8)

    heads = model.heads
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def head_loss_fn(h_model, batch_data, key=None):
        examples = [item[0] for item in batch_data]
        leaves = mx.concatenate([item[1] for item in batch_data], axis=0)
        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        logits, valid = h_model(leaves, examples, kmax)
        losses = grouped_calibrated_loss_mlx(logits, examples, "gold_distribution", "ce", key=key)
        return mx.mean(losses)

    loss_and_grad = nn.value_and_grad(heads, head_loss_fn)

    best_dev_loss = float("inf")
    output_dir = Path(args.output_dir)
    rng_key = mx.random.key(args.seed)

    init_dev, init_acc = evaluate_cached_dev(heads, cached_dev)
    print(f"Initial Dev Loss: {init_dev:.4f} | Dev Complexity Acc: {init_acc*100:.1f}%")

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
        dev_loss, dev_acc = evaluate_cached_dev(heads, cached_dev)
        dt = time.time() - t0

        star = "🌟 (Best)" if dev_loss < best_dev_loss else ""
        if (epoch + 1) % 5 == 0 or dev_loss < best_dev_loss:
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train: {train_loss:.4f} | Dev: {dev_loss:.4f} | Comp Acc: {dev_acc*100:.1f}% | Time: {dt:.2f}s {star}", flush=True)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            output_dir.mkdir(parents=True, exist_ok=True)
            # Save heads weights directly
            flat_weights = {}
            for k, v in heads.parameters().items():
                def recurse(prefix, obj):
                    if isinstance(obj, mx.array):
                        flat_weights[prefix] = obj
                    elif isinstance(obj, dict):
                        for sub_k, sub_v in obj.items():
                            recurse(f"{prefix}.{sub_k}" if prefix else sub_k, sub_v)
                recurse(k, v)
            
            # Save full weights
            root_paths = local_checkpoint_files(args.base_checkpoint)[1]
            from safetensors import safe_open
            full_weights = {}
            with safe_open(str(root_paths["weights"]), framework="numpy") as f:
                for k in f.keys():
                    if k.startswith("backbone."):
                        full_weights[k] = f.get_tensor(k)
                    elif args.head_name != "router":
                        # Preserve existing heads when training a specialized head
                        full_weights[k] = f.get_tensor(k)
            # Add deep heads with namespace
            for k, v in flat_weights.items():
                if args.head_name == "router":
                    full_weights["heads." + k] = np.array(v)
                else:
                    full_weights[f"heads.{args.head_name}.{k}"] = np.array(v)

            save_file(full_weights, str(output_dir / "best.safetensors"))

            # Optionally export lightweight standalone head (~2MB)
            if args.export_standalone_head:
                heads_dir = output_dir / "heads"
                heads_dir.mkdir(parents=True, exist_ok=True)
                standalone = {f"heads.{k}": np.array(v) for k, v in flat_weights.items()}
                save_file(standalone, str(heads_dir / f"{args.head_name}.safetensors"))
                print(f"[Pluggable] Exported standalone head: {heads_dir / f'{args.head_name}.safetensors'}")

            (output_dir / "config.json").write_text(json.dumps({
                "model": "Qwen3-0.6B",
                "heads_architecture": "deep_mlp",
                "head_name": args.head_name,
                "set_head": run_config.get("set_head", "none"),
                "max_length": args.max_length,
                "best_dev_loss": best_dev_loss,
            }, indent=2) + "\n")

    print(f"\nDone! Best Dev Loss: {best_dev_loss:.4f}")


if __name__ == "__main__":
    main()
