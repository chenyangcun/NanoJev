#!/usr/bin/env python3
"""Train MLX-Native PointerHead for Large Candidate Selection (e.g. Skill Selection).

Uses two-stage feature extraction:
- Extracts h_decide (task/question intent representation) from end of question prefix
- Extracts h_opts (candidate representations) from end of each candidate branch
- Optimizes (K(h_opts) @ Q(h_decide)) matching scores using Calibrated Cross-Entropy
- Exports standalone pluggable head to <output_dir>/heads/skill.safetensors (< 1MB)
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from safetensors.numpy import save_file
import numpy as np

from calibrated_objectives_mlx import grouped_calibrated_loss_mlx
from mlx_pointer_head import PointerHead
from predict_mlx_decisions import load_mlx_decision_model
from predict_toy_decisions import local_checkpoint_files, prepare_examples
from train_pipeline_decisions import read_training_records, validate_training_row


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


def extract_pointer_features(backbone, examples, tokenizer, pad_token_id):
    """Extract both h_decide (from prefix) and h_opts (from candidate leaves)."""
    cached = []
    for ex in examples:
        paths = ex["leaf_tokens"]
        lengths = [len(p) for p in paths]
        width = max(lengths)
        tokens_mat = [p + [pad_token_id] * (width - len(p)) for p in paths]
        tokens = mx.array(tokens_mat, dtype=mx.int32)

        # Forward through backbone
        hidden = backbone.model(tokens)
        if hasattr(hidden, "last_hidden_state"):
            hidden = hidden.last_hidden_state

        # Find prefix end for h_decide
        cand0_suffix = tokenizer.encode(
            "Candidate:\n" + ex["candidate_texts"][0] + "\nDecision:", add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        prefix_len = len(ex["leaf_tokens"][0]) - len(cand0_suffix)

        h_decide = hidden[0, prefix_len - 1]
        row_indices = mx.arange(len(paths), dtype=mx.int32)
        leaf_indices = mx.array([l - 1 for l in lengths], dtype=mx.int32)
        h_opts = hidden[row_indices, leaf_indices]

        mx.eval(h_decide, h_opts)
        cached.append((ex, h_decide, h_opts))

    return cached


def precompute_pointer_cache(backbone, examples, tokenizer, pad_token_id, batch_size=4):
    print(f"Precomputing pointer embeddings for {len(examples)} examples...", flush=True)
    t0 = time.time()
    cached = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        items = extract_pointer_features(backbone, chunk, tokenizer, pad_token_id)
        cached.extend(items)
        if (i // batch_size + 1) % 25 == 0 or (i + batch_size >= len(examples)):
            pct = min(100.0, (i + batch_size) / len(examples) * 100)
            print(f"  Processed {min(i + batch_size, len(examples))}/{len(examples)} ({pct:.1f}%) in {time.time() - t0:.1f}s", flush=True)
    print(f"Pointer embedding cache ready in {time.time() - t0:.1f}s", flush=True)
    return cached


def evaluate_pointer_dev(head, cached_dev, batch_size=16):
    if not cached_dev:
        return 0.0, 0.0
    total_loss = 0.0
    correct = 0
    total = 0

    for i in range(0, len(cached_dev), batch_size):
        chunk = cached_dev[i : i + batch_size]
        for ex, h_dec, h_opts in chunk:
            k = len(ex["candidate_ids"])
            logits, valid = head(h_opts, [ex], kmax=k, h_decide=h_dec)
            mx.eval(logits)
            loss = grouped_calibrated_loss_mlx(logits, [ex], "gold_distribution", "ce")
            total_loss += mx.sum(loss).item()

            pred_idx = int(mx.argmax(logits[0, :k]).item())
            pred_val = ex["candidate_ids"][pred_idx]
            gold_val = ex.get("gold")
            if isinstance(gold_val, dict):
                gold_val = gold_val.get(ex["qid"])
            if gold_val is not None:
                if ex["type"] == "boolean":
                    if bool(pred_idx) == bool(gold_val):
                        correct += 1
                elif str(pred_val) == str(gold_val):
                    correct += 1
            total += 1

    return total_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-checkpoint", default="checkpoints/NanoJev")
    parser.add_argument("--head-name", default="skill")
    parser.add_argument("--pointer-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-questions", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset: {args.input}", flush=True)
    records, _ = read_training_records(args.input)

    print(f"Loading base checkpoint: {args.base_checkpoint}", flush=True)
    model, tokenizer, root, run_config = load_mlx_decision_model(args.base_checkpoint)
    pad_token_id = tokenizer.pad_token_id

    pointer_head = PointerHead(hidden_size=1024, pointer_dim=args.pointer_dim)

    train_rows = [r for r in records if r["split"] == "train"]
    dev_rows = [r for r in records if r["split"] == "dev"]

    train_examples = load_examples_from_records(train_rows, tokenizer, args.max_length)
    dev_examples = load_examples_from_records(dev_rows, tokenizer, args.max_length)
    print(f"Train questions: {len(train_examples)} | Dev questions: {len(dev_examples)}", flush=True)

    cached_train = precompute_pointer_cache(model.backbone, train_examples, tokenizer, pad_token_id)
    cached_dev = precompute_pointer_cache(model.backbone, dev_examples, tokenizer, pad_token_id)

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def loss_fn(p_head, batch_items, key=None):
        losses = []
        for ex, h_dec, h_opts in batch_items:
            k = len(ex["candidate_ids"])
            logits, valid = p_head(h_opts, [ex], kmax=k, h_decide=h_dec)
            l = grouped_calibrated_loss_mlx(logits, [ex], "gold_distribution", "ce", key=key)
            losses.append(l)
        return mx.mean(mx.concatenate(losses))

    loss_and_grad = nn.value_and_grad(pointer_head, loss_fn)

    best_dev_loss = float("inf")
    output_dir = Path(args.output_dir)
    rng_key = mx.random.key(args.seed)

    init_dev, init_acc = evaluate_pointer_dev(pointer_head, cached_dev)
    print(f"Initial Dev Loss: {init_dev:.4f} | Accuracy: {init_acc*100:.1f}%\n" + "=" * 65, flush=True)

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
            loss_val, grads = loss_and_grad(pointer_head, batch_data, key=subkey)
            optimizer.update(pointer_head, grads)
            mx.eval(pointer_head.parameters(), optimizer.state)

            running_loss += loss_val.item()
            steps += 1

        train_loss = running_loss / max(1, steps)
        dev_loss, dev_acc = evaluate_pointer_dev(pointer_head, cached_dev)
        dt = time.time() - t0

        star = "🌟 (Best)" if dev_loss < best_dev_loss else ""
        if (epoch + 1) % 5 == 0 or dev_loss < best_dev_loss:
            print(
                f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train: {train_loss:.4f} | Dev: {dev_loss:.4f} | Acc: {dev_acc*100:.1f}% | Time: {dt:.2f}s {star}",
                flush=True,
            )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            output_dir.mkdir(parents=True, exist_ok=True)

            # Flatten pointer head parameters
            flat_weights = {}
            for k, v in pointer_head.parameters().items():
                def recurse(prefix, obj):
                    if isinstance(obj, mx.array):
                        flat_weights[prefix] = obj
                    elif isinstance(obj, dict):
                        for sub_k, sub_v in obj.items():
                            recurse(f"{prefix}.{sub_k}" if prefix else sub_k, sub_v)
                recurse(k, v)

            # Save standalone pluggable head to <output_dir>/heads/<head_name>.safetensors (~0.5MB)
            heads_dir = output_dir / "heads"
            heads_dir.mkdir(parents=True, exist_ok=True)
            standalone = {f"heads.{k}": np.array(v) for k, v in flat_weights.items()}
            save_file(standalone, str(heads_dir / f"{args.head_name}.safetensors"))

            # Also save updated best.safetensors containing all heads
            root_paths = local_checkpoint_files(args.base_checkpoint)[1]
            from safetensors import safe_open
            full_weights = {}
            with safe_open(str(root_paths["weights"]), framework="numpy") as f:
                for k in f.keys():
                    full_weights[k] = f.get_tensor(k)
            for k, v in flat_weights.items():
                full_weights[f"heads.{args.head_name}.{k}"] = np.array(v)
            save_file(full_weights, str(output_dir / "best.safetensors"))

            (output_dir / "config.json").write_text(
                json.dumps({
                    "model": "Qwen3-0.6B",
                    "head_type": "pointer_head",
                    "pointer_dim": args.pointer_dim,
                    "head_name": args.head_name,
                    "best_dev_loss": best_dev_loss,
                }, indent=2) + "\n"
            )

    print(f"\nTraining Complete! Best Dev Loss: {best_dev_loss:.4f}")
    print(f"Exported standalone PointerHead: {heads_dir / f'{args.head_name}.safetensors'}")


if __name__ == "__main__":
    main()
