#!/usr/bin/env python3
"""Train NanoJev decision models natively on Apple Silicon using RLCD in Apple MLX.

Algorithm:
- Reinforcement Learning for Calibrated Decisions (RLCD)
- Logit Gaussian exploration (M=4, sigma=0.3)
- Proper scoring rule composite reward (Log-Score + 0.75 * Spherical - RPS penalty)
- Advantage normalization with group baseline
- Auxiliary Cross-Entropy loss (weight=1.0)
- Precomputed backbone embeddings for fast training (100x speedup)
- Dev NLL checkpoint selection and standalone head export
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
import numpy as np
from safetensors.numpy import save_file

from calibrated_rlcd_mlx import RLCDConfig, grouped_rlcd_loss_mlx
from mlx_decision_model import MLXDecisionModel
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
            ex.update(
                t,
                state_id=row["state_id"],
                family_id=row["family_id"],
                split=row["split"],
                source=row,
                gold=row.get("gold", {}),
            )
            examples.append(ex)
    return examples


def extract_leaves_for_paths(backbone, paths, pad_token_id):
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
    leaves_np = np.array(leaves, dtype=np.float16)
    del hidden, input_ids
    return leaves_np


def precompute_embeddings(backbone, examples, pad_token_id, max_paths_per_chunk=48, cache_path: Path = None):
    if cache_path and cache_path.exists():
        print(f"Loading precomputed embeddings from cache: {cache_path}...", flush=True)
        t0 = time.time()
        data = np.load(cache_path)
        cached = []
        for i, ex in enumerate(examples):
            key = f"arr_{i}"
            if key in data:
                cached.append((ex, mx.array(data[key], dtype=mx.float32)))
            else:
                cached = []
                break
        if len(cached) == len(examples):
            print(f"Loaded {len(cached)} cached embeddings in {time.time() - t0:.2f}s", flush=True)
            return cached

    print(f"Precomputing embeddings for {len(examples)} examples (max_paths={max_paths_per_chunk})...", flush=True)
    t0 = time.time()
    cached = []
    arrays_to_save = {}
    total_paths = 0

    idx = 0
    num_ex = len(examples)

    while idx < num_ex:
        chunk = []
        chunk_paths = []
        while idx < num_ex:
            ex = examples[idx]
            n = len(ex["leaf_tokens"])
            if chunk and len(chunk_paths) + n > max_paths_per_chunk:
                break
            chunk.append(ex)
            chunk_paths.extend(ex["leaf_tokens"])
            idx += 1

        leaves_np = extract_leaves_for_paths(backbone, chunk_paths, pad_token_id)
        mx.clear_cache()

        offset = 0
        for ex in chunk:
            n = len(ex["leaf_tokens"])
            ex_np = leaves_np[offset : offset + n]
            ex_leaves = mx.array(ex_np, dtype=mx.float32)
            c_idx = len(cached)
            arrays_to_save[f"arr_{c_idx}"] = ex_np
            cached.append((ex, ex_leaves))
            offset += n
            total_paths += n

        if len(cached) % 250 == 0 or idx >= num_ex:
            pct = min(100.0, idx / num_ex * 100)
            elapsed = time.time() - t0
            rate = len(cached) / max(0.1, elapsed)
            print(f"  Processed {idx}/{num_ex} ({pct:.1f}%) | {rate:.1f} q/s | Elapsed: {elapsed:.1f}s", flush=True)

    print(f"Embedding cache ready: {len(cached)} questions, {total_paths} paths in {time.time() - t0:.1f}s", flush=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving precomputed embeddings to cache: {cache_path}...", flush=True)
        np.savez_compressed(cache_path, **arrays_to_save)
        print(f"Saved cache ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    return cached


def evaluate_cached_dev(heads, cached_dev, batch_size=16):
    if not cached_dev:
        return 0.0, 0.0
    total_nll = 0.0
    correct = 0
    total = 0

    for i in range(0, len(cached_dev), batch_size):
        chunk = cached_dev[i : i + batch_size]
        examples = [c[0] for c in chunk]
        leaves = mx.concatenate([c[1] for c in chunk], axis=0)
        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        logits, valid = heads(leaves, examples, kmax)
        mx.eval(logits)

        for ex, l in zip(examples, logits):
            k = len(ex["candidate_ids"])
            z = l[:k]
            # NLL against target
            target = ex.get("gold_distribution_probs")
            if target is None and "gold_index" in ex:
                target = [float(j == ex["gold_index"]) for j in range(k)]
            if target is not None:
                t_arr = mx.array(target, dtype=mx.float32)
                logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
                nll = -mx.sum(t_arr * logp).item()
                total_nll += nll

            pred_idx = int(mx.argmax(z).item())
            gold_val = ex.get("gold", {})
            if isinstance(gold_val, dict):
                gold_val = gold_val.get(ex["qid"])
            if gold_val is not None:
                if ex["type"] == "boolean":
                    if bool(pred_idx) == bool(gold_val):
                        correct += 1
                elif ex["type"] == "choice":
                    if pred_idx < len(ex["candidate_ids"]) and str(ex["candidate_ids"][pred_idx]) == str(gold_val):
                        correct += 1
                else:
                    if pred_idx == int(gold_val):
                        correct += 1
            total += 1

    mean_nll = total_nll / max(1, total)
    acc = correct / max(1, total)
    return mean_nll, acc


def save_mlx_checkpoint(
    output_dir: Path,
    model: MLXDecisionModel,
    run_config: dict,
    tokenizer_dir: Path,
    body_config_dir: Path,
    head_name: str = "rlcd",
    export_standalone: bool = True,
):
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
    standalone_head_weights = {}

    for k, v in flat_weights.items():
        clean_k = k
        if clean_k.startswith("backbone.model."):
            clean_k = "backbone." + clean_k[len("backbone.model.") :]
        elif clean_k.startswith("heads."):
            clean_k = clean_k[len("heads.") :]
            # Collect for standalone head
            standalone_head_weights[clean_k] = np.array(v)
        converted_weights[clean_k] = np.array(v)

    # Save full checkpoint
    save_file(converted_weights, str(output_dir / "best.safetensors"))

    # Also save standalone head if requested
    if export_standalone and standalone_head_weights:
        heads_dir = output_dir / "heads"
        heads_dir.mkdir(parents=True, exist_ok=True)
        standalone_path = heads_dir / f"{head_name}.safetensors"
        save_file(standalone_head_weights, str(standalone_path))
        print(f"Exported standalone head to: {standalone_path} ({standalone_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True, help="Path to train JSONL dataset")
    parser.add_argument("--dev-data", required=True, help="Path to dev JSONL dataset")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained model")
    parser.add_argument("--base-checkpoint", default="checkpoints/NanoJev", help="Base checkpoint dir")
    parser.add_argument("--head-name", default="rlcd", help="Name of decision head")
    parser.add_argument("--samples", type=int, default=4, help="RLCD perturbation samples (M=4)")
    parser.add_argument("--sigma", type=float, default=0.3, help="RLCD Gaussian exploration sigma")
    parser.add_argument("--ce-weight", type=float, default=1.0, help="Weight of auxiliary cross entropy loss")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for decision heads")
    parser.add_argument("--batch-questions", type=int, default=16, help="Batch size in questions")
    parser.add_argument("--max-length", type=int, default=4096, help="Max token length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cache-dir", default="data/rlcd_dataset/.cache", help="Directory to cache precomputed embeddings")
    args = parser.parse_args()

    print(f"Loading datasets:\n  Train: {args.train_data}\n  Dev:   {args.dev_data}", flush=True)
    train_records, _ = read_training_records(args.train_data)
    dev_records, _ = read_training_records(args.dev_data)

    print(f"Loading base checkpoint: {args.base_checkpoint}", flush=True)
    model, tokenizer, root, run_config = load_mlx_decision_model(args.base_checkpoint)
    pad_token_id = tokenizer.pad_token_id

    print("Tokenizing train & dev examples...", flush=True)
    train_examples = load_examples_from_records(train_records, tokenizer, args.max_length)
    dev_examples = load_examples_from_records(dev_records, tokenizer, args.max_length)
    print(f"Train questions: {len(train_examples)} | Dev questions: {len(dev_examples)}", flush=True)

    # 1. Precompute backbone embeddings
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    train_cache = cache_dir / "train_embeddings.npz" if cache_dir else None
    dev_cache = cache_dir / "dev_embeddings.npz" if cache_dir else None

    cached_train = precompute_embeddings(model.backbone, train_examples, pad_token_id, max_paths_per_chunk=48, cache_path=train_cache)
    cached_dev = precompute_embeddings(model.backbone, dev_examples, pad_token_id, max_paths_per_chunk=48, cache_path=dev_cache)

    # 2. Setup RLCD optimization
    rlcd_config = RLCDConfig(
        samples=args.samples,
        sigma=args.sigma,
        ce_weight=args.ce_weight,
    )
    heads = model.heads
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def head_loss_fn(h_model, batch_data, key=None):
        examples = [item[0] for item in batch_data]
        leaves = mx.concatenate([item[1] for item in batch_data], axis=0)
        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        logits, valid = h_model(leaves, examples, kmax)
        total_loss, telemetry = grouped_rlcd_loss_mlx(logits, examples, config=rlcd_config, key=key)
        return total_loss

    loss_and_grad = nn.value_and_grad(heads, head_loss_fn)

    best_dev_nll = float("inf")
    output_dir = Path(args.output_dir)
    rng_key = mx.random.key(args.seed)

    init_nll, init_acc = evaluate_cached_dev(heads, cached_dev)
    print(f"\nInitial Dev: NLL = {init_nll:.4f} | Accuracy = {init_acc * 100:.2f}%\n" + "=" * 65, flush=True)

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
        dev_nll, dev_acc = evaluate_cached_dev(heads, cached_dev)
        dt = time.time() - t0

        star = "🌟 (Best)" if dev_nll < best_dev_nll else ""
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Dev NLL: {dev_nll:.4f} | Dev Acc: {dev_acc * 100:.2f}% | Time: {dt:.1f}s {star}",
            flush=True,
        )

        if dev_nll < best_dev_nll:
            best_dev_nll = dev_nll
            best_cfg = {
                "model": "Qwen3-0.6B-RLCD",
                "loss": "rlcd_joint",
                "rlcd": {
                    "samples": args.samples,
                    "sigma": args.sigma,
                    "ce_weight": args.ce_weight,
                },
                "max_length": args.max_length,
                "best_dev_nll": best_dev_nll,
                "best_dev_acc": dev_acc,
                "head_name": args.head_name,
            }
            root_paths = local_checkpoint_files(args.base_checkpoint)[1]
            save_mlx_checkpoint(
                output_dir,
                model,
                best_cfg,
                root_paths["tokenizer"],
                root_paths["body_config"],
                head_name=args.head_name,
                export_standalone=True,
            )

    print("=" * 65)
    print(f"RLCD Training completed successfully! Best Dev NLL: {best_dev_nll:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
