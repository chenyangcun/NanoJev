#!/usr/bin/env python3
"""Train a general_set (CandidateSetHead) on rlcd_dataset using Apple MLX.

Architecture:
  candidate vectors [K, 1024]
    -> Linear(1024, 256)
    -> CandidateSetEncoder: 2-layer TransformerEncoder (hidden=256, heads=4, norm_first=True)
    -> Linear(256, 1024) + Residual Connection
    -> ScalarScorer: RMSNorm(1024) -> Linear(1024, 256) -> SiLU -> Linear(256, 1)

Features:
  - Loads precomputed embeddings from data/rlcd_dataset/.cache_q35/train_marker_embeddings.npz
  - Evaluates dev NLL and Accuracy after every epoch
  - Saves best checkpoint to checkpoints/dohnuts_merged_0.8b/heads/general_set.safetensors
  - Directly runs JevBench 231 to measure progress!
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Add scripts directory
sys.path.insert(0, "scripts")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import numpy as np
from safetensors.numpy import save_file

from benchmark_qwen35_suite import load_scorer_head, run_jevbench_eval
from calibrated_rlcd_mlx import RLCDConfig, rlcd_loss_single
from mlx_candidate_set_head import CandidateSetHead


def flatten_params(obj, prefix=""):
    flat = {}
    if isinstance(obj, mx.array):
        flat[prefix] = np.array(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else k
            flat.update(flatten_params(v, name))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            name = f"{prefix}.{i}" if prefix else str(i)
            flat.update(flatten_params(v, name))
    return flat


def evaluate_dev(head, cached_dev, batch_size=32):
    total_nll = 0.0
    correct = 0
    total = 0

    for i in range(0, len(cached_dev), batch_size):
        chunk = cached_dev[i : i + batch_size]
        for item in chunk:
            meta = item["meta"]
            leaves = item["leaves"]
            logits, _ = head(leaves)
            k = len(meta["candidate_ids"])
            z = logits[0, :k]
            t = meta.get("gold_distribution_probs")
            if t is not None:
                t_arr = mx.array(t, dtype=mx.float32)
                logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
                total_nll += -mx.sum(t_arr * logp).item()

            pred_idx = int(mx.argmax(z).item())
            gold_val = meta.get("gold")
            if gold_val is not None:
                if meta["type"] == "boolean":
                    if bool(pred_idx) == bool(gold_val):
                        correct += 1
                elif meta["type"] == "choice":
                    if pred_idx < len(meta["candidate_ids"]) and str(meta["candidate_ids"][pred_idx]) == str(gold_val):
                        correct += 1
                else:
                    if pred_idx == int(gold_val):
                        correct += 1
            total += 1

    return total_nll / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="general_set")
    args = parser.parse_args()

    model_dir = Path("checkpoints/dohnuts_merged_0.8b")
    heads_dir = model_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("data/rlcd_dataset/.cache_q35")
    train_cache = cache_dir / "train_marker_embeddings.npz"
    dev_cache = cache_dir / "dev_marker_embeddings.npz"

    print("Loading precomputed marker embeddings from disk...", flush=True)
    t0 = time.time()
    train_npz = np.load(train_cache, allow_pickle=True)
    train_meta = json.loads(str(train_npz["metadata"]))
    cached_train = []
    for i, meta in enumerate(train_meta):
        cached_train.append({
            "meta": meta,
            "leaves": mx.array(train_npz[f"marker_{i}"], dtype=mx.float32),
        })

    dev_npz = np.load(dev_cache, allow_pickle=True)
    dev_meta = json.loads(str(dev_npz["metadata"]))
    cached_dev = []
    for i, meta in enumerate(dev_meta):
        cached_dev.append({
            "meta": meta,
            "leaves": mx.array(dev_npz[f"marker_{i}"], dtype=mx.float32),
        })

    print(f"Loaded {len(cached_train)} train and {len(cached_dev)} dev questions in {time.time() - t0:.2f}s", flush=True)

    # Initialize CandidateSetHead
    head = CandidateSetHead(in_dim=1024, set_dim=256, num_layers=2, num_heads=4)
    print("Initialized CandidateSetHead (2-layer Transformer, hidden=256, heads=4)", flush=True)

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    cfg = RLCDConfig(samples=4, sigma=0.3, ce_weight=1.0)

    def loss_fn(h, batch, key=None):
        losses = []
        for item in batch:
            meta = item["meta"]
            leaves = item["leaves"]
            logits, _ = h(leaves)
            k = len(meta["candidate_ids"])
            z = logits[0, :k]
            t = meta["gold_distribution_probs"]
            if t is None:
                continue
            key, subkey = mx.random.split(key)
            l, _ = rlcd_loss_single(z, mx.array(t, dtype=mx.float32), qtype=meta["type"], config=cfg, key=subkey)
            losses.append(l)
        return mx.mean(mx.stack(losses))

    loss_and_grad = nn.value_and_grad(head, loss_fn)
    rng_key = mx.random.key(args.seed)

    best_dev_acc = 0.0
    best_weights = None

    init_nll, init_acc = evaluate_dev(head, cached_dev)
    print(f"\nInitial Dev: NLL = {init_nll:.4f} | Accuracy = {init_acc * 100:.2f}%\n" + "=" * 65, flush=True)

    for ep in range(args.epochs):
        t_ep = time.time()
        random.seed(args.seed + ep)
        random.shuffle(cached_train)

        losses = []
        for i in range(0, len(cached_train), args.batch_size):
            b = cached_train[i : i + args.batch_size]
            rng_key, subkey = mx.random.split(rng_key)
            l, g = loss_and_grad(head, b, key=subkey)
            opt.update(head, g)
            mx.eval(head.parameters(), opt.state)
            losses.append(l.item())

        train_loss = np.mean(losses)
        dev_nll, dev_acc = evaluate_dev(head, cached_dev)
        dt = time.time() - t_ep

        star = "🌟 (Best)" if dev_acc > best_dev_acc else ""
        print(f"Epoch {ep + 1:02d}/{args.epochs:02d} | Train: {train_loss:.4f} | Dev NLL: {dev_nll:.4f} | Dev Acc: {dev_acc * 100:.2f}% | Time: {dt:.1f}s {star}", flush=True)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_weights = flatten_params(head.parameters())
            out_file = heads_dir / f"{args.output_name}.safetensors"
            save_file(best_weights, str(out_file))

    print("=" * 65)
    print(f"Training complete! Best Dev Accuracy: {best_dev_acc * 100:.2f}%")
    print(f"Saved best weights to: {heads_dir / f'{args.output_name}.safetensors'}")

    # Run JevBench 231 Evaluation
    print("\nRunning JevBench v1.2.2 Public 231 Benchmark on new head...", flush=True)
    model, tokenizer = load(str(model_dir))
    model.freeze()
    head.load_weights([(k, mx.array(v)) for k, v in best_weights.items()], strict=True)
    temps = {"choice": 1.8172, "boolean": 3.5866, "score": 1.2567}
    res = run_jevbench_eval(model, tokenizer, head, Path("data/jevbench"), temps)

    print("\n" + "=" * 65)
    print(f"FINAL JEVBENCH ACCURACY WITH {args.output_name}: {res['overall_accuracy'] * 100:.2f}% (Dohnuts: 65.80%)")
    print(f"  Easy: {res['by_tier']['easy'][0] * 100:.1f}%")
    print(f"  Standard: {res['by_tier']['standard'][0] * 100:.1f}%")
    print(f"  Hard: {res['by_tier']['hard'][0] * 100:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
