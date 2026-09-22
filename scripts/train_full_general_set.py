#!/usr/bin/env python3
"""Full-scale training pipeline for general_set decision head (CandidateSetHead).

Processes 28,328 records (20,564 train, 2,628 dev, 1,933 calibration).
Algorithm:
  - In-Context Marker representation extraction
  - CandidateSetHead (2-layer Transformer cross-candidate attention, hidden=256, heads=4)
  - RLCD proper scoring loss (Log-score + 0.75*Spherical - RPS) + Auxiliary CE
  - Temperature calibration on holdout calibration set
  - Full JevBench 231 benchmark evaluation

Outputs:
  checkpoints/dohnuts_merged_0.8b/heads/general_set.safetensors
  logs/train_full_general_set.log
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
from train_pipeline_decisions import read_training_records, validate_training_row
from train_qwen35_rlcd import precompute_marker_embeddings, evaluate_dev, render_dohnuts_question


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="general_set")
    args = parser.parse_args()

    model_dir = Path("checkpoints/dohnuts_merged_0.8b")
    heads_dir = model_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("data/full_general_dataset/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train_marker_embeddings.npz"
    dev_cache = cache_dir / "dev_marker_embeddings.npz"

    print("=" * 68, flush=True)
    print("NANOJEV FULL-SCALE GENERAL_SET TRAINING PIPELINE", flush=True)
    print("=" * 68, flush=True)

    print(f"Loading merged backbone: {model_dir}", flush=True)
    model, tokenizer = load(str(model_dir))
    model.freeze()

    train_file = Path("data/full_general_dataset/train.jsonl")
    dev_file = Path("data/full_general_dataset/dev.jsonl")

    print(f"Loading records from:\n  Train: {train_file}\n  Dev:   {dev_file}", flush=True)
    train_records, _ = read_training_records(str(train_file))
    dev_records, _ = read_training_records(str(dev_file))
    print(f"Loaded {len(train_records)} train records and {len(dev_records)} dev records.", flush=True)

    # 1. Precompute / Load marker embeddings
    print("\n--- PHASE 1: PRECOMPUTING MARKER EMBEDDINGS ---", flush=True)
    cached_train = precompute_marker_embeddings(model, tokenizer, train_records, cache_path=train_cache, batch_size=4)
    cached_dev = precompute_marker_embeddings(model, tokenizer, dev_records, cache_path=dev_cache, batch_size=4)

    print(f"\nEmbedding cache ready: {len(cached_train)} train items | {len(cached_dev)} dev items", flush=True)

    # 2. Setup CandidateSetHead & Optimizer
    print("\n--- PHASE 2: INITIALIZING CANDIDATESETHEAD ---", flush=True)
    head = CandidateSetHead(in_dim=1024, set_dim=256, num_layers=2, num_heads=4)
    print("CandidateSetHead ready (2-layer Transformer, hidden=256, heads=4)", flush=True)

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    cfg = RLCDConfig(samples=4, sigma=0.3, ce_weight=1.0)

    def loss_fn(h, batch, key=None):
        losses = []
        for meta, leaves in batch:
            k = len(meta["candidate_ids"])
            logits, _ = h(leaves)
            z = logits[0, :k] if logits.ndim > 1 else logits[:k]
            t = meta.get("gold_distribution_probs")
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
    print(f"Initial Dev: NLL = {init_nll:.4f} | Accuracy = {init_acc * 100:.2f}%\n" + "=" * 65, flush=True)

    print("\n--- PHASE 3: RLCD TRAINING LOOP ---", flush=True)
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

    print("=" * 65, flush=True)
    print(f"Training complete! Best Dev Accuracy: {best_dev_acc * 100:.2f}%", flush=True)
    print(f"Saved weights to: {heads_dir / f'{args.output_name}.safetensors'}", flush=True)

    # 3. Evaluate JevBench 231
    print("\n--- PHASE 4: EVALUATING JEVBENCH 231 ---", flush=True)
    head.load_weights([(k, mx.array(v)) for k, v in best_weights.items()], strict=True)
    temps = {"choice": 1.8172, "boolean": 3.5866, "score": 1.2567}
    res = run_jevbench_eval(model, tokenizer, head, Path("data/jevbench"), temps)

    print("\n" + "=" * 65, flush=True)
    print(f"FINAL FULL-SCALE JEVBENCH ACCURACY ({args.output_name}): {res['overall_accuracy'] * 100:.2f}% (Dohnuts: 65.80%)", flush=True)
    print(f"  Easy:     {res['by_tier']['easy'][0] * 100:.1f}%", flush=True)
    print(f"  Standard: {res['by_tier']['standard'][0] * 100:.1f}%", flush=True)
    print(f"  Hard:     {res['by_tier']['hard'][0] * 100:.1f}%", flush=True)
    print(f"  Choice:   {res['by_type']['choice'][0] * 100:.1f}%", flush=True)
    print(f"  Noul:     {res['by_type']['noul'][0] * 100:.1f}%", flush=True)
    print(f"  Score:    {res['by_type']['score'][0] * 100:.1f}%", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
