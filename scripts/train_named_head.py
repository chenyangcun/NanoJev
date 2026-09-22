#!/usr/bin/env python3
"""Train a named decision head on top of the merged Qwen3.5-0.8B backbone using RLCD in Apple MLX.

Usage:
  python3 scripts/train_named_head.py --head-name skill --dataset data/skill_selection_v2_dataset.jsonl --epochs 15
  python3 scripts/train_named_head.py --head-name subagent --dataset data/subagent_dataset.jsonl --epochs 15
  python3 scripts/train_named_head.py --head-name router --dataset data/router_augmented_v3.jsonl --epochs 15

Outputs:
  checkpoints/dohnuts_merged_0.8b/heads/<head_name>.safetensors
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

from benchmark_qwen35_suite import load_scorer_head
from calibrated_rlcd_mlx import RLCDConfig, rlcd_loss_single
from train_pipeline_decisions import read_training_records, validate_training_row
from train_qwen35_rlcd import LinearScorerHead, render_dohnuts_question


def precompute_marker_cache(model, tokenizer, records: list, cache_path: Path):
    if cache_path and cache_path.exists():
        print(f"Loading precomputed marker embeddings from: {cache_path}...", flush=True)
        t0 = time.time()
        npz = np.load(cache_path, allow_pickle=True)
        metadata = json.loads(str(npz["metadata"]))
        cached = []
        for i, meta in enumerate(metadata):
            cached.append({
                "meta": meta,
                "leaves": mx.array(npz[f"marker_{i}"], dtype=mx.float32),
            })
        print(f"Loaded {len(cached)} cached questions in {time.time() - t0:.2f}s", flush=True)
        return cached

    marker = "<|fim_suffix|>"
    marker_id = tokenizer.convert_tokens_to_ids(marker)
    cached = []
    arrays_to_save = {}
    meta_to_save = []
    t0 = time.time()

    print(f"Precomputing marker embeddings for {len(records)} records...", flush=True)
    for idx, row in enumerate(records):
        targets = validate_training_row(row)
        for qid, q in row["questions"].items():
            t = targets[qid]
            prompt, cands = render_dohnuts_question(row["state"], qid, q, marker=marker)
            input_ids = tokenizer.encode(prompt)
            marker_positions = [pos for pos, tid in enumerate(input_ids) if tid == marker_id]

            if len(marker_positions) != len(cands) or not marker_positions:
                continue

            x = mx.array([input_ids], dtype=mx.int32)
            hidden = model.language_model.model(x)
            marker_h = hidden[0, mx.array(marker_positions)].astype(mx.float32)
            mx.eval(marker_h)
            marker_h_np = np.array(marker_h, dtype=np.float16)

            k = len(cands)
            target = t.get("gold_distribution_probs")
            if target is None and t.get("gold_index") is not None:
                target = [float(j == t["gold_index"]) for j in range(k)]

            if target is not None:
                meta = {
                    "id": f"{row['id']}:{qid}",
                    "qid": qid,
                    "type": q["type"],
                    "candidate_ids": cands,
                    "gold_distribution_probs": target,
                    "gold": row.get("gold", {}).get(qid),
                }
                c_idx = len(cached)
                arrays_to_save[f"marker_{c_idx}"] = marker_h_np
                meta_to_save.append(meta)
                cached.append({
                    "meta": meta,
                    "leaves": mx.array(marker_h_np, dtype=mx.float32),
                })

            del hidden, marker_h, x
            mx.clear_cache()

        if (idx + 1) % 150 == 0 or idx + 1 == len(records):
            print(f"  Processed {idx + 1}/{len(records)} ({len(cached)} questions) in {time.time() - t0:.1f}s", flush=True)

    print(f"Cached {len(cached)} questions in {time.time() - t0:.1f}s", flush=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays_to_save["metadata"] = json.dumps(meta_to_save, ensure_ascii=False)
        np.savez_compressed(cache_path, **arrays_to_save)
        print(f"Saved cache to: {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)

    return cached


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
    parser.add_argument("--head-name", required=True, help="Name of head to train: router, skill, subagent, memory")
    parser.add_argument("--head-type", choices=["set", "linear"], default="set", help="Head architecture: 'set' (CandidateSetHead) or 'linear' (LinearScorerHead)")
    parser.add_argument("--dataset", required=True, help="Path to input JSONL dataset")
    parser.add_argument("--model-dir", default="checkpoints/dohnuts_merged_0.8b")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    heads_dir = model_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(f"data/.cache_{args.head_name}.npz")

    print(f"Loading merged backbone: {model_dir}", flush=True)
    model, tokenizer = load(str(model_dir))
    model.freeze()

    input_file = Path(args.dataset)
    records, _ = read_training_records(str(input_file))
    print(f"Loaded {len(records)} records from {input_file}", flush=True)

    cached = precompute_marker_cache(model, tokenizer, records, cache_path)

    # Initialize head according to head_type
    if args.head_type == "set":
        from mlx_candidate_set_head import CandidateSetHead
        head = CandidateSetHead(in_dim=1024, set_dim=256, num_layers=2, num_heads=4)
        print("Initialized CandidateSetHead (2-layer Transformer cross-attention, hidden=256, heads=4)", flush=True)
    else:
        general_head_path = heads_dir / "general.safetensors"
        head = load_scorer_head(general_head_path)
        print("Initialized LinearScorerHead from general.safetensors", flush=True)

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
            key, subkey = mx.random.split(key)
            l, _ = rlcd_loss_single(z, mx.array(t, dtype=mx.float32), qtype=meta["type"], config=cfg, key=subkey)
            losses.append(l)
        return mx.mean(mx.stack(losses))

    loss_and_grad = nn.value_and_grad(head, loss_fn)
    rng_key = mx.random.key(args.seed)

    print(f"\nTraining '{args.head_name}' ({args.head_type}) head for {args.epochs} epochs (LR={args.lr})...", flush=True)
    for ep in range(args.epochs):
        t0 = time.time()
        random.seed(args.seed + ep)
        random.shuffle(cached)

        losses = []
        for i in range(0, len(cached), args.batch_size):
            b = cached[i : i + args.batch_size]
            rng_key, subkey = mx.random.split(rng_key)
            l, g = loss_and_grad(head, b, key=subkey)
            opt.update(head, g)
            mx.eval(head.parameters(), opt.state)
            losses.append(l.item())

        print(f"Epoch {ep + 1:02d}/{args.epochs:02d} | Loss: {np.mean(losses):.4f} | Time: {time.time() - t0:.1f}s", flush=True)

    out_head = heads_dir / f"{args.head_name}.safetensors"
    if args.head_type == "set":
        flat_w = flatten_params(head.parameters())
        save_file(flat_w, str(out_head))
    else:
        save_file({"proj.weight": np.array(head.proj.weight)}, str(out_head))

    print(f"\nSaved trained '{args.head_name}' head to: {out_head} ({out_head.stat().st_size / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
