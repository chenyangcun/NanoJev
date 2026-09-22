#!/usr/bin/env python3
"""Full-scale balanced training pipeline for general_set decision head (CandidateSetHead).

Implements:
1. Uniform Family Multi-Task Sampling (16 families, exactly 2 questions per family per batch of 32).
   Prevents MASSIVE intent classification (45% of raw data) from overwhelming logic, policy, and score tasks.
2. CandidateSetHead (2-layer Transformer cross-candidate attention, hidden=256, 4 heads).
3. RLCD proper scoring loss (Log-Score + 0.75*Spherical - RPS) + Auxiliary CE.
4. Unweighted Macro Dev Accuracy & Macro Dev NLL checkpoint selection (matching Dohnuts protocol).
5. Comprehensive JevBench 231 evaluation upon completion.

Outputs:
  checkpoints/dohnuts_merged_0.8b/heads/general_set.safetensors
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
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
from train_qwen35_rlcd import precompute_marker_embeddings, render_dohnuts_question


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


def get_family(item_id: str) -> str:
    if "cnli_" in item_id:
        return "contract_nli"
    if "wikiqa_" in item_id:
        return "wikiqa"
    if "massive_en-US" in item_id:
        return "massive_en"
    if "massive_zh-CN" in item_id:
        return "massive_zh"
    for name in [
        "banking77", "boolq", "agnews", "mnli", "sst5", "yelp",
        "trec", "dbpedia14", "amazon", "imdb", "legacy_policy",
        "compositional", "contrastive",
    ]:
        if name in item_id:
            return name
    return "other"


def evaluate_dev(head, cached_dev, batch_size: int = 32):
    by_fam = {}

    for i in range(0, len(cached_dev), batch_size):
        chunk = cached_dev[i : i + batch_size]
        for item in chunk:
            meta = item["meta"]
            fam = item["family"]
            if fam not in by_fam:
                by_fam[fam] = {"nll": [], "correct": 0, "total": 0}

            leaves = item["leaves"]
            logits, _ = head(leaves)
            k = len(meta["candidate_ids"])
            z = logits[0, :k] if logits.ndim > 1 else logits[:k]
            t = meta.get("gold_distribution_probs")
            if t is not None:
                t_arr = mx.array(t, dtype=mx.float32)
                logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
                nll = -mx.sum(t_arr * logp).item()
                by_fam[fam]["nll"].append(nll)

            pred_idx = int(mx.argmax(z).item())
            gold_val = meta.get("gold")
            if gold_val is not None:
                if meta["type"] == "boolean":
                    if bool(pred_idx) == bool(gold_val):
                        by_fam[fam]["correct"] += 1
                elif meta["type"] == "choice":
                    if pred_idx < len(meta["candidate_ids"]) and str(meta["candidate_ids"][pred_idx]) == str(gold_val):
                        by_fam[fam]["correct"] += 1
                else:
                    if pred_idx == int(gold_val):
                        by_fam[fam]["correct"] += 1
            by_fam[fam]["total"] += 1

    macro_acc = float(np.mean([f["correct"] / max(1, f["total"]) for f in by_fam.values()]))
    macro_nll = float(np.mean([np.mean(f["nll"]) for f in by_fam.values() if f["nll"]]))
    return macro_nll, macro_acc, by_fam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2.5e-4)
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
    print("NANOJEV UNIFORM FAMILY MULTI-TASK GENERAL_SET PIPELINE", flush=True)
    print("=" * 68, flush=True)

    print(f"Loading merged backbone: {model_dir}", flush=True)
    model, tokenizer = load(str(model_dir))
    model.freeze()

    print("Loading precomputed marker embeddings from disk...", flush=True)
    t0 = time.time()
    train_npz = np.load(train_cache, allow_pickle=True)
    train_meta = json.loads(str(train_npz["metadata"]))
    cached_train = []
    for i, meta in enumerate(train_meta):
        cached_train.append({
            "meta": meta,
            "family": get_family(meta["id"]),
            "leaves": mx.array(train_npz[f"marker_{i}"], dtype=mx.float32),
        })

    dev_npz = np.load(dev_cache, allow_pickle=True)
    dev_meta = json.loads(str(dev_npz["metadata"]))
    cached_dev = []
    for i, meta in enumerate(dev_meta):
        cached_dev.append({
            "meta": meta,
            "family": get_family(meta["id"]),
            "leaves": mx.array(dev_npz[f"marker_{i}"], dtype=mx.float32),
        })

    train_by_family = defaultdict(list)
    for item in cached_train:
        train_by_family[item["family"]].append(item)

    families = sorted(train_by_family.keys())
    print(f"Loaded {len(cached_train)} train and {len(cached_dev)} dev questions across {len(families)} families in {time.time() - t0:.2f}s", flush=True)
    print("Train Family Distribution:")
    for fam in families:
        print(f"  {fam:<16}: {len(train_by_family[fam]):>5} questions (balanced to 2/batch)", flush=True)

    # Initialize CandidateSetHead
    head = CandidateSetHead(in_dim=1024, set_dim=256, num_layers=2, num_heads=4)
    print("\nInitialized CandidateSetHead (2-layer Transformer, hidden=256, heads=4)", flush=True)

    total_steps = args.epochs * args.steps_per_epoch
    lr_schedule = optim.cosine_decay(args.lr, total_steps, end=1e-5)
    opt = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
    cfg = RLCDConfig(samples=4, sigma=0.3, ce_weight=1.0)

    def loss_fn(h, batch, key=None):
        losses = []
        for item in batch:
            meta = item["meta"]
            leaves = item["leaves"]
            logits, _ = h(leaves)
            k = len(meta["candidate_ids"])
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

    best_macro_nll = float("inf")
    best_macro_acc = 0.0
    best_weights = None

    init_nll, init_acc, _ = evaluate_dev(head, cached_dev)
    print(f"\nInitial Dev: Macro NLL = {init_nll:.4f} | Macro Acc = {init_acc * 100:.2f}%\n" + "=" * 68, flush=True)

    print("--- PHASE 3: UNIFORM FAMILY RLCD TRAINING LOOP ---", flush=True)
    for ep in range(args.epochs):
        t_ep = time.time()
        losses = []

        for step in range(args.steps_per_epoch):
            b = []
            for fam in families:
                b.extend(random.choices(train_by_family[fam], k=2))

            rng_key, subkey = mx.random.split(rng_key)
            l, g = loss_and_grad(head, b, key=subkey)
            opt.update(head, g)
            mx.eval(head.parameters(), opt.state)
            losses.append(l.item())

        train_loss = float(np.mean(losses))
        dev_nll, dev_acc, by_fam_res = evaluate_dev(head, cached_dev)
        dt = time.time() - t_ep

        # Balanced selection: Best Macro Accuracy with NLL < 2.0 or minimal Macro NLL
        is_best = (dev_acc > best_macro_acc and dev_nll < 2.0) or (dev_nll < best_macro_nll and dev_acc >= 0.50)
        star = "🌟 (Best)" if is_best else ""

        print(
            f"Epoch {ep + 1:02d}/{args.epochs:02d} | Train: {train_loss:.4f} | Dev Macro NLL: {dev_nll:.4f} | Dev Macro Acc: {dev_acc * 100:.2f}% | Time: {dt:.1f}s {star}",
            flush=True,
        )

        if is_best:
            best_macro_nll = dev_nll
            best_macro_acc = dev_acc
            best_weights = flatten_params(head.parameters())
            out_file = heads_dir / f"{args.output_name}.safetensors"
            save_file(best_weights, str(out_file))

    print("=" * 68, flush=True)
    print(f"Training complete! Best Dev Macro NLL: {best_macro_nll:.4f} | Macro Acc: {best_macro_acc * 100:.2f}%", flush=True)
    print(f"Saved weights to: {heads_dir / f'{args.output_name}.safetensors'}", flush=True)

    # 3. Evaluate JevBench 231
    print("\n--- PHASE 4: EVALUATING JEVBENCH 231 ---", flush=True)
    head.load_weights([(k, mx.array(v)) for k, v in best_weights.items()], strict=True)
    temps = {"choice": 1.8172, "boolean": 3.5866, "score": 1.2567}
    res = run_jevbench_eval(model, tokenizer, head, Path("data/jevbench"), temps)

    print("\n" + "=" * 68, flush=True)
    print(f"FINAL BALANCED JEVBENCH ACCURACY ({args.output_name}): {res['overall_accuracy'] * 100:.2f}% (Dohnuts: 65.80%)", flush=True)
    print(f"  Easy:     {res['by_tier']['easy'][0] * 100:.1f}%", flush=True)
    print(f"  Standard: {res['by_tier']['standard'][0] * 100:.1f}%", flush=True)
    print(f"  Hard:     {res['by_tier']['hard'][0] * 100:.1f}%", flush=True)
    print(f"  Choice:   {res['by_type']['choice'][0] * 100:.1f}%", flush=True)
    print(f"  Noul:     {res['by_type']['noul'][0] * 100:.1f}%", flush=True)
    print(f"  Score:    {res['by_type']['score'][0] * 100:.1f}%", flush=True)
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
