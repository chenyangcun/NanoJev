#!/usr/bin/env python3
"""Train specialized router head on top of merged Dohnuts-0.8B backbone in MLX.

Trains DeepDecisionHeads on:
- master_realistic_dataset.jsonl
- bilingual_master_dataset.jsonl
Covers: complexity, high_risk, independent.
Saves standalone head to:
  checkpoints/dohnuts_merged_0.8b/heads/router.safetensors
"""

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

from calibrated_rlcd_mlx import RLCDConfig, rlcd_loss_single
from mlx_deep_heads import DeepDecisionHeads
from train_pipeline_decisions import read_training_records, validate_training_row
from train_qwen35_rlcd import render_dohnuts_question


def main():
    model_dir = Path("checkpoints/dohnuts_merged_0.8b")
    heads_dir = model_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading merged backbone: {model_dir}", flush=True)
    model, tokenizer = load(str(model_dir))
    model.freeze()

    # Load router datasets
    input_file = Path("data/router_augmented_v2.jsonl")
    all_records, _ = read_training_records(str(input_file))
    print(f"Loaded {len(all_records)} augmented router records from {input_file}", flush=True)

    cache_path = Path("data/.cache_router_augmented.npz")
    marker = "<|fim_suffix|>"
    marker_id = tokenizer.convert_tokens_to_ids(marker)
    cached = []

    if cache_path.exists():
        print(f"Loading precomputed marker embeddings from: {cache_path}...", flush=True)
        t0 = time.time()
        npz = np.load(cache_path, allow_pickle=True)
        metadata = json.loads(str(npz["metadata"]))
        for i, meta in enumerate(metadata):
            cached.append({
                "meta": meta,
                "leaves": mx.array(npz[f"marker_{i}"], dtype=mx.float32),
            })
        print(f"Loaded {len(cached)} cached questions in {time.time() - t0:.2f}s", flush=True)
    else:
        print("Precomputing marker embeddings for router records...", flush=True)
        t0 = time.time()
        arrays_to_save = {}
        meta_to_save = []

        for idx, row in enumerate(all_records):
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

            if (idx + 1) % 200 == 0 or idx + 1 == len(all_records):
                print(f"  Processed {idx + 1}/{len(all_records)} ({len(cached)} questions) in {time.time() - t0:.1f}s", flush=True)

        print(f"Cached {len(cached)} questions in {time.time() - t0:.1f}s", flush=True)
        arrays_to_save["metadata"] = json.dumps(meta_to_save, ensure_ascii=False)
        np.savez_compressed(cache_path, **arrays_to_save)
        print(f"Saved cache to: {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)

    # Train Router Head
    head = DeepDecisionHeads(hidden_size=1024)
    optimizer = optim.AdamW(learning_rate=8e-4, weight_decay=0.01)
    rlcd_config = RLCDConfig(samples=4, sigma=0.3, ce_weight=1.0)

    def loss_fn(h_model, batch_data, key=None):
        losses = []
        for item in batch_data:
            meta = item["meta"]
            leaves = item["leaves"]
            k = len(meta["candidate_ids"])
            ex_item = {"type": meta["type"], "candidate_ids": meta["candidate_ids"], "leaf_tokens": [[1]] * k}
            logits, _ = h_model(leaves, [ex_item], kmax=k)
            z = logits[0, :k]
            t = meta["gold_distribution_probs"]
            key, subkey = mx.random.split(key)
            loss_val, _ = rlcd_loss_single(z, mx.array(t, dtype=mx.float32), qtype=meta["type"], config=rlcd_config, key=subkey)
            losses.append(loss_val)
        return mx.mean(mx.stack(losses))

    loss_and_grad = nn.value_and_grad(head, loss_fn)
    rng_key = mx.random.key(42)

    epochs = 25
    batch_size = 32
    print(f"\nTraining DeepDecisionHeads router head for {epochs} epochs...", flush=True)

    for epoch in range(epochs):
        t_ep = time.time()
        random.seed(42 + epoch)
        random.shuffle(cached)

        running_loss = 0.0
        steps = 0

        for i in range(0, len(cached), batch_size):
            batch = cached[i : i + batch_size]
            rng_key, subkey = mx.random.split(rng_key)
            loss_val, grads = loss_and_grad(head, batch, key=subkey)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)
            running_loss += loss_val.item()
            steps += 1

        avg_loss = running_loss / max(1, steps)
        print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Time: {time.time() - t_ep:.1f}s", flush=True)

    # Save head weights
    out_path = heads_dir / "router.safetensors"
    flat_weights = {}
    for k, v in head.parameters().items():
        if isinstance(v, mx.array):
            flat_weights[k] = np.array(v)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_weights[f"{k}.{sub_k}"] = np.array(sub_v)

    save_file(flat_weights, str(out_path))
    print(f"\nSaved trained router head to: {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
