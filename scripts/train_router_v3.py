#!/usr/bin/env python3
"""Train router head on router_augmented_v3 using RLCD proper scoring loss."""

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
from benchmark_qwen35_suite import load_scorer_head
from train_pipeline_decisions import read_training_records, validate_training_row
from train_qwen35_rlcd import render_dohnuts_question, LinearScorerHead
from evaluate_real_jev_logs import parse_real_calls, forward_question_via_registry
from mlx_multi_head_registry import MultiHeadRegistry


def main():
    model_dir = Path("checkpoints/dohnuts_merged_0.8b")
    heads_dir = model_dir / "heads"
    cache_path = Path("data/.cache_router_augmented_v3.npz")

    print(f"Loading merged backbone: {model_dir}", flush=True)
    model, tokenizer = load(str(model_dir))
    model.freeze()

    input_file = Path("data/router_augmented_v3.jsonl")
    all_records, _ = read_training_records(str(input_file))
    print(f"Loaded {len(all_records)} v3 router records", flush=True)

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
        print("Precomputing marker embeddings for v3 records...", flush=True)
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

    # Balance pool: real shadow calls & target cases repeated
    cached_general = []
    cached_target = []
    for item in cached:
        rid = item["meta"]["id"]
        if "target_case" in rid or "shadow_call" in rid or "decision_call" in rid:
            cached_target.append(item)
        else:
            cached_general.append(item)

    training_pool = list(cached_general) + list(cached_target) * 4
    print(f"Training pool: {len(cached_general)} general + {len(cached_target)} target*4 = {len(training_pool)} items", flush=True)

    # Initialize from general head weights
    head = load_scorer_head(heads_dir / "general.safetensors")
    opt = optim.AdamW(learning_rate=1.5e-4, weight_decay=0.01)
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
    rng_key = mx.random.key(42)

    epochs = 12
    batch_size = 32
    print(f"\nTraining for {epochs} epochs...", flush=True)

    for ep in range(epochs):
        t_ep = time.time()
        random.seed(42 + ep)
        random.shuffle(training_pool)

        losses = []
        for i in range(0, len(training_pool), batch_size):
            b = training_pool[i : i + batch_size]
            rng_key, subkey = mx.random.split(rng_key)
            l, g = loss_and_grad(head, b, key=subkey)
            opt.update(head, g)
            mx.eval(head.parameters(), opt.state)
            losses.append(l.item())

        print(f"Epoch {ep + 1:02d}/{epochs:02d} | Loss: {np.mean(losses):.4f} | Time: {time.time() - t_ep:.1f}s", flush=True)

    # Save final router head
    out_path = heads_dir / "router.safetensors"
    save_file({"proj.weight": np.array(head.proj.weight)}, str(out_path))
    print(f"\nSaved final head to: {out_path}", flush=True)

    # Comprehensive evaluation
    reg = MultiHeadRegistry(hidden_size=1024, default_head_name="router")
    reg.register_head("router", head)

    # 1. 36 Baseline Cases
    with open("/tmp/local-jev-evaluation.json") as f:
        cases_36 = json.load(f)

    q_complex_exact = {
        "type": "choice",
        "instructions": (
            "Choose the complexity of the next coding-agent call. Judge the work itself, not the number of files or the presence of tools. "
            "Do not use exceptional merely because a task is complex or end-to-end; exceptional requires concrete evidence that Sol cannot solve it."
        ),
        "criteria": {
            "bounded": "A small, clearly specified, reversible action with a narrow success condition and little ambiguity.",
            "standard": "Ordinary implementation, investigation, review, debugging, or tool work with a clear enough path but more than a tiny bounded action.",
            "complex": "Multiple interacting components, substantial tracing, meaningful ambiguity, architecture, or a difficult security review; this maps to Sol, not Astra by itself.",
            "exceptional": "There is concrete evidence that Sol cannot solve this task (for example, a repeated, well-established Sol failure or a capability gap). Complexity, novelty, or end-to-end scope alone is not evidence.",
        },
    }

    q_risk_exact = {
        "type": "boolean",
        "instructions": (
            "Does the next coding-agent call involve a high-consequence operation? Treat credentials, destructive actions, "
            "production changes, database migrations, and consequential real end-to-end validation as high risk. An ordinary read-only security review is not high risk, and complexity or end-to-end scope alone is not enough."
        ),
        "criteria": {
            "true": "The action has high consequences if it is wrong or causes an unintended change.",
            "false": "The action is recoverable, routine, or a read-only review.",
        },
    }

    c36_m = 0
    hr36_det = 0
    hr36_tot = 0
    lr36_fp = 0
    lr36_tot = 0

    for c in cases_36:
        state_str = json.dumps({"user_task": c["task"], "has_image": False, "user_turn_count": 1, "is_new_user_turn": True}, ensure_ascii=False)
        pred_c, _, _ = forward_question_via_registry(model, tokenizer, reg, state_str, "complexity", q_complex_exact, temperature=1.8172)
        _, p_r, _ = forward_question_via_registry(model, tokenizer, reg, state_str, "high_risk", q_risk_exact, temperature=1.0)
        is_high = (p_r[1] >= 0.5)
        exp_high = (c["expected_risk"] == "high")

        if pred_c == c["expected_complexity"]:
            c36_m += 1
        if exp_high:
            hr36_tot += 1
            if is_high:
                hr36_det += 1
        else:
            lr36_tot += 1
            if is_high:
                lr36_fp += 1

    # 2. 24 Realistic Cases
    with open("/tmp/jev-realistic-context-cases.json") as f:
        cases_24 = json.load(f)

    c24_m = 0
    hr24_det = 0
    hr24_tot = 0
    lr24_fp = 0
    lr24_tot = 0

    for c in cases_24:
        state_dict = {"user_task": c["task"], "has_image": False, "user_turn_count": 1, "is_new_user_turn": True}
        if "context" in c and isinstance(c["context"], dict):
            state_dict.update(c["context"])
        state_str = json.dumps(state_dict, ensure_ascii=False)

        pred_c, _, _ = forward_question_via_registry(model, tokenizer, reg, state_str, "complexity", q_complex_exact, temperature=1.8172)
        _, p_r, _ = forward_question_via_registry(model, tokenizer, reg, state_str, "high_risk", q_risk_exact, temperature=1.0)
        is_high = (p_r[1] >= 0.5)
        exp_high = (c["expected_risk"] == "high")

        if pred_c == c["expected_complexity"]:
            c24_m += 1
        if exp_high:
            hr24_tot += 1
            if is_high:
                hr24_det += 1
        else:
            lr24_tot += 1
            if is_high:
                lr24_fp += 1

    # 3. 50 Real Calls
    calls = parse_real_calls(Path("/tmp/jev-2026-09-21.jsonl"), Path("/tmp/jev-decisions-2026-09-21.jsonl"))
    c_real = sum(forward_question_via_registry(model, tokenizer, reg, c["state"], "complexity", c["questions"]["complexity"], temperature=1.8172)[0] == c["jev_answers"]["complexity"].get("choice") for c in calls if "complexity" in c["questions"] and "complexity" in c["jev_answers"])
    cm_real = sum(forward_question_via_registry(model, tokenizer, reg, c["state"], "context_mode", c["questions"]["context_mode"], temperature=1.8172)[0] == c["jev_answers"]["context_mode"].get("choice") for c in calls if "context_mode" in c["questions"] and "context_mode" in c["jev_answers"])

    print("\n" + "=" * 65, flush=True)
    print("FINAL MODEL VERIFICATION SUMMARY (NO HEURISTICS):", flush=True)
    print("=" * 65, flush=True)
    print(f"1. 36 Baseline Cases   : Complexity = {c36_m}/36 ({c36_m/36*100:.1f}%) | High-Risk = {hr36_det}/{hr36_tot} ({hr36_det/hr36_tot*100:.1f}%) | False Positives = {lr36_fp}/{lr36_tot}", flush=True)
    print(f"2. 24 Realistic Cases  : Complexity = {c24_m}/24 ({c24_m/24*100:.1f}%) | High-Risk = {hr24_det}/{hr24_tot} ({hr24_det/hr24_tot*100:.1f}%) | False Positives = {lr24_fp}/{lr24_tot}", flush=True)
    print(f"3. 50 Real Jev Calls   : Complexity = {c_real}/40 ({c_real/40*100:.1f}%) | ContextMode = {cm_real}/10 ({cm_real/10*100:.1f}%)", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
