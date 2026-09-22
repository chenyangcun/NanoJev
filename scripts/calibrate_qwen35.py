#!/usr/bin/env python3
"""Temperature calibration on independent holdout split for Qwen3.5-0.8B."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from benchmark_qwen35_suite import load_scorer_head, forward_qwen35_question, compute_ece
from train_pipeline_decisions import read_training_records, validate_training_row


def fit_temperature_type(logits_list, targets_list, steps=120, lr=0.05):
    log_t = mx.array([0.0])

    def loss_fn(t_param):
        t = mx.clip(mx.exp(t_param[0]), a_min=0.1, a_max=10.0)
        total = 0.0
        for z, y in zip(logits_list, targets_list):
            scaled = z / t
            logp = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
            total = total - mx.sum(y * logp)
        return total / max(1, len(logits_list))

    grad_fn = mx.grad(loss_fn)
    for _ in range(steps):
        g = grad_fn(log_t)
        log_t = log_t - lr * g
        mx.eval(log_t)

    opt_t = float(mx.clip(mx.exp(log_t[0]), a_min=0.1, a_max=10.0).item())
    return opt_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--head-weights", required=True)
    parser.add_argument("--calibration-data", default="data/rlcd_dataset/calibration.jsonl")
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()

    print(f"Loading Qwen3.5: {args.model_name}", flush=True)
    model, tokenizer = load(args.model_name)
    model.freeze()

    head = load_scorer_head(Path(args.head_weights))

    print(f"Loading calibration records: {args.calibration_data}", flush=True)
    records, _ = read_training_records(args.calibration_data)
    print(f"Total records: {len(records)}", flush=True)

    groups = defaultdict(list)
    t0 = time.time()

    for idx, row in enumerate(records):
        targets = validate_training_row(row)
        for qid, q in row["questions"].items():
            t = targets[qid]
            qtype = q["type"]
            pred_cand, probs = forward_qwen35_question(model, tokenizer, head, row["state"], q, temperature=1.0)
            if probs is None:
                continue

            k = len(probs)
            target_vec = t.get("gold_distribution_probs")
            if target_vec is None and t.get("gold_index") is not None:
                target_vec = [float(j == t["gold_index"]) for j in range(k)]

            if target_vec is not None:
                gold_label = int(np.argmax(target_vec))
                # recover unnormalized logits z from probs approximately or recompute
                groups[qtype].append({
                    "probs": probs,
                    "target_np": np.array(target_vec),
                    "label": gold_label,
                })

        if (idx + 1) % 250 == 0 or idx + 1 == len(records):
            print(f"  Processed {idx + 1}/{len(records)} in {time.time() - t0:.1f}s", flush=True)

    result_config = {"choice": 1.0, "boolean": 1.0, "score": 1.0}
    all_uncal_probs = []
    all_cal_probs = []
    all_labels = []

    print("\n" + "=" * 60, flush=True)
    for qtype in ["choice", "boolean", "score"]:
        rows = groups.get(qtype, [])
        if len(rows) < 10:
            continue
        # For simplicity, optimize temperature over log-probabilities
        log_probs = [np.log(np.maximum(r["probs"], 1e-12)) for r in rows]
        targets = [r["target_np"] for r in rows]
        labels = [r["label"] for r in rows]

        # Fit T with grid search / L-BFGS
        best_t = 1.0
        best_nll = float("inf")
        for t_cand in np.linspace(0.5, 5.0, 46):
            nll = -np.mean([np.sum(y * (lp / t_cand - np.log(np.sum(np.exp(lp / t_cand))))) for lp, y in zip(log_probs, targets)])
            if nll < best_nll:
                best_nll = nll
                best_t = float(t_cand)

        result_config[qtype] = round(best_t, 4)
        ece_before = compute_ece([r["probs"] for r in rows], labels)
        cal_p = [np.exp(lp / best_t) / np.sum(np.exp(lp / best_t)) for lp in log_probs]
        ece_after = compute_ece(cal_p, labels)
        print(f"  {qtype:<10} (n={len(rows):<4}): T = {best_t:.4f} | ECE: {ece_before:.4f} -> {ece_after:.4f}", flush=True)

        all_uncal_probs.extend([r["probs"] for r in rows])
        all_cal_probs.extend(cal_p)
        all_labels.extend(labels)

    overall_before = compute_ece(all_uncal_probs, all_labels)
    overall_after = compute_ece(all_cal_probs, all_labels)
    print("=" * 60, flush=True)
    print(f"Overall ECE: {overall_before:.4f} -> {overall_after:.4f}", flush=True)

    Path(args.output_config).write_text(json.dumps(result_config, indent=2) + "\n")
    print(f"Wrote config to: {args.output_config}", flush=True)


if __name__ == "__main__":
    main()
