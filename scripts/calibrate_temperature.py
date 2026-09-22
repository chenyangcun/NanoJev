#!/usr/bin/env python3
"""Temperature calibration on independent holdout split using MLX.

Fits optimal log-temperatures T_choice, T_boolean, T_score by minimizing NLL
on the calibration dataset, exactly following the Laya / Dohnuts protocol.
Outputs calibrated_config.json with per-type temperature factors and ECE metrics.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from predict_mlx_decisions import load_mlx_decision_model
from predict_toy_decisions import prepare_examples
from train_pipeline_decisions import read_training_records, validate_training_row


def compute_ece(probs_list, labels_list, num_bins=15):
    """Compute 15-Bin Expected Calibration Error (ECE) using top confidence."""
    confidences = [float(np.max(p)) for p in probs_list]
    predictions = [int(np.argmax(p)) for p in probs_list]
    accuracies = [float(pred == label) for pred, label in zip(predictions, labels_list)]

    n = len(confidences)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lo = bins[i]
        bin_hi = bins[i + 1]
        mask = [(c > bin_lo and c <= bin_hi) if i > 0 else (c >= bin_lo and c <= bin_hi) for c in confidences]
        bin_count = sum(mask)
        if bin_count > 0:
            bin_acc = sum(acc for acc, m in zip(accuracies, mask) if m) / bin_count
            bin_conf = sum(conf for conf, m in zip(confidences, mask) if m) / bin_count
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return ece


def fit_temperature_type(logits_list, targets_list, steps=120, lr=0.05):
    """Fit optimal temperature for a group of logits and targets using MLX."""
    log_t = mx.array([0.0])  # exp(0) = 1.0

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--calibration-data", required=True, help="Path to calibration.jsonl")
    parser.add_argument("--output-config", default=None, help="Path to output calibrated_config.json")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output_config) if args.output_config else checkpoint_dir / "calibrated_config.json"

    print(f"Loading checkpoint: {checkpoint_dir}", flush=True)
    model, tokenizer, _, _ = load_mlx_decision_model(str(checkpoint_dir))

    print(f"Loading calibration records: {args.calibration_data}", flush=True)
    records, _ = read_training_records(args.calibration_data)
    print(f"Total calibration records: {len(records)}", flush=True)

    # Tokenize and forward
    groups = defaultdict(list)
    t0 = time.time()

    for idx, row in enumerate(records):
        targets = validate_training_row(row)
        prepared = prepare_examples(
            {"states": [{key: row[key] for key in ("id", "state", "questions")}]},
            tokenizer,
            args.max_length,
        )
        for ex in prepared:
            qid = ex["qid"]
            t = targets[qid]
            qtype = ex["type"]
            k = len(ex["candidate_ids"])

            # Run forward pass for this example
            paths = ex["leaf_tokens"]
            width = max(len(p) for p in paths)
            tokens_mat = [p + [tokenizer.pad_token_id] * (width - len(p)) for p in paths]
            input_ids = mx.array(tokens_mat, dtype=mx.int32)
            hidden = model.backbone.model(input_ids)
            if hasattr(hidden, "last_hidden_state"):
                hidden = hidden.last_hidden_state
            leaf_idx = mx.array([len(p) - 1 for p in paths], dtype=mx.int32)
            leaves = hidden[mx.arange(len(paths), dtype=mx.int32), leaf_idx]

            logits, _ = model.heads(leaves, [ex], kmax=k)
            mx.eval(logits)
            z = logits[0, :k]

            target_vec = t.get("gold_distribution_probs")
            if target_vec is None and t.get("gold_index") is not None:
                target_vec = [float(j == t["gold_index"]) for j in range(k)]

            if target_vec is not None:
                gold_label = int(np.argmax(target_vec))
                groups[qtype].append({
                    "logits": z,
                    "target": mx.array(target_vec, dtype=mx.float32),
                    "target_np": np.array(target_vec),
                    "label": gold_label,
                    "k": k,
                })

        if (idx + 1) % 100 == 0 or idx + 1 == len(records):
            print(f"  Processed {idx + 1}/{len(records)} in {time.time() - t0:.1f}s", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Fitting temperatures per question type:", flush=True)

    result_config = {
        "choice": 1.0,
        "boolean": 1.0,
        "score": 1.0,
        "metrics": {},
    }

    all_uncal_probs = []
    all_cal_probs = []
    all_labels = []

    for qtype in ["choice", "boolean", "score"]:
        rows = groups.get(qtype, [])
        if len(rows) < 10:
            print(f"  {qtype:<10}: {len(rows)} samples (skipped, using T=1.0)", flush=True)
            continue

        logits_list = [r["logits"] for r in rows]
        targets_list = [r["target"] for r in rows]
        labels = [r["label"] for r in rows]

        # Fit T
        opt_t = fit_temperature_type(logits_list, targets_list)
        result_config[qtype] = round(opt_t, 4)

        # Uncalibrated probs
        uncal_p = [np.array(mx.softmax(z, axis=-1)) for z in logits_list]
        cal_p = [np.array(mx.softmax(z / opt_t, axis=-1)) for z in logits_list]

        ece_uncal = compute_ece(uncal_p, labels)
        ece_cal = compute_ece(cal_p, labels)

        nll_uncal = -float(np.mean([np.sum(r["target_np"] * np.log(np.maximum(p, 1e-12))) for r, p in zip(rows, uncal_p)]))
        nll_cal = -float(np.mean([np.sum(r["target_np"] * np.log(np.maximum(p, 1e-12))) for r, p in zip(rows, cal_p)]))

        result_config["metrics"][qtype] = {
            "n": len(rows),
            "temperature": round(opt_t, 4),
            "ece_before": round(ece_uncal, 4),
            "ece_after": round(ece_cal, 4),
            "nll_before": round(nll_uncal, 4),
            "nll_after": round(nll_cal, 4),
        }

        print(f"  {qtype:<10} (n={len(rows):<4}): T = {opt_t:.4f} | ECE: {ece_uncal:.4f} -> {ece_cal:.4f} | NLL: {nll_uncal:.4f} -> {nll_cal:.4f}", flush=True)

        all_uncal_probs.extend(uncal_p)
        all_cal_probs.extend(cal_p)
        all_labels.extend(labels)

    overall_ece_before = compute_ece(all_uncal_probs, all_labels)
    overall_ece_after = compute_ece(all_cal_probs, all_labels)
    result_config["overall_ece_before"] = round(overall_ece_before, 4)
    result_config["overall_ece_after"] = round(overall_ece_after, 4)

    print("=" * 60, flush=True)
    print(f"Overall Calibration ECE: {overall_ece_before:.4f} -> {overall_ece_after:.4f} (Delta: -{(overall_ece_before - overall_ece_after):.4f})", flush=True)

    output_path.write_text(json.dumps(result_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Configuration written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
