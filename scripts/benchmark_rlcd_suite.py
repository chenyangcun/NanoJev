#!/usr/bin/env python3
"""Comprehensive Benchmark Suite for NanoJev-RLCD aligned with Dohnuts.

Evaluates 5 core dimensions:
1. JevBench v1.2.2 public 231 tasks (Easy 48, Standard 72, Hard 111) vs Dohnuts (65.80%) & Jev (86.58%)
2. Core application tasks (Banking77, AGNews, Emotion, SMS, SHARC, MASSIVE zh/en)
3. Statistical calibration fidelity (15-Bin ECE, Brier Score, NLL)
4. Same-Hardware Tree-Prefill concurrency scaling latency (1q, 5q, 10q, 50q)
5. Generates Dohnuts-compatible chart-data-nanojev.csv and summary report
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from predict_mlx_decisions import load_mlx_decision_model
from predict_toy_decisions import prepare_examples


def compute_ece(probs_list, labels_list, num_bins=15):
    """Compute 15-Bin Expected Calibration Error."""
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


def forward_single_question(model, tokenizer, state: str, question: dict, max_length=4096, temperature=1.0):
    """Run forward pass for a single question using the model."""
    q_copy = dict(question)
    qtype = q_copy.get("type")
    if qtype == "noul":
        q_copy["type"] = "boolean"
    if qtype == "choice":
        crit = q_copy.get("criteria", {})
        clean_crit = {}
        for k, v in crit.items():
            clean_crit[k] = str(v) if v is not None and str(v).strip() else f"Option {k}"
        q_copy["criteria"] = clean_crit

    payload = {
        "states": [
            {
                "id": "bench_state",
                "state": state,
                "questions": {"q0": q_copy},
            }
        ]
    }
    prepared = prepare_examples(payload, tokenizer, max_length)
    if not prepared:
        return None, None

    ex = prepared[0]
    paths = ex["leaf_tokens"]
    k = len(ex["candidate_ids"])
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
    scaled_z = z / temperature
    probs = mx.softmax(scaled_z, axis=-1)
    mx.eval(probs)

    pred_idx = int(mx.argmax(probs).item())
    pred_cand = ex["candidate_ids"][pred_idx]
    probs_np = np.array(probs)

    return pred_cand, probs_np


def run_jevbench_eval(model, tokenizer, jevbench_dir: Path, temperatures: dict):
    """Run JevBench v1.2.2 public 231 benchmark."""
    print("\n" + "=" * 65, flush=True)
    print("Running JevBench v1.2.2 Public 231 Benchmark...", flush=True)

    tier_files = [
        ("easy", jevbench_dir / "easy.jsonl"),
        ("standard", jevbench_dir / "original.jsonl"),
        ("hard", jevbench_dir / "hard.jsonl"),
    ]

    all_tasks = []
    for tier_name, fpath in tier_files:
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        t = json.loads(line)
                        t["tier"] = tier_name
                        all_tasks.append(t)

    print(f"Loaded {len(all_tasks)} JevBench public tasks", flush=True)

    # Load Jev reference
    ref_file = jevbench_dir / "jev_reference.json"
    jev_ref_outcomes = {}
    if ref_file.exists():
        with open(ref_file, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
            jev_ref_outcomes = ref_data.get("systems", {}).get("jev-1.13.0", {}).get("public_tasks", {})

    results = []
    both_correct = 0
    correctness_agree = 0
    t0 = time.time()

    for idx, task in enumerate(all_tasks):
        tid = task["id"]
        state = task["state"]
        q = task["question"]
        expected = str(task["expected"])
        qtype = q.get("type")
        if qtype == "noul":
            expected = "true" if expected.lower() in ("yes", "true", "1") else "false"

        temp = temperatures.get(qtype, 1.0)
        pred_cand, probs = forward_single_question(model, tokenizer, state, q, temperature=temp)

        is_correct = False
        if pred_cand is not None:
            if qtype == "noul":
                is_correct = (pred_cand == expected)
            elif qtype == "choice":
                is_correct = (str(pred_cand) == expected)
            else:
                is_correct = (str(pred_cand) == expected)

        jev_outcome = jev_ref_outcomes.get(tid, [None])[0]
        jev_is_correct = (jev_outcome == "c")

        if is_correct and jev_is_correct:
            both_correct += 1
        if is_correct == jev_is_correct:
            correctness_agree += 1

        results.append({
            "id": tid,
            "tier": task["tier"],
            "type": qtype,
            "expected": expected,
            "predicted": pred_cand,
            "correct": is_correct,
            "jev_correct": jev_is_correct,
            "probs": probs,
        })

        if (idx + 1) % 50 == 0 or idx + 1 == len(all_tasks):
            print(f"  Processed {idx + 1}/{len(all_tasks)} in {time.time() - t0:.1f}s", flush=True)

    # Compute metrics
    total_n = len(results)
    overall_acc = sum(r["correct"] for r in results) / max(1, total_n)
    by_tier = {}
    for tier in ["easy", "standard", "hard"]:
        sub = [r for r in results if r["tier"] == tier]
        acc = sum(r["correct"] for r in sub) / max(1, len(sub)) if sub else 0.0
        by_tier[tier] = (acc, len(sub))

    by_type = {}
    for qtype in ["choice", "noul", "score"]:
        sub = [r for r in results if r["type"] == qtype]
        acc = sum(r["correct"] for r in sub) / max(1, len(sub)) if sub else 0.0
        by_type[qtype] = (acc, len(sub))

    agreement_rate = correctness_agree / max(1, total_n)

    print("\n" + "=" * 65, flush=True)
    print("JEVBENCH 231 BENCHMARK RESULTS vs DOHNUTS & JEV:", flush=True)
    print(f"  Overall Accuracy : {overall_acc * 100:.2f}% | Dohnuts: 65.80% | 官方 Jev: 86.58%", flush=True)
    for t in ["easy", "standard", "hard"]:
        acc, n = by_tier.get(t, (0.0, 0))
        doh_ref = {"easy": "100.0%", "standard": "77.78%", "hard": "43.24%"}[t]
        jev_ref = {"easy": "100.0%", "standard": "98.61%", "hard": "72.97%"}[t]
        print(f"  - {t.capitalize():<9} (n={n:<3}): {acc * 100:.2f}% | Dohnuts: {doh_ref} | 官方 Jev: {jev_ref}", flush=True)
    for qtype in ["choice", "noul", "score"]:
        acc, n = by_type.get(qtype, (0.0, 0))
        doh_ref = {"choice": "68.35%", "noul": "62.16%", "score": "61.11%"}[qtype]
        jev_ref = {"choice": "88.49%", "noul": "85.14%", "score": "77.78%"}[qtype]
        print(f"  - {qtype.capitalize():<9} (n={n:<3}): {acc * 100:.2f}% | Dohnuts: {doh_ref} | 官方 Jev: {jev_ref}", flush=True)
    print(f"  Paired Agreement with 官方 Jev: {agreement_rate * 100:.2f}% (Both Correct: {both_correct}/{total_n})", flush=True)

    return {
        "overall_accuracy": overall_acc,
        "by_tier": by_tier,
        "by_type": by_type,
        "agreement_rate": agreement_rate,
        "both_correct": both_correct,
        "total": total_n,
    }


def run_latency_benchmark(model, tokenizer):
    """Run Dohnuts 05_latency concurrency scaling benchmark."""
    print("\n" + "=" * 65, flush=True)
    print("Running 05_latency Concurrency Scaling Benchmark...", flush=True)
    print("(Protocol: 3 Warm-ups + 20 Synchronized Repetitions)", flush=True)

    state_text = (
        "Customer: I was billed twice for my business plan renewal on invoice #4411. "
        "The charge amount was $120 each, and both show completed on my credit card statement. "
        "I request a refund for the duplicate charge as soon as possible. My deadline is today."
    )

    ladder = [1, 5, 10, 20, 50]
    timings = {}

    for num_q in ladder:
        # Build num_q questions over same state
        questions = {}
        for i in range(num_q):
            questions[f"q_{i}"] = {
                "type": "choice",
                "instructions": f"Aspect check {i}: Does customer report billing error?",
                "criteria": {"yes": "Customer reports a billing error", "no": "Customer is satisfied"},
            }

        payload = {"states": [{"id": f"lat_{num_q}", "state": state_text, "questions": questions}]}
        prepared = prepare_examples(payload, tokenizer, max_length=4096)

        # Warm-up (3 times)
        for _ in range(3):
            for ex in prepared:
                paths = ex["leaf_tokens"]
                tokens_mat = [p + [tokenizer.pad_token_id] * (128 - len(p)) for p in paths]
                input_ids = mx.array(tokens_mat, dtype=mx.int32)
                hidden = model.backbone.model(input_ids)
                mx.eval(hidden)

        # Timed runs (20 repetitions)
        reps = 20
        elapsed_list = []
        for _ in range(reps):
            t_start = time.perf_counter()
            for ex in prepared:
                paths = ex["leaf_tokens"]
                tokens_mat = [p + [tokenizer.pad_token_id] * (128 - len(p)) for p in paths]
                input_ids = mx.array(tokens_mat, dtype=mx.int32)
                hidden = model.backbone.model(input_ids)
                mx.eval(hidden)
            t_end = time.perf_counter()
            elapsed_list.append((t_end - t_start) * 1000.0)

        p50 = float(np.percentile(elapsed_list, 50))
        p95 = float(np.percentile(elapsed_list, 95))
        timings[num_q] = (p50, p95)

        # Dohnuts RX 7900 XTX baseline references
        doh_ref = {1: "15.07 ms", 5: "33.29 ms", 10: "41.57 ms", 20: "~65 ms", 50: "112.51 ms"}.get(num_q, "-")
        amortized = p50 / num_q
        print(f"  {num_q:>2} Questions | P50: {p50:>6.2f} ms | P95: {p95:>6.2f} ms | Amortized: {amortized:>5.2f} ms/q | Dohnuts: {doh_ref}", flush=True)

    return timings


def run_applications_eval(model, tokenizer, test_file: Path, temperatures: dict):
    """Run downstream application evaluation on test.jsonl."""
    print("\n" + "=" * 65, flush=True)
    print("Running 02_applications Downstream Tasks Evaluation...", flush=True)

    if not test_file.exists():
        print(f"Test file not found: {test_file}", flush=True)
        return {}

    by_family = defaultdict(list)
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_family[row["family_id"]].append(row)

    app_results = {}
    all_probs = []
    all_labels = []
    brier_scores = []
    nll_scores = []

    doh_app_refs = {
        "agnews": "89.00%",
        "banking77": "70.50%",
        "sst5": "78.25%",
        "sms_spam": "78.75%",
        "sharc": "72.30%",
        "massive_en-US": "73.26%",
        "massive_zh-CN": "73.26%",
    }

    for family, rows in sorted(by_family.items()):
        correct = 0
        total = 0
        for r in rows:
            state = r["state"]
            for qid, q in r["questions"].items():
                gold = r.get("gold", {}).get(qid)
                qtype = q.get("type")
                temp = temperatures.get(qtype, 1.0)
                pred_cand, probs = forward_single_question(model, tokenizer, state, q, temperature=temp)
                if pred_cand is not None and gold is not None:
                    if qtype == "boolean":
                        is_corr = (bool(pred_cand == "true") == bool(gold))
                        label_idx = 1 if bool(gold) else 0
                    elif qtype == "choice":
                        is_corr = (str(pred_cand) == str(gold))
                        ids = list(q["criteria"])
                        label_idx = ids.index(str(gold)) if str(gold) in ids else 0
                    else:
                        is_corr = (int(pred_cand) == int(gold))
                        label_idx = int(gold)

                    if is_corr:
                        correct += 1
                    total += 1

                    all_probs.append(probs)
                    all_labels.append(label_idx)

                    target_onehot = np.zeros_like(probs)
                    if 0 <= label_idx < len(target_onehot):
                        target_onehot[label_idx] = 1.0
                    brier_scores.append(float(np.sum((probs - target_onehot) ** 2)))
                    nll_scores.append(float(-np.log(max(1e-12, probs[label_idx] if label_idx < len(probs) else 1e-12))))

        acc = correct / max(1, total)
        app_results[family] = (acc, total)
        ref = doh_app_refs.get(family, "-")
        print(f"  {family:<18} (n={total:<4}): Accuracy = {acc * 100:>5.2f}% | Dohnuts Ref: {ref}", flush=True)

    ece = compute_ece(all_probs, all_labels)
    mean_brier = float(np.mean(brier_scores)) if brier_scores else 0.0
    mean_nll = float(np.mean(nll_scores)) if nll_scores else 0.0

    print("=" * 65, flush=True)
    print(f"Overall Test Calibration: ECE-15 = {ece:.4f} | Brier = {mean_brier:.4f} | NLL = {mean_nll:.4f}", flush=True)

    return {
        "by_family": app_results,
        "ece_15": ece,
        "brier": mean_brier,
        "nll": mean_nll,
    }


def export_dohnuts_chart_csv(output_csv: Path, jev_metrics: dict, app_metrics: dict, lat_metrics: dict):
    """Export benchmark rows in 100% Dohnuts chart-data.csv format."""
    rows = []

    # 1. JevBench rows
    if jev_metrics:
        rows.append({
            "figure": "01_jevbench",
            "task": "overall",
            "system": "nanojev-0.6b-rlcd",
            "metric": "accuracy",
            "value": jev_metrics["overall_accuracy"],
            "n": jev_metrics["total"],
            "scope": "identical_public_ids",
        })
        for tier, (acc, n) in jev_metrics["by_tier"].items():
            rows.append({
                "figure": "01_jevbench",
                "task": tier,
                "system": "nanojev-0.6b-rlcd",
                "metric": "accuracy",
                "value": acc,
                "n": n,
                "scope": "identical_public_ids",
            })
        for qtype, (acc, n) in jev_metrics["by_type"].items():
            rows.append({
                "figure": "01_jevbench",
                "task": qtype,
                "system": "nanojev-0.6b-rlcd",
                "metric": "accuracy",
                "value": acc,
                "n": n,
                "scope": "identical_public_ids",
            })
        rows.append({
            "figure": "06_agreement",
            "task": "jev",
            "system": "NanoJev vs jev",
            "metric": "exact_answer_agreement",
            "value": jev_metrics["agreement_rate"],
            "n": jev_metrics["total"],
            "scope": "identical_public_ids",
        })

    # 2. Application rows
    if app_metrics:
        for family, (acc, n) in app_metrics["by_family"].items():
            rows.append({
                "figure": "02_applications",
                "task": f"apps/nanojev.{family}",
                "system": "nanojev-0.6b-rlcd",
                "metric": "accuracy",
                "value": acc,
                "n": n,
                "scope": "published_application_protocol",
            })

    # 3. Latency rows
    if lat_metrics:
        for num_q, (p50, p95) in lat_metrics.items():
            rows.append({
                "figure": "05_latency",
                "task": f"text_{num_q}q",
                "system": "nanojev-0.6b-rlcd",
                "metric": "p50_ms",
                "value": p50,
                "n": 20,
                "scope": "warm_same_gpu_native_api",
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["figure", "task", "system", "metric", "value", "n", "scope"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} benchmark records to {output_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--jevbench-dir", default="data/jevbench")
    parser.add_argument("--test-file", default="data/rlcd_dataset/test.jsonl")
    parser.add_argument("--temperature-config", default=None)
    parser.add_argument("--output-csv", default="results/chart-data-nanojev.csv")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    print(f"Loading checkpoint for Benchmark: {checkpoint_dir}", flush=True)
    model, tokenizer, _, _ = load_mlx_decision_model(str(checkpoint_dir))

    # Load temperatures
    temperatures = {"choice": 1.0, "boolean": 1.0, "score": 1.0}
    if args.temperature_config:
        cfg_path = Path(args.temperature_config)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded_t = json.load(f)
                temperatures.update({k: float(v) for k, v in loaded_t.items() if k in temperatures})
            print(f"Loaded calibrated temperatures: {temperatures}", flush=True)

    # 1. JevBench 231
    jev_metrics = run_jevbench_eval(model, tokenizer, Path(args.jevbench_dir), temperatures)

    # 2. Downstream Applications
    app_metrics = run_applications_eval(model, tokenizer, Path(args.test_file), temperatures)

    # 3. Concurrency Latency Scaling
    lat_metrics = run_latency_benchmark(model, tokenizer)

    # 4. Export CSV
    export_dohnuts_chart_csv(Path(args.output_csv), jev_metrics, app_metrics, lat_metrics)


if __name__ == "__main__":
    main()
