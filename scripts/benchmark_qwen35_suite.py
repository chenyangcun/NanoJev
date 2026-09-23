#!/usr/bin/env python3
"""Benchmark Suite for Qwen3.5-0.8B Decision Models with In-Context Marker Scoring.

Evaluates:
1. JevBench v1.2.2 public 231 tasks (48 Easy, 72 Standard, 111 Hard)
2. Downstream applications on test.jsonl (Banking77, AGNews, MASSIVE zh/en, SMS, SHARC, etc.)
3. 15-Bin ECE and Brier score
4. In-Context Latency scaling (1q, 5q, 10q, 50q)
Outputs Dohnuts-compatible results/chart-data-qwen35.csv.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from safetensors.numpy import load_file

from train_qwen35_rlcd import LinearScorerHead, render_dohnuts_question


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


def render_dohnuts_question_with_prefix(state_text: str, qid: str, q: dict, marker: str = "<|fim_suffix|>"):
    typ = q["type"]
    instr = q.get("instructions", "")
    if isinstance(instr, (dict, list)):
        instr = json.dumps(instr, ensure_ascii=False)
    else:
        instr = str(instr)

    if typ in ("boolean", "noul"):
        crit = q.get("criteria", {})
        if not isinstance(crit, dict):
            crit = {}
        f_text = crit.get("false") or "no, the statement does not hold"
        t_text = crit.get("true") or "yes, the statement holds"
        options = [f"false: {f_text}", f"true: {t_text}"]
        cands = ["false", "true"]
    elif typ == "choice":
        crit = q.get("criteria", {})
        if isinstance(crit, dict):
            cands = list(crit.keys())
            options = [f"{k}: {crit[k]}" for k in cands]
        elif isinstance(crit, list):
            cands = [str(x) for x in crit]
            options = [str(x) for x in crit]
        else:
            cands = ["0", "1"]
            options = ["Option 0", "Option 1"]
    else:  # score
        crit = q.get("criteria", [])
        if isinstance(crit, dict):
            cands = list(crit.keys())
            options = [f"level {k}: {crit[k]}" for k in cands]
        elif isinstance(crit, list):
            cands = [str(i) for i in range(len(crit))]
            options = [f"level {i}: {crit[i]}" for i in range(len(crit))]
        else:
            cands = ["0", "1"]
            options = ["level 0", "level 1"]

    state_clean = json.dumps(state_text, ensure_ascii=False) if isinstance(state_text, (dict, list)) else str(state_text)
    prompt_qtype = "noul" if typ in ("boolean", "noul") else typ
    prompt_prefix = f"State: {state_clean}\n{prompt_qtype} question: {instr}\nOptions:\n"
    prompt = prompt_prefix + "".join(f"- {opt}{marker}" for opt in options)
    return prompt, prompt_prefix, cands


def forward_qwen35_question(model, tokenizer, head, state: str, question: dict, marker: str = "<|fim_suffix|>", temperature: float = 1.0):
    prompt, pfx, cands = render_dohnuts_question_with_prefix(state, "q0", question, marker=marker)
    pfx_ids = tokenizer.encode(pfx)
    input_ids = tokenizer.encode(prompt)
    marker_id = tokenizer.convert_tokens_to_ids(marker)
    marker_positions = [pos for pos, tid in enumerate(input_ids) if tid == marker_id]
    state_pos = len(pfx_ids) - 1

    if not marker_positions:
        return None, None
    if len(marker_positions) != len(cands):
        cands = cands[: len(marker_positions)]

    x = mx.array([input_ids], dtype=mx.int32)
    hidden = model.language_model.model(x)

    all_positions = [state_pos] + marker_positions
    tokens = hidden[0, mx.array(all_positions)].astype(mx.float32)

    k = len(cands)
    mock_ex = [{"type": question.get("type"), "candidate_ids": cands, "leaf_tokens": [[1]] * k}]
    logits, _ = head(tokens, mock_ex, kmax=k)
    mx.eval(logits)
    z = logits[0] if logits.ndim > 1 else logits
    z = z[:k] / max(1e-4, temperature)
    probs = mx.softmax(z, axis=-1)
    mx.eval(probs)

    pred_idx = int(mx.argmax(probs).item())
    pred_cand = cands[pred_idx]
    probs_np = np.array(probs)
    return pred_cand, probs_np


def run_jevbench_eval(model, tokenizer, head, jevbench_dir: Path, temperatures: dict):
    print("\n" + "=" * 65, flush=True)
    print("Running JevBench v1.2.2 Public 231 Benchmark on Qwen3.5-0.8B...", flush=True)

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
        pred_cand, probs = forward_qwen35_question(model, tokenizer, head, state, q, temperature=temp)

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
    print("QWEN3.5-0.8B JEVBENCH 231 RESULTS vs DOHNUTS & JEV:", flush=True)
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


def run_applications_eval(model, tokenizer, head, test_file: Path, temperatures: dict):
    print("\n" + "=" * 65, flush=True)
    print("Running Downstream Application Benchmark on test.jsonl...", flush=True)

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
                pred_cand, probs = forward_qwen35_question(model, tokenizer, head, state, q, temperature=temp)
                if pred_cand is not None and gold is not None:
                    if qtype == "boolean":
                        is_corr = (bool(pred_cand == "true") == bool(gold))
                        label_idx = 1 if bool(gold) else 0
                    elif qtype == "choice":
                        is_corr = (str(pred_cand) == str(gold))
                        ids = list(q["criteria"].keys()) if isinstance(q["criteria"], dict) else list(q["criteria"])
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

    return {"by_family": app_results, "ece_15": ece, "brier": mean_brier, "nll": mean_nll}


def run_latency_benchmark(model, tokenizer, head):
    print("\n" + "=" * 65, flush=True)
    print("Running In-Context Latency Scaling Benchmark on Qwen3.5...", flush=True)

    state_text = "Customer: I was billed twice for invoice #4411. I request a full refund today."
    ladder = [1, 5, 10, 20, 50]
    timings = {}

    for num_q in ladder:
        # Build prompt with num_q options
        options = [f"option {i}: Intent {i}" for i in range(num_q)]
        prompt = f"State: {state_text}\nchoice question: Select matching intent\nOptions:\n" + "".join(f"- {opt}<|fim_suffix|>" for opt in options)
        input_ids = tokenizer.encode(prompt)
        x = mx.array([input_ids], dtype=mx.int32)

        # Warmup
        for _ in range(3):
            h = model.language_model.model(x)
            mx.eval(h)

        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            h = model.language_model.model(x)
            mx.eval(h)
        dt = (time.perf_counter() - t0) * 1000.0 / reps

        doh_ref = {1: "15.07 ms", 5: "33.29 ms", 10: "41.57 ms", 20: "~65 ms", 50: "112.51 ms"}.get(num_q, "-")
        amortized = dt / num_q
        print(f"  {num_q:>2} Options | Latency: {dt:>6.2f} ms | Amortized: {amortized:>5.2f} ms/opt | Dohnuts: {doh_ref}", flush=True)
        timings[num_q] = (dt, dt)

    return timings


def load_scorer_head(head_path: Path, hidden_size: int = 1024):
    """Load LinearScorerHead, StateGuidedCandidateSetHead, ResidualCandidateSetHead, or CandidateSetHead."""
    weights = load_file(str(head_path))
    if "base_w" in weights:
        from mlx_state_guided_head import StateGuidedCandidateSetHead
        head = StateGuidedCandidateSetHead(base_weight=mx.array(weights["base_w"]), in_dim=hidden_size)
        head.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=False)
        print(f"Loaded StateGuidedCandidateSetHead from: {head_path}", flush=True)
    elif "proj.weight" in weights and len(weights) == 1:
        head = LinearScorerHead(hidden_size=hidden_size)
        head.proj.weight = mx.array(weights["proj.weight"])
        print(f"Loaded LinearScorerHead from: {head_path}", flush=True)
    elif any("set_encoder" in k or "proj_in" in k for k in weights):
        from mlx_candidate_set_head import CandidateSetHead
        head = CandidateSetHead(in_dim=hidden_size, set_dim=256, num_layers=2, num_heads=4)
        head.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=False)
        print(f"Loaded CandidateSetHead from: {head_path}", flush=True)
    else:
        from mlx_deep_heads import DeepDecisionHeads
        head = DeepDecisionHeads(hidden_size=hidden_size)
        for k, v in weights.items():
            parts = k.split(".")
            mod = head
            for p in parts[:-1]:
                if hasattr(mod, p):
                    mod = getattr(mod, p)
            if hasattr(mod, parts[-1]):
                setattr(mod, parts[-1], mx.array(v))
        print(f"Loaded DeepDecisionHeads from: {head_path}", flush=True)
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--head-weights", required=True)
    parser.add_argument("--temperature-config", default=None, help="Path to calibrated_config.json")
    parser.add_argument("--jevbench-dir", default="data/jevbench")
    parser.add_argument("--test-file", default="data/rlcd_dataset/test.jsonl")
    parser.add_argument("--output-csv", default="results/chart-data-qwen35.csv")
    args = parser.parse_args()

    print(f"Loading base Qwen3.5 model: {args.model_name}", flush=True)
    model, tokenizer = load(args.model_name)
    model.freeze()

    head = load_scorer_head(Path(args.head_weights))

    temperatures = {"choice": 1.0, "boolean": 1.0, "score": 1.0}
    if args.temperature_config:
        cfg_path = Path(args.temperature_config)
        if cfg_path.exists():
            loaded_t = json.loads(cfg_path.read_text())
            temperatures.update({k: float(v) for k, v in loaded_t.items() if k in temperatures})
            print(f"Loaded calibrated temperatures: {temperatures}", flush=True)

    # 1. JevBench 231
    jev_metrics = run_jevbench_eval(model, tokenizer, head, Path(args.jevbench_dir), temperatures)

    # 2. Downstream Applications
    app_metrics = run_applications_eval(model, tokenizer, head, Path(args.test_file), temperatures)

    # 3. Latency
    lat_metrics = run_latency_benchmark(model, tokenizer, head)

    # 4. Export CSV
    from benchmark_rlcd_suite import export_dohnuts_chart_csv
    export_dohnuts_chart_csv(Path(args.output_csv), jev_metrics, app_metrics, lat_metrics)

    print("\nBenchmark completed successfully!", flush=True)


if __name__ == "__main__":
    main()
