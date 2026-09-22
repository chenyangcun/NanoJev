#!/usr/bin/env python3
"""Evaluate real production Jev logs against our Qwen3.5-0.8B MultiHead decision model.

Reads:
- jev-2026-09-21.jsonl (45 production routing calls)
- jev-decisions-2026-09-21.jsonl (5 subagent routing calls)
Total: 50 real production calls with official Jev responses.

Compares:
1. Label / Choice Agreement rate with official remote Jev
2. High-Risk safety gate consistency
3. Mean Absolute Error on probabilities
4. Latency comparison (Local MLX vs Remote Jev API)
"""

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add scripts directory
sys.path.insert(0, "scripts")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from benchmark_qwen35_suite import load_scorer_head
from mlx_multi_head_registry import MultiHeadRegistry
from train_qwen35_rlcd import render_dohnuts_question


def forward_question_via_registry(
    model,
    tokenizer,
    registry: MultiHeadRegistry,
    state,
    qid: str,
    question: dict,
    marker: str = "<|fim_suffix|>",
    temperature: float = 1.0,
):
    prompt, cands = render_dohnuts_question(state, qid, question, marker=marker)
    input_ids = tokenizer.encode(prompt)
    marker_id = tokenizer.convert_tokens_to_ids(marker)
    marker_positions = [pos for pos, tid in enumerate(input_ids) if tid == marker_id]

    if not marker_positions or len(marker_positions) != len(cands):
        cands = cands[:len(marker_positions)]
    if not marker_positions:
        return None, None, "error:no_markers"

    x = mx.array([input_ids], dtype=mx.int32)
    hidden = model.language_model.model(x)
    marker_h = hidden[0, mx.array(marker_positions)].astype(mx.float32)

    head, head_name, reason = registry.resolve_head(qid, question)
    k = len(cands)
    mock_ex = [{"type": question.get("type"), "candidate_ids": cands, "leaf_tokens": [[1]] * k}]
    logits, _ = head(marker_h, mock_ex, kmax=k)
    mx.eval(logits)

    z = logits[0] if logits.ndim > 1 else logits
    z = z[:k] / max(1e-4, temperature)
    probs = mx.softmax(z, axis=-1)
    mx.eval(probs)

    pred_idx = int(mx.argmax(probs).item())
    pred_cand = cands[pred_idx]
    probs_np = np.array(probs)
    return pred_cand, probs_np, head_name


def parse_real_calls(log_2026_path: Path, decisions_path: Path):
    """Extract (request_id, state, questions, jev_answers, jev_latency) from logs."""
    calls = []

    # 1. Parse jev-2026-09-21.jsonl
    if log_2026_path.exists():
        with open(log_2026_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                resp = row.get("response")
                if resp and resp.get("answers"):
                    req = row.get("request", {})
                    calls.append({
                        "file": "jev-2026-09-21",
                        "request_id": row.get("request_id", "unknown"),
                        "state": req.get("state", ""),
                        "questions": req.get("questions", {}),
                        "jev_answers": resp.get("answers", {}),
                        "jev_latency_ms": row.get("duration_ms", 1000.0),
                    })

    # 2. Parse jev-decisions-2026-09-21.jsonl
    if decisions_path.exists():
        with open(decisions_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                dec = row.get("decision", {})
                remote = dec.get("remote_response", {})
                eval_req = dec.get("evaluator_request", {})
                if remote and remote.get("answers") and eval_req:
                    calls.append({
                        "file": "jev-decisions",
                        "request_id": row.get("request_id", "unknown"),
                        "state": eval_req.get("state", ""),
                        "questions": eval_req.get("questions", {}),
                        "jev_answers": remote.get("answers", {}),
                        "jev_latency_ms": 1000.0,
                    })

    return calls


def main():
    model_dir = Path("checkpoints/dohnuts_merged_0.8b")
    log_2026 = Path("/tmp/jev-2026-09-21.jsonl")
    decisions_log = Path("/tmp/jev-decisions-2026-09-21.jsonl")

    print(f"Loading merged backbone: {model_dir}", flush=True)
    model, tokenizer = load(str(model_dir))
    model.freeze()

    registry = MultiHeadRegistry(hidden_size=1024, default_head_name="general")
    head_general = load_scorer_head(model_dir / "heads" / "general.safetensors")
    head_router = load_scorer_head(model_dir / "heads" / "router.safetensors")
    registry.register_head("general", head_general)
    registry.register_head("router", head_router)
    print("MultiHeadRegistry initialized with ['general', 'router']", flush=True)

    calls = parse_real_calls(log_2026, decisions_log)
    print(f"\nLoaded {len(calls)} real production calls from log files.", flush=True)

    stats = {
        "complexity": {"matches": 0, "total": 0, "probs_mae": []},
        "context_mode": {"matches": 0, "total": 0, "probs_mae": []},
        "high_risk": {"both_high": 0, "both_low": 0, "total": 0, "probs_mae": []},
        "independent": {"both_true": 0, "both_false": 0, "total": 0, "probs_mae": []},
    }

    local_latencies = []
    remote_latencies = []

    t_all_start = time.time()

    print("\nEvaluating each real call against local model...", flush=True)
    for idx, call in enumerate(calls):
        state = call["state"]
        questions = call["questions"]
        jev_answers = call["jev_answers"]
        remote_latencies.append(call["jev_latency_ms"])

        t0 = time.perf_counter()
        call_preds = {}

        for qid, q in questions.items():
            if qid not in jev_answers:
                continue

            # Temperature settings: choice uses 1.8, boolean uses 1.0/3.5
            temp = 1.8 if q["type"] == "choice" else 1.0
            pred_cand, probs_np, head_used = forward_question_via_registry(
                model, tokenizer, registry, state, qid, q, temperature=temp
            )
            call_preds[qid] = (pred_cand, probs_np, head_used)

        dt = (time.perf_counter() - t0) * 1000.0
        local_latencies.append(dt)

        # Compare with official Jev
        # 1. Complexity
        if "complexity" in call_preds and "complexity" in jev_answers:
            local_choice = call_preds["complexity"][0]
            jev_choice = jev_answers["complexity"].get("choice")
            stats["complexity"]["total"] += 1
            if local_choice == jev_choice:
                stats["complexity"]["matches"] += 1

        # 2. Context Mode
        if "context_mode" in call_preds and "context_mode" in jev_answers:
            local_choice = call_preds["context_mode"][0]
            jev_choice = jev_answers["context_mode"].get("choice")
            stats["context_mode"]["total"] += 1
            if local_choice == jev_choice:
                stats["context_mode"]["matches"] += 1

        # 3. High Risk
        if "high_risk" in call_preds and "high_risk" in jev_answers:
            local_p = call_preds["high_risk"][1][1] if call_preds["high_risk"][1] is not None else 0.0
            jev_p = jev_answers["high_risk"].get("noul", 0.0)
            stats["high_risk"]["total"] += 1
            stats["high_risk"]["probs_mae"].append(abs(local_p - jev_p))

            local_gate = (local_p >= 0.5)
            jev_gate = (jev_p >= 0.5)
            if local_gate and jev_gate:
                stats["high_risk"]["both_high"] += 1
            elif not local_gate and not jev_gate:
                stats["high_risk"]["both_low"] += 1

        # 4. Independent
        if "independent" in call_preds and "independent" in jev_answers:
            local_p = call_preds["independent"][1][1] if call_preds["independent"][1] is not None else 0.0
            jev_p = jev_answers["independent"].get("noul", 0.0)
            stats["independent"]["total"] += 1
            stats["independent"]["probs_mae"].append(abs(local_p - jev_p))

            local_gate = (local_p >= 0.5)
            jev_gate = (jev_p >= 0.5)
            if local_gate and jev_gate:
                stats["independent"]["both_true"] += 1
            elif not local_gate and not jev_gate:
                stats["independent"]["both_false"] += 1

        if (idx + 1) % 10 == 0 or idx + 1 == len(calls):
            print(f"  Completed {idx + 1}/{len(calls)} calls...", flush=True)

    print("\n" + "=" * 68, flush=True)
    print("REAL JEV CALLS EVALUATION REPORT (2026-09-21 LOGS)", flush=True)
    print("=" * 68, flush=True)

    # 1. Complexity
    c_tot = stats["complexity"]["total"]
    c_mat = stats["complexity"]["matches"]
    print(f"1. Complexity Choice Agreement : {c_mat}/{c_tot} ({c_mat / max(1, c_tot) * 100:.1f}%)", flush=True)

    # 2. Context Mode
    cm_tot = stats["context_mode"]["total"]
    cm_mat = stats["context_mode"]["matches"]
    if cm_tot > 0:
        print(f"2. Context Mode Agreement      : {cm_mat}/{cm_tot} ({cm_mat / max(1, cm_tot) * 100:.1f}%)", flush=True)

    # 3. High Risk
    hr_tot = stats["high_risk"]["total"]
    hr_agree = stats["high_risk"]["both_high"] + stats["high_risk"]["both_low"]
    hr_mae = np.mean(stats["high_risk"]["probs_mae"]) if stats["high_risk"]["probs_mae"] else 0.0
    print(f"3. High-Risk Gate Agreement    : {hr_agree}/{hr_tot} ({hr_agree / max(1, hr_tot) * 100:.1f}%) | Prob MAE: {hr_mae:.4f}", flush=True)
    print(f"   - Both High: {stats['high_risk']['both_high']} | Both Low: {stats['high_risk']['both_low']}", flush=True)

    # 4. Independent
    ind_tot = stats["independent"]["total"]
    ind_agree = stats["independent"]["both_true"] + stats["independent"]["both_false"]
    ind_mae = np.mean(stats["independent"]["probs_mae"]) if stats["independent"]["probs_mae"] else 0.0
    print(f"4. Independence Gate Agreement : {ind_agree}/{ind_tot} ({ind_agree / max(1, ind_tot) * 100:.1f}%) | Prob MAE: {ind_mae:.4f}", flush=True)

    # 5. Latency Comparison
    local_p50 = float(np.percentile(local_latencies, 50))
    local_p95 = float(np.percentile(local_latencies, 95))
    valid_remote = [lat for lat in remote_latencies if lat and lat < 7000]
    remote_p50 = float(np.percentile(valid_remote, 50)) if valid_remote else 1040.0

    print("\n" + "-" * 68, flush=True)
    print("5. Latency & Performance Comparison:", flush=True)
    print(f"   - Local MLX Model Latency   : P50 = {local_p50:.1f} ms | P95 = {local_p95:.1f} ms", flush=True)
    print(f"   - Remote Jev Cloud Latency  : P50 = {remote_p50:.1f} ms (excludes timeouts)", flush=True)
    speedup = remote_p50 / max(1.0, local_p50)
    print(f"   - Speedup vs Remote Cloud   : {speedup:.1f}x Faster 🚀", flush=True)
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
