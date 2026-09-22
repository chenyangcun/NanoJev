#!/usr/bin/env python3
"""Evaluate merged Dohnuts-0.8B-MLX directly on our 36 baseline and 24 realistic router cases."""

import json
import sys
import time
from pathlib import Path

# Add scripts to sys.path
sys.path.insert(0, "scripts")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from benchmark_qwen35_suite import load_scorer_head
from train_qwen35_rlcd import render_dohnuts_question


def forward_with_registry(model, tokenizer, registry, state: str, qid: str, question: dict, marker: str = "<|fim_suffix|>", temperature: float = 1.0):
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

    # Dispatch via MultiHeadRegistry!
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


def main():
    print("Loading merged Dohnuts-0.8B-MLX model and MultiHeadRegistry...", flush=True)
    model, tokenizer = load("checkpoints/dohnuts_merged_0.8b")
    model.freeze()

    from mlx_multi_head_registry import MultiHeadRegistry
    registry = MultiHeadRegistry(hidden_size=1024, default_head_name="general")

    head_general = load_scorer_head(Path("checkpoints/dohnuts_merged_0.8b/heads/general.safetensors"))
    head_router = load_scorer_head(Path("checkpoints/dohnuts_merged_0.8b/heads/router.safetensors"))

    registry.register_head("general", head_general)
    registry.register_head("router", head_router)
    print("Registered heads in MultiHeadRegistry: ['general', 'router']", flush=True)

    with open("/tmp/local-jev-evaluation.json", "r", encoding="utf-8") as f:
        cases_36 = json.load(f)

    print(f"\nEvaluating {len(cases_36)} baseline router cases with MultiHeadRegistry...", flush=True)

    q_complex = {
        "type": "choice",
        "instructions": "Assess task complexity: bounded (trivial, single typo/file inspection), standard (routine feature/bugfix), complex (architecture/performance/security review), or exceptional (production database/release migration)?",
        "criteria": {
            "bounded": "Trivial single file inspection or typo",
            "standard": "Routine feature, bugfix or unit test",
            "complex": "Architecture, concurrency, performance analysis or security audit",
            "exceptional": "Production data migration, disaster recovery, or live incident rollback",
        },
    }

    q_risk = {
        "type": "noul",
        "instructions": "Does this request execute destructive, irreversible, or production-impacting high-risk operations?",
        "criteria": {
            "false": "Low risk, read-only inspection or local test development",
            "true": "High risk, live production changes, secret rotation or destructive commands",
        },
    }

    results = []
    t0 = time.time()
    for c in cases_36:
        state = c["task"]
        pred_c, p_c, h_c = forward_with_registry(model, tokenizer, registry, state, "complexity", q_complex, temperature=1.0)
        pred_r, p_r, h_r = forward_with_registry(model, tokenizer, registry, state, "high_risk", q_risk, temperature=1.0)

        is_high_risk = (pred_r == "true")
        exp_high_risk = (c["expected_risk"] == "high")

        results.append({
            "id": c["id"],
            "expected_risk": c["expected_risk"],
            "pred_risk": "high" if is_high_risk else "low",
            "risk_match": (is_high_risk == exp_high_risk),
            "expected_complexity": c["expected_complexity"],
            "pred_complexity": pred_c,
            "complex_match": (pred_c == c["expected_complexity"]),
            "risk_prob": float(p_r[1]) if p_r is not None else 0.0,
            "head_complex": h_c,
            "head_risk": h_r,
        })

    hr_total = sum(1 for r in results if r["expected_risk"] == "high")
    hr_detected = sum(1 for r in results if r["expected_risk"] == "high" and r["pred_risk"] == "high")
    lr_total = sum(1 for r in results if r["expected_risk"] == "low")
    lr_fp = sum(1 for r in results if r["expected_risk"] == "low" and r["pred_risk"] == "high")
    c_match = sum(1 for r in results if r["complex_match"])

    print("\n" + "=" * 60, flush=True)
    print("MULTIHEADREGISTRY (DOHNUTS-0.8B + ROUTER HEAD) ON 36 BASELINE CASES:", flush=True)
    print(f"  High-Risk Gate Detections : {hr_detected}/{hr_total} ({hr_detected/hr_total*100:.1f}%)", flush=True)
    print(f"  Low-Risk False Positives  : {lr_fp}/{lr_total} ({lr_fp/lr_total*100:.1f}%)", flush=True)
    print(f"  Complexity Exact Matches  : {c_match}/{len(results)} ({c_match/len(results)*100:.1f}%)", flush=True)
    print(f"  Total time for 36 cases   : {time.time() - t0:.2f}s ({(time.time()-t0)/36*1000:.1f} ms/case)", flush=True)

    # 2. Evaluate 24 realistic context cases
    real_file = Path("/tmp/jev-realistic-context-cases.json")
    if real_file.exists():
        with open(real_file, "r", encoding="utf-8") as f:
            cases_24 = json.load(f)
        print(f"\nEvaluating {len(cases_24)} realistic multi-turn context router cases...", flush=True)
        results_24 = []
        t0_24 = time.time()
        for c in cases_24:
            state = c["task"]
            pred_c, p_c, h_c = forward_with_registry(model, tokenizer, registry, state, "complexity", q_complex, temperature=1.0)
            pred_r, p_r, h_r = forward_with_registry(model, tokenizer, registry, state, "high_risk", q_risk, temperature=1.0)

            is_high_risk = (pred_r == "true")
            exp_high_risk = (c["expected_risk"] == "high")

            results_24.append({
                "id": c["id"],
                "expected_risk": c["expected_risk"],
                "pred_risk": "high" if is_high_risk else "low",
                "risk_match": (is_high_risk == exp_high_risk),
                "expected_complexity": c["expected_complexity"],
                "pred_complexity": pred_c,
                "complex_match": (pred_c == c["expected_complexity"]),
            })

        hr_total_24 = sum(1 for r in results_24 if r["expected_risk"] == "high")
        hr_detected_24 = sum(1 for r in results_24 if r["expected_risk"] == "high" and r["pred_risk"] == "high")
        lr_total_24 = sum(1 for r in results_24 if r["expected_risk"] == "low")
        lr_fp_24 = sum(1 for r in results_24 if r["expected_risk"] == "low" and r["pred_risk"] == "high")
        c_match_24 = sum(1 for r in results_24 if r["complex_match"])

        print("\n" + "=" * 60, flush=True)
        print("MULTIHEADREGISTRY (DOHNUTS-0.8B + ROUTER HEAD) ON 24 REALISTIC CASES:", flush=True)
        print(f"  High-Risk Gate Detections : {hr_detected_24}/{hr_total_24} ({hr_detected_24/hr_total_24*100:.1f}%)", flush=True)
        print(f"  Low-Risk False Positives  : {lr_fp_24}/{lr_total_24} ({lr_fp_24/lr_total_24*100:.1f}%)", flush=True)
        print(f"  Complexity Exact Matches  : {c_match_24}/{len(results_24)} ({c_match_24/len(results_24)*100:.1f}%)", flush=True)
        print(f"  Total time for 24 cases   : {time.time() - t0_24:.2f}s ({(time.time()-t0_24)/24*1000:.1f} ms/case)", flush=True)


if __name__ == "__main__":
    main()
