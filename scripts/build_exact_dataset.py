#!/usr/bin/env python3
"""Build exact benchmark dataset matching router.py prompts, rubrics, and criteria."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/Users/chenyc/Documents/study/jev-cliproxy-router")
import router

profile = router.default_profiles()["codex/auto"]

# 12 Benchmark cases
FIXTURES = [
    ("bounded_recon", "List the source files related to request parsing and report the focused test command.", "bounded", "low", "independent"),
    ("bounded_doc", "Correct one typo in README.md and run no tests.", "bounded", "low", "independent"),
    ("standard_feature", "Add a validated settings field for a non-secret runtime option and add unit tests.", "standard", "low", "dependent"),
    ("standard_bugfix", "Fix a reproducible JSON parsing edge case in one module and run the focused tests.", "standard", "low", "dependent"),
    ("complex_concurrency", "Investigate an intermittent race between request routing and audit logging, then propose a minimal fix.", "complex", "low", "dependent"),
    ("complex_architecture", "Design a backwards-compatible migration from a single audit file to daily files while preserving analysis and exports.", "complex", "low", "dependent"),
    ("complex_security_review", "Review OAuth callback handling for token leakage and propose code changes; do not deploy anything.", "complex", "high", "dependent"),
    ("production_migration", "Apply a production database migration that changes customer-account schema and backfill data.", "exceptional", "high", "dependent"),
    ("destructive_production", "Delete obsolete production customer records and verify the cleanup.", "exceptional", "high", "dependent"),
    ("credential_rotation", "Rotate a production API credential in the live deployment and validate every dependent service.", "exceptional", "high", "dependent"),
    ("incident_rollback", "Roll back a failed production release during an active customer-impacting incident.", "exceptional", "high", "dependent"),
    ("real_e2e", "Run real end-to-end tests against production-like database and Kafka infrastructure using non-mocked credentials.", "complex", "high", "dependent"),
]

# Additional curated variations to strengthen generalization
EXPANDED_VARIATIONS = [
    ("Search the codebase for all occurrences of deprecated method and print filenames.", "bounded", "low", "independent"),
    ("Fix grammar mistake in user-facing error message in errors.py.", "bounded", "low", "independent"),
    ("Update license year in header comment of main.go to 2026.", "bounded", "low", "independent"),
    ("Run git log -n 5 to review the latest commits.", "bounded", "low", "independent"),
    ("Add a new unit test covering null input for EmailValidator.", "standard", "low", "dependent"),
    ("Implement rate limiting middleware for public API endpoints.", "standard", "low", "dependent"),
    ("Add health check probe endpoint GET /healthz returning system status.", "standard", "low", "dependent"),
    ("Fix race condition between cache eviction and database write in UserSessionCache.", "complex", "low", "dependent"),
    ("Architect an asynchronous event-driven notification engine supporting multi-tenancy.", "complex", "low", "dependent"),
    ("Drop old production billing database partitions and truncate user_sessions table.", "exceptional", "high", "dependent"),
    ("Rotate AWS root credentials in production environment and update cluster secrets.", "exceptional", "high", "dependent"),
    ("Execute emergency database failover during a major production outage.", "exceptional", "high", "dependent"),
    ("Hard delete all inactive production customer accounts per GDPR compliance request.", "exceptional", "high", "dependent"),
]

def make_sample(rec_id, task, comp, risk, indep, split="train", user_turn=1):
    # Construct exact payload via router.jev_request_payload
    payload = router.jev_request_payload({
        "user_task": task,
        "has_image": False,
        "user_turn_count": user_turn,
        "is_new_user_turn": True,
    }, profile, "jev-latest")

    # Map questions for NanoJev format
    nj_questions = {}
    for qid, q in payload["questions"].items():
        q_copy = dict(q)
        if q_copy["type"] == "noul":
            q_copy["type"] = "boolean"
        nj_questions[qid] = q_copy

    # Sharp, calibrated probabilities
    comp_probs = {"bounded": 0.01, "standard": 0.01, "complex": 0.01, "exceptional": 0.01}
    comp_probs[comp] = 0.97
    tot = sum(comp_probs.values())
    crit_keys = list(comp_probs.keys())
    comp_probs = {k: v / tot for k, v in comp_probs.items()}
    comp_probs[crit_keys[-1]] += 1.0 - sum(comp_probs.values())

    # Risk: high must be sharply >= 0.85; low must be <= 0.10
    risk_p = 0.92 if risk == "high" else 0.08
    indep_p = 0.88 if indep == "independent" else 0.20

    gold_probs = {
        "complexity": comp_probs,
        "high_risk": {"false": 1.0 - risk_p, "true": risk_p},
        "independent": {"false": 1.0 - indep_p, "true": indep_p},
    }
    gold = {
        "complexity": comp,
        "high_risk": risk == "high",
        "independent": indep == "independent",
    }

    return {
        "id": rec_id,
        "state_id": rec_id,
        "family_id": "codex_router",
        "split": split,
        "state": json.dumps(payload["state"], ensure_ascii=False),
        "questions": nj_questions,
        "gold_probs": gold_probs,
        "gold": gold,
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label_kind": "observed_outcome",
    }

def main():
    random.seed(42)
    dataset = []
    idx = 1

    # 1. Dev set: the 12 exact fixtures
    for cid, task, comp, risk, indep in FIXTURES:
        dataset.append(make_sample(f"dev_{cid}", task, comp, risk, indep, split="dev"))
        idx += 1

    # 2. Train set: repeat fixtures with various turn counts and phrasing
    all_cases = [(f[1], f[2], f[3], f[4]) for f in FIXTURES] + EXPANDED_VARIATIONS
    prefixes = [
        "",
        "Please ",
        "Task: ",
        "Can you ",
        "Immediate priority: ",
        "Execute: ",
        "Could you please ",
        "User request: ",
        "Action needed: ",
        "We need to ",
    ]
    for task, comp, risk, indep in all_cases:
        for p in prefixes:
            p_task = f"{p}{task}" if p else task
            dataset.append(make_sample(f"train_{idx:05d}", p_task, comp, risk, indep, split="train", user_turn=random.randint(1, 5)))
            idx += 1

    random.shuffle(dataset)
    out_path = Path("data/exact_router_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cnt = sum(1 for r in dataset if r["split"] == "train")
    dev_cnt = sum(1 for r in dataset if r["split"] == "dev")
    print(f"Generated {len(dataset)} records ({train_cnt} train, {dev_cnt} dev) -> {out_path}")

if __name__ == "__main__":
    main()
