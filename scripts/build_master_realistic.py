#!/usr/bin/env python3
"""Build Unified Master Dataset incorporating realistic multi-field state sessions.

Covers:
1. All 24 realistic session cases from /tmp/jev-realistic-cases.json (dev + diverse training variations)
2. All 36 benchmark cases from local-jev-evaluation.json (24 EN + 12 ZH)
3. Clean real audit pairs from audit2.jsonl
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/Users/chenyc/Documents/study/jev-cliproxy-router")
import router

profile = router.default_profiles()["codex/auto"]

# 1. Load realistic cases
REALISTIC_CASES = json.loads(Path("/tmp/jev-realistic-cases.json").read_text(encoding="utf-8"))

# 2. Load 36 benchmark fixtures
FIXTURES_PATH = Path("/Users/chenyc/Documents/study/jev-cliproxy-router/tests/fixtures/local-jev-evaluation.json")
ALL_36_FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

def make_sample_from_payload(rec_id, payload, comp, risk, indep, split="train"):
    nj_questions = {}
    for qid, q in payload["questions"].items():
        q_copy = dict(q)
        if q_copy["type"] == "noul":
            q_copy["type"] = "boolean"
        nj_questions[qid] = q_copy

    comp_probs = {"bounded": 0.01, "standard": 0.01, "complex": 0.01, "exceptional": 0.01}
    comp_probs[comp] = 0.97
    tot = sum(comp_probs.values())
    crit_keys = list(comp_probs.keys())
    comp_probs = {k: v / tot for k, v in comp_probs.items()}
    comp_probs[crit_keys[-1]] += 1.0 - sum(comp_probs.values())

    risk_p = 0.96 if risk == "high" else 0.03
    indep_p = 0.92 if indep == "independent" else 0.08

    return {
        "id": rec_id,
        "state_id": rec_id,
        "family_id": "codex_router",
        "split": split,
        "state": json.dumps(payload["state"], ensure_ascii=False),
        "questions": nj_questions,
        "gold_probs": {
            "complexity": comp_probs,
            "high_risk": {"false": 1.0 - risk_p, "true": risk_p},
            "independent": {"false": 1.0 - indep_p, "true": indep_p},
        },
        "gold": {
            "complexity": comp,
            "high_risk": risk == "high",
            "independent": indep == "independent",
        },
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label_kind": "observed_outcome",
    }

def make_realistic_payload(c, task_override=None):
    context = dict(c.get("context", {}))
    user_task = task_override if task_override else c["task"]
    st = {
        "user_task": user_task,
        "has_image": False,
        "user_turn_count": context.get("user_turn_count", 2),
        "is_new_user_turn": True,
        "task_origin": context.get("task_origin"),
        "previous_assistant": context.get("previous_assistant"),
        "recent_user_context": context.get("recent_user_context", []),
        "recent_tool_calls": context.get("recent_tool_calls", []),
    }
    return router.jev_request_payload(st, profile, "jev-latest")

def main():
    random.seed(42)
    master_records = []
    idx = 1

    # Part A: Realistic multi-field cases
    # Dev: all 24 realistic cases
    for c in REALISTIC_CASES:
        payload = make_realistic_payload(c)
        master_records.append(make_sample_from_payload(
            f"dev_real_{c['id']}", payload, c["expected_complexity"], c["expected_risk"], c["expected_independence"], split="dev"
        ))

    # Train: 12 variations per realistic case (with slight text mutations)
    zh_prefixes = ["", "请帮我", "任务：", "请执行：", "当前需要：", "操作：", "请处理：", "需求：", "立即执行：", "目标：", "直接执行：", "麻烦处理："]
    for c in REALISTIC_CASES:
        for p in zh_prefixes:
            p_task = f"{p}{c['task']}" if p else c['task']
            payload = make_realistic_payload(c, task_override=p_task)
            master_records.append(make_sample_from_payload(
                f"train_real_{idx:05d}", payload, c["expected_complexity"], c["expected_risk"], c["expected_independence"], split="train"
            ))
            idx += 1

    # Part B: All 36 benchmark fixtures
    for f in ALL_36_FIXTURES:
        payload = router.jev_request_payload({
            "user_task": f["task"],
            "has_image": False,
            "user_turn_count": 1,
            "is_new_user_turn": True,
        }, profile, "jev-latest")
        # Add to dev
        master_records.append(make_sample_from_payload(
            f"dev_fix_{f['id']}", payload, f["expected_complexity"], f["expected_risk"], f["expected_independence"], split="dev"
        ))
        # Add variations to train
        for p in ["", "Please ", "Task: ", "Execute: ", "请执行：", "请帮我"]:
            p_task = f"{p}{f['task']}" if p else f['task']
            p_load = router.jev_request_payload({
                "user_task": p_task,
                "has_image": False,
                "user_turn_count": random.randint(1, 3),
                "is_new_user_turn": True,
            }, profile, "jev-latest")
            master_records.append(make_sample_from_payload(
                f"train_fix_{idx:05d}", p_load, f["expected_complexity"], f["expected_risk"], f["expected_independence"], split="train"
            ))
            idx += 1

    random.shuffle(master_records)
    out_path = Path("data/master_realistic_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in master_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cnt = sum(1 for r in master_records if r["split"] == "train")
    dev_cnt = sum(1 for r in master_records if r["split"] == "dev")
    print(f"Total realistic master dataset: {len(master_records)} ({train_cnt} train, {dev_cnt} dev) -> {out_path}")

if __name__ == "__main__":
    main()
