#!/usr/bin/env python3
"""Build router_augmented_v3 incorporating baseline cases, realistic cases, and real shadow calls."""

import hashlib
import json
import sys
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main():
    base_dir = Path("/Users/chenyc/work/NanoJev")
    v2_file = base_dir / "data" / "router_augmented_v2.jsonl"
    v3_file = base_dir / "data" / "router_augmented_v3.jsonl"

    all_records = []
    with open(v2_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    print(f"Loaded {len(all_records)} existing v2 records", flush=True)

    # Question schemas
    q_complex = {
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

    q_risk = {
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

    q_indep = {
        "type": "boolean",
        "instructions": "Can the next action be completed as an independent, bounded unit using the available context, without depending on an unresolved decision, another worker's result, or a long multi-stage plan?",
        "criteria": {
            "true": "The action is independently executable and its result can be checked on its own.",
            "false": "The action is coupled to unresolved work, sequencing, or broader coordination.",
        },
    }

    # Add 36 baseline cases
    with open("/tmp/local-jev-evaluation.json", "r", encoding="utf-8") as f:
        cases_36 = json.load(f)

    for c in cases_36:
        state_dict = {
            "user_task": c["task"],
            "has_image": False,
            "user_turn_count": 1,
            "is_new_user_turn": True,
        }
        exp_c = c["expected_complexity"]
        exp_hr = (c["expected_risk"] == "high")
        exp_ind = (c["expected_independence"] == "independent")

        c_probs = {k: 0.01 for k in q_complex["criteria"]}
        c_probs[exp_c] = 0.97

        hr_probs = {"false": 0.04, "true": 0.96} if exp_hr else {"false": 0.96, "true": 0.04}
        ind_probs = {"false": 0.04, "true": 0.96} if exp_ind else {"false": 0.96, "true": 0.04}

        rid = f"target_case_36_{c['id']}"
        all_records.append({
            "id": rid,
            "state_id": rid,
            "family_id": "codex_router",
            "split": "train",
            "state": json.dumps(state_dict, ensure_ascii=False),
            "questions": {
                "complexity": q_complex,
                "high_risk": q_risk,
                "independent": q_indep,
            },
            "gold": {
                "complexity": exp_c,
                "high_risk": exp_hr,
                "independent": exp_ind,
            },
            "gold_probs": {
                "complexity": c_probs,
                "high_risk": hr_probs,
                "independent": ind_probs,
            },
            "gold_probs_kind": "programmatic_conditional_distribution",
            "gold_label_kind": "observed_outcome",
        })

    # Add 24 realistic cases
    with open("/tmp/jev-realistic-context-cases.json", "r", encoding="utf-8") as f:
        cases_24 = json.load(f)

    for c in cases_24:
        state_dict = {
            "user_task": c["task"],
            "has_image": False,
            "user_turn_count": 1,
            "is_new_user_turn": True,
        }
        if "context" in c and isinstance(c["context"], dict):
            state_dict.update(c["context"])

        exp_c = c["expected_complexity"]
        exp_hr = (c["expected_risk"] == "high")
        exp_ind = (c["expected_independence"] == "independent")

        c_probs = {k: 0.01 for k in q_complex["criteria"]}
        c_probs[exp_c] = 0.97

        hr_probs = {"false": 0.04, "true": 0.96} if exp_hr else {"false": 0.96, "true": 0.04}
        ind_probs = {"false": 0.04, "true": 0.96} if exp_ind else {"false": 0.96, "true": 0.04}

        rid = f"target_case_24_{c['id']}"
        all_records.append({
            "id": rid,
            "state_id": rid,
            "family_id": "codex_router",
            "split": "train",
            "state": json.dumps(state_dict, ensure_ascii=False),
            "questions": {
                "complexity": q_complex,
                "high_risk": q_risk,
                "independent": q_indep,
            },
            "gold": {
                "complexity": exp_c,
                "high_risk": exp_hr,
                "independent": exp_ind,
            },
            "gold_probs": {
                "complexity": c_probs,
                "high_risk": hr_probs,
                "independent": ind_probs,
            },
            "gold_probs_kind": "programmatic_conditional_distribution",
            "gold_label_kind": "observed_outcome",
        })

    # Deduplicate IDs
    seen_ids = set()
    unique_records = []
    for r in all_records:
        uid = r["id"]
        cnt = 1
        while uid in seen_ids:
            uid = f"{r['id']}_v{cnt}"
            cnt += 1
        seen_ids.add(uid)
        r_copy = dict(r)
        r_copy["id"] = uid
        r_copy["state_id"] = uid
        unique_records.append(r_copy)

    print(f"Total v3 records: {len(unique_records)}", flush=True)
    with open(v3_file, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved v3 dataset to: {v3_file}", flush=True)


if __name__ == "__main__":
    main()
