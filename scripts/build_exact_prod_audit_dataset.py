#!/usr/bin/env python3
"""Build authentic production fine-tuning dataset directly from the 315 pairs in jev-router-audit.jsonl.

Preserves the exact:
1. Multi-field state: user_task, previous_assistant, recent_user_context, recent_tool_calls, previous_route
2. Exact router instructions and criteria
3. Exact teacher probability targets from official Jev judgments
"""
import json
import random
from pathlib import Path
from typesafe_adapter import typesafe_request_to_nanojev

def main():
    audit_file = "/Users/chenyc/Downloads/router/jev-router-audit.jsonl"
    out_path = Path("data/exact_production_audit_train.jsonl")

    records = []
    skipped = 0

    with open(audit_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                d = json.loads(line)
                req = d.get("jev_request")
                judg = d.get("judgments")
                if not req or not judg:
                    continue

                # Filter out crazy outliers > 3500 chars to avoid memory issues
                st_len = len(json.dumps(req.get("state", {}), ensure_ascii=False))
                if st_len > 3500:
                    skipped += 1
                    continue

                nj_payload, meta = typesafe_request_to_nanojev(req)
                st = nj_payload["states"][0]
                questions = st["questions"]

                gold_probs = {}
                gold = {}

                for qid, q in questions.items():
                    j_ans = judg.get(qid)
                    if not j_ans: continue

                    if q["type"] == "boolean":
                        p_true = float(j_ans.get("noul", 0.5))
                        gold_probs[qid] = {"false": 1.0 - p_true, "true": p_true}
                        gold[qid] = p_true >= 0.5

                    elif q["type"] == "choice":
                        probs = j_ans.get("probabilities", {})
                        crit_keys = list(q["criteria"].keys())
                        norm_p = {k: float(probs.get(k, 1.0 / len(crit_keys))) for k in crit_keys}
                        tot = sum(norm_p.values())
                        norm_p = {k: v / tot for k, v in norm_p.items()}
                        norm_p[crit_keys[-1]] += 1.0 - sum(norm_p.values())
                        gold_probs[qid] = norm_p
                        gold[qid] = j_ans.get("choice", crit_keys[0])

                if "complexity" in gold_probs:
                    records.append({
                        "id": f"prod_audit_{len(records):04d}",
                        "state_id": f"prod_audit_{len(records):04d}",
                        "family_id": "codex_router",
                        "split": "train",
                        "state": st["state"],
                        "questions": questions,
                        "gold_probs": gold_probs,
                        "gold": gold,
                        "gold_probs_kind": "programmatic_conditional_distribution",
                        "gold_label_kind": "observed_outcome",
                    })
            except Exception:
                pass

    print(f"Extracted {len(records)} authentic production records (skipped {skipped} long outliers).")

    # Split into 85% train and 15% dev
    random.seed(42)
    random.shuffle(records)
    n_dev = int(len(records) * 0.15)
    for i, r in enumerate(records):
        r["split"] = "dev" if i < n_dev else "train"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cnt = sum(1 for r in records if r["split"] == "train")
    dev_cnt = sum(1 for r in records if r["split"] == "dev")
    print(f"Saved: {len(records)} total ({train_cnt} train, {dev_cnt} dev) -> {out_path}")

if __name__ == "__main__":
    main()
