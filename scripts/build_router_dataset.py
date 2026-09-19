#!/usr/bin/env python3
"""Convert jev-router-audit.jsonl into NanoJev standard training JSONL format."""
import argparse
import json
import random
from pathlib import Path
from typesafe_adapter import typesafe_request_to_nanojev


def convert_audit_to_training(audit_path: str, output_path: str, dev_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)
    records = []
    skipped = 0

    with open(audit_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            req = data.get("jev_request")
            judgments = data.get("judgments")
            if not req or not judgments:
                skipped += 1
                continue

            # Convert request via adapter
            try:
                nj_payload, meta = typesafe_request_to_nanojev(req)
            except Exception as e:
                skipped += 1
                continue

            state_row = nj_payload["states"][0]
            questions = state_row["questions"]

            # Build gold_probs and gold mappings
            gold_probs = {}
            gold = {}

            for qid, q in questions.items():
                j_ans = judgments.get(qid)
                if not j_ans:
                    continue

                q_type = q["type"]
                if q_type == "boolean":
                    # In TypeSafe, noul returns probability of yes/true
                    p_true = float(j_ans.get("noul", 0.5))
                    p_false = max(0.0, min(1.0, 1.0 - p_true))
                    gold_probs[qid] = {"false": round(p_false, 4), "true": round(p_true, 4)}
                    gold[qid] = p_true >= 0.5

                elif q_type == "choice":
                    probs = j_ans.get("probabilities", {})
                    # Ensure sum is 1.0 and matches criteria
                    crit_keys = list(q["criteria"].keys())
                    norm_probs = {}
                    total_p = sum(float(probs.get(k, 0.0)) for k in crit_keys)
                    if total_p <= 0:
                        norm_probs = {k: 1.0 / len(crit_keys) for k in crit_keys}
                    else:
                        for k in crit_keys:
                            norm_probs[k] = float(probs.get(k, 0.0)) / total_p
                    # Ensure exact sum = 1.0
                    total_p2 = sum(norm_probs.values())
                    diff = 1.0 - total_p2
                    norm_probs[crit_keys[0]] += diff
                    gold_probs[qid] = {k: round(v, 6) for k, v in norm_probs.items()}
                    gold[qid] = j_ans.get("choice", crit_keys[0])

                elif q_type == "score":
                    probs = j_ans.get("probabilities", {})
                    n_levels = len(q["criteria"])
                    norm_probs = {str(i): float(probs.get(str(i), 1.0 / n_levels)) for i in range(n_levels)}
                    gold_probs[qid] = norm_probs
                    gold[qid] = int(j_ans.get("level", 0))

            if not gold_probs:
                skipped += 1
                continue

            record_id = f"router_rec_{len(records):04d}"
            records.append({
                "id": record_id,
                "state_id": record_id,
                "family_id": "codex_router",
                "split": "train",  # placeholder, split later
                "state": state_row["state"],
                "questions": questions,
                "gold_probs": gold_probs,
                "gold": gold,
                "gold_probs_kind": "programmatic_conditional_distribution",
                "gold_label_kind": "observed_outcome",
                "metadata": {
                    "source_audit_line": idx,
                    "at": data.get("at"),
                    "chosen_model": data.get("chosen_model"),
                }
            })

    # Shuffle and split into train and dev
    random.shuffle(records)
    n_dev = int(len(records) * dev_ratio)
    for i, r in enumerate(records):
        r["split"] = "dev" if i < n_dev else "train"

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_count = sum(1 for r in records if r["split"] == "train")
    dev_count = sum(1 for r in records if r["split"] == "dev")
    print(f"Generated {len(records)} records ({train_count} train, {dev_count} dev, {skipped} skipped) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-log", default="/Users/chenyc/Downloads/router/jev-router-audit.jsonl")
    parser.add_argument("--output", default="data/router_train.jsonl")
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    args = parser.parse_args()
    convert_audit_to_training(args.audit_log, args.output, args.dev_ratio)
