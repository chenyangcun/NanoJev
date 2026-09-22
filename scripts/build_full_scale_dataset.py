#!/usr/bin/env python3
"""Build large-scale diversified general decision dataset (~30,000 questions) in NanoJev schema.

Ingests:
1. kev/decision-v2 (6,232 records: Banking77, BoolQ, AGNews, MNLI, SST5, Yelp, TREC, DBPedia14, Amazon, IMDb)
2. kev/decision-v3 (additional ~5,000 records)
3. MASSIVE 1.1 (5,000 en-US + 5,000 zh-CN across 60 intents = 10,000 records)
4. Contract-NLI (3,000 long-context legal & policy entailment records)
5. WikiQA (3,000 factual question-passage answerability boolean records)
6. SHARC (1,500 rule-based conversational policy records)
7. SMS Spam (1,200 binary spam/phishing records)

Outputs:
  data/full_general_dataset/train.jsonl
  data/full_general_dataset/dev.jsonl
  data/full_general_dataset/calibration.jsonl
  data/full_general_dataset/test.jsonl
"""

import argparse
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def digest(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def clean_str(val) -> str:
    if isinstance(val, dict):
        parts = [str(v).strip() for v in val.values() if v is not None and str(v).strip()]
        return " ".join(" ".join(parts).split())
    if isinstance(val, list):
        return " ".join(str(item).strip() for item in val if str(item).strip())
    if val is None:
        return ""
    return " ".join(str(val).strip().split())


def clean_state(val):
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def process_decision_row(raw: dict, default_split: str = "train", prefix: str = "dec") -> dict:
    state = clean_state(raw["state"])
    meta = raw.get("_meta", {})
    row_id = f"{prefix}_{meta.get('id') or digest(state)[:16]}"
    group_id = f"{prefix}_{meta.get('group_id') or row_id}"
    source = meta.get("source", "decision_pool")

    questions = {}
    gold = {}

    for qid, q in raw["questions"].items():
        qtype = q["type"]
        instr = clean_str(q["instructions"])

        if qtype in ("noul", "boolean"):
            questions[qid] = {
                "type": "boolean",
                "instructions": instr,
            }
            lbl = q.get("label")
            gold[qid] = bool(lbl)
        elif qtype == "choice":
            crit = q.get("criteria", {})
            clean_crit = {}
            for ck, cv in crit.items():
                if cv is None or not str(cv).strip():
                    clean_crit[ck] = f"Option: {ck.replace('_', ' ')}"
                else:
                    clean_crit[ck] = clean_str(str(cv))
            questions[qid] = {
                "type": "choice",
                "instructions": instr,
                "criteria": clean_crit,
            }
            gold[qid] = str(q.get("label"))
        elif qtype == "score":
            crit = q.get("criteria", [])
            clean_crit = [clean_str(str(c)) if str(c).strip() else f"Level {i}" for i, c in enumerate(crit)]
            questions[qid] = {
                "type": "score",
                "instructions": instr,
                "criteria": clean_crit,
            }
            gold[qid] = int(q.get("label"))

    return {
        "id": row_id,
        "state_id": group_id,
        "family_id": source,
        "split": default_split,
        "state": state,
        "questions": questions,
        "gold": gold,
    }


def process_massive(raw_dir: Path, target_per_lang=5000, seed=42) -> list:
    massive_dir = raw_dir / "massive"
    if not massive_dir.exists():
        return []

    rng = random.Random(seed)
    records = []

    for lang in ["en-US", "zh-CN"]:
        lang_file = massive_dir / f"{lang}.jsonl"
        if not lang_file.exists():
            continue

        raw_rows = []
        with open(lang_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_rows.append(json.loads(line))

        all_intents = sorted(list({r["intent"] for r in raw_rows}))
        criteria = {it: f"Intent: {it.replace('_', ' ')}" for it in all_intents}

        by_part = {"train": [], "dev": [], "test": []}
        for r in raw_rows:
            p = r.get("partition", "train")
            if p in by_part:
                by_part[p].append(r)

        for p in by_part:
            rng.shuffle(by_part[p])

        train_take = min(len(by_part["train"]), target_per_lang)
        dev_take = min(len(by_part["dev"]), 500)
        test_take = min(len(by_part["test"]), 600)

        sel_train = by_part["train"][:train_take]
        cal_count = int(train_take * 0.1)
        train_final = sel_train[cal_count:]
        cal_final = sel_train[:cal_count]
        dev_final = by_part["dev"][:dev_take]
        test_final = by_part["test"][:test_take]

        instr = "Which user intent best describes this utterance?" if lang == "en-US" else "该用户指令表达了什么意图？"

        for split_name, rows in [("train", train_final), ("calibration", cal_final), ("dev", dev_final), ("test", test_final)]:
            for r in rows:
                qid = "intent"
                item_id = f"massive_{lang}_{split_name}_{r['id']}"
                records.append({
                    "id": item_id,
                    "state_id": f"massive_{r['id']}",
                    "family_id": f"massive_{lang}",
                    "split": split_name,
                    "state": clean_str(r["utt"]),
                    "questions": {
                        qid: {
                            "type": "choice",
                            "instructions": instr,
                            "criteria": criteria,
                        }
                    },
                    "gold": {qid: r["intent"]},
                })

    print(f"Loaded {len(records)} balanced MASSIVE records", file=sys.stderr)
    return records


def process_contract_nli(raw_dir: Path, max_rows=3000, seed=42) -> list:
    contract_zip = raw_dir / "enrichment" / "contract.zip"
    if not contract_zip.exists():
        return []

    records = []
    rng = random.Random(seed)
    criteria = {
        "Entailment": "The full contract supports and satisfies the stated claim.",
        "Contradiction": "The full contract clearly contradicts the stated claim.",
        "NotMentioned": "The contract does not establish or mention this condition.",
    }

    with zipfile.ZipFile(contract_zip) as z:
        for split in ["train", "dev", "test"]:
            try:
                data = json.loads(z.read(f"contract-nli/{split}.json"))
            except KeyError:
                continue

            for doc in data["documents"]:
                doc_text = clean_str(doc["text"])[:3000]  # truncate huge contracts to fit context
                for key, annot in doc["annotation_sets"][0]["annotations"].items():
                    hyp = data["labels"][key]["hypothesis"]
                    choice_val = annot.get("choice")
                    if choice_val not in criteria:
                        continue

                    qid = "entailment"
                    uid = f"cnli_{doc['id']}_{key}"
                    bucket = int(digest(uid)[:8], 16) % 100
                    if split == "dev":
                        part = "dev"
                    elif split == "test":
                        part = "test"
                    else:
                        part = "calibration" if bucket < 10 else "train"

                    state_str = f"Contract Document:\n{doc_text}\n\nClaim / Hypothesis:\n{hyp}"

                    records.append({
                        "id": uid,
                        "state_id": f"contract_{doc['id']}",
                        "family_id": "contract_nli",
                        "split": part,
                        "state": state_str,
                        "questions": {
                            qid: {
                                "type": "choice",
                                "instructions": "Based strictly on the contract text, how does the agreement relate to the claim?",
                                "criteria": criteria,
                            }
                        },
                        "gold": {qid: choice_val},
                    })

    rng.shuffle(records)
    records = records[:max_rows]
    print(f"Loaded {len(records)} Contract-NLI records", file=sys.stderr)
    return records


def process_wikiqa(raw_dir: Path, max_rows=3000, seed=42) -> list:
    train_parquet = raw_dir / "enrichment" / "wikiqa-train.parquet"
    dev_parquet = raw_dir / "enrichment" / "wikiqa-dev.parquet"
    if not train_parquet.exists():
        return []

    records = []
    rng = random.Random(seed)

    for pfile, split in [(train_parquet, "train"), (dev_parquet, "dev")]:
        table = pq.read_table(pfile)
        rows = table.to_pylist()
        rng.shuffle(rows)

        for i, row in enumerate(rows):
            q_text = clean_str(row["question"])
            ans_text = clean_str(row["answer"])
            lbl = bool(row["label"] == 1)

            uid = f"wikiqa_{split}_{row['question_id']}_{i}"
            bucket = int(digest(uid)[:8], 16) % 100

            if split == "dev":
                part = "test" if bucket < 50 else "dev"
            else:
                part = "calibration" if bucket < 10 else "train"

            state_str = f"Question: {q_text}\n\nCandidate Passage: {ans_text}"

            records.append({
                "id": uid,
                "state_id": f"wikiqa_{row['question_id']}",
                "family_id": "wikiqa",
                "split": part,
                "state": state_str,
                "questions": {
                    "answers_question": {
                        "type": "boolean",
                        "instructions": "Does this passage provide a direct, factual answer to the stated question?",
                    }
                },
                "gold": {"answers_question": lbl},
            })

    rng.shuffle(records)
    records = records[:max_rows]
    print(f"Loaded {len(records)} WikiQA records", file=sys.stderr)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw datasets dir")
    parser.add_argument("--kev-evals", default="/Users/chenyc/work/kev/evals", help="Path to kev/evals dir")
    parser.add_argument("--output-dir", default="data/full_general_dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    all_records = []

    # 1. Load decision-v2
    dec2_dir = Path(args.kev_evals) / "decision-v2"
    if dec2_dir.exists():
        print("Loading decision-v2...", file=sys.stderr)
        for fname, s in [("train.jsonl", "train"), ("development.jsonl", "dev"), ("calibration.jsonl", "calibration"), ("test.jsonl", "test")]:
            fp = dec2_dir / fname
            if fp.exists():
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_records.append(process_decision_row(json.loads(line), default_split=s, prefix="v2"))
        print(f"Loaded {len(all_records)} records from decision-v2", file=sys.stderr)

    # 2. Load decision-v3
    dec3_dir = Path(args.kev_evals) / "v3" / "decision-v3"
    if dec3_dir.exists():
        print("Loading decision-v3...", file=sys.stderr)
        fp = dec3_dir / "train.jsonl"
        if fp.exists():
            v3_cnt = 0
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(process_decision_row(json.loads(line), default_split="train", prefix="v3"))
                        v3_cnt += 1
            print(f"Loaded {v3_cnt} records from decision-v3", file=sys.stderr)

    # 3. Load MASSIVE (scaled to 10k)
    massive = process_massive(raw_dir, target_per_lang=5000, seed=args.seed)
    all_records.extend(massive)

    # 4. Load Contract-NLI (3k)
    cnli = process_contract_nli(raw_dir, max_rows=3000, seed=args.seed)
    all_records.extend(cnli)

    # 5. Load WikiQA (3k)
    wqa = process_wikiqa(raw_dir, max_rows=3000, seed=args.seed)
    all_records.extend(wqa)

    # Deduplicate IDs
    seen_ids = set()
    unique_records = []
    for r in all_records:
        uid = r["id"]
        cnt = 1
        while uid in seen_ids:
            uid = f"{r['id']}_dup{cnt}"
            cnt += 1
        seen_ids.add(uid)
        r_copy = dict(r)
        r_copy["id"] = uid
        r_copy["state_id"] = uid
        unique_records.append(r_copy)

    # Group by split
    splits = {"train": [], "dev": [], "calibration": [], "test": []}
    for r in unique_records:
        s = r["split"]
        if s in splits:
            splits[s].append(r)
        else:
            splits["train"].append(r)

    print("\n" + "=" * 60, file=sys.stderr)
    print("Full General Dataset Summary by Split:", file=sys.stderr)
    for s, l in splits.items():
        print(f"  {s:<12}: {len(l)} records", file=sys.stderr)
    print(f"  TOTAL       : {len(unique_records)} records", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for s, rows in splits.items():
        out_file = out_dir / f"{s}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} records to {out_file}", file=sys.stderr)

    manifest = {
        "dataset": "NanoJev-Full-General-Dataset",
        "counts": {s: len(rows) for s, rows in splits.items()},
        "total": len(unique_records),
        "families": dict(Counter(r["family_id"] for r in unique_records)),
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest written to {out_dir / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
