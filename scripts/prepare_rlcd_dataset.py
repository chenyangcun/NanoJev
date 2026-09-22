#!/usr/bin/env python3
"""Build calibrated RLCD dataset combining decision-v2 and Dohnuts pure-text sources.

Sources included:
- kev/decision-v2: Banking77, BoolQ, AGNews, MNLI, SST-5, Yelp, TREC, DBPedia14, Amazon, IMDb
- Dohnuts massive 1.1: bilingual zh-CN and en-US intent classification (60 intents)
- Dohnuts enrichment/sms: binary unsolicited spam detection
- Dohnuts enrichment/sharc: conversational rule & policy reasoning

All multimodal (images/bounding boxes) are strictly excluded.
Outputs strict NanoJev schema to:
  data/rlcd_dataset/train.jsonl
  data/rlcd_dataset/dev.jsonl
  data/rlcd_dataset/calibration.jsonl
  data/rlcd_dataset/test.jsonl
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


def process_decision_v2_row(raw: dict, default_split: str = "train") -> dict:
    """Normalize a row from kev/decision-v2 into NanoJev format."""
    state = clean_state(raw["state"])
    meta = raw.get("_meta", {})
    row_id = meta.get("id") or digest(state)[:16]
    group_id = meta.get("group_id") or row_id
    source = meta.get("source", "decision_v2")
    split = default_split

    questions = {}
    gold = {}

    for qid, q in raw["questions"].items():
        qtype = q["type"]
        instr = clean_str(q["instructions"])

        if qtype == "noul":
            # Convert noul to NanoJev boolean
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
        "split": split,
        "state": state,
        "questions": questions,
        "gold": gold,
    }


def process_massive(raw_dir: Path, target_per_lang=1200, seed=42) -> list:
    """Extract balanced bilingual MASSIVE intent classification rows."""
    massive_dir = raw_dir / "massive"
    if not massive_dir.exists():
        print("MASSIVE raw directory not found, skipping...", file=sys.stderr)
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

        # Collect all 60 unique intents
        all_intents = sorted(list({r["intent"] for r in raw_rows}))
        criteria = {it: f"Intent: {it.replace('_', ' ')}" for it in all_intents}

        # Group by partition
        by_partition = {"train": [], "dev": [], "test": []}
        for r in raw_rows:
            p = r.get("partition", "train")
            if p in by_partition:
                by_partition[p].append(r)

        # Shuffle
        for p in by_partition:
            rng.shuffle(by_partition[p])

        # Subsample to keep dataset balanced
        train_take = min(len(by_partition["train"]), target_per_lang)
        dev_take = min(len(by_partition["dev"]), 200)
        test_take = min(len(by_partition["test"]), 250)

        selected_train = by_partition["train"][:train_take]
        # Split a fraction of train into calibration
        calib_count = int(train_take * 0.1)
        train_final = selected_train[calib_count:]
        calib_final = selected_train[:calib_count]
        dev_final = by_partition["dev"][:dev_take]
        test_final = by_partition["test"][:test_take]

        split_map = [
            ("train", train_final),
            ("calibration", calib_final),
            ("dev", dev_final),
            ("test", test_final),
        ]

        instr = "Which user intent best describes this utterance?" if lang == "en-US" else "该用户指令表达了什么意图？"

        for split_name, rows in split_map:
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


def process_sms(raw_dir: Path, max_rows=700, seed=42) -> list:
    """Extract SMS spam detection rows."""
    sms_zip = raw_dir / "enrichment" / "sms.zip"
    if not sms_zip.exists():
        return []

    records = []
    rng = random.Random(seed)
    with zipfile.ZipFile(sms_zip) as z:
        lines = z.read("SMSSpamCollection").decode("utf-8", errors="ignore").splitlines()

    rows = []
    for i, line in enumerate(lines):
        parts = line.split("\t", 1)
        if len(parts) == 2:
            rows.append((i, parts[0].strip().lower(), clean_str(parts[1])))

    rng.shuffle(rows)
    rows = rows[:max_rows]

    for i, label, text in rows:
        bucket = int(digest(f"sms_{i}_{text}")[:8], 16) % 100
        if bucket < 15:
            split = "test"
        elif bucket < 25:
            split = "dev"
        elif bucket < 35:
            split = "calibration"
        else:
            split = "train"

        records.append({
            "id": f"sms_{i}",
            "state_id": f"sms_state_{i}",
            "family_id": "sms_spam",
            "split": split,
            "state": text,
            "questions": {
                "is_spam": {
                    "type": "boolean",
                    "instructions": "Is this message unsolicited promotional spam or phishing?",
                }
            },
            "gold": {"is_spam": bool(label == "spam")},
        })

    print(f"Loaded {len(records)} SMS Spam records", file=sys.stderr)
    return records


def process_sharc(raw_dir: Path, max_rows=600, seed=42) -> list:
    """Extract SHARC conversational rule reasoning rows."""
    sharc_zip = raw_dir / "enrichment" / "sharc.zip"
    if not sharc_zip.exists():
        return []

    records = []
    rng = random.Random(seed)
    criteria = {
        "yes": "The rules clearly support answering yes.",
        "no": "The rules clearly support answering no.",
        "irrelevant": "The rules do not address this question.",
        "ask_follow_up": "Request additional information before deciding.",
    }

    with zipfile.ZipFile(sharc_zip) as z:
        for split in ["train", "dev"]:
            try:
                data = json.loads(z.read(f"sharc1-official/json/sharc_{split}.json"))
            except KeyError:
                continue

            for row in data:
                ans = str(row.get("answer", "")).strip().casefold()
                if ans not in ["yes", "no", "irrelevant"]:
                    label = "ask_follow_up"
                else:
                    label = ans

                snippet = clean_str(row.get("snippet", ""))
                scenario = clean_str(row.get("scenario", ""))
                question = clean_str(row.get("question", ""))
                state_text = f"Rule Snippet:\n{snippet}\nScenario:\n{scenario}\nQuestion:\n{question}"

                uid = row.get("utterance_id", digest(state_text)[:12])
                bucket = int(digest(f"sharc_{uid}")[:8], 16) % 100
                if split == "dev":
                    out_split = "test" if bucket < 50 else "dev"
                else:
                    out_split = "calibration" if bucket < 10 else "train"

                records.append({
                    "id": f"sharc_{uid}",
                    "state_id": f"sharc_state_{uid}",
                    "family_id": "sharc",
                    "split": out_split,
                    "state": state_text,
                    "questions": {
                        "action": {
                            "type": "choice",
                            "instructions": "Given these rules and scenario, what should the assistant do?",
                            "criteria": criteria,
                        }
                    },
                    "gold": {"action": label},
                })

    rng.shuffle(records)
    records = records[:max_rows]
    print(f"Loaded {len(records)} SHARC rule records", file=sys.stderr)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-v2-dir", default="/Users/chenyc/work/kev/evals/decision-v2", help="Path to decision-v2 dir")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw datasets dir")
    parser.add_argument("--output-dir", default="data/rlcd_dataset", help="Path to output dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    # 1. Load decision-v2
    dec2_dir = Path(args.decision_v2_dir)
    if dec2_dir.exists():
        print(f"Loading decision-v2 from {dec2_dir}...", file=sys.stderr)
        split_files = [
            ("train.jsonl", "train"),
            ("development.jsonl", "dev"),
            ("calibration.jsonl", "calibration"),
            ("test.jsonl", "test"),
        ]
        for fname, target_split in split_files:
            fpath = dec2_dir / fname
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            row = json.loads(line)
                            all_records.append(process_decision_v2_row(row, default_split=target_split))
        print(f"Loaded {len(all_records)} records from decision-v2", file=sys.stderr)

    # 2. Load MASSIVE
    raw_dir = Path(args.raw_dir)
    massive_records = process_massive(raw_dir, target_per_lang=1200, seed=args.seed)
    all_records.extend(massive_records)

    # 3. Load SMS
    sms_records = process_sms(raw_dir, max_rows=700, seed=args.seed)
    all_records.extend(sms_records)

    # 4. Load SHARC
    sharc_records = process_sharc(raw_dir, max_rows=600, seed=args.seed)
    all_records.extend(sharc_records)

    # Split counts and write
    splits = {"train": [], "dev": [], "calibration": [], "test": []}
    for r in all_records:
        s = r["split"]
        if s in splits:
            splits[s].append(r)
        else:
            splits["train"].append(r)

    print("\n" + "=" * 60, file=sys.stderr)
    print("Dataset Summary by Split:", file=sys.stderr)
    for s, l in splits.items():
        print(f"  {s:<12}: {len(l)} records", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Write output JSONL
    for s, rows in splits.items():
        out_file = out_dir / f"{s}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} records to {out_file}", file=sys.stderr)

    # Also generate manifest
    manifest = {
        "dataset": "NanoJev-RLCD-Combined",
        "counts": {s: len(rows) for s, rows in splits.items()},
        "total": sum(len(rows) for rows in splits.values()),
        "families": dict(Counter(r["family_id"] for r in all_records)),
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest written to {out_dir / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
