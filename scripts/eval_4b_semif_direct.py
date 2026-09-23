import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

def build_semif_row(task):
    qtype = task.question["type"]
    crit = task.question.get("criteria")
    if qtype == "noul":
        options = [{"id": k, "description": (crit or {}).get(k, f"The proposition is {k}.")}
                   for k in ("true", "false")]
    elif qtype == "choice":
        options = [{"id": k, "description": v or k} for k, v in crit.items()]
    else:
        options = [{"id": str(i), "description": lvl} for i, lvl in enumerate(crit)]
    for o in options:
        o["description"] = o["id"] + ": " + o["description"]
    return {
        "id": task.id,
        "state": task.state,
        "question": task.question["instructions"],
        "options": options,
        "type": qtype
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints/qwen35_4b_8bit")
    parser.add_argument("--cache-file", default=None)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    default_cache = os.path.join(os.path.dirname(__file__), "..", "results", f"eval_cache_{Path(checkpoint_dir).name}.jsonl")
    cache_file = args.cache_file or default_cache

    jb_path = os.path.expanduser("~/work/NanoJev/data/jevbench_v130")
    if os.path.exists(jb_path):
        sys.path.insert(0, jb_path)
    from jevbench.tasks import load_jsonl
    from jevbench.metrics import ece_top_label, brier_score
    from jevbench import composite_v13 as score

    print("=" * 75)
    print(f"EVALUATING {checkpoint_dir} ON JEVBENCH v1.3.0")
    print("=" * 75)

    print(f"Loading model from {checkpoint_dir} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(checkpoint_dir, tokenizer_config={"trust_remote_code": False})
    print(f"Model loaded in {time.time()-t0:.2f}s!\n", flush=True)

    easy = load_jsonl(os.path.join(jb_path, "datasets/public/easy.jsonl"))
    orig = load_jsonl(os.path.join(jb_path, "datasets/public/original.jsonl"))
    hard = load_jsonl(os.path.join(jb_path, "datasets/public/hard.jsonl"))
    tasks = easy + orig + hard
    print(f"Loaded {len(tasks)} tasks: {len(easy)} Easy, {len(orig)} Standard, {len(hard)} Hard.\n")

    # We evaluate with a calibrated temperature T (SemIf uses T=1.8 ~ 2.0 on hard/general)
    temperatures = [1.0, 1.8]

    # Pre-encode all tasks
    print("Pre-encoding prompts and verifying single-token answer slots...", flush=True)
    encoded_items = []
    t_enc = time.time()
    for task in tasks:
        row = build_semif_row(task)
        payload = {
            "evidence": row["state"],
            "criterion": row["question"],
            "options": [
                {"letter": LETTERS[i], "description": opt["description"]}
                for i, opt in enumerate(row["options"])
            ]
        }
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Cap to 4096 tokens if needed
        if len(input_ids) > 4096:
            input_ids = input_ids[:4096]
        slots = [tokenizer.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(len(row["options"]))]
        encoded_items.append((task, row, input_ids, slots))
    print(f"Pre-encoded {len(encoded_items)} tasks in {time.time()-t_enc:.2f}s.\n")

    # Run forward passes once to get logits
    print("Running forward passes on Apple Metal GPU...", flush=True)
    cached_logits = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            for line in f:
                d = json.loads(line)
                cached_logits[d["id"]] = d

    print(f"Found {len(cached_logits)} cached results.", flush=True)

    raw_logits_records = []
    latencies = []
    t_start = time.time()

    cache_fh = open(cache_file, "a", encoding="utf-8")

    for idx, (task, row, input_ids, slots) in enumerate(encoded_items):
        if task.id in cached_logits:
            item = cached_logits[task.id]
            selected = item["selected"]
            lat = item["lat"]
            latencies.append(lat)
            raw_logits_records.append((task, row, selected, len(input_ids), lat))
            continue

        t_fwd = time.perf_counter()
        x = mx.array([input_ids], dtype=mx.int32)
        logits = model(x)[:, -1, :]
        mx.eval(logits)
        lat = time.perf_counter() - t_fwd
        latencies.append(lat)

        selected = logits[0, mx.array(slots)].tolist()
        record = {
            "id": task.id,
            "selected": selected,
            "lat": lat,
            "tokens": len(input_ids)
        }
        cache_fh.write(json.dumps(record) + "\n")
        cache_fh.flush()

        raw_logits_records.append((task, row, selected, len(input_ids), lat))

        if (idx + 1) % 10 == 0 or idx + 1 == len(encoded_items) or lat > 1.0:
            print(f"  Processed {idx + 1:3d} / {len(encoded_items)}: {task.id:<30} ({len(input_ids):4d} tok, {lat*1000:6.1f}ms) in {time.time()-t_start:.1f}s", flush=True)

    cache_fh.close()
    print(f"\nAll {len(encoded_items)} forward passes complete in {time.time()-t_start:.1f}s!")

    # Latency percentiles
    lat_sorted = sorted(latencies)
    p50_s = lat_sorted[len(lat_sorted) // 2]
    p95_s = lat_sorted[int(len(lat_sorted) * 0.95)]
    mean_s = sum(latencies) / len(latencies)
    print("\n" + "=" * 75)
    print("INFERENCE SPEED & LATENCY (Apple M1 Mac Studio - Qwen3.5-4B 8-bit):")
    print(f"  p50 (Median)   : {p50_s * 1000:6.1f} ms")
    print(f"  p95 (95th %)   : {p95_s * 1000:6.1f} ms")
    print(f"  Mean Latency   : {mean_s * 1000:6.1f} ms")
    print("=" * 75)

    for temp in temperatures:
        print(f"\nEvaluating JevBench v1.3.0 Metrics at Temperature T = {temp:.2f}...")
        easy_corr = 0
        std_corr = 0
        hard_corr = 0
        pairs = []
        briers = []

        for task, row, selected, tok_len, lat in raw_logits_records:
            scaled = [v / temp for v in selected]
            probs_list = mx.softmax(mx.array(scaled)).tolist()
            probs = dict(zip([opt["id"] for opt in row["options"]], probs_list))

            if row["type"] == "noul":
                res_probs = {"yes": probs["true"], "no": probs["false"]}
                pred = "yes" if probs["true"] >= 0.5 else "no"
            else:
                res_probs = probs
                pred = max(probs.keys(), key=lambda k: probs[k])

            gold = str(task.expected)
            is_corr = (str(pred) == gold)
            if is_corr:
                if task.id.startswith("easy-"):
                    easy_corr += 1
                elif task.id.startswith("original-"):
                    std_corr += 1
                elif task.id.startswith("hard-"):
                    hard_corr += 1

            conf = max(res_probs.values())
            pairs.append((conf, is_corr))
            bs = brier_score(res_probs, gold, task.labels)
            briers.append(bs)

        tot_corr = easy_corr + std_corr + hard_corr
        acc_easy = easy_corr / len(easy)
        acc_std = std_corr / len(orig)
        acc_hard = hard_corr / len(hard)
        acc_tot = tot_corr / len(tasks)

        print(f"  Accuracy:")
        print(f"    Easy      : {easy_corr:3d} / {len(easy):3d} = {acc_easy * 100:.2f}%")
        print(f"    Standard  : {std_corr:3d} / {len(orig):3d} = {acc_std * 100:.2f}%")
        print(f"    Hard      : {hard_corr:3d} / {len(hard):3d} = {acc_hard * 100:.2f}%")
        print(f"    Total     : {tot_corr:3d} / {len(tasks):3d} = {acc_tot * 100:.2f}%")

        tiers = {
            "easy": acc_easy,
            "standard": acc_std,
            "hard": acc_hard,
        }
        intel = score.intelligence(tiers)

        ece_res = ece_top_label(pairs)
        ece_val = ece_res["ece"]
        calib = score.calibration(ece_val)
        mean_brier = sum(briers) / len(briers)
        mean_conf = sum(p[0] for p in pairs) / len(pairs)

        # Speed score
        spd_gpu = score.speed(p50_s, p95_s, endpoint_kind="gpu")
        spd_api = score.speed(p50_s, p95_s, endpoint_kind="api")

        # Cost: deepinfra Qwen3.5-4B reference list price is $0.03/M in, ~400 tokens -> ~$0.012 per 1000 decisions
        cost_usd_1k = 0.0120
        cst = score.cost(cost_usd_1k)

        axes_gpu = {
            "intelligence": intel,
            "calibration": calib,
            "speed": spd_gpu,
            "cost": cst,
        }
        jb_score = score.jevbench_score(axes_gpu)

        print(f"  Axes (0-100):")
        print(f"    Intelligence : {intel:.2f} (Chance-corrected: Hard={score.chance_corrected_accuracy(acc_hard, score.TIER_CHANCES['hard']):.1f}, Std={score.chance_corrected_accuracy(acc_std, score.TIER_CHANCES['standard']):.1f})")
        print(f"    Calibration  : {calib:.2f} (ECE={ece_val*100:.2f}%, Brier={mean_brier:.4f}, MeanConf={mean_conf*100:.2f}%)")
        print(f"    Speed (GPU)  : {spd_gpu:.2f} (p50={p50_s*1000:.1f}ms, p95={p95_s*1000:.1f}ms)")
        print(f"    Cost         : {cst:.2f} (${cost_usd_1k:.4f} / 1k decisions)")
        print(f"  --> JEVBENCH v1.3.0 SCORE: {jb_score:.2f}")

    print("\n" + "=" * 75)
    print("EVALUATION FINISHED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
