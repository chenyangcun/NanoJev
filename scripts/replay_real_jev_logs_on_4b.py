#!/usr/bin/env python3
"""Replay 2-Day Real Jev Production Logs against NanoJev-4B (port 8770) and compare accuracy, latency, and divergences."""

import http.client
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

LOG_FILES = [
    "/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-09-22.jsonl",
    "/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-09-23.jsonl"
]

ENDPOINT_HOST = "192.168.123.88"
ENDPOINT_PORT = 8770
ENDPOINT_PATH = "/v1/systemone"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5, help="Sample every N-th request across 48h")
    parser.add_argument("--limit", type=int, default=60, help="Max requests to replay")
    parser.add_argument("--cache-file", default="/tmp/nanojev_4b_real_log_cache.jsonl")
    args = parser.parse_args()

    print("=" * 80)
    print("REPLAYING REAL 2-DAY PRODUCTION JEV AUDIT LOGS ON NANOJEV-4B (port 8770)")
    print("=" * 80)

    # 1. Load log entries
    raw_entries = []
    for fpath in LOG_FILES:
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found.")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("http_status") == 200 and d.get("request") and d.get("response"):
                        raw_entries.append(d)
                except Exception:
                    pass

    # Sample uniformly across the whole 2 days
    all_entries = raw_entries[::args.stride][:args.limit]
    print(f"Total available entries: {len(raw_entries)} across 2 days (2026-09-22 & 2026-09-23).")
    print(f"Uniformly sampled {len(all_entries)} requests (stride={args.stride}, limit={args.limit}).")
    total_q_count = sum(len(e["request"].get("questions", {})) for e in all_entries)
    print(f"Total individual decisions to evaluate: {total_q_count}\n")

    # Connect with persistent HTTP/1.1 connection
    conn = http.client.HTTPConnection(ENDPOINT_HOST, ENDPOINT_PORT, timeout=30)
    headers = {"Content-Type": "application/json"}

    cache_file = "/tmp/nanojev_4b_real_log_cache.jsonl"
    cached = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cached[d["idx"]] = d
    print(f"Found {len(cached)} previously cached replay records.")

    cache_fh = open(cache_file, "a", encoding="utf-8")

    replayed = []
    official_latencies = []
    nanojev_latencies = []

    t_start = time.time()
    for idx, entry in enumerate(all_entries):
        req_body = entry["request"]
        off_resp = entry["response"]
        off_dur = entry.get("duration_ms", 0.0)

        if idx in cached:
            item = cached[idx]
            replayed.append(item)
            official_latencies.append(item["official_duration_ms"])
            nanojev_latencies.append(item["nanojev_duration_ms"])
            continue

        body_bytes = json.dumps(req_body, ensure_ascii=False).encode("utf-8")

        t0 = time.perf_counter()
        try:
            conn.request("POST", ENDPOINT_PATH, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            res_data = json.loads(resp.read().decode("utf-8"))
            lat_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            conn.close()
            conn = http.client.HTTPConnection(ENDPOINT_HOST, ENDPOINT_PORT, timeout=30)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            res_data = {"error": str(e)}

        official_latencies.append(off_dur)
        nanojev_latencies.append(lat_ms)

        item = {
            "idx": idx,
            "at": entry.get("at"),
            "request": req_body,
            "official_response": off_resp,
            "official_duration_ms": off_dur,
            "nanojev_response": res_data,
            "nanojev_duration_ms": lat_ms
        }
        replayed.append(item)
        cache_fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        cache_fh.flush()

        if (idx + 1) % 25 == 0 or idx + 1 == len(all_entries):
            print(f"  Replayed {idx + 1:3d} / {len(all_entries)} requests in {time.time()-t_start:.1f}s (latest lat: {lat_ms:.1f}ms)", flush=True)

    cache_fh.close()
    print(f"\nAll {len(replayed)} requests replayed in {time.time()-t_start:.1f}s!\n")

    # 2. Latency Analysis
    print("=" * 80)
    print("1. RESPONSE TIME & LATENCY COMPARISON:")
    print("=" * 80)

    def p_tile(vals, q):
        s = sorted(vals)
        return s[int(len(s) * q)]

    off_p50 = p_tile(official_latencies, 0.5)
    off_p90 = p_tile(official_latencies, 0.9)
    off_p95 = p_tile(official_latencies, 0.95)
    off_mean = sum(official_latencies) / len(official_latencies)

    nj_p50 = p_tile(nanojev_latencies, 0.5)
    nj_p90 = p_tile(nanojev_latencies, 0.9)
    nj_p95 = p_tile(nanojev_latencies, 0.95)
    nj_mean = sum(nanojev_latencies) / len(nanojev_latencies)

    print(f"{'Metric':<16} {'Official Jev Cloud API':<26} {'NanoJev-4B Local Server':<26} {'Speedup'}")
    print("-" * 80)
    print(f"{'Median (p50)':<16} {off_p50:8.1f} ms             {nj_p50:8.1f} ms             {off_p50 / nj_p50:.2f}x")
    print(f"{'90th % (p90)':<16} {off_p90:8.1f} ms             {nj_p90:8.1f} ms             {off_p90 / nj_p90:.2f}x")
    print(f"{'95th % (p95)':<16} {off_p95:8.1f} ms             {nj_p95:8.1f} ms             {off_p95 / nj_p95:.2f}x")
    print(f"{'Mean':<16} {off_mean:8.1f} ms             {nj_mean:8.1f} ms             {off_mean / nj_mean:.2f}x")

    # 3. Accuracy & Agreement Analysis
    print("\n" + "=" * 80)
    print("2. ACCURACY & AGREEMENT WITH OFFICIAL JEV ANSWERS:")
    print("=" * 80)

    q_stats = defaultdict(lambda: {"total": 0, "exact_match": 0, "close_match": 0, "divergences": []})
    risk_stats = {"high_risk_official": 0, "high_risk_nanojev": 0, "both_high_risk": 0, "diffs": []}

    for item in replayed:
        off_ans = item["official_response"].get("answers", {})
        nj_ans = item["nanojev_response"].get("answers", {})
        state = item["request"].get("state", {})
        user_task = state.get("user_task", "") if isinstance(state, dict) else str(state)

        for qid, off_q in off_ans.items():
            if qid not in nj_ans:
                continue
            nj_q = nj_ans[qid]
            qtype = off_q.get("type")

            q_stats[qid]["total"] += 1

            if qtype == "choice":
                off_choice = off_q.get("choice")
                nj_choice = nj_q.get("choice")
                is_exact = (off_choice == nj_choice)
                if is_exact:
                    q_stats[qid]["exact_match"] += 1
                    q_stats[qid]["close_match"] += 1
                else:
                    # Check close match (e.g. standard vs complex, medium vs high)
                    is_close = False
                    if qid == "complexity" and {off_choice, nj_choice} <= {"standard", "complex"}:
                        is_close = True
                    elif qid == "effort" and {off_choice, nj_choice} <= {"medium", "high"}:
                        is_close = True
                    elif qid in ("luna_effort", "sol_effort") and {off_choice, nj_choice} <= {"medium", "high"}:
                        is_close = True

                    if is_close:
                        q_stats[qid]["close_match"] += 1

                    q_stats[qid]["divergences"].append({
                        "idx": item["idx"],
                        "task_snippet": user_task[:150].replace("\n", " "),
                        "off_choice": off_choice,
                        "off_prob": off_q.get("probabilities", {}).get(off_choice),
                        "nj_choice": nj_choice,
                        "nj_prob": nj_q.get("probabilities", {}).get(nj_choice),
                        "is_close": is_close
                    })

            elif qtype == "noul":
                off_p = off_q.get("noul", 0.0)
                nj_p = nj_q.get("noul", 0.0)
                off_bool = (off_p >= 0.5)
                nj_bool = (nj_p >= 0.5)
                is_exact = (off_bool == nj_bool)
                if is_exact:
                    q_stats[qid]["exact_match"] += 1
                    q_stats[qid]["close_match"] += 1
                else:
                    q_stats[qid]["divergences"].append({
                        "idx": item["idx"],
                        "task_snippet": user_task[:150].replace("\n", " "),
                        "off_p": off_p,
                        "nj_p": nj_p
                    })

                if qid == "high_risk":
                    if off_bool: risk_stats["high_risk_official"] += 1
                    if nj_bool: risk_stats["high_risk_nanojev"] += 1
                    if off_bool and nj_bool: risk_stats["both_high_risk"] += 1

    print(f"{'Question':<24} {'Total':<8} {'Exact Match':<14} {'Exact %':<10} {'Close/Adjacent %'}")
    print("-" * 80)
    for qid in sorted(q_stats.keys()):
        s = q_stats[qid]
        tot = s["total"]
        ex = s["exact_match"]
        cl = s["close_match"]
        ex_pct = ex / tot * 100 if tot else 0
        cl_pct = cl / tot * 100 if tot else 0
        print(f"{qid:<24} {tot:<8} {ex:<14} {ex_pct:6.1f}%     {cl_pct:6.1f}%")

    # 4. Detailed Divergence Deep-Dive
    print("\n" + "=" * 80)
    print("3. DIVERGENCE ANALYSIS & EXAMPLES:")
    print("=" * 80)

    for qid in ["complexity", "effort", "high_risk", "independent"]:
        diffs = q_stats[qid]["divergences"]
        print(f"\n--- [{qid.upper()}] Disagreements ({len(diffs)} / {q_stats[qid]['total']}) ---")
        if not diffs:
            print("  None! 100% agreement.")
            continue

        # Count pattern breakdown
        if qid in ("complexity", "effort"):
            patterns = Counter(f"{d['off_choice']} -> {d['nj_choice']}" for d in diffs)
            print(f"  Distribution of Disagreements: {dict(patterns)}")

        for d in diffs[:4]:
            print(f"  [Item #{d['idx']}]")
            print(f"    Task    : {d['task_snippet']}...")
            if "off_choice" in d:
                print(f"    Official: {d['off_choice']} (P={d['off_prob']})")
                print(f"    NanoJev : {d['nj_choice']} (P={d['nj_prob']})")
                print(f"    Adjacent: {'Yes (borderline boundary)' if d['is_close'] else 'No (true disagreement)'}")
            else:
                print(f"    Official: P(yes)={d['off_p']:.3f}")
                print(f"    NanoJev : P(yes)={d['nj_p']:.3f}")

    # Save full evaluation output
    out_file = "/tmp/nanojev_4b_real_log_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "latency": {
                "official": {"p50": off_p50, "p90": off_p90, "p95": off_p95, "mean": off_mean},
                "nanojev_4b": {"p50": nj_p50, "p90": nj_p90, "p95": nj_p95, "mean": nj_mean}
            },
            "summary": {qid: {"total": s["total"], "exact": s["exact_match"], "close": s["close_match"]} for qid, s in q_stats.items()},
            "replayed": replayed
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull report and individual exchange comparison saved to: {out_file}")

if __name__ == "__main__":
    main()
