#!/usr/bin/env python3
"""Simulate and compare NanoJev-4B (port 8770) vs Official Jev using real production logs."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

# Ensure jev-cliproxy-router is importable for native h2c transport
ROUTER_PATH = Path("/Users/chenyc/Documents/study/jev-cliproxy-router")
if str(ROUTER_PATH) not in sys.path:
    sys.path.insert(0, str(ROUTER_PATH))

from http_transport import post_systemone_json

TARGET_URL = "http://192.168.123.88:8770/v1/systemone"


def load_logs(file_paths, limit=None):
    items = []
    for fp in file_paths:
        path = Path(fp)
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("request") and d.get("duration_ms"):
                        items.append(d)
                except Exception:
                    continue
    if limit:
        items = items[:limit]
    return items


def compare_answers(ans_4b, ans_off):
    matches = 0
    total = 0
    details = {}
    for qid, q_off in ans_off.items():
        if qid not in ans_4b:
            continue
        q_4b = ans_4b[qid]
        total += 1
        val_off = q_off.get("choice", q_off.get("noul", q_off.get("score")))
        val_4b = q_4b.get("choice", q_4b.get("noul", q_4b.get("score")))

        is_match = False
        if isinstance(val_off, (int, float)) and isinstance(val_4b, (int, float)):
            # For probabilities (noul): binary agreement on threshold 0.5
            is_match = (val_off >= 0.5) == (val_4b >= 0.5)
        else:
            is_match = str(val_off).strip().lower() == str(val_4b).strip().lower()

        if is_match:
            matches += 1
        details[qid] = {"off": val_off, "4b": val_4b, "match": is_match}
    return matches, total, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60, help="Max requests to replay (default: 60)")
    parser.add_argument("--url", default=TARGET_URL)
    args = parser.parse_args()

    files = [
        "/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-09-22.jsonl",
        "/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-09-23.jsonl",
    ]

    print(f"Loading production requests from past 2 days (limit={args.limit})...")
    items = load_logs(files, limit=args.limit)
    print(f"Loaded {len(items)} real production requests. Starting sequential replay against 4B...")
    print(f"Target: {args.url} [h2c-only Native Multiplexed Transport]")
    print("-" * 80)

    results = []
    headers = {"Content-Type": "application/json", "X-Decision-Head": "router"}

    for idx, item in enumerate(items, 1):
        req = item["request"]
        off_dur = item.get("duration_ms", 0.0)
        off_resp = item.get("response") or {}
        off_ans = off_resp.get("answers", {})
        step_type = item.get("step_type", "unknown")
        q_count = len(req.get("questions", {}))

        t0 = time.perf_counter()
        resp = post_systemone_json(args.url, req, headers, timeout=20.0)
        wall_dt = (time.perf_counter() - t0) * 1000

        resp_payload = resp.payload
        ans_4b = resp_payload.get("answers", {})
        m, tot, details = compare_answers(ans_4b, off_ans)

        # Inspect headers for cache hit
        # Note: http_transport may not expose custom headers directly in JSONResponse,
        # but we can deduce hit from latency or compare
        results.append({
            "id": idx,
            "step": step_type,
            "q_count": q_count,
            "off_dur": off_dur,
            "4b_dur": wall_dt,
            "status": resp.status,
            "matches": m,
            "total_qs": tot,
        })

        speedup = off_dur / wall_dt if wall_dt > 0 else 1.0
        spd_tag = f"{speedup:.2f}x" if speedup >= 1.0 else f"0.{int(speedup*100):02d}x"
        acc_tag = f"{m}/{tot}" if tot > 0 else "0/0"
        print(f"[{idx:>2}/{len(items)}] {step_type:<10} ({q_count}题) | 官方: {off_dur:>7.1f}ms | 4B: {wall_dt:>7.1f}ms | 相对: {spd_tag:<6} | 一致: {acc_tag}")

    # Summary Statistics
    off_durs = [r["off_dur"] for r in results if r["status"] == 200]
    b4_durs = [r["4b_dur"] for r in results if r["status"] == 200]

    def q(arr, pct):
        s = sorted(arr)
        return s[int(len(s) * pct)]

    print("\n" + "=" * 80)
    print("📊 NanoJev-4B (88 本地服务) vs 官方 Jev (云端 API) 生产耗时对比报告")
    print("=" * 80)
    print(f"{'指标':<18} | {'官方 Jev (云端 API)':<22} | {'NanoJev-4B (88 本地)':<22} | {'对比评价':<12}")
    print("-" * 80)

    p25_off, p25_4b = q(off_durs, 0.25), q(b4_durs, 0.25)
    p50_off, p50_4b = q(off_durs, 0.50), q(b4_durs, 0.50)
    mean_off, mean_4b = statistics.mean(off_durs), statistics.mean(b4_durs)
    p75_off, p75_4b = q(off_durs, 0.75), q(b4_durs, 0.75)
    p90_off, p90_4b = q(off_durs, 0.90), q(b4_durs, 0.90)
    p95_off, p95_4b = q(off_durs, 0.95), q(b4_durs, 0.95)
    max_off, max_4b = max(off_durs), max(b4_durs)

    print(f"{'P25 耗时':<18} | {p25_off:>18.1f} ms | {p25_4b:>18.1f} ms | {f'{p25_off/p25_4b:.2f}x':<12}")
    print(f"{'P50 中位耗时':<18} | {p50_off:>18.1f} ms | {p50_4b:>18.1f} ms | {f'{p50_off/p50_4b:.2f}x':<12}")
    print(f"{'平均耗时':<18} | {mean_off:>18.1f} ms | {mean_4b:>18.1f} ms | {f'{mean_off/mean_4b:.2f}x':<12}")
    print(f"{'P75 耗时':<18} | {p75_off:>18.1f} ms | {p75_4b:>18.1f} ms | {f'{p75_off/p75_4b:.2f}x':<12}")
    print(f"{'P90 耗时':<18} | {p90_off:>18.1f} ms | {p90_4b:>18.1f} ms | {f'{p90_off/p90_4b:.2f}x':<12}")
    print(f"{'P95 耗时':<18} | {p95_off:>18.1f} ms | {p95_4b:>18.1f} ms | {f'{p95_off/p95_4b:.2f}x':<12}")
    print(f"{'最大耗时 Max':<18} | {max_off:>18.1f} ms | {max_4b:>18.1f} ms | {'长尾对比':<12}")
    print("=" * 80)

    # Sub-category Breakdown
    for step in ("tool_step", "user_turn"):
        sub = [r for r in results if r["step"] == step and r["status"] == 200]
        if not sub:
            continue
        sub_off = [r["off_dur"] for r in sub]
        sub_4b = [r["4b_dur"] for r in sub]
        print(f"\n场景细分 [{step}] (N={len(sub)}):")
        print(f"  P50 中位数: 官方={q(sub_off, 0.5):.1f}ms | 4B={q(sub_4b, 0.5):.1f}ms (比值: {q(sub_off, 0.5)/q(sub_4b, 0.5):.2f}x)")
        print(f"  P90 长尾:   官方={q(sub_off, 0.9):.1f}ms | 4B={q(sub_4b, 0.9):.1f}ms")
        print(f"  平均耗时:   官方={statistics.mean(sub_off):.1f}ms | 4B={statistics.mean(sub_4b):.1f}ms")

    # Decision Agreement Rate
    total_q = sum(r["total_qs"] for r in results)
    total_m = sum(r["matches"] for r in results)
    print("\n" + "-" * 80)
    print(f"🎯 决策对齐度 (Decision Agreement with Official Jev):")
    print(f"  总决策题目数: {total_q}")
    print(f"  与官方一致数: {total_m}")
    print(f"  总体一致率:   {total_m/total_q*100:.1f}%")
    print("-" * 80)


if __name__ == "__main__":
    main()
