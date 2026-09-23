#!/usr/bin/env python3
"""Benchmark and compare Official Jev vs 0.8B (port 8769) vs 4B (port 8770) on personal-memory webpage review logs."""
import argparse
import glob
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

URL_08B = "http://192.168.123.88:8769/v1/systemone"
URL_4B = "http://192.168.123.88:8770/v1/systemone"


def load_memory_logs(limit=None):
    files = sorted(glob.glob("data/personal_memory_logs/jev-2026-*.jsonl"))
    items = []
    for fp in files:
        fname = Path(fp).name
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    req = d.get("request", {})
                    resp = d.get("response", {})
                    if req and resp and "answers" in resp:
                        items.append((fname, d))
                except Exception:
                    continue
    if limit:
        items = items[:limit]
    return items


def is_match(val_a, val_b):
    if val_a is None or val_b is None:
        return False
    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        # For probabilities/noul: threshold at 0.5
        # For scores (0, 1, 2): round to nearest integer
        if abs(val_a - round(val_a)) < 0.05 and abs(val_b - round(val_b)) < 0.05:
            return round(val_a) == round(val_b)
        return (val_a >= 0.5) == (val_b >= 0.5)
    return str(val_a).strip().lower() == str(val_b).strip().lower()


def evaluate_answers(pred_ans, off_ans):
    res = {}
    for qid, q_off in off_ans.items():
        if qid not in pred_ans:
            continue
        q_pred = pred_ans[qid]
        v_off = q_off.get("choice", q_off.get("noul", q_off.get("score")))
        v_pred = q_pred.get("choice", q_pred.get("noul", q_pred.get("score")))
        m = is_match(v_pred, v_off)
        res[qid] = {"pred": v_pred, "off": v_off, "match": m}
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40, help="Number of records to test (default: 40)")
    args = parser.parse_args()

    items = load_memory_logs(limit=args.limit)
    print("=" * 85)
    print(f"📊 Personal-Memory 网页评分与质量审查日志 3-Way 测评对比")
    print(f"数据量: {len(items)} 条真实网页抓取审查记录 (来自 192.168.123.253 /opt/personal-memory)")
    print(f"基准端点: 官方 Jev (云端) vs 0.8B (8769) vs 4B (8770)")
    print("=" * 85)

    headers = {"Content-Type": "application/json", "X-Decision-Head": "memory"}

    records = []

    # Warmup
    print("Warming up 0.8B and 4B...")
    _ = post_systemone_json(URL_08B, items[0][1]["request"], headers, timeout=10.0)
    _ = post_systemone_json(URL_4B, items[0][1]["request"], headers, timeout=10.0)
    print("Warmup complete. Starting benchmark runs...\n")

    for idx, (fname, item) in enumerate(items, 1):
        req = item["request"]
        off_dur = item.get("duration_ms", 0.0)
        off_ans = item.get("response", {}).get("answers", {})
        q_count = len(req.get("questions", {}))

        # Title snippet
        page = req.get("state", {}).get("page", {})
        title = page.get("title") or page.get("url") or "Untitled"
        title_snippet = title[:28].replace("\n", " ")

        # 1. Call 0.8B
        t0 = time.perf_counter()
        resp_08b = post_systemone_json(URL_08B, req, headers, timeout=15.0)
        dt_08b = (time.perf_counter() - t0) * 1000
        ans_08b = resp_08b.payload.get("answers", {})

        # 2. Call 4B
        t0 = time.perf_counter()
        resp_4b = post_systemone_json(URL_4B, req, headers, timeout=15.0)
        dt_4b = (time.perf_counter() - t0) * 1000
        ans_4b = resp_4b.payload.get("answers", {})

        eval_08b = evaluate_answers(ans_08b, off_ans)
        eval_4b = evaluate_answers(ans_4b, off_ans)

        match_08b = sum(1 for v in eval_08b.values() if v["match"])
        match_4b = sum(1 for v in eval_4b.values() if v["match"])
        tot_q = len(eval_4b)

        records.append({
            "id": idx,
            "title": title_snippet,
            "q_count": q_count,
            "off_dur": off_dur,
            "dt_08b": dt_08b,
            "dt_4b": dt_4b,
            "eval_08b": eval_08b,
            "eval_4b": eval_4b,
            "match_08b": match_08b,
            "match_4b": match_4b,
            "tot_q": tot_q,
            "status_08b": resp_08b.status,
            "status_4b": resp_4b.status,
        })

        acc_tag_08b = f"{match_08b}/{tot_q}"
        acc_tag_4b = f"{match_4b}/{tot_q}"
        print(
            f"[{idx:>2}/{len(items)}] {title_snippet:<30} | "
            f"官方: {off_dur:>6.1f}ms | 0.8B: {dt_08b:>6.1f}ms ({acc_tag_08b}) | "
            f"4B: {dt_4b:>6.1f}ms ({acc_tag_4b})"
        )

    # Statistics
    def q(arr, pct):
        s = sorted(arr)
        return s[int(len(s) * pct)]

    d_off = [r["off_dur"] for r in records if r["status_08b"] == 200 and r["status_4b"] == 200]
    d_08b = [r["dt_08b"] for r in records if r["status_08b"] == 200 and r["status_4b"] == 200]
    d_4b = [r["dt_4b"] for r in records if r["status_08b"] == 200 and r["status_4b"] == 200]

    print("\n" + "=" * 85)
    print("📊 网页评分场景耗时分布 (Latency Distribution)")
    print("=" * 85)
    print(f"{'指标':<18} | {'官方 Jev (云端)':<18} | {'NanoJev-0.8B (8769)':<20} | {'NanoJev-4B (8770)':<18}")
    print("-" * 85)
    print(f"{'P25 耗时':<18} | {q(d_off, 0.25):>15.1f} ms | {q(d_08b, 0.25):>17.1f} ms | {q(d_4b, 0.25):>15.1f} ms")
    print(f"{'P50 中位耗时':<18} | {q(d_off, 0.50):>15.1f} ms | {q(d_08b, 0.50):>17.1f} ms | {q(d_4b, 0.50):>15.1f} ms")
    print(f"{'平均耗时 (Mean)':<18} | {statistics.mean(d_off):>15.1f} ms | {statistics.mean(d_08b):>17.1f} ms | {statistics.mean(d_4b):>15.1f} ms")
    print(f"{'P75 耗时':<18} | {q(d_off, 0.75):>15.1f} ms | {q(d_08b, 0.75):>17.1f} ms | {q(d_4b, 0.75):>15.1f} ms")
    print(f"{'P90 耗时':<18} | {q(d_off, 0.90):>15.1f} ms | {q(d_08b, 0.90):>17.1f} ms | {q(d_4b, 0.90):>15.1f} ms")
    print(f"{'P95 耗时':<18} | {q(d_off, 0.95):>15.1f} ms | {q(d_08b, 0.95):>17.1f} ms | {q(d_4b, 0.95):>15.1f} ms")
    print(f"{'最大耗时 (Max)':<18} | {max(d_off):>15.1f} ms | {max(d_08b):>17.1f} ms | {max(d_4b):>15.1f} ms")
    print("=" * 85)

    # Accuracy
    total_q = sum(r["tot_q"] for r in records)
    total_m_08b = sum(r["match_08b"] for r in records)
    total_m_4b = sum(r["match_4b"] for r in records)

    print("\n" + "=" * 85)
    print("🎯 网页评分决策一致率对比 (Agreement with Official Jev)")
    print("=" * 85)
    print(f"总决策题目数: {total_q} 题 (来自 {len(records)} 篇网页快照评测)")
    print(f"  ● NanoJev-0.8B 一致数: {total_m_08b:>3} / {total_q} -> 总体一致率: {total_m_08b/total_q*100:.1f}%")
    print(f"  ● NanoJev-4B   一致数: {total_m_4b:>3} / {total_q} -> 总体一致率: {total_m_4b/total_q*100:.1f}% 🏆 (+{(total_m_4b - total_m_08b)/total_q*100:.1f}%)")
    print("-" * 85)

    # Per-Question Accuracy Breakdown
    by_q_08b = {}
    by_q_4b = {}
    by_q_tot = {}

    for r in records:
        for qid, res in r["eval_08b"].items():
            by_q_tot[qid] = by_q_tot.get(qid, 0) + 1
            if res["match"]:
                by_q_08b[qid] = by_q_08b.get(qid, 0) + 1
        for qid, res in r["eval_4b"].items():
            if res["match"]:
                by_q_4b[qid] = by_q_4b.get(qid, 0) + 1

    print(f"{'评分维度 (Question)':<25} | {'样本数':<8} | {'0.8B 一致率':<15} | {'4B 一致率':<15} | {'4B 提升对比':<10}")
    print("-" * 85)
    for qid in sorted(by_q_tot.keys()):
        tot = by_q_tot[qid]
        acc_08b = (by_q_08b.get(qid, 0) / tot) * 100
        acc_4b = (by_q_4b.get(qid, 0) / tot) * 100
        diff = acc_4b - acc_08b
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        flag = "🏆" if diff > 0 else ("-" if diff == 0 else "▼")
        print(f"{qid:<25} | {tot:<8} | {acc_08b:>13.1f}% | {acc_4b:>13.1f}% | {diff_str:>8} {flag}")
    print("=" * 85)


if __name__ == "__main__":
    main()
