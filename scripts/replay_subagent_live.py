#!/usr/bin/env python3
"""Replay all subagent routing requests from router logs against NanoJev-4B and compare with Official Jev."""
import glob
import json
from pathlib import Path
import statistics
import sys
import time

ROUTER_PATH = Path("/Users/chenyc/Documents/study/jev-cliproxy-router")
if str(ROUTER_PATH) not in sys.path:
    sys.path.insert(0, str(ROUTER_PATH))

from http_transport import post_systemone_json

TARGET_URL = "http://192.168.123.88:8770/v1/systemone"

def load_subagent_requests():
    files = sorted(glob.glob("/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-*.jsonl"))
    sub_requests = []
    for fp in files:
        fname = Path(fp).name
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    req = d.get("request", {})
                    qs = req.get("questions", {})
                    if "context_mode" in qs or d.get("profile") == "subagent_route" or d.get("operation") == "subagent_route":
                        sub_requests.append((fname, d))
                except Exception:
                    continue
    return sub_requests

def main():
    items = load_subagent_requests()
    print("=" * 85)
    print(f"📊 Subagent 上下文选择决策回放与对比 (共 {len(items)} 条生产真实调用)")
    print(f"目标端点: {TARGET_URL} (h2c-only Native Multiplexing)")
    print("=" * 85)

    headers = {
        "Content-Type": "application/json",
        "X-Decision-Head": "subagent",
    }

    records = []

    for idx, (fname, item) in enumerate(items, 1):
        req = item["request"]
        off_dur = item.get("duration_ms", 0.0)
        off_resp = item.get("response") or {}
        off_ans = off_resp.get("answers", {}).get("context_mode", {}) if isinstance(off_resp, dict) else {}
        off_choice = off_ans.get("choice")
        off_conf = off_ans.get("confidence", 0.0)

        # Attach session id hint if task origin exists
        st = req.get("state", {})
        task = ""
        if isinstance(st, dict):
            task = st.get("task_origin") or st.get("user_task") or st.get("subagent_task") or ""
        elif isinstance(st, str):
            task = st[:200]
        
        import hashlib
        sid = hashlib.md5(str(task).encode("utf-8")).hexdigest()[:16] if task else item.get("request_id")
        cur_headers = dict(headers)
        if sid:
            cur_headers["X-Session-ID"] = str(sid)

        t0 = time.perf_counter()
        resp = post_systemone_json(TARGET_URL, req, cur_headers, timeout=15.0)
        wall_dt = (time.perf_counter() - t0) * 1000

        b4_ans = resp.payload.get("answers", {}).get("context_mode", {})
        b4_choice = b4_ans.get("choice")
        b4_conf = b4_ans.get("confidence", 0.0)
        is_match = (str(b4_choice).strip().lower() == str(off_choice).strip().lower())

        records.append({
            "id": idx,
            "file": fname,
            "off_dur": off_dur,
            "b4_dur": wall_dt,
            "off_choice": off_choice,
            "b4_choice": b4_choice,
            "off_conf": off_conf,
            "b4_conf": b4_conf,
            "match": is_match,
            "status": resp.status,
        })

        speedup = off_dur / wall_dt if wall_dt > 0 else 1.0
        spd_tag = f"{speedup:.2f}x" if speedup >= 1.0 else f"0.{int(speedup*100):02d}x"
        match_tag = "✅ 一致" if is_match else f"❌ 差异 ({off_choice} vs {b4_choice})"

        print(f"[{idx:>2}/{len(items)}] {fname[-15:]} | 官方: {off_dur:>6.1f}ms [{off_choice}] | 4B: {wall_dt:>6.1f}ms [{b4_choice}] | 相对: {spd_tag:<6} | {match_tag}")

    # Summary Statistics
    off_durs = [r["off_dur"] for r in records if r["status"] == 200]
    b4_durs = [r["b4_dur"] for r in records if r["status"] == 200]

    def q(arr, pct):
        s = sorted(arr)
        return s[int(len(s) * pct)]

    matches = sum(1 for r in records if r["match"])
    total = len(records)

    print("\n" + "=" * 85)
    print("📊 耗时与准确度综合统计 (Subagent Context Mode Decisions)")
    print("=" * 85)
    print(f"● 决策总数: {total} 笔")
    print(f"● 与官方决策一致率: {matches} / {total} -> {matches/total*100:.1f}% 🎯")
    print("-" * 85)
    print(f"{'指标':<18} | {'官方 Jev (云端 API)':<22} | {'NanoJev-4B (88 本地)':<22} | {'对比评价':<12}")
    print("-" * 85)

    p25_off, p25_4b = q(off_durs, 0.25), q(b4_durs, 0.25)
    p50_off, p50_4b = q(off_durs, 0.50), q(b4_durs, 0.50)
    mean_off, mean_4b = statistics.mean(off_durs), statistics.mean(b4_durs)
    p75_off, p75_4b = q(off_durs, 0.75), q(b4_durs, 0.75)
    p90_off, p90_4b = q(off_durs, 0.90), q(b4_durs, 0.90)
    p95_off, p95_4b = q(off_durs, 0.95), q(b4_durs, 0.95)

    print(f"{'P25 耗时':<18} | {p25_off:>18.1f} ms | {p25_4b:>18.1f} ms | {f'{p25_off/p25_4b:.2f}x':<12}")
    print(f"{'P50 中位耗时':<18} | {p50_off:>18.1f} ms | {p50_4b:>18.1f} ms | {f'{p50_off/p50_4b:.2f}x':<12} ⚡")
    print(f"{'平均耗时 (Mean)':<18} | {mean_off:>18.1f} ms | {mean_4b:>18.1f} ms | {f'{mean_off/mean_4b:.2f}x':<12}")
    print(f"{'P75 耗时':<18} | {p75_off:>18.1f} ms | {p75_4b:>18.1f} ms | {f'{p75_off/p75_4b:.2f}x':<12}")
    print(f"{'P90 耗时':<18} | {p90_off:>18.1f} ms | {p90_4b:>18.1f} ms | {f'{p90_off/p90_4b:.2f}x':<12}")
    print(f"{'P95 耗时':<18} | {p95_off:>18.1f} ms | {p95_4b:>18.1f} ms | {f'{p95_off/p95_4b:.2f}x':<12}")
    print(f"{'最大耗时 (Max)':<18} | {max(off_durs):>18.1f} ms | {max(b4_durs):>18.1f} ms | {'长尾稳定':<12}")
    print("=" * 85)

if __name__ == "__main__":
    main()
