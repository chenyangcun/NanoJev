#!/usr/bin/env python3
"""Run full audit comparison against /Users/chenyc/Downloads/jev-router-audit.jsonl.

Computes aggregate metrics:
1. Total valid cases compared
2. Complexity Choice Agreement Rate (Exact match %)
3. High Risk Gate Agreement Rate (both agree risk >= 0.5 or both < 0.5)
4. High Risk MAE (Mean Absolute Error)
5. Independent Noul MAE
6. Latency distribution (median, p95)
"""
import argparse
import json
import statistics
import time
import urllib.request


def evaluate_audit_dataset(audit_path: str, server_url: str, limit: int = 0):
    endpoint = f"{server_url.rstrip('/')}/v1/systemone"
    cases = []

    print(f"Loading real cases from {audit_path}...")
    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if "jev_request" in d and d.get("judgments"):
                    cases.append({
                        "request": d["jev_request"],
                        "jev_judgments": d["judgments"],
                        "chosen_model": d.get("chosen_model"),
                        "at": d.get("at"),
                    })
                    if limit and len(cases) >= limit:
                        break
            except Exception:
                pass

    print(f"Loaded {len(cases)} valid real production request/judgment pairs.")
    print(f"Querying live MLX server at {endpoint}...\n")

    comp_matches = 0
    risk_gate_agreements = 0
    risk_errors = []
    indep_errors = []
    latencies = []
    disagreements = []

    for i, c in enumerate(cases):
        req = c["request"]
        body = json.dumps(req).encode("utf-8")
        http_req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(http_req, timeout=10) as resp:
                nano_res = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.time() - t0) * 1000
            latencies.append(elapsed)
        except Exception as e:
            print(f"❌ Case {i+1} failed: {e}")
            continue

        nano_answers = nano_res.get("answers", {})
        jev_answers = c["jev_judgments"]

        # 1. Complexity Comparison
        j_choice = jev_answers.get("complexity", {}).get("choice")
        n_choice = nano_answers.get("complexity", {}).get("choice")
        n_conf = nano_answers.get("complexity", {}).get("confidence", 0.0)

        if j_choice and n_choice:
            if j_choice == n_choice:
                comp_matches += 1
            else:
                disagreements.append({
                    "idx": i + 1,
                    "task": str(req.get("state", {}).get("user_task", ""))[:80],
                    "jev": j_choice,
                    "nano": n_choice,
                    "nano_conf": n_conf,
                })

        # 2. High Risk Comparison
        j_hr = jev_answers.get("high_risk", {}).get("noul")
        n_hr = nano_answers.get("high_risk", {}).get("noul")
        if j_hr is not None and n_hr is not None:
            risk_errors.append(abs(j_hr - n_hr))
            # Did both agree on risk gate? (both >= 0.5 or both < 0.5)
            j_gate = j_hr >= 0.5
            n_gate = n_hr >= 0.5
            if j_gate == n_gate:
                risk_gate_agreements += 1

        # 3. Independent Comparison
        j_ind = jev_answers.get("independent", {}).get("noul")
        n_ind = nano_answers.get("independent", {}).get("noul")
        if j_ind is not None and n_ind is not None:
            indep_errors.append(abs(j_ind - n_ind))

        if (i + 1) % 50 == 0 or (i + 1 == len(cases)):
            print(f"Processed [{i+1}/{len(cases)}] cases...", flush=True)

    total_valid = len(latencies)
    print("\n" + "=" * 70)
    print("📊 线上真实历史请求 (jev-router-audit.jsonl) 全量对比报告")
    print("=" * 70)
    print(f"对比有效请求总数: {total_valid} 条")
    print(f"• 复杂度 (Complexity Choice) 一致率 : {comp_matches}/{total_valid} ({comp_matches/total_valid*100:.1f}%)")
    print(f"• 高风险门禁 (Risk Gate >= 0.5) 一致率: {risk_gate_agreements}/{total_valid} ({risk_gate_agreements/total_valid*100:.1f}%)")
    print(f"• 高风险概率绝对误差 (Risk MAE)       : {statistics.mean(risk_errors):.4f}")
    print(f"• 独立性概率绝对误差 (Indep MAE)      : {statistics.mean(indep_errors):.4f}")
    print(f"• 响应延迟分布                       : 中位={statistics.median(latencies):.1f}ms, P95={sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")

    if disagreements:
        print("\n🔍 抽样不一致案例 (前 5 条):")
        for d in disagreements[:5]:
            print(f"  [Case #{d['idx']}] {d['task']}...")
            print(f"     官方 Jev: {d['jev']}  <--->  NanoJev: {d['nano']} (conf: {d['nano_conf']})")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-log", default="/Users/chenyc/Downloads/router/jev-router-audit.jsonl")
    parser.add_argument("--server-url", default="http://192.168.123.88:8769")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    evaluate_audit_dataset(args.audit_log, args.server_url, args.limit)
