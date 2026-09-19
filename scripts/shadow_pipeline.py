#!/usr/bin/env python3
"""Daily Shadow Evaluation & Disagreement Harvesting Pipeline.

Pairs logs/jev-YYYY-MM-DD.jsonl and logs/jev-local-88-YYYY-MM-DD.jsonl via request_id.
1. Computes operational metrics (Agreement rate, Risk MAE, Latency speedup).
2. Extracts hard disagreement cases into training datasets for continual distillation.
"""
import argparse
import datetime
import json
import statistics
from pathlib import Path


def process_daily_logs(router_logs_dir: str, date_str: str = None, output_train_file: str = None):
    logs_dir = Path(router_logs_dir).resolve()
    if not date_str:
        # Default to today (or latest available)
        date_str = datetime.date.today().strftime("%Y-%m-%d")

    jev_file = logs_dir / f"jev-{date_str}.jsonl"
    local_file = logs_dir / f"jev-local-88-{date_str}.jsonl"

    if not jev_file.exists() or not local_file.exists():
        print(f"Waiting for log files: {jev_file.name} and {local_file.name}...")
        return None

    # 1. Index official Jev logs by request_id
    jev_map = {}
    with open(jev_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                rid = d.get("request_id")
                if rid and d.get("response", {}).get("answers"):
                    jev_map[rid] = d
            except Exception:
                pass

    # 2. Match with local 88 logs
    matched_pairs = []
    disagreements = []
    latencies_jev = []
    latencies_88 = []
    risk_diffs = []

    with open(local_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d88 = json.loads(line)
                rid = d88.get("request_id")
                if rid in jev_map:
                    j_entry = jev_map[rid]
                    j_ans = j_entry["response"]["answers"]
                    n_ans = d88["response"].get("answers", {})

                    j_dur = j_entry.get("duration_ms", 0)
                    n_dur = d88.get("duration_ms", 0)
                    latencies_jev.append(j_dur)
                    latencies_88.append(n_dur)

                    # Compare Complexity
                    j_comp = j_ans.get("complexity", {}).get("choice")
                    n_comp = n_ans.get("complexity", {}).get("choice")

                    # Compare Risk
                    j_risk = j_ans.get("high_risk", {}).get("noul")
                    n_risk = n_ans.get("high_risk", {}).get("noul")

                    is_disagreement = False
                    if j_comp != n_comp:
                        is_disagreement = True

                    if j_risk is not None and n_risk is not None:
                        diff = abs(j_risk - n_risk)
                        risk_diffs.append(diff)
                        if (j_risk >= 0.5) != (n_risk >= 0.5):
                            is_disagreement = True

                    matched_pairs.append({
                        "request_id": rid,
                        "task": str(d88.get("request", {}).get("state", {}).get("user_task", ""))[:120],
                        "jev_comp": j_comp,
                        "nano_comp": n_comp,
                        "jev_risk": j_risk,
                        "nano_risk": n_risk,
                        "speedup": round(j_dur / max(1, n_dur), 1),
                    })

                    if is_disagreement:
                        disagreements.append((j_entry["request"], j_ans, rid))
            except Exception:
                pass

    # 3. Print Report
    total = len(matched_pairs)
    print("\n" + "=" * 70)
    print(f"📊 Daily Shadow 对齐与自动化分析报告 ({date_str})")
    print("=" * 70)
    print(f"已捕获并成功配对的请求总数: {total}")

    if total > 0:
        agreed_comp = sum(1 for p in matched_pairs if p["jev_comp"] == p["nano_comp"])
        print(f"• 复杂度决策一致率 (Choice Agreement): {agreed_comp}/{total} ({agreed_comp/total*100:.1f}%)")
        if risk_diffs:
            print(f"• 高风险概率平均误差 (Risk MAE):       {statistics.mean(risk_diffs):.4f}")
        print(f"• 决策耗时对比 (平均):")
        print(f"    - 云端官方 Jev: {statistics.mean(latencies_jev):.1f} ms")
        print(f"    - 88 本地 NanoJev: {statistics.mean(latencies_88):.1f} ms  (提速约 {statistics.mean(latencies_jev)/max(1, statistics.mean(latencies_88)):.1f} 倍⚡)")

        if disagreements:
            print(f"\n🔍 发现 {len(disagreements)} 个分歧样本，已自动萃取用于增量强化学习。")
            for req, j_ans, rid in disagreements[:3]:
                task_txt = str(req.get("state", {}).get("user_task", ""))[:80]
                print(f"  • [RID: {rid[:8]}] 任务: {task_txt}...")
                print(f"    官方 Jev 标注: complexity='{j_ans.get('complexity',{}).get('choice')}', high_risk={j_ans.get('high_risk',{}).get('noul')}")
        else:
            print("\n✨ 今日全部捕获流量决策 100% 吻合，无分歧！")

    # 4. Harvest disagreements to training format
    if output_train_file and disagreements:
        out_p = Path(output_train_file).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        from typesafe_adapter import typesafe_request_to_nanojev

        harvested_rows = []
        for req, judg, rid in disagreements:
            try:
                nj_payload, _ = typesafe_request_to_nanojev(req)
                st = nj_payload["states"][0]
                questions = st["questions"]

                gold_probs = {}
                gold = {}
                for qid, q in questions.items():
                    j_a = judg.get(qid)
                    if not j_a:
                        continue
                    if q["type"] == "boolean":
                        p_true = float(j_a.get("noul", 0.5))
                        gold_probs[qid] = {"false": 1.0 - p_true, "true": p_true}
                        gold[qid] = p_true >= 0.5
                    elif q["type"] == "choice":
                        probs = j_a.get("probabilities", {})
                        crit_keys = list(q["criteria"].keys())
                        norm_p = {k: float(probs.get(k, 1.0 / len(crit_keys))) for k in crit_keys}
                        tot = sum(norm_p.values())
                        norm_p = {k: v / tot for k, v in norm_p.items()}
                        norm_p[crit_keys[-1]] += 1.0 - sum(norm_p.values())
                        gold_probs[qid] = norm_p
                        gold[qid] = j_a.get("choice", crit_keys[0])

                harvested_rows.append({
                    "id": f"harvest_{rid[:12]}",
                    "state_id": f"harvest_{rid[:12]}",
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

        if harvested_rows:
            with open(out_p, "a", encoding="utf-8") as f:
                for r in harvested_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"💾 已将 {len(harvested_rows)} 条真实对齐样本追加沉淀至: {out_p}")

    return {
        "matched": total,
        "disagreements": len(disagreements),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="/Users/chenyc/Documents/study/jev-cliproxy-router/logs")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format (defaults to latest)")
    parser.add_argument("--harvest-out", default="data/harvested_shadow_data.jsonl", help="Path to save harvested hard cases")
    args = parser.parse_args()
    process_daily_logs(args.logs_dir, args.date, args.harvest_out)
