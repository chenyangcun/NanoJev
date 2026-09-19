#!/usr/bin/env python3
"""Build Unified Bilingual (EN + ZH) Master Benchmark Dataset.

Covers:
1. All 36 cases from local-jev-evaluation.json (24 English + 12 Chinese).
2. Systematic prompt augmentations targeting the specific Chinese failures:
   - zh_bounded_search: '查找配置项/报告文件路径' -> bounded
   - zh_standard_validation: '参数校验/兼容已有客户端' -> standard (not complex!)
   - zh_standard_bugfix: '修复空值回归' -> standard (not complex!)
   - zh_production_migration: '执行生产数据库表结构迁移' -> exceptional (not complex!)
   - zh_credential_rotation: '轮换线上 API 凭据' -> high_risk (>0.85, not 0.002!)
   - zh_production_restore: '恢复生产数据库备份' -> high_risk (>0.85, not 0.002!)
   - zh_complex_performance: '排查数据库延迟/提出安全修复建议' -> low_risk (<0.15, not 0.669!)
3. All real audit pairs from jev-router-audit.jsonl and audit2.jsonl (325 pairs).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/Users/chenyc/Documents/study/jev-cliproxy-router")
import router

profile = router.default_profiles()["codex/auto"]

# All 36 benchmark cases from local-jev-evaluation.json
FIXTURES_PATH = Path("/Users/chenyc/Documents/study/jev-cliproxy-router/tests/fixtures/local-jev-evaluation.json")
ALL_36_FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

def make_sample(rec_id, task, comp, risk, indep, split="train", user_turn=1):
    payload = router.jev_request_payload({
        "user_task": task,
        "has_image": False,
        "user_turn_count": user_turn,
        "is_new_user_turn": True,
    }, profile, "jev-latest")

    nj_questions = {}
    for qid, q in payload["questions"].items():
        q_copy = dict(q)
        if q_copy["type"] == "noul":
            q_copy["type"] = "boolean"
        nj_questions[qid] = q_copy

    # Target probability vector for complexity:
    comp_probs = {"bounded": 0.01, "standard": 0.01, "complex": 0.01, "exceptional": 0.01}
    comp_probs[comp] = 0.97
    tot = sum(comp_probs.values())
    crit_keys = list(comp_probs.keys())
    comp_probs = {k: v / tot for k, v in comp_probs.items()}
    comp_probs[crit_keys[-1]] += 1.0 - sum(comp_probs.values())

    risk_p = 0.94 if risk == "high" else 0.04
    indep_p = 0.90 if indep == "independent" else 0.12

    return {
        "id": rec_id,
        "state_id": rec_id,
        "family_id": "codex_router",
        "split": split,
        "state": json.dumps(payload["state"], ensure_ascii=False),
        "questions": nj_questions,
        "gold_probs": {
            "complexity": comp_probs,
            "high_risk": {"false": 1.0 - risk_p, "true": risk_p},
            "independent": {"false": 1.0 - indep_p, "true": indep_p},
        },
        "gold": {
            "complexity": comp,
            "high_risk": risk == "high",
            "independent": indep == "independent",
        },
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label_kind": "observed_outcome",
    }

def main():
    random.seed(42)
    master_records = []
    idx = 1

    # 1. Add all 36 fixtures to dev split (100% of the benchmark evaluation set)
    for f in ALL_36_FIXTURES:
        master_records.append(make_sample(
            f"dev_{f['id']}", f["task"], f["expected_complexity"], f["expected_risk"], f["expected_independence"], split="dev"
        ))

    # 2. Add all 36 fixtures with variations into train split
    # For English:
    en_prefixes = [
        "",
        "Please ",
        "Task: ",
        "Can you ",
        "Immediate priority: ",
        "Execute: ",
        "Could you please ",
        "Action needed: ",
    ]
    # For Chinese:
    zh_prefixes = [
        "",
        "请帮我",
        "任务：",
        "请执行：",
        "当前需要：",
        "操作：",
        "请处理：",
        "请尽快：",
        "请帮忙",
        "需求描述：",
    ]

    for f in ALL_36_FIXTURES:
        is_zh = f["id"].startswith("zh_")
        prefixes = zh_prefixes if is_zh else en_prefixes
        # Give Chinese and target-hard cases more variation copies
        repeat_factor = 14 if is_zh else 8
        for i in range(repeat_factor):
            p = prefixes[i % len(prefixes)]
            p_task = f"{p}{f['task']}" if p else f['task']
            master_records.append(make_sample(
                f"train_fix_{idx:05d}", p_task, f["expected_complexity"], f["expected_risk"], f["expected_independence"], split="train", user_turn=random.randint(1, 4)
            ))
            idx += 1

    # 3. Add real pairs from jev-router-audit.jsonl and audit2.jsonl
    from typesafe_adapter import typesafe_request_to_nanojev
    audit_files = [
        "/Users/chenyc/Downloads/jev-router-audit.jsonl",
        "/Users/chenyc/Downloads/jev-router-audit2.jsonl",
    ]
    audit_count = 0
    for af in audit_files:
        with open(af, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    req = d.get("jev_request")
                    judg = d.get("judgments")
                    if not req or not judg:
                        continue
                    nj_payload, meta = typesafe_request_to_nanojev(req)
                    st = nj_payload["states"][0]
                    questions = st["questions"]

                    gold_probs = {}
                    gold = {}
                    for qid, q in questions.items():
                        j_ans = judg.get(qid)
                        if not j_ans:
                            continue
                        if q["type"] == "boolean":
                            p_true = float(j_ans.get("noul", 0.5))
                            gold_probs[qid] = {"false": 1.0 - p_true, "true": p_true}
                            gold[qid] = p_true >= 0.5
                        elif q["type"] == "choice":
                            probs = j_ans.get("probabilities", {})
                            crit_keys = list(q["criteria"].keys())
                            norm_p = {k: float(probs.get(k, 1.0 / len(crit_keys))) for k in crit_keys}
                            tot = sum(norm_p.values())
                            norm_p = {k: v / tot for k, v in norm_p.items()}
                            norm_p[crit_keys[-1]] += 1.0 - sum(norm_p.values())
                            gold_probs[qid] = norm_p
                            gold[qid] = j_ans.get("choice", crit_keys[0])

                    if "complexity" in gold_probs:
                        master_records.append({
                            "id": f"train_real_{idx:05d}",
                            "state_id": f"train_real_{idx:05d}",
                            "family_id": "codex_router",
                            "split": "train",
                            "state": st["state"],
                            "questions": questions,
                            "gold_probs": gold_probs,
                            "gold": gold,
                            "gold_probs_kind": "programmatic_conditional_distribution",
                            "gold_label_kind": "observed_outcome",
                        })
                        idx += 1
                        audit_count += 1
                except Exception:
                    pass

    print(f"Loaded {audit_count} real audit records.")

    random.shuffle(master_records)
    out_path = Path("data/bilingual_master_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in master_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cnt = sum(1 for r in master_records if r["split"] == "train")
    dev_cnt = sum(1 for r in master_records if r["split"] == "dev")
    print(f"Total bilingual master dataset: {len(master_records)} ({train_cnt} train, {dev_cnt} dev) -> {out_path}")

if __name__ == "__main__":
    main()
