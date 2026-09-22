#!/usr/bin/env python3
"""Build training dataset for the 'subagent' decision head.

Covers:
1. 'context_mode': choice ('none', 'recent_3', 'recent_6', 'needs_summary')
   - none: self-contained task packages, independent reviews, fresh commands
   - recent_3: immediate follow-ups referencing last 1-3 turns (e.g. "fix error from last run")
   - recent_6: multi-step tracing across 4-6 turns
   - needs_summary: long discussions requiring condensed parent summary
2. 'agent_role': choice ('opencode-worker', 'codex_native', 'reviewer')
3. 'requires_parent_summary': boolean

Outputs:
  data/subagent_dataset.jsonl (with train/dev/calibration splits)
"""

import hashlib
import json
import random
import sys
from pathlib import Path

CONTEXT_CHOICES = ["none", "recent_3", "recent_6", "needs_summary"]

SUBAGENT_CRITERIA = {
    "none": "The provided task package is self-contained. The worker can proceed without parent conversation turns.",
    "recent_3": "The worker needs the complete supplied turns 1 through 3, with turn 1 newest, to resolve the immediate context.",
    "recent_6": "The worker needs the complete supplied turns 1 through 6, with turn 1 newest, to resolve multi-step history.",
    "needs_summary": "Older or broader parent context matters, but a focused parent-written summary should be supplied instead of raw turns.",
}

ROLE_CRITERIA = {
    "opencode-worker": "Default general-purpose subagent for execution, inspection, and implementation.",
    "codex_native": "Native inline execution without subagent delegation.",
    "reviewer": "Read-only inspection, architectural audit, or security review.",
}


def build_subagent_state(task: str, task_kind: str = "exploration", turns_count: int = 0, facts: list = None):
    recent_turns = []
    if turns_count == 3:
        recent_turns = [
            {"role": "assistant", "content": "Ran unit test, found 1 assertion failure in test_parser.py:42."},
            {"role": "user", "content": "Please fix that test failure."},
            {"role": "assistant", "content": "Inspecting lines 40-50 in parser.py."},
        ]
    elif turns_count == 6:
        recent_turns = [
            {"role": "user", "content": "Step 1: check routes"},
            {"role": "assistant", "content": "Routes checked."},
            {"role": "user", "content": "Step 2: add validation"},
            {"role": "assistant", "content": "Validation added."},
            {"role": "user", "content": "Step 3: run integration tests"},
            {"role": "assistant", "content": "Tests failed on timeout."},
        ]

    state = {
        "task": task,
        "task_kind": task_kind,
        "scope": "Confined to target files and local tests only.",
        "known_facts": facts or ["Repository is clean", "Python 3.12 environment"],
        "recent_turns": recent_turns,
        "recent_decision_digest": "Not provided." if not recent_turns else "Recent steps performed.",
        "policy": (
            "Use only this task package. Do not assume, guess, or invent omitted parent-history details. "
            "Use only the explicitly supplied recent_turns. Prefer the smallest context option."
        ),
    }
    return json.dumps(state, ensure_ascii=False)


# Task bank by expected context_mode
TASKS_NONE = [
    ("只读分析当前路由模块代码结构，仅列出文件路径和核心类定义。", "exploration"),
    ("检查 README.md 中的拼写错误，这是一个自包含的任务。", "implementation"),
    ("查询系统中已注册的 Skill 列表并导出 JSON，无需父会话上下文。", "exploration"),
    ("运行 focused unit test: pytest tests/test_parser.py 并报告结果。", "verification"),
    ("在当前目录查找所有包含 MAX_OPTIONS 常量的 Python 文件。", "exploration"),
    ("阅读并总结 docs/API.md 的主要接口契约。", "exploration"),
    ("格式化单个文件 scripts/model.py，无需任何历史信息。", "implementation"),
    ("检查远程服务器 192.168.123.88 的 8769 端口健康检查返回值。", "verification"),
]

TASKS_RECENT_3 = [
    ("根据上一轮测试输出的报错信息，修复 parser.py 中刚才失败的断言。", "implementation"),
    ("继续上一步的调试，查看刚才那条失败命令对应的日志堆栈。", "diagnosis"),
    ("将上一轮讨论中提到的那个函数参数默认值修改为 0.35。", "implementation"),
    ("在上一步发现的语法错误行处加上 try-except 异常处理。", "implementation"),
    ("基于刚才讨论的代码片段，补写一个对应的单元测试用例。", "implementation"),
]

TASKS_RECENT_6 = [
    ("结合我们前面 3 轮的修改和测试排错过程，汇总一份完整的回归测试结果。", "verification"),
    ("根据前序 5 轮逐步推进的重构步骤，完成最终的跨模块集成验证。", "verification"),
    ("检查从最初的接口设计到刚才几轮测试中是否有未处理的边界遗漏。", "diagnosis"),
]

TASKS_NEEDS_SUMMARY = [
    ("经过我们这一整轮长时间的方案探讨，请提炼并总结出最终的架构决策结论。", "review"),
    ("前序长篇讨论中包含了大量分支逻辑，请将其精简总结为一份任务交接卡片。", "review"),
    ("请将之前讨论的各种方案优缺点梳理为一份背景摘要，以便新智能体接入。", "review"),
]


def generate_subagent_dataset(output_file: Path, seed: int = 42):
    rng = random.Random(seed)
    records = []

    # 1. Generate synthetic variations
    categories = [
        (TASKS_NONE, "none", 0, 80),
        (TASKS_RECENT_3, "recent_3", 3, 70),
        (TASKS_RECENT_6, "recent_6", 6, 60),
        (TASKS_NEEDS_SUMMARY, "needs_summary", 0, 70),
    ]

    idx = 0
    for task_list, exp_mode, turns, repeat_cnt in categories:
        for _ in range(repeat_cnt):
            base_task, task_kind = rng.choice(task_list)
            # Add small random phrasing variation
            prefixes = ["", "当前子任务：", "请协助处理：", "执行独立工作项：", "任务要求："]
            pfx = rng.choice(prefixes)
            task_text = f"{pfx}{base_task}"

            # Role logic
            if task_kind == "review":
                exp_role = "reviewer"
            elif exp_mode == "none":
                exp_role = rng.choice(["opencode-worker", "opencode-worker", "reviewer"])
            else:
                exp_role = "opencode-worker"

            state_str = build_subagent_state(task_text, task_kind=task_kind, turns_count=turns)

            probs_map = {c: 0.01 for c in CONTEXT_CHOICES}
            probs_map[exp_mode] = 0.97

            role_probs = {r: 0.01 for r in ROLE_CRITERIA}
            role_probs[exp_role] = 0.98

            rec_id = f"subagent_gen_{idx:04d}"
            split = "test" if idx % 10 == 0 else ("dev" if idx % 10 == 1 else "train")

            records.append({
                "id": rec_id,
                "state_id": rec_id,
                "family_id": "subagent_dispatch",
                "split": split,
                "state": state_str,
                "questions": {
                    "context_mode": {
                        "type": "choice",
                        "instructions": "Based only on the supplied task package, which smallest parent conversation context should a subagent receive? Prefer none. Choose needs_summary whenever older context can be distilled into a focused task-package summary rather than inherited conversation turns.",
                        "criteria": SUBAGENT_CRITERIA,
                    },
                    "agent_role": {
                        "type": "choice",
                        "instructions": "Which subagent role is best suited to execute this task package?",
                        "criteria": ROLE_CRITERIA,
                    },
                },
                "gold": {
                    "context_mode": exp_mode,
                    "agent_role": exp_role,
                },
                "gold_probs": {
                    "context_mode": probs_map,
                    "agent_role": role_probs,
                },
                "gold_probs_kind": "programmatic_conditional_distribution",
                "gold_label_kind": "observed_outcome",
            })
            idx += 1

    # Ingest real decision calls from jev-decisions-2026-09-21
    real_decisions_file = Path("/tmp/jev-decisions-2026-09-21.jsonl")
    if real_decisions_file.exists():
        with open(real_decisions_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                dec = row.get("decision", {})
                remote = dec.get("remote_response", {})
                eval_req = dec.get("evaluator_request", {})
                if remote and eval_req:
                    st_str = json.dumps(eval_req.get("state", {}), ensure_ascii=False)
                    rec_id = f"subagent_real_{row.get('request_id', idx)}"
                    ans = remote.get("answers", {}).get("context_mode", {})
                    choice_val = ans.get("choice", "none")
                    probs = ans.get("probabilities", {"none": 0.98, "needs_summary": 0.02, "recent_3": 0.0, "recent_6": 0.0})

                    # Add real decision records
                    for repeat in range(4):
                        records.append({
                            "id": f"{rec_id}_r{repeat}",
                            "state_id": f"{rec_id}_r{repeat}",
                            "family_id": "subagent_dispatch",
                            "split": "train",
                            "state": st_str,
                            "questions": {
                                "context_mode": {
                                    "type": "choice",
                                    "instructions": "Based only on the supplied task package, which smallest parent conversation context should a subagent receive? Prefer none. Choose needs_summary whenever older context can be distilled into a focused task-package summary rather than inherited conversation turns.",
                                    "criteria": SUBAGENT_CRITERIA,
                                }
                            },
                            "gold": {"context_mode": choice_val},
                            "gold_probs": {"context_mode": probs},
                            "gold_probs_kind": "programmatic_conditional_distribution",
                            "gold_label_kind": "observed_outcome",
                        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} subagent records to: {output_file}", flush=True)


if __name__ == "__main__":
    generate_subagent_dataset(Path("data/subagent_dataset.jsonl"))
