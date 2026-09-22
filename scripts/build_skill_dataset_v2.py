#!/usr/bin/env python3
"""Build Unified Skill Selection Dataset matching V2 Production Schema.

Includes:
Phase 1: shortlist (Choice) + 3 gating nouls:
  - shortlist: choice (none + candidate skills)
  - acts_on_user_system: noul
  - would_follow_documented_procedure: noul
  - prose_suffices: noul

Phase 2: winner (Choice) + per-candidate fits_<skill_id> (Noul):
  - winner: choice (none + shortlisted skills)
  - fits_<skill_id>: noul
"""
import copy
import json
import random
import sys
from pathlib import Path

ROUTER_DIR = Path("/Users/chenyc/Documents/study/jev-cliproxy-router").resolve()
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))

from skill_selection import (
    load_registry, normalize_request, build_shortlist_request, build_verify_request
)
from internal_api import approved_roots_from_env

REGISTRY_PATH = ROUTER_DIR / "skill-registry.json"
registry = load_registry(REGISTRY_PATH, approved_roots_from_env())

ALL_SKILL_ENTRIES = registry.entries
ENABLED_SKILL_ENTRIES = registry.enabled()

EVAL_CASES = [
    ("drawio", "请把系统模块关系画成可继续编辑的 Draw.io 架构图。", "drawio-skill"),
    ("chrome", "请检查我当前打开的 Chrome 页面，定位控制台报错。", "chrome-cdp"),
    ("openai_docs", "查询 OpenAI Responses API 最新官方参数，并给出迁移建议。", "openai-docs"),
    ("opencli", "使用 OpenCLI 搜索并读取这个网站的页面内容。", "opencli"),
    ("plugin_create", "帮我创建一个新的 Codex 插件，包含 manifest 和本地市场配置。", "plugin-creator"),
    ("skill_create", "把这套重复工作流整理成一个新的 Codex Skill。", "skill-creator"),
    ("skill_install", "从一个 GitHub 仓库安装 Codex Skill。", "skill-installer"),
    ("video_parse", "提取这个 Bilibili 视频的字幕，没有字幕就做本地语音转写。", "video-parse"),
    ("typesafe", "用 Jev 为应用设计一个返回概率的结构化路由决策。", "typesafe-ai"),
    ("find_skill", "有没有可以管理 Notion 的 Agent Skill，帮我查找一下。", "find-skills"),
    ("none_python", "解释 Python 列表推导式和普通 for 循环的区别。", "none"),
    ("none_java", "修复这段 Java 代码中的空指针异常并补单元测试。", "none"),
    ("combo_diagram", "设计系统架构并交付一份可编辑的架构图。", "drawio-skill"),
    ("adjacent_docs_search", "查阅 OpenAI 官方文档，确认 Codex 的最新配置方式。", "openai-docs"),
]

SKILL_TASK_BANK = {
    "drawio-skill": [
        "请把系统模块关系画成可继续编辑的 Draw.io 架构图。",
        "设计系统架构并交付一份可编辑的架构图。",
        "帮我画一个订单系统的微服务架构时序图，需要生成 drawio 文件。",
        "画一个用户注册登录与双因素认证的 UML 状态机图，输出 drawio 格式。",
        "把数据库里的这几张表的关联关系画成 ER 实体关系图，保存为 .drawio 文件。",
        "制作系统微服务网络拓扑架构图，支持后续在 draw.io 中继续编辑调整。",
        "帮我生成系统的数据流图 (DFD)，格式要求为可编辑的 drawio 图表。",
    ],
    "chrome-cdp": [
        "请检查我当前打开的 Chrome 页面，定位控制台报错。",
        "连接本机 Chrome 浏览器，查看当前标签页的网络请求是否有 500 错误。",
        "调试 Chrome 页面中前端按钮点击没有反应的问题，查看控制台输出日志。",
        "检查已打开的 Chrome 浏览器中的 DOM 树，找到登录表单的输入框 ID。",
    ],
    "openai-docs": [
        "查询 OpenAI Responses API 最新官方参数，并给出迁移建议。",
        "查阅 OpenAI 官方文档，确认 Codex 的最新配置方式。",
        "查阅官方文档确认 GPT-4o 结构化输出 (Structured Outputs) 的 json_schema 传参规范。",
        "帮我查询 OpenAI Embeddings API 的最新推荐模型和计费单价。",
        "根据 OpenAI 官方文档，如何正确配置 Reasoning Effort 参数？",
    ],
    "opencli": [
        "使用 OpenCLI 搜索并读取这个网站的页面内容。",
        "用 opencli 工具抓取 Hacker News 首页最热门的 10 条讨论。",
        "使用 OpenCLI 适配器搜索 GitHub Trending 今天的 Python 热门开源项目。",
        "通过 opencli search 检索关于 Rust 异步运行时的最新技术博客文章。",
    ],
    "typesafe-ai": [
        "用 Jev 为应用设计一个返回概率的结构化路由决策。",
        "接入 TypeSafe System One API，实现对客服工单紧急程度的概率判断。",
        "使用 TypeSafe Jev 模型构建零输出解码的布尔命题验证逻辑。",
        "使用 TypeSafe API 的 Choice 原语对用户意图进行多选项概率分布打分。",
    ],
    "video-parse": [
        "提取这个 Bilibili 视频的字幕，没有字幕就做本地语音转写。",
        "解析这个 YouTube 视频的文案和元数据，生成一段内容摘要。",
        "下载这个 B 站教学视频的音频，并使用本地 Faster-Whisper 转写成文字。",
        "提取视频中的中文字幕轨道，并保存为 .vtt 文件。",
    ],
    "none": [
        "解释 Python 列表推导式和普通 for 循环的区别。",
        "修复这段 Java 代码中的空指针异常并补单元测试。",
        "写一个快速排序算法并分析其平均时间复杂度。",
        "在当前文件里增加一个辅助函数，计算两个日期的天数差。",
        "帮我给这段业务代码写单元测试，覆盖边界异常情况。",
        "优化这段数组去重算法的时间复杂度，从 O(N^2) 降到 O(N)。",
    ]
}


def make_record(row_id: str, state_dict: dict, questions: dict, gold_probs: dict, gold: dict, split: str = "train"):
    # Convert noul to boolean in question schemas for NanoJev contract
    nj_questions = {}
    for qid, q in questions.items():
        q_copy = dict(q)
        if q_copy["type"] == "noul":
            q_copy["type"] = "boolean"
        nj_questions[qid] = q_copy

    return {
        "id": row_id,
        "state_id": row_id,
        "family_id": "skill_selection_v2",
        "split": split,
        "state": json.dumps(state_dict, ensure_ascii=False),
        "questions": nj_questions,
        "gold_probs": gold_probs,
        "gold": gold,
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label_kind": "observed_outcome",
    }


def main():
    random.seed(42)
    output_path = Path("data/skill_selection_v2_dataset.jsonl")
    records = []
    idx = 1

    # Generate samples for both shortlist and verify phases
    all_tasks = [(cid, task, exp) for cid, task, exp in EVAL_CASES]
    for sk, tlist in SKILL_TASK_BANK.items():
        for t in tlist:
            all_tasks.append((sk, t, sk))

    prefixes = ["", "请帮我", "任务：", "请执行：", "需求：", "指令："]

    for orig_id, task, expected in all_tasks:
        for p in prefixes:
            p_task = f"{p}{task}" if p else task
            req = normalize_request({"task": p_task, "trigger": "user_prompt", "conversation_key": f"train-{idx}"}, registry)

            # --- Phase 1: Shortlist payload ---
            shortlist_payload = build_shortlist_request(req, registry, "jev-latest")
            st_state = shortlist_payload["state"]
            st_questions = shortlist_payload["questions"]

            cand_keys = list(st_questions["shortlist"]["criteria"].keys())
            target_cand = expected if expected in cand_keys else "none"

            # Prob distribution
            p_short = {k: 0.01 for k in cand_keys}
            p_short[target_cand] = 0.96
            tot = sum(p_short.values())
            p_short = {k: v / tot for k, v in p_short.items()}
            p_short[cand_keys[-1]] += 1.0 - sum(p_short.values())

            # Gating nouls
            is_none = (target_cand == "none")
            p_act = 0.10 if is_none else 0.85
            p_proc = 0.05 if is_none else 0.90
            p_prose = 0.92 if is_none else 0.08

            gold_probs_1 = {
                "shortlist": p_short,
                "acts_on_user_system": {"false": 1.0 - p_act, "true": p_act},
                "would_follow_documented_procedure": {"false": 1.0 - p_proc, "true": p_proc},
                "prose_suffices": {"false": 1.0 - p_prose, "true": p_prose},
            }
            gold_1 = {
                "shortlist": target_cand,
                "acts_on_user_system": p_act >= 0.5,
                "would_follow_documented_procedure": p_proc >= 0.5,
                "prose_suffices": p_prose >= 0.5,
            }

            records.append(make_record(
                f"skill_v2_p1_{idx:05d}", st_state, st_questions, gold_probs_1, gold_1,
                split="dev" if (p == "" and orig_id in [c[0] for c in EVAL_CASES]) else "train"
            ))
            idx += 1

            # --- Phase 2: Verify payload ---
            verify_ids = [target_cand] if target_cand != "none" else ["chrome-cdp"]
            verify_payload = build_verify_request(req, registry, verify_ids, "jev-latest")
            v_state = verify_payload["state"]
            v_questions = verify_payload["questions"]

            win_keys = list(v_questions["winner"]["criteria"].keys())
            p_win = {k: 0.01 for k in win_keys}
            p_win[target_cand] = 0.96
            tot_w = sum(p_win.values())
            p_win = {k: v / tot_w for k, v in p_win.items()}
            p_win[win_keys[-1]] += 1.0 - sum(p_win.values())

            gold_probs_2 = {"winner": p_win}
            gold_2 = {"winner": target_cand}

            for sid in verify_ids:
                q_fit = f"fits_{sid.replace('-', '_')}"
                if q_fit in v_questions:
                    is_match = (sid == target_cand)
                    p_fit = 0.96 if is_match else 0.02
                    gold_probs_2[q_fit] = {"false": 1.0 - p_fit, "true": p_fit}
                    gold_2[q_fit] = is_match

            records.append(make_record(
                f"skill_v2_p2_{idx:05d}", v_state, v_questions, gold_probs_2, gold_2,
                split="dev" if (p == "" and orig_id in [c[0] for c in EVAL_CASES]) else "train"
            ))
            idx += 1

    random.shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cnt = sum(1 for r in records if r["split"] == "train")
    dev_cnt = sum(1 for r in records if r["split"] == "dev")
    print(f"Generated {len(records)} V2 skill records ({train_cnt} train, {dev_cnt} dev) -> {output_path}")


if __name__ == "__main__":
    main()
