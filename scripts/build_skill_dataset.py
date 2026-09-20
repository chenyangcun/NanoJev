#!/usr/bin/env python3
"""Build Skill Selection Training & Validation Dataset for NanoJev.

Generates realistic two-stage System One queries:
1. `shortlist` (Choice):
   - State contains `task`, `trigger: "user_prompt"`, `candidate_skills` list, and `selection_budget`.
   - Question `shortlist`: Choice with options `none` + candidate skill IDs and descriptions.
   - Target probability: sharp mass (~0.95) on the ground truth skill (or `none`).
2. `verify_<skill_id>` (Boolean/Noul):
   - State contains `task`, `trigger: "user_prompt"`, `candidate_skills` list.
   - Question `verify_<skill_id>`: Boolean.
   - Target probability: `true` (0.95) when skill matches task, `false` (0.02) when irrelevant.
"""
import copy
import json
import random
import sys
from pathlib import Path

# Add router directory
ROUTER_DIR = Path("/Users/chenyc/Documents/study/jev-cliproxy-router").resolve()
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))

from skill_selection import (
    load_registry, normalize_request, build_shortlist_request, build_verify_request
)
from internal_api import approved_roots_from_env

REGISTRY_PATH = ROUTER_DIR / "skill-registry.json"
registry = load_registry(REGISTRY_PATH, approved_roots_from_env())

ALL_SKILL_ENTRIES = registry.entries # all 10 skills
ENABLED_SKILL_ENTRIES = registry.enabled() # 6 currently enabled skills

# Benchmark cases from evaluate-skill-selection-live.py
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

# Rich pool of tasks per skill for training data augmentation
SKILL_TASK_BANK = {
    "drawio-skill": [
        "请把系统模块关系画成可继续编辑的 Draw.io 架构图。",
        "设计系统架构并交付一份可编辑的架构图。",
        "帮我画一个订单系统的微服务架构时序图，需要生成 drawio 文件。",
        "画一个用户注册登录与双因素认证的 UML 状态机图，输出 drawio 格式。",
        "把数据库里的这几张表的关联关系画成 ER 实体关系图，保存为 .drawio 文件。",
        "需要绘制支付退款的核心业务状态流转流程图。",
        "请用 draw.io 绘制当前项目的模块架构与类依赖关系图。",
        "制作系统微服务网络拓扑架构图，支持后续在 draw.io 中继续编辑调整。",
        "帮我生成系统的数据流图 (DFD)，格式要求为可编辑的 drawio 图表。",
        "将业务用例交互绘制成序列图并导出 drawio 图像。",
    ],
    "chrome-cdp": [
        "请检查我当前打开的 Chrome 页面，定位控制台报错。",
        "连接本机 Chrome 浏览器，查看当前标签页的网络请求是否有 500 错误。",
        "调试 Chrome 页面中前端按钮点击没有反应的问题，查看控制台输出日志。",
        "检查已打开的 Chrome 浏览器中的 DOM 树，找到登录表单的输入框 ID。",
        "通过 Chrome DevTools 协议查看当前打开页面的 localStorage 和 Cookie 缓存内容。",
        "捕获当前 Chrome 页面渲染性能卡顿的原因，导出 Performance 追踪面板结果。",
        "检查当前 Chrome 页面的控制台是否有未捕获的 JavaScript 异常。",
        "定位已打开网页中 CSS 样式覆盖导致的排版错乱问题。",
    ],
    "openai-docs": [
        "查询 OpenAI Responses API 最新官方参数，并给出迁移建议。",
        "查阅 OpenAI 官方文档，确认 Codex 的最新配置方式。",
        "查阅官方文档确认 GPT-4o 结构化输出 (Structured Outputs) 的 json_schema 传参规范。",
        "帮我查询 OpenAI Embeddings API 的最新推荐模型和计费单价。",
        "根据 OpenAI 官方文档，如何正确配置 Reasoning Effort 参数？",
        "迁移旧版 Chat Completions 到最新的 Responses API，需要修改哪些字段？",
        "查阅 OpenAI 官方文档，了解关于 Assistants API 的最新功能更新。",
        "确认 OpenAI API 的速率限制 (Rate Limits) 和层级规则。",
    ],
    "opencli": [
        "使用 OpenCLI 搜索并读取这个网站的页面内容。",
        "用 opencli 工具抓取 Hacker News 首页最热门的 10 条讨论。",
        "使用 OpenCLI 适配器搜索 GitHub Trending 今天的 Python 热门开源项目。",
        "通过 opencli search 检索关于 Rust 异步运行时的最新技术博客文章。",
        "调用 opencli 抓取指定技术文档网页并转换为 Markdown 格式供我阅读。",
        "使用 opencli 搜索 V2EX 上的最新相关讨论帖子。",
        "用 opencli 读取外部文档网站的更新公告页面。",
    ],
    "typesafe-ai": [
        "用 Jev 为应用设计一个返回概率的结构化路由决策。",
        "接入 TypeSafe System One API，实现对客服工单紧急程度的概率判断。",
        "使用 TypeSafe Jev 模型构建零输出解码的布尔命题验证逻辑。",
        "使用 TypeSafe API 的 Choice 原语对用户意图进行多选项概率分布打分。",
        "设计一套基于 Jev 模型的任务复杂度四档分类器方案。",
        "基于 TypeSafe 规范开发一个微型概率判决服务。",
        "利用 Jev 的 Score 原语评估代码质量评分。",
    ],
    "video-parse": [
        "提取这个 Bilibili 视频的字幕，没有字幕就做本地语音转写。",
        "解析这个 YouTube 视频的文案和元数据，生成一段内容摘要。",
        "下载这个 B 站教学视频的音频，并使用本地 Faster-Whisper 转写成文字。",
        "提取视频中的中文字幕轨道，并保存为 .vtt 文件。",
        "分析 YouTube 视频的标题、发布者和视频时长等基本信息。",
        "解析视频文件中的音频内容，产出结构化的字幕文本。",
    ],
    "plugin-creator": [
        "帮我创建一个新的 Codex 插件，包含 manifest 和本地市场配置。",
        "按照 Codex 规范新建一个插件脚手架，编写 plugin.json 清单文件。",
        "创建并更新一个本地插件，配置其依赖工具和扩展声明。",
    ],
    "skill-creator": [
        "把这套重复工作流整理成一个新的 Codex Skill。",
        "新建一个 Codex Skill，编写结构规范的 SKILL.md 文档与规则说明。",
        "重构现有 Skill 的指令与触发条件，定义清晰的执行边界与工作流。",
    ],
    "skill-installer": [
        "从一个 GitHub 仓库安装 Codex Skill。",
        "安装社区精选的 markdown 格式化 Skill 到本机环境中。",
        "通过 Git URL 添加并安装一个第三方的 Agent Skill。",
    ],
    "find-skills": [
        "有没有可以管理 Notion 的 Agent Skill，帮我查找一下。",
        "帮我检索是否有支持操作 Figma 设计稿的可安装 Codex Skill。",
        "寻找一个能够对接 Slack 发送告警消息的现有 Skill。",
        "查找适合做自动化测试的 Agent Skill。",
    ],
    "none": [
        "解释 Python 列表推导式和普通 for 循环的区别。",
        "修复这段 Java 代码中的空指针异常并补单元测试。",
        "写一个快速排序算法并分析其平均时间复杂度。",
        "在当前文件里增加一个辅助函数，计算两个日期的天数差。",
        "帮我给这段业务代码写单元测试，覆盖边界异常情况。",
        "解释一下什么是数据库索引的 B+ 树结构以及聚簇索引。",
        "重构这段长函数，减少 if-else 嵌套并提高代码可读性。",
        "写一个 SQL 查询语句，统计每个部门近三个月的平均销售额。",
        "解释 HTTP 状态码 301 永久重定向和 302 临时重定向的区别。",
        "帮我写一个正则表达式，匹配标准格式的中国大陆手机号码。",
        "在当前的 Makefile 中增加一个 clean 目标，删除 build 目录。",
        "优化这段数组去重算法的时间复杂度，从 O(N^2) 降到 O(N)。",
    ]
}


def make_training_row(row_id: str, state_dict: dict, questions_dict: dict, gold_probs: dict, gold: dict, split: str = "train"):
    return {
        "id": row_id,
        "state_id": row_id,
        "family_id": "skill_selection",
        "split": split,
        "state": json.dumps(state_dict, ensure_ascii=False),
        "questions": questions_dict,
        "gold_probs": gold_probs,
        "gold": gold,
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label_kind": "observed_outcome",
    }


def generate_skill_dataset(output_path: Path):
    random.seed(42)
    records = []
    idx = 1

    # 1. First, build samples for all 14 official evaluation benchmark cases
    # Both shortlist (Choice) and verify (Boolean)
    for cid, task, expected_skill in EVAL_CASES:
        # Use currently enabled skills (the 6 enabled ones)
        cand_entries = list(ENABLED_SKILL_ENTRIES)
        cand_ids = [e.id for e in cand_entries]

        # Target skill
        if expected_skill in cand_ids:
            target_choice = expected_skill
        else:
            target_choice = "none"

        # A. Shortlist request
        state_dict = {
            "task": task,
            "trigger": "user_prompt",
            "candidate_skills": [{"id": e.id, "description": e.description} for e in cand_entries],
            "selection_budget": {"max_selected": 3, "max_injection_chars": 5000}
        }
        shortlist_crit = {"none": "No registered Skill is relevant enough for this task."}
        shortlist_crit.update({e.id: e.description for e in cand_entries})
        
        # Build probability distribution: sharp mass on target_choice
        crit_keys = list(shortlist_crit.keys())
        p_dist = {k: 0.01 for k in crit_keys}
        p_dist[target_choice] = 0.95
        tot = sum(p_dist.values())
        p_dist = {k: v / tot for k, v in p_dist.items()}
        p_dist[crit_keys[-1]] += 1.0 - sum(p_dist.values())

        q_shortlist = {
            "shortlist": {
                "type": "choice",
                "instructions": "Produce a broad shortlist of registered Skills relevant to the task. Choose the strongest candidate or none. The probabilities are shortlist relevance scores, not final injection decisions.",
                "criteria": shortlist_crit
            }
        }
        records.append(make_training_row(
            f"eval_shortlist_{cid}_{idx:04d}", state_dict, q_shortlist,
            {"shortlist": p_dist}, {"shortlist": target_choice}, split="dev"
        ))
        idx += 1

        # B. Verify requests (for both matching skill and non-matching skills)
        test_verify_skills = [target_choice] if target_choice != "none" else [cand_ids[0]]
        for v_skill in test_verify_skills:
            v_qid = f"verify_{v_skill.replace('-', '_')}"
            is_match = (v_skill == expected_skill)
            p_true = 0.96 if is_match else 0.02
            q_verify = {
                v_qid: {
                    "type": "boolean",
                    "instructions": f"Is the registered Skill '{v_skill}' directly relevant to the task and useful to inject? Return noul as the probability that the answer is yes.",
                    "criteria": {
                        "true": "The Skill is directly relevant and useful for this task.",
                        "false": "The Skill is not directly relevant or should not be injected."
                    }
                }
            }
            records.append(make_training_row(
                f"eval_verify_{cid}_{v_skill}_{idx:04d}", state_dict, q_verify,
                {v_qid: {"false": 1.0 - p_true, "true": p_true}}, {v_qid: is_match}, split="dev"
            ))
            idx += 1

    print(f"Built {len(records)} dev benchmark samples.")

    # 2. Build training samples from SKILL_TASK_BANK with permutations & variations
    prefixes = [
        "", "请帮我", "任务：", "请执行：", "当前操作：", "请处理：",
        "需求如下：", "请帮忙：", "需要实现：", "指令："
    ]

    all_registered = list(ALL_SKILL_ENTRIES)
    all_registered_ids = [e.id for e in all_registered]

    for skill_name, task_list in SKILL_TASK_BANK.items():
        for task in task_list:
            for p in random.sample(prefixes, 5):
                p_task = f"{p}{task}" if p else task

                # Form candidate sets:
                # 50% chance: currently enabled 6 skills
                # 50% chance: all 10 skills
                if random.random() < 0.5:
                    cand_entries = list(ENABLED_SKILL_ENTRIES)
                else:
                    cand_entries = list(ALL_SKILL_ENTRIES)

                cand_ids = [e.id for e in cand_entries]

                target = skill_name if skill_name in cand_ids else "none"

                state_dict = {
                    "task": p_task,
                    "trigger": "user_prompt",
                    "candidate_skills": [{"id": e.id, "description": e.description} for e in cand_entries],
                    "selection_budget": {"max_selected": 3, "max_injection_chars": 5000}
                }

                # A. Shortlist Query
                shortlist_crit = {"none": "No registered Skill is relevant enough for this task."}
                shortlist_crit.update({e.id: e.description for e in cand_entries})
                crit_keys = list(shortlist_crit.keys())
                p_dist = {k: 0.01 for k in crit_keys}
                p_dist[target] = 0.95
                tot = sum(p_dist.values())
                p_dist = {k: v / tot for k, v in p_dist.items()}
                p_dist[crit_keys[-1]] += 1.0 - sum(p_dist.values())

                q_shortlist = {
                    "shortlist": {
                        "type": "choice",
                        "instructions": "Produce a broad shortlist of registered Skills relevant to the task. Choose the strongest candidate or none. The probabilities are shortlist relevance scores, not final injection decisions.",
                        "criteria": shortlist_crit
                    }
                }
                records.append(make_training_row(
                    f"train_shortlist_{idx:05d}", state_dict, q_shortlist,
                    {"shortlist": p_dist}, {"shortlist": target}, split="train"
                ))
                idx += 1

                # B. Verify Queries (both positive and negative)
                # Sample 1 positive (if target != none) and 1 negative
                skills_to_verify = []
                if target != "none":
                    skills_to_verify.append((target, True))
                # Negative sample
                other_cands = [sid for sid in cand_ids if sid != target]
                if other_cands:
                    skills_to_verify.append((random.choice(other_cands), False))

                for v_skill, is_match in skills_to_verify:
                    v_qid = f"verify_{v_skill.replace('-', '_')}"
                    p_true = 0.96 if is_match else 0.02
                    q_verify = {
                        v_qid: {
                            "type": "boolean",
                            "instructions": f"Is the registered Skill '{v_skill}' directly relevant to the task and useful to inject? Return noul as the probability that the answer is yes.",
                            "criteria": {
                                "true": "The Skill is directly relevant and useful for this task.",
                                "false": "The Skill is not directly relevant or should not be injected."
                            }
                        }
                    }
                    records.append(make_training_row(
                        f"train_verify_{idx:05d}", state_dict, q_verify,
                        {v_qid: {"false": 1.0 - p_true, "true": p_true}}, {v_qid: is_match}, split="train"
                    ))
                    idx += 1

    random.shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cnt = sum(1 for r in records if r["split"] == "train")
    dev_cnt = sum(1 for r in records if r["split"] == "dev")
    print(f"Generated {len(records)} total skill selection samples ({train_cnt} train, {dev_cnt} dev) -> {output_path}")


if __name__ == "__main__":
    out_file = Path("data/skill_selection_dataset.jsonl")
    generate_skill_dataset(out_file)
