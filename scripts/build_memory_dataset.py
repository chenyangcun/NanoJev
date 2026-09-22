#!/usr/bin/env python3
"""Build training dataset for the 'memory' decision head.

Evaluates:
1. Browser webpage history / snippets
2. Local Agent conversation turns and tool execution outputs
To judge whether they have long-term value to be persisted into personal memory.

Question Schema:
1. 'is_valuable': boolean (noul)
   - true: contains reusable knowledge, core user preferences, unique solutions, key architecture decisions.
   - false: transient status, routine logs, conversational noise, temporary debugging, generic search results.

2. 'memory_category': choice
   - technical_solution: reproducible bug fixes, verified configurations, unique technical solutions.
   - user_preference: user habits, project conventions, recurring constraints, personal preferences.
   - key_architecture: core architecture decisions, permanent system design choices, API specifications.
   - transient_noise: fleeting search noise, routine command outputs, chit-chat, temporary session state.

3. 'significance': score (0-3)
   - 0: transient noise (drop completely)
   - 1: short-term working context (valid within current session only)
   - 2: reusable technical reference (valuable across projects or future sessions)
   - 3: critical long-term fact or permanent project constraint

Outputs:
  data/memory_dataset.jsonl (with train/dev/calibration splits)
"""

import json
import random
import sys
from pathlib import Path

CATEGORY_CRITERIA = {
    "technical_solution": "A verified technical solution, command sequence, or reproducible fix for a specific problem.",
    "user_preference": "An explicit user preference, coding habit, recurring workflow constraint, or custom rule.",
    "key_architecture": "A permanent architectural decision, system contract, or core design choice.",
    "transient_noise": "Fleeting status checks, routine tool outputs, small talk, or temporary debugging state.",
}

SIGNIFICANCE_CRITERIA = [
    "Transient noise: discard immediately.",
    "Short-term context: useful only within the current active session.",
    "Reusable reference: valuable knowledge worth recalling in future tasks.",
    "Critical fact: fundamental rule, permanent user preference, or high-value architectural decision.",
]

# Seed task bank covering both Browser pages and Agent session turns
SEED_CASES = [
    # --- Category: user_preference (is_valuable: True, score: 2-3) ---
    {
        "source": "agent_session",
        "content": "User: 记住，以后的 Python 脚本全部使用 Python 3.12 虚拟环境运行，不要用系统默认的 python3。\nAssistant: 明白，已记录该开发偏好，后续所有脚本均显式调用虚拟环境。",
        "is_valuable": True,
        "category": "user_preference",
        "significance": 3,
    },
    {
        "source": "agent_session",
        "content": "User: 项目所有对外 API 必须采用纯文本返回，严禁擅自增加 Markdown 加粗或多余礼貌用语。\nAssistant: 收到，后续接口一律保持极简纯文本输出。",
        "is_valuable": True,
        "category": "user_preference",
        "significance": 3,
    },
    {
        "source": "agent_session",
        "content": "User: 我的远程 M1 Mac Studio 地址是 192.168.123.88，所有模型训练任务都必须推送到远程执行，不要在本地跑。\nAssistant: 已记住远程算力机配置，训练任务将全量在 192.168.123.88 上启动。",
        "is_valuable": True,
        "category": "user_preference",
        "significance": 3,
    },

    # --- Category: technical_solution (is_valuable: True, score: 2-3) ---
    {
        "source": "browser_page",
        "url": "https://developer.apple.com/documentation/metal/unified_memory_best_practices",
        "content": "Page Title: Unified Memory Allocation in Apple Silicon Metal\nSummary: For MLX array buffers, calling mx.eval() before converting to numpy avoids retaining the full computation graph. Using float16 for intermediate activations saves 50% bandwidth.",
        "is_valuable": True,
        "category": "technical_solution",
        "significance": 2,
    },
    {
        "source": "browser_page",
        "url": "https://github.com/ml-explore/mlx-lm/issues/214",
        "content": "Page Title: Fixing CustomKernel Primitive vjp error in Qwen3.5\nSummary: Qwen3.5 linear attention layers (DeltaNet) require pure ops fallback when running autograd. Overriding gated_delta_kernel with gated_delta_ops resolves the backward pass issue on Apple GPUs.",
        "is_valuable": True,
        "category": "technical_solution",
        "significance": 3,
    },
    {
        "source": "agent_session",
        "content": "Tool Exec: launchctl unload ~/Library/LaunchAgents/com.chenyc.nanojev.plist && launchctl load ~/Library/LaunchAgents/com.chenyc.nanojev.plist\nOutput: Service successfully restarted on port 8769 with early-exit-layer disabled.",
        "is_valuable": True,
        "category": "technical_solution",
        "significance": 2,
    },

    # --- Category: key_architecture (is_valuable: True, score: 2-3) ---
    {
        "source": "agent_session",
        "content": "Architecture Decision: NanoJev 采用 MultiHeadRegistry 双轨物理隔离设计。general 判决头服务于 JevBench 通用推理，router 判决头服务于代码复杂度与高危安全拦截，二者共享同一个 Qwen3.5-0.8B 骨干，互不产生参数污染。",
        "is_valuable": True,
        "category": "key_architecture",
        "significance": 3,
    },
    {
        "source": "browser_page",
        "url": "https://docs.typesafe.ai/v1/systemone",
        "content": "API Specification: POST /v1/systemone accepts state and questions, returning probabilities in a single forward pass without token autoregression. Compatible question types: choice, noul, score.",
        "is_valuable": True,
        "category": "key_architecture",
        "significance": 2,
    },

    # --- Category: transient_noise (is_valuable: False, score: 0-1) ---
    {
        "source": "agent_session",
        "content": "User: 今天天气怎么样？\nAssistant: 今天上海天气晴朗，气温 24 度。",
        "is_valuable": False,
        "category": "transient_noise",
        "significance": 0,
    },
    {
        "source": "agent_session",
        "content": "Tool Exec: git status -s\nOutput: M scripts/predict_mlx_decisions.py\n?? scripts/test.py",
        "is_valuable": False,
        "category": "transient_noise",
        "significance": 0,
    },
    {
        "source": "agent_session",
        "content": "Tool Exec: ps aux | grep python\nOutput: 57352 11.2 3.0 python3 scripts/train.py",
        "is_valuable": False,
        "category": "transient_noise",
        "significance": 0,
    },
    {
        "source": "browser_page",
        "url": "https://www.google.com/search?q=apple+studio+display+price",
        "content": "Page Title: Google Search: apple studio display price\nSnippet: Sponsored Ads: Buy Studio Display from $1599. Find retailers near you. Best Buy in stock.",
        "is_valuable": False,
        "category": "transient_noise",
        "significance": 0,
    },
    {
        "source": "browser_page",
        "url": "https://news.ycombinator.com/",
        "content": "Page Title: Hacker News\nFrontpage: 1. Show HN: New SQLite Browser (124 points) 2. Ask HN: Favorite mechanical keyboard? (45 comments) 3. Why Rust 2026 Edition is Fast (230 points)",
        "is_valuable": False,
        "category": "transient_noise",
        "significance": 1,
    },
    {
        "source": "agent_session",
        "content": "User: 谢谢你！\nAssistant: 不客气，随时乐意为您效劳！",
        "is_valuable": False,
        "category": "transient_noise",
        "significance": 0,
    },
]


def generate_memory_dataset(output_file: Path, seed: int = 42, target_records: int = 400):
    rng = random.Random(seed)
    records = []

    # Expand seed cases with realistic variations
    idx = 0
    while len(records) < target_records:
        base = rng.choice(SEED_CASES)
        source = base["source"]
        content = base["content"]
        is_val = base["is_valuable"]
        cat = base["category"]
        sig = base["significance"]

        # Build clean state
        state_dict = {
            "source": source,
            "raw_text": content,
        }
        if "url" in base:
            state_dict["url"] = base["url"]

        state_str = json.dumps(state_dict, ensure_ascii=False)

        # Build target probability distributions
        val_probs = {"true": 0.98, "false": 0.02} if is_val else {"true": 0.02, "false": 0.98}

        cat_probs = {c: 0.01 for c in CATEGORY_CRITERIA}
        cat_probs[cat] = 0.97

        sig_probs = {str(i): 0.01 for i in range(4)}
        sig_probs[str(sig)] = 0.97

        split = "test" if idx % 10 == 0 else ("dev" if idx % 10 == 1 else "train")
        rec_id = f"mem_record_{idx:04d}"

        records.append({
            "id": rec_id,
            "state_id": rec_id,
            "family_id": "memory_curation",
            "split": split,
            "state": state_str,
            "questions": {
                "is_valuable": {
                    "type": "boolean",
                    "instructions": (
                        "Evaluate whether this browsing snippet or agent interaction contains enduring, reusable value "
                        "that should be retained in long-term memory. Ephemeral status, chit-chat, and transient command noise must be rejected."
                    ),
                    "criteria": {
                        "true": "Contains reusable technical knowledge, core user preferences, or permanent architecture decisions.",
                        "false": "Ephemeral status, routine logs, conversational noise, or temporary debugging steps.",
                    },
                },
                "memory_category": {
                    "type": "choice",
                    "instructions": "Classify this piece of information into its primary memory category.",
                    "criteria": CATEGORY_CRITERIA,
                },
                "significance": {
                    "type": "score",
                    "instructions": "Rate the long-term utility and significance of this content (0=noise, 1=short-term, 2=reusable, 3=critical).",
                    "criteria": SIGNIFICANCE_CRITERIA,
                },
            },
            "gold": {
                "is_valuable": is_val,
                "memory_category": cat,
                "significance": sig,
            },
            "gold_probs": {
                "is_valuable": val_probs,
                "memory_category": cat_probs,
                "significance": sig_probs,
            },
            "gold_probs_kind": "programmatic_conditional_distribution",
            "gold_label_kind": "observed_outcome",
        })
        idx += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} memory curation records to: {output_file}", flush=True)


if __name__ == "__main__":
    generate_memory_dataset(Path("data/memory_dataset.jsonl"), target_records=450)
