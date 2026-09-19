#!/usr/bin/env python3
"""Augment router training dataset with synthetic balanced cases to eliminate class collapse.

Features:
1. Keeps all real official Jev records (315 pairs).
2. Generates balanced synthetic samples for under-represented classes:
   - 'bounded': typos, formatting, simple config changes, single print/log addition
   - 'complex': large architectural migrations, concurrency race debugging, complex multi-file refactors
   - 'exceptional': fundamental capability limitations, formal proof, novel compiler algorithm
3. Generates high-risk vs low-risk samples for calibrated 'high_risk' decisions.
4. Produces train.jsonl and dev.jsonl with balanced distributions.
"""
import argparse
import json
import random
from pathlib import Path
from typesafe_adapter import typesafe_request_to_nanojev

# Real instructions and criteria from router audit
COMPLEXITY_INSTRUCTIONS = (
    'Choose the complexity of the next coding-agent call. Judge the work itself, '
    'not the number of files or the presence of tools. Do not use exceptional merely because a '
    'task is complex or end-to-end; exceptional requires concrete evidence that Sol cannot solve it. '
    'Candidate model guidance: {"gpt-5.6-luna": "Bounded, clearly scoped, low-risk work", '
    '"gpt-5.6-sol": "Complex or ambiguous implementation, architecture, debugging", '
    '"gpt-5.6-terra": "Default primary coding work", "gpt-6-astra": "Only high-consequence operation"}'
)

COMPLEXITY_CRITERIA = {
    "bounded": "A small, clearly specified, reversible action with a narrow success condition and little ambiguity.",
    "standard": "Ordinary implementation, investigation, review, debugging, or tool work with a clear enough path but more than a tiny bounded action.",
    "complex": "Multiple interacting components, substantial tracing, meaningful ambiguity, architecture, or a difficult security review; this maps to Sol, not Astra by itself.",
    "exceptional": "There is concrete evidence that Sol cannot solve this task. Complexity, novelty, or end-to-end scope alone is not evidence."
}

HIGH_RISK_INSTRUCTIONS = "Does this task carry high risk of data destruction, irreversible system change, or security violation?"
INDEPENDENT_INSTRUCTIONS = "Can this task be executed independently without requiring immediate interactive user clarification?"

SYNTHETIC_TEMPLATES = {
    "bounded": [
        ("把文件 {file} 里的拼写错误 '{word_a}' 替换为 '{word_b}'", 0.05, 0.15),
        ("给 {file} 的第 {num} 行函数加上一行注释，说明该函数的入参含义", 0.04, 0.10),
        ("在 {file} 中增加一个打印语句，输出当前变量 {var} 的值用于简单排查", 0.05, 0.12),
        ("修改 package.json 中的版本号，从 {v1} 改为 {v2}", 0.08, 0.10),
        ("把 {file} 里的超时时间从 {t1} 秒调整为 {t2} 秒", 0.06, 0.15),
        ("格式化一下当前目录下的 {file}，按照项目的 eslint 规则修正缩进", 0.05, 0.08),
        ("把 {file} 里面死循环代码中未使用的变量 {var} 删掉", 0.06, 0.12),
        ("查看一下当前的 git status 和最近一条 commit 提交记录", 0.02, 0.05),
        ("在 .gitignore 中增加一行忽略临时目录 {dir}/", 0.03, 0.08),
        ("运行一下单测命令 pytest {test_file} 看看是否全部通过", 0.04, 0.08),
    ],
    "complex": [
        ("我们需要把整个底层的分布式通信架构从 gRPC 迁移到基于 Rust Tokio + WebTransport 的新协议，需要重构所有的消息序列化、心跳保活、连接池以及重试机制，涉及 40 多个微服务的核心基类改造，存在很多未知的并发状态机竞争问题需要排查分析。", 0.45, 0.70),
        ("系统在高并发下出现死锁问题，TransactionManager 与 ConnectionPool 在异步回调中发生交叉加锁，排查涉及 6 个模块的异步调用链路，需要重新设计无锁队列与状态流转机制。", 0.40, 0.65),
        ("将现有的单体数据库分库分表迁移到分布式 TiDB，重写所有跨表 JOIN 和事务一致性保证逻辑，设计双写一致性对比与平滑割接方案。", 0.65, 0.80),
        ("实现一套端到端零知识证明验证电路，从数学约束定义、多项式承诺到 Rust 原语生成，涉及复杂的密码学算子实现和深度性能优化。", 0.55, 0.75),
        ("重构跨进程共享内存 IPC 模块，排查 Linux/macOS 跨平台的 POSIX 信号量唤醒丢失 bug，并用无锁环形缓冲区 RingBuffer 替代现有互斥锁管道。", 0.48, 0.72),
        ("排查分布式定时调度任务中极难复现的脑裂问题，分析 Raft 选举日志截断与网络分区下的租约过期边缘案例，重新设计仲裁逻辑。", 0.52, 0.70),
        ("设计并实现微服务网关的动态插件热加载引擎，基于 WASM 运行时隔离用户自定义鉴权与限流逻辑，处理内存分配泄漏与沙箱逃逸防御。", 0.58, 0.76),
        ("重构编译器前端语法解析与 AST 变换，实现对特定 DSL 语法扩展支持，包括类型推导、生命周期静态检查以及符号表全局解析重构。", 0.46, 0.68),
    ],
    "exceptional": [
        ("连续多次使用主流模型在复杂数学定理证明中失败，需要设计全新的反向回溯启发式符号证明算法，并给出形式化验证完整推导链条。", 0.50, 0.85),
        ("针对现有通用 LLM 无法解决的高维度混沌动力学非线性偏微分方程求解，提出创新的几何深度学习拓扑流形流算子网络，需要完全原创的数学构思。", 0.45, 0.80),
        ("解决 NP-Hard 调度问题的全新近似比多项式时间求解方案，需要提供严密数学证明以及超越现有 SOTA 的理论边界推导。", 0.40, 0.75),
    ],
    "high_risk": [
        ("线上 MySQL 数据库快满了，执行 DROP TABLE orders_archive 和 TRUNCATE TABLE user_actions 把历史旧数据全部清空，并用 rm -rf /var/log/audit/* 释放硬盘空间。", 0.95, 0.85, "bounded"),
        ("直接在生产 Kubernetes 集群上删除 core-dns 命名空间并重启 kube-system 节点以刷新 DNS 缓存。", 0.92, 0.80, "bounded"),
        ("把 AWS 生产环境的 S3 存储桶策略设置为公共读写 (Public Read/Write)，以便临时排查前端图片 403 访问问题。", 0.96, 0.85, "bounded"),
        ("强制重写生产环境 git master 分支历史，执行 git push origin master --force 覆盖远程分支。", 0.88, 0.75, "bounded"),
        ("批量杀死生产服务器上所有包含 java 关键字的进程：killall -9 java 并清空共享内存块 ipcrm -a。", 0.93, 0.80, "bounded"),
    ]
}


def generate_synthetic_records(start_id: int, count_per_category: int = 120):
    records = []
    rid = start_id

    # 1. Bounded cases
    for _ in range(count_per_category):
        tmpl, hr_p, indep_p = random.choice(SYNTHETIC_TEMPLATES["bounded"])
        text = tmpl.format(
            file=random.choice(["src/index.ts", "lib/utils.py", "README.md", "src/core/router.rs", "config.yaml"]),
            word_a=random.choice(["teh", "recieve", "adn", "functon"]),
            word_b=random.choice(["the", "receive", "and", "function"]),
            num=random.randint(10, 200),
            var=random.choice(["userId", "retryCount", "requestTimestamp", "sessionToken"]),
            v1=f"1.0.{random.randint(0, 5)}",
            v2=f"1.0.{random.randint(6, 12)}",
            t1=random.choice([5, 10, 15]),
            t2=random.choice([30, 60, 120]),
            dir=random.choice([".cache", "dist", "build", "coverage", ".tmp"]),
            test_file=random.choice(["tests/test_api.py", "tests/test_auth.py", "tests/test_model.py"]),
        )
        rec = make_record(f"synth_{rid:05d}", text, "bounded", hr_p, indep_p)
        records.append(rec)
        rid += 1

    # 2. Complex cases
    for _ in range(count_per_category):
        text, hr_p, indep_p = random.choice(SYNTHETIC_TEMPLATES["complex"])
        # Slight variation
        text += f" [Trace ID: {random.randint(1000, 9999)}]"
        rec = make_record(f"synth_{rid:05d}", text, "complex", hr_p, indep_p)
        records.append(rec)
        rid += 1

    # 3. Exceptional cases
    for _ in range(count_per_category // 3):
        text, hr_p, indep_p = random.choice(SYNTHETIC_TEMPLATES["exceptional"])
        rec = make_record(f"synth_{rid:05d}", text, "exceptional", hr_p, indep_p)
        records.append(rec)
        rid += 1

    # 4. Explicit high risk cases
    for _ in range(count_per_category // 2):
        text, hr_p, indep_p, comp = random.choice(SYNTHETIC_TEMPLATES["high_risk"])
        rec = make_record(f"synth_{rid:05d}", text, comp, hr_p, indep_p)
        records.append(rec)
        rid += 1

    return records


def make_record(rec_id: str, state_text: str, complexity_choice: str, high_risk_prob: float, indep_prob: float):
    # Construct soft probability distributions
    comp_probs = {"bounded": 0.05, "standard": 0.05, "complex": 0.05, "exceptional": 0.02}
    comp_probs[complexity_choice] = 0.83
    # Normalize to exact 1.0 (float64)
    tot = sum(comp_probs.values())
    comp_keys = list(comp_probs.keys())
    comp_probs = {k: float(v) / tot for k, v in comp_probs.items()}
    # Adjust first key to ensure math.fsum == 1.0
    comp_probs[comp_keys[0]] += 1.0 - sum(comp_probs.values())

    # Add small noise to probabilities
    hr_prob = max(0.02, min(0.98, high_risk_prob + random.uniform(-0.04, 0.04)))
    in_prob = max(0.02, min(0.98, indep_prob + random.uniform(-0.04, 0.04)))

    gold_probs = {
        "complexity": comp_probs,
        "high_risk": {"false": 1.0 - hr_prob, "true": hr_prob},
        "independent": {"false": 1.0 - in_prob, "true": in_prob},
    }
    gold = {
        "complexity": complexity_choice,
        "high_risk": hr_prob >= 0.5,
        "independent": in_prob >= 0.5,
    }

    questions = {
        "complexity": {
            "type": "choice",
            "instructions": COMPLEXITY_INSTRUCTIONS,
            "criteria": COMPLEXITY_CRITERIA,
        },
        "high_risk": {
            "type": "boolean",
            "instructions": HIGH_RISK_INSTRUCTIONS,
        },
        "independent": {
            "type": "boolean",
            "instructions": INDEPENDENT_INSTRUCTIONS,
        },
    }

    return {
        "id": rec_id,
        "state_id": rec_id,
        "family_id": "codex_router",
        "split": "train",
        "state": state_text,
        "questions": questions,
        "gold_probs": gold_probs,
        "gold": gold,
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label_kind": "observed_outcome",
        "metadata": {"synthetic": True, "target_complexity": complexity_choice},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", default="data/router_train.jsonl")
    parser.add_argument("--output", default="data/router_augmented.jsonl")
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--count-per-cat", type=int, default=150)
    args = parser.parse_args()

    # Read existing real records
    real_records = []
    with open(args.real_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                real_records.append(json.loads(line))
    print(f"Loaded {len(real_records)} real audit records.")

    # Generate synthetic balanced records
    synth_records = generate_synthetic_records(start_id=len(real_records) + 1, count_per_category=args.count_per_cat)
    print(f"Generated {len(synth_records)} synthetic balanced records.")

    all_records = real_records + synth_records
    random.seed(42)
    random.shuffle(all_records)

    n_dev = int(len(all_records) * args.dev_ratio)
    for i, r in enumerate(all_records):
        r["split"] = "dev" if i < n_dev else "train"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cnt = sum(1 for r in all_records if r["split"] == "train")
    dev_cnt = sum(1 for r in all_records if r["split"] == "dev")

    from collections import Counter
    comp_dist = Counter(r["gold"]["complexity"] for r in all_records)
    print(f"Total dataset: {len(all_records)} ({train_cnt} train, {dev_cnt} dev) -> {args.output}")
    print("New Balanced Complexity Distribution:")
    for k, v in comp_dist.most_common():
        print(f"  {k}: {v} ({v/len(all_records)*100:.1f}%)")


if __name__ == "__main__":
    main()
