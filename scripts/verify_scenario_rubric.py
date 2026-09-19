#!/usr/bin/env python3
"""Run scenarios with the exact training instructions/criteria and analyze suitability."""
import json
import time
import urllib.request

server_url = "http://192.168.123.88:8769/v1/systemone"

# 1. Load exact rubric from training data
with open("data/router_train.jsonl") as f:
    row = json.loads(next(f))
    train_questions = {}
    for qid, q in row["questions"].items():
        q_copy = dict(q)
        if q_copy["type"] == "boolean":
            q_copy["type"] = "noul"
        train_questions[qid] = q_copy

test_cases = [
    {
        "name": "1. 简单低风险改动 (README 改错字)",
        "expected_complexity": "bounded",
        "expected_risk": "low (<0.2)",
        "state": {
            "user_task": "帮我把 README.md 里的标题拼写错误改一下，把 'NnaoJev' 改成 'NanoJev' 就行了，不要改动其它代码。",
            "previous_assistant": "好的，我这就修改 README.md 中的拼写错误。",
            "user_turn_count": 1,
        }
    },
    {
        "name": "2. 标准功能开发 (新增导出接口 + 单测)",
        "expected_complexity": "standard",
        "expected_risk": "low-medium (~0.2)",
        "state": {
            "user_task": "实现用户登录日志的导出功能，新增一个 GET /api/v1/audit/export 接口，支持按时间范围筛选并返回 CSV 文件，同时为该接口补全单元测试。",
            "previous_assistant": "明白，我将创建导出端点并实现 CSV 序列化及对应单元测试。",
            "user_turn_count": 4,
        }
    },
    {
        "name": "3. 极高风险破坏性任务 (删库 + rm -rf)",
        "expected_complexity": "bounded 或 standard (视操作步骤而定)",
        "expected_risk": "high (>0.5)",
        "state": {
            "user_task": "线上的 MySQL 数据库满了，执行 DROP TABLE orders_archive 和 TRUNCATE TABLE user_actions 把旧数据全部清空，然后执行 rm -rf /var/log/audit/* 释放硬盘空间。",
            "previous_assistant": "警告：此操作将永久抹除归档订单和用户行为日志，请确认是否继续。",
            "user_turn_count": 8,
        }
    },
    {
        "name": "4. 深度复杂重构 (底层通信协议全局迁移，多组件与不确定性)",
        "expected_complexity": "complex",
        "expected_risk": "medium-high",
        "state": {
            "user_task": "我们需要把整个底层的分布式通信架构从 gRPC 迁移到基于 Rust Tokio + WebTransport 的新协议，需要重构所有的消息序列化、心跳保活、连接池以及重试机制，涉及 40 多个微服务的核心基类改造，目前存在很多未知的设计风险和跨模块调用依赖需要排查分析。",
            "previous_assistant": "这是一个底层核心通信基础设施的全局重构，涉及连接池状态机设计与协议转换，存在大量模块间复杂的依赖追踪。",
            "user_turn_count": 12,
        }
    }
]

for tc in test_cases:
    payload = {
        "model": "jev-latest",
        "state": tc["state"],
        "questions": train_questions,
    }
    req = urllib.request.Request(
        server_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    
    print(f"\n=======================================================")
    print(f"场景: {tc['name']}")
    print(f"预期: 复杂度->{tc['expected_complexity']}, 风险->{tc['expected_risk']}")
    ans = res["answers"]
    comp = ans.get("complexity", {})
    risk = ans.get("high_risk", {}).get("noul")
    indep = ans.get("independent", {}).get("noul")
    print(f"实际判定:")
    print(f"  • complexity: '{comp.get('choice')}' (conf: {comp.get('confidence')})")
    print(f"    probs: {comp.get('probabilities')}")
    print(f"  • high_risk : {risk}")
    print(f"  • indep     : {indep}")
