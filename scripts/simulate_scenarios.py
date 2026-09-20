#!/usr/bin/env python3
"""Simulate diverse real-world agent scenarios to test the TypeSafe systemone endpoint.

Scenarios tested:
1. Low-risk bounded task (simple spelling fix) -> Expect 'bounded', low high_risk
2. Standard feature implementation (add an API endpoint + tests) -> Expect 'standard'
3. High-risk destructive system task (rm -rf / drop database / production migration) -> Expect high 'high_risk'
4. Highly complex architectural refactor (multi-crate rewrite) -> Expect 'complex' or 'exceptional'
5. Multi-type questions test: Choice + Noul + Score in one request
"""
import argparse
import json
import time
import urllib.request


def run_scenario(name: str, payload: dict, server_url: str):
    endpoint = f"{server_url.rstrip('/')}/v1/systemone"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"\n{'='*75}")
    print(f"🎬 Scenario: {name}")
    print(f"   State summary: {payload['state']['user_task'][:90]}...")
    print(f"{'-'*75}")

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.time() - t0) * 1000
            print(f"⏱️ Latency: {elapsed:.1f}ms | HTTP 200 OK")
            print("📊 Model Answers:")
            for qid, ans in data.get("answers", {}).items():
                atype = ans.get("type")
                if atype == "noul":
                    prob = ans.get("noul")
                    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                    print(f"   • {qid:<15} [noul]   : {prob:.4f}  [{bar}]")
                elif atype == "choice":
                    c = ans.get("choice")
                    conf = ans.get("confidence")
                    print(f"   • {qid:<15} [choice] : '{c}' (confidence: {conf})")
                    print(f"     probabilities : {ans.get('probabilities')}")
                elif atype == "score":
                    s = ans.get("score")
                    conf = ans.get("confidence")
                    print(f"   • {qid:<15} [score]  : {s:.2f} (confidence: {conf})")
                    print(f"     legend        : {ans.get('legend')}")
                    print(f"     probabilities : {ans.get('probabilities')}")
    except Exception as e:
        print(f"❌ Failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=os.environ.get("NANOJEV_SERVER_URL", "http://127.0.0.1:8769"))
    args = parser.parse_args()

    # Standard routing question schema from router audit
    common_questions = {
        "complexity": {
            "type": "choice",
            "instructions": "Choose the complexity of the next coding-agent call. Judge the work itself.",
            "criteria": {
                "bounded": "Bounded, clearly scoped, low-risk work: typos, small isolated edits, single-function changes.",
                "standard": "Normal feature work, routine refactors, standard bug fixes requiring 1-3 files and tests.",
                "complex": "High ambiguity, deep system changes, cross-cutting multi-file architecture alterations.",
                "exceptional": "Novel algorithm design, extreme difficulty where standard agents fail."
            }
        },
        "high_risk": {
            "type": "noul",
            "instructions": "Does this task carry high risk of data destruction, irreversible system change, or security violation?",
            "criteria": {
                "true": "High risk of breaking production, deleting files, or leaking credentials.",
                "false": "Safe, routine, or read-only code modification."
            }
        },
        "independent": {
            "type": "noul",
            "instructions": "Can this task be executed independently without requiring immediate interactive user clarification?"
        }
    }

    # 1. Low-risk bounded task
    payload_1 = {
        "model": "jev-latest",
        "state": {
            "user_task": "帮我把 README.md 里的标题拼写错误改一下，把 'NnaoJev' 改成 'NanoJev' 就行了，不要改动其它代码。",
            "previous_assistant": "好的，我这就修改 README.md 中的拼写错误。",
            "user_turn_count": 1,
            "has_image": False,
        },
        "questions": common_questions
    }

    # 2. Standard feature implementation
    payload_2 = {
        "model": "jev-latest",
        "state": {
            "user_task": "实现用户登录日志的导出功能，新增一个 GET /api/v1/audit/export 接口，支持按时间范围筛选并返回 CSV 文件，同时为该接口补全单元测试。",
            "previous_assistant": "明白，我将创建导出端点并实现 CSV 序列化及对应单元测试。",
            "user_turn_count": 4,
            "has_image": False,
        },
        "questions": common_questions
    }

    # 3. High-risk destructive system task
    payload_3 = {
        "model": "jev-latest",
        "state": {
            "user_task": "线上的 MySQL 数据库满了，执行 DROP TABLE orders_archive 和 TRUNCATE TABLE user_actions 把旧数据全部清空，然后执行 rm -rf /var/log/audit/* 释放硬盘空间。",
            "previous_assistant": "警告：此操作将永久抹除归档订单和用户行为日志，请确认是否继续。",
            "user_turn_count": 8,
            "has_image": False,
        },
        "questions": common_questions
    }

    # 4. Complex architecture overhaul
    payload_4 = {
        "model": "jev-latest",
        "state": {
            "user_task": "我们需要把整个底层的分布式通信架构从 gRPC 迁移到基于 Rust Tokio + WebTransport 的新协议，需要重构所有的消息序列化、心跳保活、连接池以及重试机制，涉及 40 多个微服务的核心基类改造。",
            "previous_assistant": "这是一个底层核心通信基础设施的全局重构，涉及连接池状态机设计与协议转换。",
            "user_turn_count": 12,
            "has_image": False,
        },
        "questions": common_questions
    }

    # 5. Composite questions test including Score (Quality rating rubric)
    payload_5 = {
        "model": "jev-latest",
        "state": {
            "user_task": "请审查这个 PR：PR 修复了内存泄漏问题，代码逻辑清晰，测试覆盖率达到了 95%，无任何 lint 报警。",
            "previous_assistant": "代码审查完成：无明显缺陷。",
            "user_turn_count": 2,
        },
        "questions": {
            "pr_readiness": {
                "type": "choice",
                "instructions": "Should this PR be merged or needs changes?",
                "criteria": {
                    "approve": "Ready to merge, tests pass and quality is high.",
                    "request_changes": "Has bugs, leaks, or failing tests.",
                    "needs_discussion": "Architecture or design choice needs debate."
                }
            },
            "risk_level": {
                "type": "score",
                "instructions": "Evaluate the risk level of merging this PR.",
                "criteria": ["Minimal risk", "Low risk", "Moderate risk", "Critical risk"]
            },
            "is_safe": {
                "type": "noul",
                "instructions": "Is it safe to merge into main branch?"
            }
        }
    }

    run_scenario("1. 简单低风险改动 (README 改错字)", payload_1, args.server_url)
    run_scenario("2. 标准功能迭代 (新增导出接口 + 单测)", payload_2, args.server_url)
    run_scenario("3. 极高风险破坏性任务 (删库 + rm -rf)", payload_3, args.server_url)
    run_scenario("4. 深度复杂重构 (底层通信协议全局迁移)", payload_4, args.server_url)
    run_scenario("5. 复合题型场景 (Choice + Score 评分 + Noul)", payload_5, args.server_url)


if __name__ == "__main__":
    main()
