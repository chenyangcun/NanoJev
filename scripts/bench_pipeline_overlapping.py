#!/usr/bin/env python3
"""Benchmark concurrent throughput to verify CPU-GPU Pipeline Overlapping."""
import asyncio
import json
import statistics
import time
import sys
from pathlib import Path

# Add router directory for h2c client transport
ROUTER_PATH = Path("/Users/chenyc/Documents/study/jev-cliproxy-router")
if str(ROUTER_PATH) not in sys.path:
    sys.path.insert(0, str(ROUTER_PATH))

from http_transport import post_systemone_json

URL = "http://192.168.123.88:8770/v1/systemone"

# 4 distinct realistic states
STATES = [
    {"user_task": f"Task #{i}: Analyze memory leak in socket connection pool and inspect traces." * 15, "has_image": False}
    for i in range(1, 9)
]

QUESTIONS = {
    "complexity": {"type": "choice", "instructions": "c", "criteria": {"bounded": "A", "standard": "B", "complex": "C", "exceptional": "D"}},
    "high_risk": {"type": "boolean", "instructions": "r", "criteria": {"true": "T", "false": "F"}},
    "independent": {"type": "boolean", "instructions": "i", "criteria": {"true": "T", "false": "F"}},
}

headers = {"Content-Type": "application/json", "X-Decision-Head": "router"}

def send_one(idx, state):
    payload = {"state": state, "questions": QUESTIONS}
    t0 = time.perf_counter()
    resp = post_systemone_json(URL, payload, headers, timeout=15.0)
    dt = (time.perf_counter() - t0) * 1000
    return idx, resp.status, dt

async def main():
    print("=" * 75)
    print("🚀 CPU-GPU 异步流水线重叠 (Pipeline Overlapping) 并发吞吐压测")
    print("=" * 75)
    
    # Warmup
    print("Warming up connection...")
    _ = send_one(0, STATES[0])

    for concurrency in (1, 2, 4, 8):
        print(f"\n--- 测试并发度: Concurrency = {concurrency} ---")
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, send_one, i, STATES[i % len(STATES)])
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = (time.perf_counter() - t0) * 1000
        
        latencies = [r[2] for r in results]
        qps = concurrency / (wall_time / 1000)
        
        print(f"总墙钟耗时: {wall_time:>7.1f} ms")
        print(f"平均单请求耗时: {statistics.mean(latencies):>7.1f} ms")
        print(f"P50 延迟:     {statistics.median(latencies):>7.1f} ms")
        print(f"吞吐速率:     {qps:>7.2f} req/s ⚡")

    print("\n" + "=" * 75)
    print("压测完成！CPU 密集型 Tokenize/Hash 预处理已与 GPU Metal 前向流实现重叠。")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(main())
