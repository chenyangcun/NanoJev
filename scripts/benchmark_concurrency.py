#!/usr/bin/env python3
"""Concurrency & Multiplexing Benchmark: HTTP/2 (Multiplexed) vs HTTP/1.1 (Pipelining / Sequential).

Simulates high-load scenarios sending 20 concurrent requests simultaneously.
"""
import asyncio
import json
import statistics
import time
import httpx

URL = "http://192.168.123.88:8769/v1/systemone"

SAMPLE_PAYLOAD = {
    "model": "jev-latest",
    "state": {
        "user_task": "修正 README.md 中的一个错别字，不修改任何代码，也不需要运行测试。",
        "has_image": False,
        "user_turn_count": 1,
        "is_new_user_turn": True,
    },
    "questions": {
        "complexity": {
            "type": "choice",
            "instructions": "Choose complexity.",
            "criteria": {"bounded": "small", "complex": "large"},
        }
    },
}


async def test_concurrent_h2(total_reqs=20):
    async with httpx.AsyncClient(http2=True, timeout=15) as client:
        t0 = time.perf_counter()
        tasks = [client.post(URL, json=SAMPLE_PAYLOAD) for _ in range(total_reqs)]
        responses = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - t0) * 1000
        statuses = [r.status_code for r in responses]
    return total_time, statuses


async def test_concurrent_h1(total_reqs=20):
    async with httpx.AsyncClient(http2=False, timeout=15) as client:
        t0 = time.perf_counter()
        tasks = [client.post(URL, json=SAMPLE_PAYLOAD) for _ in range(total_reqs)]
        responses = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - t0) * 1000
        statuses = [r.status_code for r in responses]
    return total_time, statuses


async def main():
    print(f"Running Concurrency Benchmark (20 simultaneous requests) against {URL}...\n")

    # Warmup
    async with httpx.AsyncClient() as c:
        await c.get("http://192.168.123.88:8769/api/health")

    n = 20
    print(f"1. Testing 20 Concurrent Requests via HTTP/1.1 Pool...")
    t_h1, st_h1 = await test_concurrent_h1(n)
    print(f"   HTTP/1.1 Total Elapsed: {t_h1:.1f}ms (Throughput: {n/(t_h1/1000):.1f} req/s)")

    print(f"2. Testing 20 Concurrent Requests via HTTP/2 Multiplexing...")
    t_h2, st_h2 = await test_concurrent_h2(n)
    print(f"   HTTP/2.0 Total Elapsed: {t_h2:.1f}ms (Throughput: {n/(t_h2/1000):.1f} req/s)")

    print("\n" + "=" * 65)
    print(f"⚡ 高并发压测对比 (20 请求同时到达)")
    print("=" * 65)
    print(f"• HTTP/1.1 并发池总耗时 : {t_h1:.1f} ms")
    print(f"• HTTP/2.0 单连接复用总耗时: {t_h2:.1f} ms")
    print(f"• 吞吐性能收益          : {t_h1 / t_h2:.2f}x 吞吐提升 🚀")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
