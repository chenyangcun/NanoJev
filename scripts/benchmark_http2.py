#!/usr/bin/env python3
"""Benchmark HTTP/2.0 (h2c) vs HTTP/1.1 on NanoJev Server (192.168.123.88:8769).

Compares:
1. TCP Short Connection (HTTP/1.1 no keep-alive: repeated connect + disconnect)
2. HTTP/1.1 Keep-Alive Connection Pool
3. Native HTTP/2.0 Multiplexed Connection (single socket, binary stream framing)
"""
import json
import socket
import statistics
import time
import urllib.request
import h2.config
import h2.connection
import h2.events


HOST = "192.168.123.88"
PORT = 8769
URL = f"http://{HOST}:{PORT}/v1/systemone"

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
            "instructions": "Choose the complexity of the next coding-agent call.",
            "criteria": {
                "bounded": "Small isolated fix",
                "standard": "Normal feature work",
                "complex": "Architecture migration",
                "exceptional": "Production incident",
            },
        },
        "high_risk": {
            "type": "noul",
            "instructions": "Does this task carry high risk?",
        },
    },
}

BODY_BYTES = json.dumps(SAMPLE_PAYLOAD).encode("utf-8")


def bench_http1_short(n_calls=15):
    """Scenario 1: HTTP/1.1 with connection closed after each request."""
    latencies = []
    for _ in range(n_calls):
        req = urllib.request.Request(
            URL,
            data=BODY_BYTES,
            headers={
                "Content-Type": "application/json",
                "Connection": "close",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def bench_http1_keepalive(n_calls=15):
    """Scenario 2: HTTP/1.1 with persistent TCP Keep-Alive connection."""
    import http.client
    conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
    latencies = []
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    for _ in range(n_calls):
        t0 = time.perf_counter()
        conn.request("POST", "/v1/systemone", body=BODY_BYTES, headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        latencies.append((time.perf_counter() - t0) * 1000)
    conn.close()
    return latencies


def bench_http2_native(n_calls=15):
    """Scenario 3: Native binary HTTP/2.0 stream over single persistent socket."""
    sock = socket.create_connection((HOST, PORT))
    config = h2.config.H2Configuration(client_side=True)
    conn = h2.connection.H2Connection(config=config)
    conn.initiate_connection()
    sock.sendall(conn.data_to_send())

    latencies = []
    stream_id = 1

    for _ in range(n_calls):
        t0 = time.perf_counter()
        headers = [
            (":method", "POST"),
            (":authority", f"{HOST}:{PORT}"),
            (":scheme", "http"),
            (":path", "/v1/systemone"),
            ("content-type", "application/json"),
            ("content-length", str(len(BODY_BYTES))),
        ]
        conn.send_headers(stream_id=stream_id, headers=headers, end_stream=False)
        conn.send_data(stream_id=stream_id, data=BODY_BYTES, end_stream=True)
        sock.sendall(conn.data_to_send())

        # Receive response stream
        resp_body = bytearray()
        stream_ended = False
        while not stream_ended:
            data = sock.recv(65535)
            if not data:
                break
            events = conn.receive_data(data)
            for event in events:
                if isinstance(event, h2.events.DataReceived):
                    resp_body.extend(event.data)
                    conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                elif isinstance(event, h2.events.StreamEnded):
                    if event.stream_id == stream_id:
                        stream_ended = True
                        break

        latencies.append((time.perf_counter() - t0) * 1000)
        stream_id += 2  # Client initiated stream IDs must be odd numbers

    sock.close()
    return latencies


def main():
    print(f"Running End-to-End Speed Benchmark against {URL}...\n")
    print("Warmup calls...")
    _ = bench_http1_short(2)

    n = 20
    print(f"1. Testing HTTP/1.1 Short Connection (Connection: close, {n} calls)...")
    h1_short = bench_http1_short(n)

    print(f"2. Testing HTTP/1.1 Persistent Keep-Alive Connection ({n} calls)...")
    h1_keep = bench_http1_keepalive(n)

    print(f"3. Testing Native HTTP/2.0 Binary Stream Multiplexing ({n} calls)...")
    h2_stream = bench_http2_native(n)

    print("\n" + "=" * 75)
    print("🚀 HTTP 网络协议与传输层速度基准对比报告")
    print("=" * 75)
    print(f"{'协议与连接方式':<30} | {'中位延迟':<10} | {'平均延迟':<10} | {'P95 延迟':<10} | {'相对提速':<8}")
    print("-" * 75)

    med_short = statistics.median(h1_short)
    med_keep = statistics.median(h1_keep)
    med_h2 = statistics.median(h2_stream)

    print(f"{'HTTP/1.1 (短连接重连)':<30} | {med_short:<8.1f}ms | {statistics.mean(h1_short):<8.1f}ms | {sorted(h1_short)[int(n*0.95)]:<8.1f}ms | {'基准 (1.0x)':<8}")
    print(f"{'HTTP/1.1 (Keep-Alive 长连接)':<30} | {med_keep:<8.1f}ms | {statistics.mean(h1_keep):<8.1f}ms | {sorted(h1_keep)[int(n*0.95)]:<8.1f}ms | {f'{med_short/med_keep:.2f}x':<8}")
    print(f"{'HTTP/2.0 (原生二进制流复用)':<30} | {med_h2:<8.1f}ms | {statistics.mean(h2_stream):<8.1f}ms | {sorted(h2_stream)[int(n*0.95)]:<8.1f}ms | {f'{med_short/med_h2:.2f}x ⚡':<8}")
    print("=" * 75)


if __name__ == "__main__":
    main()
