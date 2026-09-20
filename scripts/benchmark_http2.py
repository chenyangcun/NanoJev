#!/usr/bin/env python3
"""Benchmark HTTP/2.0 (h2c) vs HTTP/1.1 on NanoJev Server.

Compares:
1. TCP Short Connection (HTTP/1.1 no keep-alive: repeated connect + disconnect)
2. HTTP/1.1 Keep-Alive Connection Pool
3. Native HTTP/2.0 Multiplexed Connection (single socket, binary stream framing)
"""
import argparse
import json
import os
import socket
import statistics
import time
import urllib.request
import h2.config
import h2.connection
import h2.events


DEFAULT_HOST = os.environ.get("NANOJEV_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("NANOJEV_PORT", "8769"))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--calls", type=int, default=20)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/systemone"
    print(f"Running End-to-End Speed Benchmark against {url}...\n")
    print("Warmup calls...")

    # helper with args
    def run_h1_short(n):
        latencies = []
        for _ in range(n):
            req = urllib.request.Request(
                url, data=BODY_BYTES,
                headers={"Content-Type": "application/json", "Connection": "close"},
                method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req) as resp:
                _ = json.loads(resp.read().decode("utf-8"))
            latencies.append((time.perf_counter() - t0) * 1000)
        return latencies

    def run_h1_keep(n):
        import http.client
        conn = http.client.HTTPConnection(args.host, args.port, timeout=10)
        latencies = []
        headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
        for _ in range(n):
            t0 = time.perf_counter()
            conn.request("POST", "/v1/systemone", body=BODY_BYTES, headers=headers)
            resp = conn.getresponse()
            _ = json.loads(resp.read().decode("utf-8"))
            latencies.append((time.perf_counter() - t0) * 1000)
        conn.close()
        return latencies

    def run_h2_native(n):
        sock = socket.create_connection((args.host, args.port))
        config = h2.config.H2Configuration(client_side=True)
        conn = h2.connection.H2Connection(config=config)
        conn.initiate_connection()
        sock.sendall(conn.data_to_send())
        latencies = []
        stream_id = 1
        for _ in range(n):
            t0 = time.perf_counter()
            headers = [
                (":method", "POST"),
                (":authority", f"{args.host}:{args.port}"),
                (":scheme", "http"),
                (":path", "/v1/systemone"),
                ("content-type", "application/json"),
                ("content-length", str(len(BODY_BYTES))),
            ]
            conn.send_headers(stream_id=stream_id, headers=headers, end_stream=False)
            conn.send_data(stream_id=stream_id, data=BODY_BYTES, end_stream=True)
            sock.sendall(conn.data_to_send())

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
            stream_id += 2
        sock.close()
        return latencies

    _ = run_h1_short(2)
    n = args.calls
    print(f"1. Testing HTTP/1.1 Short Connection (Connection: close, {n} calls)...")
    h1_short = run_h1_short(n)

    print(f"2. Testing HTTP/1.1 Persistent Keep-Alive Connection ({n} calls)...")
    h1_keep = run_h1_keep(n)

    print(f"3. Testing Native HTTP/2.0 Binary Stream Multiplexing ({n} calls)...")
    h2_stream = run_h2_native(n)

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
