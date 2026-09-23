import asyncio
import json
import time

import h2.config
import h2.connection
import h2.events

HOST, PORT = "192.168.123.88", 8770

with open("/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-09-23.jsonl") as f:
    REQ = json.loads(f.readline())["request"]


async def send_health(name, delay_before=0.0):
    if delay_before > 0:
        await asyncio.sleep(delay_before)
    reader, writer = await asyncio.open_connection(HOST, PORT)
    cfg = h2.config.H2Configuration(client_side=True)
    conn = h2.connection.H2Connection(config=cfg)
    conn.initiate_connection()
    writer.write(conn.data_to_send())
    await writer.drain()

    headers = [
        (":method", "GET"),
        (":authority", f"{HOST}:{PORT}"),
        (":scheme", "http"),
        (":path", "/api/health"),
    ]
    t0 = time.perf_counter()
    conn.send_headers(1, headers, end_stream=True)
    writer.write(conn.data_to_send())
    await writer.drain()

    done = False
    while not done:
        data = await reader.read(65535)
        if not data:
            break
        for ev in conn.receive_data(data):
            if isinstance(ev, h2.events.StreamEnded):
                done = True
    dt = (time.perf_counter() - t0) * 1000
    writer.close()
    await writer.wait_closed()
    print(f"[{name}] Completed in {dt:.1f}ms")
    return dt


async def send_infer(name):
    reader, writer = await asyncio.open_connection(HOST, PORT)
    cfg = h2.config.H2Configuration(client_side=True)
    conn = h2.connection.H2Connection(config=cfg)
    conn.initiate_connection()
    writer.write(conn.data_to_send())
    await writer.drain()

    body = json.dumps(REQ).encode("utf-8")
    headers = [
        (":method", "POST"),
        (":authority", f"{HOST}:{PORT}"),
        (":scheme", "http"),
        (":path", "/v1/systemone"),
        ("content-type", "application/json"),
        ("content-length", str(len(body))),
    ]
    t0 = time.perf_counter()
    conn.send_headers(1, headers, end_stream=False)
    conn.send_data(1, body, end_stream=True)
    writer.write(conn.data_to_send())
    await writer.drain()

    done = False
    while not done:
        data = await reader.read(65535)
        if not data:
            break
        for ev in conn.receive_data(data):
            if isinstance(ev, h2.events.StreamEnded):
                done = True
    dt = (time.perf_counter() - t0) * 1000
    writer.close()
    await writer.wait_closed()
    print(f"[{name}] Heavy inference completed in {dt:.1f}ms")
    return dt


async def main():
    print("Testing Event Loop Non-blocking Behavior during heavy inference:")
    # Launch 1 heavy inference and simultaneously send 3 health checks with 0.1s, 0.3s, 0.5s delays
    t0 = time.perf_counter()
    results = await asyncio.gather(
        send_infer("HeavyInfer-1"),
        send_health("Health-1", delay_before=0.1),
        send_health("Health-2", delay_before=0.3),
        send_health("Health-3", delay_before=0.5),
    )
    total_time = (time.perf_counter() - t0) * 1000
    print(f"Total wall time: {total_time:.1f}ms")


asyncio.run(main())
