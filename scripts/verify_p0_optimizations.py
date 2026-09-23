import json
import socket
import time

import h2.config
import h2.connection
import h2.events

HOST, PORT = "192.168.123.88", 8770


def create_h2_client():
    sock = socket.create_connection((HOST, PORT), timeout=15)
    config = h2.config.H2Configuration(client_side=True)
    conn = h2.connection.H2Connection(config=config)
    conn.initiate_connection()
    sock.sendall(conn.data_to_send())
    return sock, conn


def post_req(sock, conn, stream_id, path, payload):
    data = json.dumps(payload).encode("utf-8")
    headers = [
        (":method", "POST"),
        (":authority", f"{HOST}:{PORT}"),
        (":scheme", "http"),
        (":path", path),
        ("content-type", "application/json"),
        ("content-length", str(len(data))),
    ]
    t0 = time.perf_counter()
    conn.send_headers(stream_id=stream_id, headers=headers, end_stream=False)
    conn.send_data(stream_id=stream_id, data=data, end_stream=True)
    sock.sendall(conn.data_to_send())

    resp_headers = {}
    resp_body = bytearray()
    done = False
    while not done:
        chunk = sock.recv(65535)
        if not chunk:
            break
        for event in conn.receive_data(chunk):
            if isinstance(event, h2.events.ResponseReceived):
                resp_headers = dict((k.decode(), v.decode()) for k, v in event.headers)
            elif isinstance(event, h2.events.DataReceived):
                resp_body.extend(event.data)
                conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
            elif isinstance(event, h2.events.StreamEnded) and event.stream_id == stream_id:
                done = True
    sock.sendall(conn.data_to_send())
    dt = (time.perf_counter() - t0) * 1000
    res_obj = json.loads(bytes(resp_body).decode("utf-8")) if resp_body else {}
    return resp_headers, res_obj, dt


def get_req(sock, conn, stream_id, path):
    headers = [
        (":method", "GET"),
        (":authority", f"{HOST}:{PORT}"),
        (":scheme", "http"),
        (":path", path),
    ]
    t0 = time.perf_counter()
    conn.send_headers(stream_id=stream_id, headers=headers, end_stream=True)
    sock.sendall(conn.data_to_send())

    resp_headers = {}
    resp_body = bytearray()
    done = False
    while not done:
        chunk = sock.recv(65535)
        if not chunk:
            break
        for event in conn.receive_data(chunk):
            if isinstance(event, h2.events.ResponseReceived):
                resp_headers = dict((k.decode(), v.decode()) for k, v in event.headers)
            elif isinstance(event, h2.events.DataReceived):
                resp_body.extend(event.data)
                conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
            elif isinstance(event, h2.events.StreamEnded) and event.stream_id == stream_id:
                done = True
    sock.sendall(conn.data_to_send())
    dt = (time.perf_counter() - t0) * 1000
    res_obj = json.loads(bytes(resp_body).decode("utf-8")) if resp_body else {}
    return resp_headers, res_obj, dt


def main():
    sock, conn = create_h2_client()
    stream_id = 1

    # 1. Health check over h2c
    h, body, dt = get_req(sock, conn, stream_id, "/api/health")
    stream_id += 2
    print(f"Health Check (h2c): {h.get(':status')} in {dt:.1f}ms")
    print("Health payload:", body)

    # 2. Test LRU Cache with 3 distinct states
    # Load 3 requests from real logs
    requests = []
    with open("/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-09-23.jsonl") as f:
        for _ in range(3):
            requests.append(json.loads(f.readline())["request"])

    print("\n--- Testing P0 Multi-State LRU KV Cache ---")
    # Turn 1: State A (cold)
    h, b, dt = post_req(sock, conn, stream_id, "/v1/systemone", requests[0])
    stream_id += 2
    print(f"Req 1 (State A, cold): status={h.get(':status')}, hit={h.get('x-cache-hit')}, infer={h.get('x-inference-time-ms')}ms, total={dt:.1f}ms")

    # Turn 2: State B (cold)
    h, b, dt = post_req(sock, conn, stream_id, "/v1/systemone", requests[1])
    stream_id += 2
    print(f"Req 2 (State B, cold): status={h.get(':status')}, hit={h.get('x-cache-hit')}, infer={h.get('x-inference-time-ms')}ms, total={dt:.1f}ms")

    # Turn 3: State A again (SHOULD HIT in multi-state LRU! In single-state it would have missed!)
    h, b, dt = post_req(sock, conn, stream_id, "/v1/systemone", requests[0])
    stream_id += 2
    print(f"Req 3 (State A, LRU HIT): status={h.get(':status')}, hit={h.get('x-cache-hit')}, infer={h.get('x-inference-time-ms')}ms, total={dt:.1f}ms")

    # Turn 4: State B again (SHOULD HIT!)
    h, b, dt = post_req(sock, conn, stream_id, "/v1/systemone", requests[1])
    stream_id += 2
    print(f"Req 4 (State B, LRU HIT): status={h.get(':status')}, hit={h.get('x-cache-hit')}, infer={h.get('x-inference-time-ms')}ms, total={dt:.1f}ms")

    # Check updated health stats
    h, body, dt = get_req(sock, conn, stream_id, "/api/health")
    print("\nUpdated Health Stats:", {k: body[k] for k in ("lru_cache_slots", "lru_cache_capacity", "cache_hit_rate", "total_requests")})

    sock.close()


if __name__ == "__main__":
    main()
