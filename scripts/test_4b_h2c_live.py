import json
import socket
import time

import h2.config
import h2.connection
import h2.events

host, port = "192.168.123.88", 8770

with open("/Users/chenyc/Documents/study/jev-cliproxy-router/logs/jev-2026-09-23.jsonl") as f:
    d = json.loads(f.readline())
req_body = d["request"]
body_bytes = json.dumps(req_body).encode("utf-8")

sock = socket.create_connection((host, port), timeout=30)
config = h2.config.H2Configuration(client_side=True)
conn = h2.connection.H2Connection(config=config)
conn.initiate_connection()
sock.sendall(conn.data_to_send())


def do_post(stream_id, data):
    t0 = time.perf_counter()
    headers = [
        (":method", "POST"),
        (":authority", f"{host}:{port}"),
        (":scheme", "http"),
        (":path", "/v1/systemone"),
        ("content-type", "application/json"),
        ("content-length", str(len(data))),
    ]
    conn.send_headers(stream_id=stream_id, headers=headers, end_stream=False)
    conn.send_data(stream_id=stream_id, data=data, end_stream=True)
    sock.sendall(conn.data_to_send())

    resp_body = bytearray()
    status_code = None
    ended = False
    while not ended:
        frame = sock.recv(65535)
        if not frame:
            break
        events = conn.receive_data(frame)
        for event in events:
            if isinstance(event, h2.events.ResponseReceived):
                for name, val in event.headers:
                    if name == b":status":
                        status_code = val.decode()
            elif isinstance(event, h2.events.DataReceived):
                resp_body.extend(event.data)
                conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
            elif isinstance(event, h2.events.StreamEnded):
                if event.stream_id == stream_id:
                    ended = True
    return status_code, bytes(resp_body), (time.perf_counter() - t0) * 1000


status1, body1, lat1 = do_post(1, body_bytes)
r1 = json.loads(body1.decode())
status2, body2, lat2 = do_post(3, body_bytes)

try:
    import http.client

    c = http.client.HTTPConnection(host, port, timeout=5)
    c.request("POST", "/v1/systemone", body=body_bytes, headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    print("HTTP/1.1 fallback status:", resp.status)
    c.close()
except Exception as e:
    print("HTTP/1.1 rejected as expected (h2c-only):", type(e).__name__)

off_lat = d.get("duration_ms", 0.0)
ans_summary = {}
for k, v in r1["answers"].items():
    ans_summary[k] = v.get("choice", v.get("noul"))

print(f"Official Jev (5 questions): {off_lat:.1f} ms")
print(f"h2c Call #1 (prefill + 5 Qs): {status1}, {lat1:.1f} ms")
print(f"h2c Call #2 (prefix cache hit): {status2}, {lat2:.1f} ms")
print("Answers:", ans_summary)
sock.close()
