"""NanoJev-4B High-Performance Asynchronous ASGI Application.
Supports h2c (HTTP/2 cleartext) and HTTP/1.1 via Hypercorn.

Features:
- Full parity with /v1/systemone (TypeSafe API) and /api/evaluate.
- Multi-State LRU KV Cache (8 slots, bounded unified memory, high hit rate).
- Decoupled Async Event Loop (asyncio.Lock + worker thread offloading).
- Tail-Preserving Context Truncation (Head 500 + Tail 3000 tokens).
- Integrated Request Logging (Request Size, Prefill Tokens, Latency Breakdown, Cache Hits).
- Web Management & Live Observability Dashboard (GET /logs, /api/logs).
- Calibrated temperature scaling and conservative tie-breaking.
- Detailed Telemetry (X-Inference-Time-Ms, X-Cache-Hit, X-Prefill-Tokens).
- Integrated Apple MPS Multimodal Vision Engine.
"""
import asyncio
from collections import deque, OrderedDict
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import mimetypes
from pathlib import Path
import statistics
import time
from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
CALIBRATED_TEMPS = {"choice": 1.8172, "boolean": 3.5866, "score": 1.2567}
MAX_PREFIX_TOKENS = 3500
HEAD_PRESERVE_TOKENS = 500
TAIL_PRESERVE_TOKENS = 3000
DEFAULT_LRU_CAPACITY = 8
LOCAL_TZ = timezone(timedelta(hours=8))  # Beijing Time UTC+8


class NanoJev4BASGIApp:
    def __init__(
        self,
        checkpoint_dir: str,
        web_root: str = "web",
        default_temperature: float = 1.0,
        allow_h1_fallback: bool = False,
        lru_capacity: int = DEFAULT_LRU_CAPACITY,
        log_dir: str = "logs",
    ):
        from mlx_lm import load

        print(f"[NanoJev-4B] Loading model from: {checkpoint_dir} ...", flush=True)
        t0 = time.time()
        self.model, self.tokenizer = load(checkpoint_dir, tokenizer_config={"trust_remote_code": False})
        print(f"[NanoJev-4B] Model loaded in {time.time()-t0:.2f}s!", flush=True)

        self.slot_tokens = [self.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]
        self.web_root = Path(web_root).resolve()
        self.default_temperature = default_temperature
        self.allow_h1_fallback = allow_h1_fallback

        # P0: Multi-State LRU Cache
        self.lru_capacity = lru_capacity
        self.lru_cache: OrderedDict[str, any] = OrderedDict()
        self.total_requests = 0
        self.cache_hits = 0
        self.start_time = time.time()

        # P0: Async Event Loop Decoupling Lock
        self._inference_lock = asyncio.Lock()

        # Request Logging Storage
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "requests.jsonl"
        self.recent_logs = deque(maxlen=500)
        self.req_counter = 0

        # Preload historical logs if file exists
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                d = json.loads(line)
                                self.recent_logs.append(d)
                                self.req_counter = max(self.req_counter, d.get("id", 0))
                            except Exception:
                                pass
            except Exception as e:
                print(f"[NanoJev-4B] Notice: could not preload existing log: {e}", flush=True)

        # Vision engine (Apple MPS)
        import os
        import sys
        nj_scripts = os.path.expanduser("~/work/NanoJev/scripts")
        if nj_scripts not in sys.path:
            sys.path.insert(0, nj_scripts)
        try:
            from vision_decision_engine import VisionDecisionEngine
            vision_ckpt = os.path.expanduser("~/work/NanoJev/checkpoints/dohnuts_merged_0.8b")
            self.vision_engine = VisionDecisionEngine(checkpoint_dir=vision_ckpt)
            print("[NanoJev-4B] Multimodal Vision Decision Engine ready on Apple MPS!", flush=True)
        except Exception as e:
            print(f"[NanoJev-4B] Vision engine not loaded ({e}); continuing text-only.", flush=True)
            self.vision_engine = None

    def _record_log(self, entry: dict):
        self.req_counter += 1
        entry["id"] = self.req_counter
        self.recent_logs.append(entry)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[NanoJev-4B] Log write error: {e}", flush=True)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return

        if scope["type"] != "http":
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "")
        http_version = scope.get("http_version", "1.1")

        # Allow HTTP/1.1 for GET and HEAD requests (browsers, dashboards, health checks, static assets)
        if method not in ("GET", "HEAD") and http_version != "2" and not self.allow_h1_fallback:
            body = json.dumps({
                "error": "HTTP/1.1 is not supported on this endpoint; use HTTP/2 over cleartext (h2c)",
                "required_protocol": "h2c",
            }).encode("utf-8")
            await self.send_response(send, 505, body, "application/json; charset=utf-8")
            return

        if method in ("GET", "HEAD"):
            if path == "/api/health":
                hit_rate = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0.0
                body = json.dumps({
                    "ready": True,
                    "model": "nanojev-4b",
                    "engine": "nanojev-4b-hybrid",
                    "architecture": "Qwen3.5-4B oQ4e-FP16 Metal GPU (2.3GB) + Multimodal Vision MPS",
                    "quantization": "oq4e-fp16",
                    "http_version": http_version,
                    "prefix_sharing": True,
                    "lru_cache_slots": len(self.lru_cache),
                    "lru_cache_capacity": self.lru_capacity,
                    "cache_hit_rate": round(hit_rate, 4),
                    "total_requests": self.total_requests,
                    "vision_available": self.vision_engine is not None,
                    "status": "healthy",
                }).encode("utf-8")
                await self.send_response(send, 200, body, "application/json; charset=utf-8")
                return

            if path == "/api/logs":
                await self.handle_api_logs(scope, send)
                return

            # Serve Dashboard / Logs Web Page
            if path in ("/", "/logs", "/logs.html", "/dashboard", "/dashboard.html"):
                logs_html = self.web_root / "logs.html"
                if logs_html.is_file():
                    data = logs_html.read_bytes()
                    await self.send_response(send, 200, data, "text/html; charset=utf-8")
                    return
                # Fallback to index if logs.html doesn't exist
                index_html = self.web_root / "index.html"
                if index_html.is_file():
                    data = index_html.read_bytes()
                    await self.send_response(send, 200, data, "text/html; charset=utf-8")
                    return

            # Serve static files from web_root
            rel = path.lstrip("/")
            target = (self.web_root / rel).resolve()
            if target.is_relative_to(self.web_root) and target.is_file():
                mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                data = target.read_bytes()
                if method == "HEAD":
                    await self.send_response(send, 200, b"", mime, content_length=len(data))
                else:
                    await self.send_response(send, 200, data, mime)
                return

            await self.send_response(send, 404, b'{"error":"Not found"}', "application/json; charset=utf-8")
            return

        if method == "POST":
            body = bytearray()
            while True:
                msg = await receive()
                body.extend(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break

            if path in ("/v1/systemone", "/api/evaluate"):
                await self.handle_typesafe_systemone(scope, send, bytes(body))
                return

            await self.send_response(send, 404, b'{"error":"Unknown endpoint"}', "application/json; charset=utf-8")

    async def handle_api_logs(self, scope, send):
        """API endpoint returning live request metrics and paginated request history."""
        query_str = scope.get("query_string", b"").decode("utf-8")
        page = 1
        page_size = 20
        filter_type = None
        search_kw = None
        if query_str:
            parts = query_str.split("&")
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    if k in ("limit", "page_size") and v.isdigit():
                        page_size = min(200, max(1, int(v)))
                    elif k == "page" and v.isdigit():
                        page = max(1, int(v))
                    elif k == "filter":
                        filter_type = v.strip().lower()
                    elif k == "search":
                        search_kw = v.strip().lower()

        # Compute summary stats across all recent logs
        logs_list = list(self.recent_logs)
        total_count = len(logs_list)
        hits_count = sum(1 for x in logs_list if x.get("cache_hit"))
        latencies = [x["total_ms"] for x in logs_list if "total_ms" in x and x.get("status") == 200]
        sizes = [x["request_bytes"] for x in logs_list if "request_bytes" in x]

        p50 = round(statistics.median(latencies), 1) if latencies else 0.0
        p95 = round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0.0
        avg_lat = round(statistics.mean(latencies), 1) if latencies else 0.0
        avg_size = round(statistics.mean(sizes), 1) if sizes else 0.0
        hit_rate = round(hits_count / total_count, 4) if total_count > 0 else 0.0

        # Filter logs
        filtered = logs_list
        if filter_type == "hit":
            filtered = [x for x in filtered if x.get("cache_hit")]
        elif filter_type == "miss":
            filtered = [x for x in filtered if not x.get("cache_hit")]
        elif filter_type == "error":
            filtered = [x for x in filtered if x.get("status", 200) >= 400]

        if search_kw:
            filtered = [
                x for x in filtered
                if search_kw in str(x.get("decisions", "")).lower()
                or search_kw in str(x.get("questions", "")).lower()
                or search_kw in str(x.get("path", "")).lower()
                or search_kw in str(x.get("state_preview", "")).lower()
            ]

        # Pagination calculations
        total_records = len(filtered)
        total_pages = max(1, (total_records + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages

        reversed_filtered = list(reversed(filtered))
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_records)
        page_slice = reversed_filtered[start_idx:end_idx]

        payload = {
            "stats": {
                "total_requests": self.total_requests or total_count,
                "recent_window_requests": total_count,
                "cache_hits": self.cache_hits or hits_count,
                "cache_hit_rate": hit_rate,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "avg_latency_ms": avg_lat,
                "avg_request_size_bytes": avg_size,
                "lru_slots_used": len(self.lru_cache),
                "lru_capacity": self.lru_capacity,
                "uptime_seconds": round(time.time() - self.start_time, 1),
            },
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_records": total_records,
                "total_pages": total_pages,
                "start_index": start_idx + 1 if total_records > 0 else 0,
                "end_index": end_idx,
                "has_prev": page > 1,
                "has_next": page < total_pages,
            },
            "logs": page_slice,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self.send_response(send, 200, body, "application/json; charset=utf-8")

    async def handle_typesafe_systemone(self, scope, send, body_bytes: bytes):
        t0_req = time.perf_counter()
        now_dt = datetime.now(LOCAL_TZ)
        now_iso = now_dt.isoformat()
        time_str = now_dt.strftime("%H:%M:%S")

        client_ip = ""
        client = scope.get("client")
        if client and len(client) >= 1:
            client_ip = str(client[0])

        http_ver = scope.get("http_version", "1.1")
        req_size = len(body_bytes)

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
            temp = float(payload.get("temperature", self.default_temperature))
            state = payload.get("state", "")
            raw_questions = payload.get("questions", {})

            # Format state snippet for logs
            if isinstance(state, (dict, list)):
                state_prev = json.dumps(state, ensure_ascii=False)[:200]
            else:
                state_prev = str(state).strip().replace("\n", " ")[:200]

            # Multimodal Vision Lane
            if self.vision_engine and self.vision_engine.has_image(state):
                try:
                    from typesafe_adapter import typesafe_request_to_nanojev, nanojev_response_to_typesafe
                    nj_payload, meta = typesafe_request_to_nanojev(payload)
                    t0_vis = time.perf_counter()
                    nj_res = self.vision_engine.predict(nj_payload, temperature=temp)
                    vis_ms = (time.perf_counter() - t0_vis) * 1000
                    ts_res = nanojev_response_to_typesafe(nj_res, meta)
                    resp_bytes = json.dumps(ts_res, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    total_ms = (time.perf_counter() - t0_req) * 1000

                    # Record log
                    log_entry = {
                        "timestamp": now_iso,
                        "time_str": time_str,
                        "method": "POST",
                        "path": scope.get("path", "/v1/systemone"),
                        "status": 200,
                        "http_version": http_ver,
                        "client_ip": client_ip,
                        "request_bytes": req_size,
                        "response_bytes": len(resp_bytes),
                        "total_ms": round(total_ms, 1),
                        "infer_ms": round(vis_ms, 1),
                        "cache_hit": False,
                        "prefill_tokens": 0,
                        "lane": "vision-mps",
                        "questions_count": len(raw_questions),
                        "questions": list(raw_questions.keys()),
                        "decisions": {k: v.get("choice", v.get("noul")) for k, v in ts_res.get("answers", {}).items()},
                        "state_preview": state_prev,
                    }
                    self._record_log(log_entry)

                    extra_headers = [
                        (b"x-inference-time-ms", f"{vis_ms:.1f}".encode("utf-8")),
                        (b"x-cache-hit", b"0"),
                        (b"x-lane", b"vision-mps"),
                    ]
                    await self.send_response(send, 200, resp_bytes, "application/json; charset=utf-8", extra_headers=extra_headers)
                except (ValueError, TypeError, KeyError) as exc:
                    err = json.dumps({"error": str(exc)}).encode("utf-8")
                    await self.send_response(send, 422, err, "application/json; charset=utf-8")
                except Exception as exc:
                    err = json.dumps({"error": f"Vision inference error: {exc}"}).encode("utf-8")
                    await self.send_response(send, 500, err, "application/json; charset=utf-8")
                return

            # P0: Non-blocking async queue - event loop remains responsive
            async with self._inference_lock:
                t0_infer = time.perf_counter()
                answers, total_tokens, cache_hit = await asyncio.to_thread(
                    self.predict_multi_questions, state, raw_questions, temperature=temp
                )
                infer_ms = (time.perf_counter() - t0_infer) * 1000

            resp = {
                "model": "nanojev-4b",
                "answers": answers,
                "usage": {"input_tokens": total_tokens, "output_tokens": 0},
            }
            resp_bytes = json.dumps(resp, ensure_ascii=False, allow_nan=False).encode("utf-8")
            total_ms = (time.perf_counter() - t0_req) * 1000

            decisions_summary = {}
            confidences_summary = {}
            for qk, qv in answers.items():
                decisions_summary[qk] = qv.get("choice", qv.get("noul", qv.get("score")))
                confidences_summary[qk] = qv.get("confidence", 0.0)

            # Record request log
            log_entry = {
                "timestamp": now_iso,
                "time_str": time_str,
                "method": "POST",
                "path": scope.get("path", "/v1/systemone"),
                "status": 200,
                "http_version": http_ver,
                "client_ip": client_ip,
                "request_bytes": req_size,
                "response_bytes": len(resp_bytes),
                "total_ms": round(total_ms, 1),
                "infer_ms": round(infer_ms, 1),
                "cache_hit": cache_hit,
                "prefill_tokens": total_tokens,
                "lane": "metal-gpu-4b",
                "questions_count": len(raw_questions),
                "questions": list(raw_questions.keys()),
                "decisions": decisions_summary,
                "confidences": confidences_summary,
                "state_preview": state_prev,
            }
            self._record_log(log_entry)

            extra_headers = [
                (b"x-inference-time-ms", f"{infer_ms:.1f}".encode("utf-8")),
                (b"x-cache-hit", b"1" if cache_hit else b"0"),
                (b"x-prefill-tokens", str(total_tokens).encode("utf-8")),
                (b"x-lane", b"metal-gpu-4b"),
            ]
            await self.send_response(send, 200, resp_bytes, "application/json; charset=utf-8", extra_headers=extra_headers)
        except (ValueError, TypeError, KeyError) as exc:
            total_ms = (time.perf_counter() - t0_req) * 1000
            self._record_log({
                "timestamp": now_iso,
                "time_str": time_str,
                "method": "POST",
                "path": scope.get("path", "/v1/systemone"),
                "status": 422,
                "http_version": http_ver,
                "client_ip": client_ip,
                "request_bytes": req_size,
                "response_bytes": 0,
                "total_ms": round(total_ms, 1),
                "infer_ms": 0.0,
                "cache_hit": False,
                "prefill_tokens": 0,
                "error": str(exc),
            })
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            await self.send_response(send, 422, err, "application/json; charset=utf-8")
        except Exception as exc:
            total_ms = (time.perf_counter() - t0_req) * 1000
            self._record_log({
                "timestamp": now_iso,
                "time_str": time_str,
                "method": "POST",
                "path": scope.get("path", "/v1/systemone"),
                "status": 500,
                "http_version": http_ver,
                "client_ip": client_ip,
                "request_bytes": req_size,
                "response_bytes": 0,
                "total_ms": round(total_ms, 1),
                "infer_ms": 0.0,
                "cache_hit": False,
                "prefill_tokens": 0,
                "error": str(exc),
            })
            err = json.dumps({"error": f"Internal error: {str(exc)}"}).encode("utf-8")
            await self.send_response(send, 500, err, "application/json; charset=utf-8")

    def _truncate_prefix_tokens(self, tokens: list[int]) -> list[int]:
        """P0: Tail-preserving context truncation."""
        if len(tokens) <= MAX_PREFIX_TOKENS:
            return tokens
        head = tokens[:HEAD_PRESERVE_TOKENS]
        tail = tokens[-TAIL_PRESERVE_TOKENS:]
        return head + tail

    def predict_multi_questions(self, state, raw_questions: dict, temperature: float = 1.0) -> Tuple[dict, int, bool]:
        state_text = json.dumps(state, ensure_ascii=False) if isinstance(state, (dict, list)) else str(state)
        prefix_text = f"<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\nEvidence:\n{state_text}\n\n"
        raw_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_tokens = self._truncate_prefix_tokens(raw_tokens)

        state_hash = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

        self.total_requests += 1
        cache_hit = False

        # P0: Multi-State LRU Cache Lookup
        if state_hash in self.lru_cache:
            base_cache = self.lru_cache.pop(state_hash)
            self.lru_cache[state_hash] = base_cache
            self.cache_hits += 1
            cache_hit = True
        else:
            cache = make_prompt_cache(self.model)
            x_prefix = mx.array([prefix_tokens], dtype=mx.int32)
            self.model(x_prefix, cache=cache)
            mx.eval([e.state for e in cache])
            mx.synchronize()

            if len(self.lru_cache) >= self.lru_capacity:
                _oldest_key, oldest_val = self.lru_cache.popitem(last=False)
                del oldest_val
                mx.metal.clear_cache()

            self.lru_cache[state_hash] = cache
            base_cache = cache

        answers = {}
        total_tokens = len(prefix_tokens)

        for qid, q in raw_questions.items():
            qtype = q.get("type", "choice")
            crit = q.get("criteria")
            instr = q.get("instructions", "")

            if qtype in ("noul", "boolean"):
                options = [
                    {"id": "yes", "description": (crit or {}).get("true", "The proposition is true.")},
                    {"id": "no", "description": (crit or {}).get("false", "The proposition is false.")},
                ]
            elif qtype == "choice":
                if isinstance(crit, dict):
                    options = [{"id": k, "description": v or k} for k, v in crit.items()]
                else:
                    options = [{"id": str(i), "description": str(opt)} for i, opt in enumerate(crit or [])]
            else:
                if isinstance(crit, list):
                    options = [{"id": str(i), "description": str(lvl)} for i, lvl in enumerate(crit)]
                else:
                    options = [{"id": str(i), "description": f"Level {i}"} for i in range(5)]

            opts_lines = [LETTERS[i] + ": " + o["id"] + ": " + o["description"] for i, o in enumerate(options)]
            opts_text = "\n".join(opts_lines)
            suffix_text = (
                f"Criterion:\n{instr}\n\nOptions:\n{opts_text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            suffix_tokens = self.tokenizer.encode(suffix_text, add_special_tokens=False)
            total_tokens += len(suffix_tokens)

            branch = copy.deepcopy(base_cache)
            logits = self.model(mx.array([suffix_tokens], dtype=mx.int32), cache=branch)[:, -1, :]
            mx.eval(logits)

            slots = self.slot_tokens[: len(options)]
            selected = logits[0, mx.array(slots)].tolist()

            calib_factor = CALIBRATED_TEMPS.get(qtype, 1.0)
            eff_temp = calib_factor * max(1e-4, temperature)
            scaled = [v / eff_temp for v in selected]
            probs_list = mx.softmax(mx.array(scaled)).tolist()
            probs = dict(zip([o["id"] for o in options], [round(p, 4) for p in probs_list]))

            if qtype in ("noul", "boolean"):
                p_yes = probs.get("yes", 0.5)
                answers[qid] = {
                    "type": "noul",
                    "noul": p_yes,
                    "confidence": round(max(p_yes, 1.0 - p_yes), 4),
                    "probabilities": {"yes": p_yes, "no": round(1.0 - p_yes, 4)},
                }
            elif qtype == "choice":
                pred = max(probs.keys(), key=lambda k: probs[k])
                answers[qid] = {
                    "type": "choice",
                    "choice": pred,
                    "probabilities": probs,
                    "confidence": probs[pred],
                }
            else:
                pred = max(probs.keys(), key=lambda k: probs[k])
                answers[qid] = {
                    "type": "score",
                    "score": int(pred),
                    "probabilities": probs,
                    "confidence": probs[pred],
                }

        return answers, total_tokens, cache_hit

    async def send_response(
        self,
        send,
        status: int,
        body: bytes,
        content_type: str,
        content_length: Optional[int] = None,
        extra_headers: Optional[list] = None,
    ):
        cl = len(body) if content_length is None else content_length
        headers = [
            (b"content-type", content_type.encode("utf-8")),
            (b"content-length", str(cl).encode("utf-8")),
            (b"cache-control", b"no-store"),
            (b"access-control-allow-origin", b"*"),
        ]
        if extra_headers:
            headers.extend(extra_headers)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})
