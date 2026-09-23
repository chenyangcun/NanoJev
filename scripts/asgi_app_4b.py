"""NanoJev-4B High-Performance Asynchronous ASGI Application (P0 Optimized).
Supports h2c (HTTP/2 cleartext) and HTTP/1.1 via Hypercorn.

Features:
- Full parity with /v1/systemone (TypeSafe API) and /api/evaluate.
- Multi-State LRU KV Cache (8 slots, bounded unified memory, high hit rate).
- Decoupled Async Event Loop (asyncio.Lock + worker thread offloading).
- Tail-Preserving Context Truncation (Head 500 + Tail 3000 tokens).
- Calibrated temperature scaling and conservative tie-breaking.
- Detailed Telemetry (X-Inference-Time-Ms, X-Cache-Hit, X-Prefill-Tokens).
- Integrated Apple MPS Multimodal Vision Engine.
"""
import asyncio
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
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


class NanoJev4BASGIApp:
    def __init__(
        self,
        checkpoint_dir: str,
        web_root: str = "web",
        default_temperature: float = 1.0,
        allow_h1_fallback: bool = False,
        lru_capacity: int = DEFAULT_LRU_CAPACITY,
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

        # P0: Async Event Loop Decoupling Lock
        self._inference_lock = asyncio.Lock()

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

        http_version = scope.get("http_version", "1.1")
        if http_version != "2" and not self.allow_h1_fallback:
            body = json.dumps({
                "error": "HTTP/1.1 is not supported on this endpoint; use HTTP/2 over cleartext (h2c)",
                "required_protocol": "h2c",
            }).encode("utf-8")
            await self.send_response(send, 505, body, "application/json; charset=utf-8")
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

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

    async def handle_typesafe_systemone(self, scope, send, body_bytes: bytes):
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
            temp = float(payload.get("temperature", self.default_temperature))
            state = payload.get("state", "")
            raw_questions = payload.get("questions", {})

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
            extra_headers = [
                (b"x-inference-time-ms", f"{infer_ms:.1f}".encode("utf-8")),
                (b"x-cache-hit", b"1" if cache_hit else b"0"),
                (b"x-prefill-tokens", str(total_tokens).encode("utf-8")),
                (b"x-lane", b"metal-gpu-4b"),
            ]
            await self.send_response(send, 200, resp_bytes, "application/json; charset=utf-8", extra_headers=extra_headers)
        except (ValueError, TypeError, KeyError) as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            await self.send_response(send, 422, err, "application/json; charset=utf-8")
        except Exception as exc:
            err = json.dumps({"error": f"Internal error: {str(exc)}"}).encode("utf-8")
            await self.send_response(send, 500, err, "application/json; charset=utf-8")

    def _truncate_prefix_tokens(self, tokens: list[int]) -> list[int]:
        """P0: Tail-preserving context truncation."""
        if len(tokens) <= MAX_PREFIX_TOKENS:
            return tokens
        # Preserve head (system instruction + metadata) and tail (recent evidence & user command)
        head = tokens[:HEAD_PRESERVE_TOKENS]
        tail = tokens[-TAIL_PRESERVE_TOKENS:]
        return head + tail

    def predict_multi_questions(self, state, raw_questions: dict, temperature: float = 1.0) -> Tuple[dict, int, bool]:
        state_text = json.dumps(state, ensure_ascii=False) if isinstance(state, (dict, list)) else str(state)
        prefix_text = f"<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\nEvidence:\n{state_text}\n\n"
        raw_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_tokens = self._truncate_prefix_tokens(raw_tokens)

        # Hash prefix tokens to identify unique state contexts
        state_hash = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

        self.total_requests += 1
        cache_hit = False

        # P0: Multi-State LRU Cache Lookup
        if state_hash in self.lru_cache:
            base_cache = self.lru_cache.pop(state_hash)
            self.lru_cache[state_hash] = base_cache  # Move to most recently used
            self.cache_hits += 1
            cache_hit = True
        else:
            cache = make_prompt_cache(self.model)
            x_prefix = mx.array([prefix_tokens], dtype=mx.int32)
            self.model(x_prefix, cache=cache)
            mx.eval([e.state for e in cache])
            mx.synchronize()

            # Evict oldest entry if at capacity
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
        ]
        if extra_headers:
            headers.extend(extra_headers)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})
