"""NanoJev-4B High-Performance Asynchronous ASGI Application.
Supports h2c (HTTP/2 cleartext) and HTTP/1.1 via Hypercorn.

Features:
- Full parity with /v1/systemone (TypeSafe API) and /api/evaluate.
- Dynamic KV Cache state prefix sharing across questions in one request.
- Inter-request state prefix caching for identical multi-turn contexts.
- Calibrated temperature scaling and conservative tie-breaking.
- Integrated Apple MPS Multimodal Vision Engine.
"""
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
CALIBRATED_TEMPS = {"choice": 1.8172, "boolean": 3.5866, "score": 1.2567}


class NanoJev4BASGIApp:
    def __init__(
        self,
        checkpoint_dir: str,
        web_root: str = "web",
        default_temperature: float = 1.0,
        allow_h1_fallback: bool = False,
    ):
        from mlx_lm import load

        print(f"[NanoJev-4B] Loading model from: {checkpoint_dir} ...", flush=True)
        t0 = time.time()
        self.model, self.tokenizer = load(checkpoint_dir, tokenizer_config={"trust_remote_code": False})
        print(f"[NanoJev-4B] Model loaded in {time.time()-t0:.2f}s!", flush=True)

        self.slot_tokens = [self.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]
        self.cached_state_hash = None
        self.cached_kv = None
        self.web_root = Path(web_root).resolve()
        self.default_temperature = default_temperature
        self.allow_h1_fallback = allow_h1_fallback

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
                body = json.dumps({
                    "ready": True,
                    "model": "nanojev-4b",
                    "engine": "nanojev-4b-hybrid",
                    "architecture": "Qwen3.5-4B oQ4e-FP16 Metal GPU (2.3GB) + Multimodal Vision MPS",
                    "quantization": "oq4e-fp16",
                    "http_version": scope.get("http_version", "1.1"),
                    "prefix_sharing": True,
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

            if path == "/v1/systemone":
                await self.handle_typesafe_systemone(scope, send, bytes(body))
                return
            if path == "/api/evaluate":
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
                    nj_res = self.vision_engine.predict(nj_payload, temperature=temp)
                    ts_res = nanojev_response_to_typesafe(nj_res, meta)
                    resp_bytes = json.dumps(ts_res, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    await self.send_response(send, 200, resp_bytes, "application/json; charset=utf-8")
                except (ValueError, TypeError, KeyError) as exc:
                    err = json.dumps({"error": str(exc)}).encode("utf-8")
                    await self.send_response(send, 422, err, "application/json; charset=utf-8")
                except Exception as exc:
                    err = json.dumps({"error": f"Vision inference error: {exc}"}).encode("utf-8")
                    await self.send_response(send, 500, err, "application/json; charset=utf-8")
                return

            answers, total_tokens = self.predict_multi_questions(state, raw_questions, temperature=temp)
            resp = {
                "model": "nanojev-4b",
                "answers": answers,
                "usage": {"input_tokens": total_tokens, "output_tokens": 0},
            }
            resp_bytes = json.dumps(resp, ensure_ascii=False, allow_nan=False).encode("utf-8")
            await self.send_response(send, 200, resp_bytes, "application/json; charset=utf-8")
        except (ValueError, TypeError, KeyError) as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            await self.send_response(send, 422, err, "application/json; charset=utf-8")
        except Exception as exc:
            err = json.dumps({"error": f"Internal error: {str(exc)}"}).encode("utf-8")
            await self.send_response(send, 500, err, "application/json; charset=utf-8")

    def predict_multi_questions(self, state, raw_questions: dict, temperature: float = 1.0):
        state_text = json.dumps(state, ensure_ascii=False) if isinstance(state, (dict, list)) else str(state)
        prefix_text = f"<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\nEvidence:\n{state_text}\n\n"
        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_tokens) > 3500:
            prefix_tokens = prefix_tokens[:3500]

        state_hash = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

        if self.cached_state_hash == state_hash and self.cached_kv is not None:
            base_cache = self.cached_kv
        else:
            cache = make_prompt_cache(self.model)
            x_prefix = mx.array([prefix_tokens], dtype=mx.int32)
            self.model(x_prefix, cache=cache)
            mx.eval([e.state for e in cache])
            mx.synchronize()
            self.cached_state_hash = state_hash
            self.cached_kv = cache
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

        return answers, total_tokens

    async def send_response(self, send, status: int, body: bytes, content_type: str, content_length: Optional[int] = None):
        cl = len(body) if content_length is None else content_length
        headers = [
            (b"content-type", content_type.encode("utf-8")),
            (b"content-length", str(cl).encode("utf-8")),
            (b"cache-control", b"no-store"),
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})
