"""NanoJev High-Performance Asynchronous ASGI Application.
Supports HTTP/1.1 and HTTP/2 with persistent long connection pooling via Hypercorn.

Features:
- Full parity with /v1/systemone (TypeSafe API) and /api/evaluate.
- Integrated Dynamic Adaptive Temperature.
- Integrated Dynamic Early Exit.
- Seamless ASGI interface for Hypercorn.
"""
import asyncio
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from predict_mlx_decisions import MLXDecisionPredictor
from predict_toy_decisions import reject_nonfinite, unique_object, validate_request
from typesafe_adapter import nanojev_response_to_typesafe, typesafe_request_to_nanojev


class NanoJevASGIApp:
    def __init__(
        self,
        checkpoint_dir: str,
        web_root: str = "web",
        max_length: int = 4096,
        default_temperature: float = 0.35,
        enable_adaptive_temp: bool = True,
        early_exit_layer: int = 0,
        early_exit_confidence: float = 0.98,
    ):
        self.engine = MLXDecisionPredictor(
            checkpoint_dir,
            max_length=max_length,
            enable_adaptive_temp=enable_adaptive_temp,
            early_exit_layer=early_exit_layer,
            early_exit_confidence=early_exit_confidence,
        )
        self.web_root = Path(web_root).resolve()
        self.default_temperature = default_temperature

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

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        if method == "GET":
            if path == "/api/health":
                body = json.dumps({
                    "ready": True,
                    "engine": "mlx-asgi-h2",
                    "http_version": scope.get("http_version", "1.1"),
                    "model_loaded_once": True,
                    "provider_calls": 0,
                    "adaptive_temperature": self.engine.enable_adaptive_temp,
                    "early_exit_layer": self.engine.early_exit_layer,
                }).encode("utf-8")
                await self.send_response(send, 200, body, "application/json; charset=utf-8")
                return

            # Static files
            rel = path.lstrip("/") or "index.html"
            target = (self.web_root / rel).resolve()
            if target.is_relative_to(self.web_root) and target.is_file():
                mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                await self.send_response(send, 200, target.read_bytes(), mime)
                return

            await self.send_response(send, 404, b'{"error":"File not found"}', "application/json; charset=utf-8")
            return

        if method == "POST":
            body = bytearray()
            while True:
                msg = await receive()
                body.extend(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break

            if path == "/v1/systemone":
                await self.handle_typesafe_systemone(send, bytes(body))
                return

            if path == "/api/evaluate":
                await self.handle_api_evaluate(send, bytes(body))
                return

            await self.send_response(send, 404, b'{"error":"Unknown endpoint"}', "application/json; charset=utf-8")
            return

    async def handle_typesafe_systemone(self, send, body_bytes: bytes):
        try:
            ts_req = json.loads(body_bytes.decode("utf-8"))
            temp = float(ts_req.get("temperature", self.default_temperature))
            nj_payload, meta = typesafe_request_to_nanojev(ts_req)
            # MLX streams are thread-local; execute directly on main event loop thread
            nj_res = self.engine.predict(nj_payload, temperature=temp)
            ts_res = nanojev_response_to_typesafe(nj_res, meta)
            resp_bytes = json.dumps(ts_res, ensure_ascii=False, allow_nan=False).encode("utf-8")
            await self.send_response(send, 200, resp_bytes, "application/json; charset=utf-8")
        except (ValueError, TypeError, KeyError) as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            await self.send_response(send, 422, err, "application/json; charset=utf-8")
        except Exception as exc:
            err = json.dumps({"error": f"Internal error: {str(exc)}"}).encode("utf-8")
            await self.send_response(send, 500, err, "application/json; charset=utf-8")

    async def handle_api_evaluate(self, send, body_bytes: bytes):
        try:
            payload = json.loads(
                body_bytes.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_nonfinite,
            )
            states = validate_request(payload)
            t0 = time.perf_counter()
            result = self.engine.predict(payload)
            result["execution"]["server_evaluation_seconds"] = time.perf_counter() - t0
            resp_bytes = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            await self.send_response(send, 200, resp_bytes, "application/json; charset=utf-8")
        except (ValueError, TypeError, KeyError) as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            await self.send_response(send, 400, err, "application/json; charset=utf-8")
        except Exception as exc:
            err = json.dumps({"error": f"MLX inference error: {str(exc)}"}).encode("utf-8")
            await self.send_response(send, 500, err, "application/json; charset=utf-8")

    async def send_response(self, send, status: int, body: bytes, content_type: str):
        headers = [
            (b"content-type", content_type.encode("utf-8")),
            (b"content-length", str(len(body)).encode("utf-8")),
            (b"cache-control", b"no-store"),
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})
