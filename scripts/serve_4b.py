#!/usr/bin/env python3
"""Persistent TypeSafe API (/v1/systemone) HTTP Server for NanoJev-4B on Apple Silicon MLX.

Features:
- Native Apple Metal GPU execution using Qwen3.5-4B oQ4e-FP16 (2.3GB) quantized model.
- Dynamic KV Cache Prefix Sharing: state is prefilled ONCE across all questions in a request!
- Inter-request state prefix caching for multi-turn conversations and re-evaluation.
- Zero-token autoregressive decoding: single forward pass for entire decision.
- Calibrated temperature scaling and conservative tie-breaking from NanoJev-0.8B.
- Full parity with TypeSafe official POST /v1/systemone and GET /api/health.
- Integrated Apple MPS Multimodal Vision Engine for image/screenshot tasks.
"""

import argparse
import copy
import hashlib
import http.server
import json
import os
import sys
import time
from urllib.parse import urlparse

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

class NanoJev4BPredictor:
    def __init__(self, checkpoint_dir: str):
        print(f"Loading Qwen3.5-4B from: {checkpoint_dir} ...", flush=True)
        t0 = time.time()
        self.model, self.tokenizer = load(checkpoint_dir, tokenizer_config={"trust_remote_code": False})
        print(f"Model loaded in {time.time()-t0:.2f}s!", flush=True)

        self.slot_tokens = [self.tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]
        self.cached_state_hash = None
        self.cached_kv = None
        self.calibrated_temps = {"choice": 1.8172, "boolean": 3.5866, "score": 1.2567}

        # Initialize Multimodal Vision Engine (Apple MPS Qwen3.5)
        nj_scripts = os.path.expanduser("~/work/NanoJev/scripts")
        if nj_scripts not in sys.path:
            sys.path.insert(0, nj_scripts)
        try:
            from vision_decision_engine import VisionDecisionEngine
            vision_ckpt = os.path.expanduser("~/work/NanoJev/checkpoints/dohnuts_merged_0.8b")
            self.vision_engine = VisionDecisionEngine(checkpoint_dir=vision_ckpt)
            print("Multimodal Vision Decision Engine ready on Apple MPS!", flush=True)
        except Exception as e:
            print(f"Vision Decision Engine not loaded ({e}); continuing with pure-text lanes.", flush=True)
            self.vision_engine = None

    def predict_multi_questions(self, state: object, raw_questions: dict, temperature: float = 1.0) -> tuple[dict, int]:
        state_text = json.dumps(state, ensure_ascii=False) if isinstance(state, (dict, list)) else str(state)
        prefix_text = f"<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\nEvidence:\n{state_text}\n\n"
        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_tokens) > 3500:
            prefix_tokens = prefix_tokens[:3500]

        state_hash = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

        # 1. Prefill state prefix into KV cache (or reuse if matching state)
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

        # 2. Evaluate each question against branched KV cache
        for qid, q in raw_questions.items():
            qtype = q.get("type", "choice")
            crit = q.get("criteria")
            instr = q.get("instructions", "")

            if qtype in ("noul", "boolean"):
                options = [
                    {"id": "yes", "description": (crit or {}).get("true", "The proposition is true.")},
                    {"id": "no", "description": (crit or {}).get("false", "The proposition is false.")}
                ]
            elif qtype == "choice":
                if isinstance(crit, dict):
                    options = [{"id": k, "description": v or k} for k, v in crit.items()]
                else:
                    options = [{"id": str(i), "description": str(opt)} for i, opt in enumerate(crit or [])]
            else: # score
                if isinstance(crit, list):
                    options = [{"id": str(i), "description": str(lvl)} for i, lvl in enumerate(crit)]
                else:
                    options = [{"id": str(i), "description": f"Level {i}"} for i in range(5)]

            opts_lines = []
            for i, opt in enumerate(options):
                opts_lines.append(LETTERS[i] + ": " + opt["id"] + ": " + opt["description"])
            opts_text = "\n".join(opts_lines)

            suffix_text = f"Criterion:\n{instr}\n\nOptions:\n{opts_text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            suffix_tokens = self.tokenizer.encode(suffix_text, add_special_tokens=False)
            total_tokens += len(suffix_tokens)

            # Branch the prefilled cache (instantaneous copy)
            branch = copy.deepcopy(base_cache)
            logits = self.model(mx.array([suffix_tokens], dtype=mx.int32), cache=branch)[:, -1, :]
            mx.eval(logits)

            slots = self.slot_tokens[:len(options)]
            selected = logits[0, mx.array(slots)].tolist()

            calib_factor = self.calibrated_temps.get(qtype, 1.0)
            eff_temp = calib_factor * max(1e-4, temperature)
            scaled = [v / eff_temp for v in selected]
            probs_list = mx.softmax(mx.array(scaled)).tolist()
            probs = dict(zip([opt["id"] for opt in options], [round(p, 4) for p in probs_list]))

            if qtype in ("noul", "boolean"):
                p_yes = probs.get("yes", 0.5)
                # Conservative negative-default tie breaker
                pred = "yes" if p_yes > 0.505 else "no"
                ans = {
                    "type": "noul",
                    "noul": p_yes,
                    "confidence": round(max(p_yes, 1.0 - p_yes), 4),
                    "probabilities": {"yes": p_yes, "no": round(1.0 - p_yes, 4)}
                }
            elif qtype == "choice":
                pred = max(probs.keys(), key=lambda k: probs[k])
                ans = {
                    "type": "choice",
                    "choice": pred,
                    "probabilities": probs,
                    "confidence": probs[pred]
                }
            else: # score
                pred = max(probs.keys(), key=lambda k: probs[k])
                ans = {
                    "type": "score",
                    "score": int(pred),
                    "probabilities": probs,
                    "confidence": probs[pred]
                }

            answers[qid] = ans

        return answers, total_tokens

def make_handler(predictor: NanoJev4BPredictor, default_temp: float = 1.0):
    class NanoJevHandler(http.server.BaseHTTPRequestHandler):
        def send_json(self, status_code: int, data: dict):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/api/health", "/health"):
                self.send_json(200, {
                    "ready": True,
                    "model": "nanojev-4b",
                    "engine": "nanojev-4b-hybrid",
                    "architecture": "Qwen3.5-4B oQ4e-FP16 Metal GPU (2.3GB) + Multimodal Vision MPS",
                    "quantization": "oq4e-fp16",
                    "prefix_sharing": True,
                    "vision_available": predictor.vision_engine is not None,
                    "status": "healthy"
                })
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ("/v1/systemone", "/api/evaluate"):
                self.send_json(404, {"error": "Unknown endpoint"})
                return

            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self.send_json(400, {"error": "Missing body"})
                return

            body_bytes = self.rfile.read(length)
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self.send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            state = payload.get("state", "")
            raw_questions = payload.get("questions", {})
            temp = float(payload.get("temperature", default_temp))

            # 1. Multimodal Vision Lane: if state contains an image, route to VisionDecisionEngine
            if predictor.vision_engine and predictor.vision_engine.has_image(state):
                try:
                    from typesafe_adapter import typesafe_request_to_nanojev, nanojev_response_to_typesafe
                    nj_payload, meta = typesafe_request_to_nanojev(payload)
                    nj_res = predictor.vision_engine.predict(nj_payload, temperature=temp)
                    resp = nanojev_response_to_typesafe(nj_res, meta)
                    self.send_json(200, resp)
                    return
                except Exception as exc:
                    self.send_json(500, {"error": f"Vision inference error: {exc}"})
                    return

            # 2. Text Decision Lane with KV Cache Prefix Sharing
            try:
                answers, total_tokens = predictor.predict_multi_questions(state, raw_questions, temperature=temp)
                resp = {
                    "model": "nanojev-4b",
                    "answers": answers,
                    "usage": {
                        "input_tokens": total_tokens,
                        "output_tokens": 0
                    }
                }
                self.send_json(200, resp)
            except Exception as exc:
                self.send_json(500, {"error": f"Inference error: {exc}"})

        def log_message(self, fmt, *args):
            pass

    return NanoJevHandler

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/qwen35_4b_oq4e_fp16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    predictor = NanoJev4BPredictor(args.checkpoint_dir)
    handler = make_handler(predictor, default_temp=args.temperature)

    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"NanoJev-4B server listening on http://{args.host}:{args.port} (PID: {os.getpid()})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...", flush=True)
        server.server_close()

if __name__ == "__main__":
    main()
