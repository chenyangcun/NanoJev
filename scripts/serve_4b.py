#!/usr/bin/env python3
"""Persistent TypeSafe API (/v1/systemone) HTTP Server for NanoJev-4B on Apple Silicon MLX.

Features:
- Native Apple Metal GPU execution using Qwen3.5-4B 8-bit quantized model.
- SemIf direct option-letter readout + NanoJev calibrated probability distributions.
- Zero-token autoregressive decoding: single forward pass for entire decision.
- Full parity with TypeSafe official POST /v1/systemone and GET /api/health.
"""

import argparse
import http.server
import json
import os
import sys
import time
from urllib.parse import urlparse

import mlx.core as mx
from mlx_lm import load

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

    def predict_question(self, state: str, qid: str, q: dict, temperature: float = 1.0) -> dict:
        qtype = q.get("type", "choice")
        crit = q.get("criteria")
        instr = q.get("instructions", "")

        if qtype == "noul" or qtype == "boolean":
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

        for o in options:
            o["description"] = o["id"] + ": " + o["description"]

        payload = {
            "evidence": state,
            "criterion": instr,
            "options": [
                {"letter": LETTERS[i], "description": opt["description"]}
                for i, opt in enumerate(options)
            ]
        }
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(input_ids) > 4096:
            input_ids = input_ids[:4096]

        slots = [self.tokenizer.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(len(options))]

        x = mx.array([input_ids], dtype=mx.int32)
        logits = self.model(x)[:, -1, :]
        mx.eval(logits)

        selected = logits[0, mx.array(slots)].tolist()
        scaled = [v / max(1e-4, temperature) for v in selected]
        probs_list = mx.softmax(mx.array(scaled)).tolist()
        probs = dict(zip([opt["id"] for opt in options], [round(p, 4) for p in probs_list]))

        if qtype in ("noul", "boolean"):
            p_yes = probs.get("yes", 0.5)
            pred = "yes" if p_yes >= 0.5 else "no"
            ans = {
                "type": "noul",
                "noul": p_yes,
                "confidence": max(p_yes, 1.0 - p_yes),
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

        return ans, len(input_ids)

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
                    "engine": "qwen35-4b-mlx",
                    "architecture": "Qwen3.5-4B 8-bit Metal GPU",
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

            answers = {}
            total_tokens = 0
            try:
                for qid, q in raw_questions.items():
                    ans, tok_cnt = predictor.predict_question(state, qid, q, temperature=temp)
                    answers[qid] = ans
                    total_tokens += tok_cnt

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
            # Suppress noisy request logs
            pass

    return NanoJevHandler

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/qwen35_4b_8bit")
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
