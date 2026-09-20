#!/usr/bin/env python3
"""Serve NanoJev decisions via Hypercorn ASGI server with HTTP/2 and HTTP/1.1 support.

Features:
- Native HTTP/2 multiplexing and long persistent connection pooling.
- Dual endpoint support: TypeSafe official /v1/systemone & NanoJev /api/evaluate.
- Integrated Dynamic Adaptive Temperature.
- Integrated Dynamic Early Exit.
"""
import argparse
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve

from asgi_app import NanoJevASGIApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--disable-adaptive-temp", action="store_true", help="Disable adaptive temperature")
    parser.add_argument("--early-exit-layer", type=int, default=14, help="Intermediate layer for dynamic early exit probe (0 to disable)")
    parser.add_argument("--early-exit-confidence", type=float, default=0.98, help="Confidence threshold for early exit")
    parser.add_argument("--disable-ane", action="store_true", help="Disable ANE fast lane")
    parser.add_argument("--ane-checkpoint-dir", default="checkpoints/laya_multilingual_ane", help="Path to ANE CoreML model")
    args = parser.parse_args()

    app = NanoJevASGIApp(
        checkpoint_dir=args.checkpoint_dir,
        web_root=args.web_root,
        max_length=args.max_length,
        default_temperature=args.temperature,
        enable_adaptive_temp=not args.disable_adaptive_temp,
        early_exit_layer=args.early_exit_layer,
        early_exit_confidence=args.early_exit_confidence,
        enable_ane=not args.disable_ane,
        ane_checkpoint_dir=args.ane_checkpoint_dir,
    )

    config = Config()
    config.bind = [f"{args.host}:{args.port}"]
    config.accesslog = "-"
    config.errorlog = "-"
    config.keep_alive_timeout = 120
    # Enable HTTP/2 over h2c (cleartext HTTP)
    config.alpn_protocols = ["h2", "http/1.1"]

    print(f"Starting Hypercorn HTTP/2 + HTTP/1.1 NanoJev server on {args.host}:{args.port}...", flush=True)
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
