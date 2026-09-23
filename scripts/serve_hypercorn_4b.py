#!/usr/bin/env python3
"""Serve NanoJev-4B decisions via Hypercorn ASGI server with h2c (HTTP/2 cleartext) support.

Features:
- h2c-only or h2c + HTTP/1.1 dual protocol (HTTP/2 over cleartext, no TLS).
- Native HTTP/2 multiplexing and long persistent connection pooling.
- KV Cache state prefix sharing for multi-question requests.
- Dual endpoint support: TypeSafe /v1/systemone & NanoJev /api/evaluate.
- Integrated Apple MPS Multimodal Vision Engine.
"""
import argparse
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve

from asgi_app_4b import NanoJev4BASGIApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/qwen35_4b_oq4e_fp16")
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--http",
        default="h2c",
        choices=("h2c", "h2c-preferred", "h1"),
        help="h2c = HTTP/2 cleartext only (rejects HTTP/1.1); "
             "h2c-preferred = HTTP/2 with HTTP/1.1 fallback; "
             "h1 = HTTP/1.1 only",
    )
    args = parser.parse_args()

    app = NanoJev4BASGIApp(
        checkpoint_dir=args.checkpoint_dir,
        web_root=args.web_root,
        default_temperature=args.temperature,
        allow_h1_fallback=(args.http != "h2c"),
    )

    config = Config()
    config.bind = [f"{args.host}:{args.port}"]
    config.accesslog = "-"
    config.errorlog = "-"
    config.keep_alive_timeout = 120

    if args.http == "h2c":
        config.alpn_protocols = ["h2"]
    elif args.http == "h2c-preferred":
        config.alpn_protocols = ["h2", "http/1.1"]
    else:
        config.alpn_protocols = ["http/1.1"]

    proto_label = {"h2c": "h2c-only (HTTP/2 cleartext)", "h2c-preferred": "h2c + HTTP/1.1 fallback", "h1": "HTTP/1.1 only"}[args.http]
    print(f"Starting Hypercorn NanoJev-4B server on {args.host}:{args.port} [{proto_label}]...", flush=True)
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
