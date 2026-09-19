#!/usr/bin/env python3
"""Quantize NanoJev Qwen3 backbone using Apple MLX native quantization.

Supports:
- 4-bit and 8-bit weight quantization (group_size=64 by default)
- Keeps the classifier heads (DeepDecisionHeads) in full precision (float16/float32) for zero accuracy loss
- Generates a standalone quantized checkpoint
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors.numpy import save_file
import numpy as np

from predict_mlx_decisions import load_mlx_decision_model


def quantize_nanojev_checkpoint(source_dir: str, target_dir: str, bits: int = 8, group_size: int = 64):
    src = Path(source_dir).resolve()
    dst = Path(target_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loading full precision model from: {src}")
    model, tokenizer, root, run_config = load_mlx_decision_model(str(src))

    # 1. Quantize only the backbone linear layers, keep heads unquantized
    def predicate(path, module):
        # Do not quantize decision heads (keep classification sharp and precise)
        if "heads" in path:
            return False
        # Do not quantize embeddings or rms norms
        if isinstance(module, (nn.Embedding, nn.RMSNorm, nn.LayerNorm)):
            return False
        return isinstance(module, nn.Linear)

    print(f"Quantizing backbone linear layers to {bits}-bit (group_size={group_size})...")
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=predicate,
    )
    mx.eval(model.parameters())

    # 2. Copy configs and tokenizer
    tok_dst = dst / "tokenizer"
    if tok_dst.exists():
        shutil.rmtree(tok_dst)
    shutil.copytree(src / "tokenizer", tok_dst)

    cfg_dst = dst / "backbone_config"
    if cfg_dst.exists():
        shutil.rmtree(cfg_dst)
    shutil.copytree(src / "backbone_config", cfg_dst)

    # Update backbone_config with quantization info
    bb_cfg_path = cfg_dst / "config.json"
    bb_cfg = json.loads(bb_cfg_path.read_text(encoding="utf-8"))
    bb_cfg["quantization"] = {"group_size": group_size, "bits": bits}
    bb_cfg_path.write_text(json.dumps(bb_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Update run config.json
    run_cfg_path = dst / "config.json"
    run_cfg = dict(run_config)
    run_cfg["quantization"] = {"group_size": group_size, "bits": bits}
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 3. Export quantized weights
    flat_weights = {}

    def recurse(prefix, obj):
        if isinstance(obj, mx.array):
            flat_weights[prefix] = obj
        elif isinstance(obj, dict):
            for sub_k, sub_v in obj.items():
                recurse(f"{prefix}.{sub_k}" if prefix else sub_k, sub_v)
        elif isinstance(obj, list):
            for i, sub_v in enumerate(obj):
                recurse(f"{prefix}.{i}" if prefix else str(i), sub_v)

    for k, v in model.parameters().items():
        recurse(k, v)

    converted_weights = {}
    for k, v in flat_weights.items():
        clean_k = k
        if clean_k.startswith("backbone.model."):
            clean_k = "backbone." + clean_k[len("backbone.model.") :]
        elif clean_k.startswith("heads."):
            clean_k = clean_k[len("heads.") :]
        converted_weights[clean_k] = np.array(v)

    out_weights = dst / "best.safetensors"
    save_file(converted_weights, str(out_weights))

    orig_size_mb = (src / "best.safetensors").stat().st_size / (1024 * 1024)
    new_size_mb = out_weights.stat().st_size / (1024 * 1024)
    print(f"\nCheckpoint quantized successfully!")
    print(f"Original size: {orig_size_mb:.1f} MB -> Quantized ({bits}-bit): {new_size_mb:.1f} MB (Reduction: {(1 - new_size_mb/orig_size_mb)*100:.1f}%)")
    print(f"Saved to: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="checkpoints/router_realistic_mlp")
    parser.add_argument("--target-dir", default="checkpoints/router_quant_8bit")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    quantize_nanojev_checkpoint(args.source_dir, args.target_dir, args.bits, args.group_size)
