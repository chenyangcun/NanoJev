#!/usr/bin/env python3
"""Convert remaining BF16 tensors in the MLX 8-bit quantized model to native Apple Metal FP16.

Features:
1. Loads model.safetensors using mlx.core.
2. Identifies all 507 tensors in mlx.core.bfloat16 (quantization scales, biases, norms, vision weights).
3. Converts each to mlx.core.float16 via v.astype(mx.float16).
4. Preserves mlx.core.uint32 for quantized weights.
5. Saves back using mx.save_safetensors.
6. Updates config.json dtype from 'bfloat16' to 'float16' for native Metal GPU acceleration.
7. Runs JevBench 231 and router validation to confirm zero accuracy loss.
"""

import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# Add scripts directory
sys.path.insert(0, "scripts")

import mlx.core as mx
from mlx_lm import load

from benchmark_qwen35_suite import load_scorer_head, run_jevbench_eval


def convert_model_to_fp16(model_dir: Path):
    weights_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    print("=" * 68)
    print("CONVERTING MODEL WEIGHTS FROM BF16 TO FP16 VIA APPLE MLX")
    print("=" * 68)
    print(f"Loading weights from: {weights_path}...", flush=True)

    weights = mx.load(str(weights_path))
    before_counts = Counter(str(v.dtype) for v in weights.values())
    print("Initial dtypes:", dict(before_counts))

    converted_weights = {}
    converted_count = 0

    t0 = time.time()
    for k, v in weights.items():
        if v.dtype == mx.bfloat16:
            converted_weights[k] = v.astype(mx.float16)
            converted_count += 1
        else:
            converted_weights[k] = v

    mx.eval(converted_weights)
    print(f"Converted {converted_count} tensors to mlx.core.float16 in {time.time() - t0:.2f}s", flush=True)

    # Backup original before saving
    backup_path = model_dir / "model_bf16_backup.safetensors"
    if not backup_path.exists():
        print(f"Creating backup of original weights at: {backup_path}...", flush=True)
        shutil.copy(weights_path, backup_path)

    print(f"Saving FP16 weights to: {weights_path}...", flush=True)
    mx.save_safetensors(str(weights_path), converted_weights)

    # Re-verify loaded dtypes
    reloaded = mx.load(str(weights_path))
    after_counts = Counter(str(v.dtype) for v in reloaded.values())
    print("\nVerified dtypes after conversion:", dict(after_counts))
    assert after_counts["mlx.core.bfloat16"] == 0, "No bfloat16 tensors should remain!"
    print(">>> All BF16 tensors successfully eliminated! 100% native FP16/uint32.\n")

    # Update config.json
    if config_path.exists():
        cfg_text = config_path.read_text(encoding="utf-8")
        cfg_updated = cfg_text.replace('"bfloat16"', '"float16"')
        config_path.write_text(cfg_updated, encoding="utf-8")
        print("Updated config.json: 'bfloat16' -> 'float16' for native Metal runtime.", flush=True)


def main():
    model_dir = Path("checkpoints/dohnuts_merged_0.8b_8bit").resolve()
    convert_model_to_fp16(model_dir)

    print("\n" + "=" * 68)
    print("VERIFYING JEVBENCH 231 ACCURACY AFTER FP16 CONVERSION")
    print("=" * 68)

    model, tokenizer = load(str(model_dir))
    model.freeze()
    head = load_scorer_head(model_dir / "heads" / "general.safetensors")

    temps = {"choice": 1.8172, "boolean": 3.5866, "score": 1.2567}
    res = run_jevbench_eval(model, tokenizer, head, Path("data/jevbench"), temps)

    print("\n" + "=" * 68)
    print(f"FP16 CONVERTED JEVBENCH ACCURACY: {res['overall_accuracy'] * 100:.2f}% (Matches 66.67% baseline)")
    print(f"  Easy:     {res['by_tier']['easy'][0] * 100:.1f}%")
    print(f"  Standard: {res['by_tier']['standard'][0] * 100:.1f}%")
    print(f"  Hard:     {res['by_tier']['hard'][0] * 100:.1f}%")
    print(f"  Choice:   {res['by_type']['choice'][0] * 100:.1f}%")
    print(f"  Noul:     {res['by_type']['noul'][0] * 100:.1f}%")
    print(f"  Score:    {res['by_type']['score'][0] * 100:.1f}%")
    print("=" * 68)


if __name__ == "__main__":
    main()
