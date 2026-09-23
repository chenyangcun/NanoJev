#!/usr/bin/env python3
"""Quantize Qwen3.5-4B using oMLX Universal Dynamic Quantization (oQ4e FP16)."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")
from omlx.oq import quantize_oq_streaming

def main():
    source_dir = Path("checkpoints/qwen35_4b_bf16").resolve()
    output_dir = Path("checkpoints/qwen35_4b_oq4e_fp16").resolve()

    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}. Removing for fresh quantization...")
        import shutil
        shutil.rmtree(output_dir)

    print("=" * 70)
    print("OMLX STREAMING QUANTIZATION: Qwen3.5-4B -> oQ4e-fp16")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70, flush=True)

    t0 = time.time()

    def progress(phase, pct, detail="", meta=None):
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {phase:<15} {pct:5.1f}% {detail} ({elapsed:.1f}s)", flush=True)

    quantize_oq_streaming(
        model_path=str(source_dir),
        output_path=str(output_dir),
        oq_level=4,
        group_size=64,
        progress_callback=progress,
        text_only=True,
        dtype="float16",
        enhanced=True,
        auto_proxy_sensitivity=True,
        imatrix_reuse_cache=True,
    )

    print(f"\noQ4e FP16 QUANTIZATION COMPLETED SUCCESSFULLY IN {time.time()-t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
