#!/usr/bin/env python3
"""Quantize Qwen3.5-4B using oMLX Universal Dynamic Quantization (oQ3e FP16).

Non-critical layers are quantized to 3-bit, while sensitive/attention layers
are boosted to 4/5-bit using data-driven sensitivity and cached imatrix.
Target BPW: 3.5 ~ 3.7 (yielding ~1.7GB model size and ~20% faster inference).
"""
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")
from omlx.oq import quantize_oq_streaming


def main():
    source_dir = Path("checkpoints/qwen35_4b_bf16").resolve()
    output_dir = Path("checkpoints/qwen35_4b_oq3e_fp16").resolve()
    imatrix_cache = Path("checkpoints/.oqe_imatrix/qwen35_4b_bf16-oQe-s128-l512.npz").resolve()

    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}. Removing for fresh quantization...")
        import shutil

        shutil.rmtree(output_dir)

    print("=" * 70)
    print("OMLX STREAMING QUANTIZATION: Qwen3.5-4B -> oQ3e-FP16 (3-Bit Non-Critical)")
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print(f"iMatrix: {imatrix_cache} (reused={imatrix_cache.exists()})")
    print("=" * 70, flush=True)

    t0 = time.time()

    def progress(phase, pct, detail="", meta=None):
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {phase:<15} {pct:5.1f}% {detail} ({elapsed:.1f}s)", flush=True)

    quantize_oq_streaming(
        model_path=str(source_dir),
        output_path=str(output_dir),
        oq_level=3,
        group_size=64,
        progress_callback=progress,
        text_only=True,
        dtype="float16",
        enhanced=True,
        auto_proxy_sensitivity=True,
        imatrix_cache_path=str(imatrix_cache) if imatrix_cache.exists() else "",
        imatrix_reuse_cache=True,
    )

    print(f"\noQ3e FP16 QUANTIZATION COMPLETED SUCCESSFULLY IN {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
