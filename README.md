# NanoJev-MLX — Native Parallel Decision Model on Apple Silicon

**English** | [简体中文](README.zh-CN.md)

**A 0.6B Apple Silicon native parallel decision model. States and questions in, complete probability distributions out — with zero autoregressive token decoding.**

NanoJev-MLX is forked and extended from the original project [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev), reproducing and advancing [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) into a fully local, high-throughput, and energy-efficient System One decision engine. Built with Apple MLX and Core ML ANE, it delivers **~5ms ultra-low latency on Neural Engine (ANE)** and **~230ms deep multi-turn agent reasoning on 8-bit Metal GPU**.

[Upstream Project (TianyuCodings/NanoJev)](https://github.com/TianyuCodings/NanoJev) · [Optimization & Architecture Report](docs/OPTIMIZATION_SUMMARY.md) · [TypeSafe API Wire Spec](docs/TYPESAFE_CONTRACT.md) · [Benchmarks](docs/DEVELOPMENT_RESULTS.md)

---

## ⚡ Highlights & Key Innovations

- **Heterogeneous Dual-Engine Concurrency (`DualEngineRouter`)**:
  - **ANE Fast-Lane (`~5ms P50`)**: Routes short (<90 tokens), latency-sensitive boolean/choice checks to the Apple Neural Engine (NPU) with zero GPU memory allocation and only ~0.15 J energy per decision.
  - **Metal GPU Deep-Lane (`~250ms`)**: Routes rich, multi-turn agent sessions with tool call context to an 8-bit quantized Qwen3-0.6B backbone on MLX.
- **Hierarchical Cross-Question State Sharing (Tree-Prefill)**:
  - Long multi-field `state` text is prefilled **only ONCE** per request across all questions.
  - Candidate options fork in parallel via zero-copy tensor broadcasting. Cuts backbone computation tokens by **84.5%**.
- **Static In-Place KV-Cache Memory Pool**:
  - Global pre-allocated prompt cache resets via in-place `.trim(offset)` with **Zero Dynamic Memory Allocations and zero GC pauses**.
- **MLX 8-Bit Native Quantization**:
  - Backbone linear layers compressed from 2.27GB down to **1.06GB (53% memory reduction)**, running smoothly alongside other 30B+ LLMs without OOM.
- **Adaptive Multi-Feature Confidence Pooling**:
  - Merges Top-1 probability, decision margin ($\Delta = \text{Top1} - \text{Top2}$), normalized Shannon entropy complement, and capacity factors to dynamically tune temperature ($T = 0.25 \sim 0.85$), eliminating overconfidence on ambiguous tasks while achieving **0.99~1.00 confidence** on clear decisions.
- **Pluggable Multi-Head Architecture (`MultiHeadRegistry`)**:
  - Decoupled `DeepDecisionHeads` for independent task domains: `router` (complexity & risk), `skill` (tool selection), `agent` (subagent delegation), and `news` (editorial scoring).
  - Supports lightweight standalone head hot-plugging (~2MB per head).
- **Production HTTP/2 & HTTP/1.1 ASGI Serving**:
  - Powered by Hypercorn with persistent connection pooling and binary frame multiplexing (**91.9 req/s throughput**).
  - 100% wire-compatible with TypeSafe official `POST /v1/systemone` and batch `POST /api/evaluate`.
- **Automated Dual-Log Shadow Distillation (Continual Self-Learning)**:
  - Automated hourly cron sync (`scripts/daily_shadow_auto_loop.sh`) matching production router shadow logs against cloud Jev via `request_id`.
  - Automatically harvests hard disagreement samples into distillation datasets and triggers background retraining.

---

## 📊 Evaluation & Benchmark Results

Verified against the official 36-case Bilingual Benchmark and 24-case Realistic Compressed Session Benchmark:

| Evaluation Dimension | Baseline (Original NanoJev) | NanoJev-MLX (Current) | Status |
| :--- | :---: | :---: | :---: |
| **Realistic Session Benchmark (24 Cases)** | 10 / 24 (41.7%) ❌ | **24 / 24 (100.0%)** 🌟 | **100% Match** |
| **Bilingual Benchmark (36 Cases)** | 2 / 12 (16.7%) ❌ | **35 / 36 (97.2%)** 🌟 | **Production Ready** |
| **High-Risk Production Gate Recall (`>=0.5`)** | 0 / 6 (0.0%) ❌ (Missed) | **100.0% (17/17 all caught)** 🚨 | **Zero Leakage** |
| **Low-Risk False Positive Alarm Rate** | 7 False Alarms (41%) ❌ | **0 False Alarms (0.0%)** ✅ | **Zero Noise** |
| **Decision Confidence Level** | 0.18 ~ 0.23 (Too low) | **0.98 ~ 1.00 (Calibrated)** ⚡ | **Decisive** |
| **Physical Memory Footprint** | 2.27 GB (Heavy) | **1.06 GB (53% saved)** | **Ultra-Light** |
| **End-to-End Latency (P50)** | ~870 ms (P95 ~4.7s) | **~258 ms (P95 <380ms)** | **3.3× Faster** |

---

## 🚀 Quick Start on Apple Silicon (macOS)

### 1. Environment Setup

Requirements: macOS 14+ (macOS 15+ recommended), Apple Silicon (M1/M2/M3/M4), Python 3.11+.

```bash
git clone git@github.com:chenyangcun/NanoJev-MLX.git
cd NanoJev-MLX

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mlx.txt
```

### 2. Download Pre-trained Weights

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='C-Tianyu/NanoJev', local_dir='checkpoints/NanoJev')
"
```

### 3. Start High-Performance HTTP/2 System One Server

```bash
# Serves with Dual-Engine (ANE Fast-Lane + 8-bit MLX Metal GPU) on port 8769
python3 scripts/serve_hypercorn.py \
  --checkpoint-dir checkpoints/router_quant_8bit \
  --host 0.0.0.0 \
  --port 8769 \
  --temperature 0.35 \
  --max-length 4096
```

Health check:
```bash
curl -s http://127.0.0.1:8769/api/health
```

---

## 📡 API Usage (TypeSafe System One Compatible)

NanoJev-MLX exposes the standard TypeSafe System One wire protocol at `POST /v1/systemone`:

```bash
curl -X POST http://127.0.0.1:8769/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": {
      "user_task": "Review OAuth callback handling for token leakage and propose code changes; do not deploy anything."
    },
    "questions": {
      "complexity": {
        "type": "choice",
        "instructions": "Choose the complexity of the next coding-agent call.",
        "criteria": {
          "bounded": "Small isolated fix",
          "standard": "Normal feature work",
          "complex": "Architecture migration, concurrency, or security review",
          "exceptional": "Production outage intervention"
        }
      },
      "high_risk": {
        "type": "noul",
        "instructions": "Does the next coding-agent call involve a high-consequence operation?"
      }
    }
  }'
```

**Response**:
```json
{
  "model": "jev-latest",
  "answers": {
    "complexity": {
      "type": "choice",
      "choice": "complex",
      "probabilities": {
        "bounded": 0.0,
        "standard": 0.0,
        "complex": 1.0,
        "exceptional": 0.0
      },
      "confidence": 1.0
    },
    "high_risk": {
      "type": "noul",
      "noul": 0.9998
    }
  },
  "usage": {
    "input_tokens": 573,
    "output_tokens": 0
  }
}
```

---

## 🛠️ CLI & Management Commands

### Service Daemon Management (LaunchAgent)
On macOS servers (e.g. M1 Mac Studio):
```bash
nanojev-service status   # Check service status, PID, and health
nanojev-service restart  # Restart persistent daemon
nanojev-service logs     # Stream live inference & access logs
```

### Benchmark & Validation
```bash
# Evaluate against full realistic session cases
python3 /path/to/evaluate-local-jev.py --cases /tmp/jev-realistic-cases.json

# HTTP/2 vs HTTP/1.1 latency & throughput benchmark
python3 scripts/benchmark_http2.py
python3 scripts/benchmark_concurrency.py

# MultiHeadRegistry unit tests
python3 scripts/test_multi_head_registry.py
```

### Model Quantization & Pruning
```bash
# Export 8-bit MLX Quantized checkpoint (53% memory reduction)
python3 scripts/quantize_mlx_model.py \
  --source-dir checkpoints/router_realistic_mlp \
  --target-dir checkpoints/router_quant_8bit \
  --bits 8

# Structural depth pruning (e.g. prune 28 layers down to 14 layers)
python3 scripts/prune_mlx_model.py \
  --source-dir checkpoints/router_realistic_mlp \
  --target-dir checkpoints/router_pruned_14l \
  --target-layers 14
```

---

## 📂 Project Structure

```text
NanoJev-MLX/
├── checkpoints/                 # Local model checkpoints (8-bit quantized, pluggable heads)
├── data/                        # Curated distillation datasets and harvested shadow pairs
├── docs/
│   ├── OPTIMIZATION_SUMMARY.md  # Deep technical breakdown of all 15 optimizations
│   ├── TYPESAFE_CONTRACT.md     # TypeSafe System One compatibility specification
│   └── DEVELOPMENT_RESULTS.md   # Original navigation & game benchmark results
├── scripts/
│   ├── dual_engine_router.py    # ANE (NPU) + MLX (GPU) heterogeneous concurrency router
│   ├── cross_question_sharing_engine.py  # Level-1/2/3 hierarchical prefix sharing engine
│   ├── fast_decision_engine.py  # Static in-place KV-cache memory pool & JIT kernels
│   ├── mlx_multi_head_registry.py # Pluggable multi-head registry with heuristic audit logs
│   ├── adaptive_confidence.py   # Multi-signal confidence feature engineering
│   ├── serve_hypercorn.py       # High-concurrency HTTP/2 + HTTP/1.1 ASGI server
│   ├── asgi_app.py              # Asynchronous ASGI endpoint router
│   ├── quantize_mlx_model.py    # Native 8-bit/4-bit MLX quantization tool
│   ├── shadow_pipeline.py       # Dual-log pairing & automated disagreement harvester
│   └── daily_shadow_auto_loop.sh # Hourly cron automation script
└── requirements-mlx.txt         # Clean Apple Silicon native dependencies
```

---

## 📄 License & Attribution

- Core implementation licensed under **Apache-2.0**.
- Forked and evolved from the original [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) by Tianyu Chen.
- Built on top of the open-weight **Qwen3-0.6B** backbone by Alibaba Qwen Team.
- System One specification and input semantics inspired by [TypeSafe AI](https://typesafe.ai).
- ANE and Core ML design elements adapted from [Laya](https://github.com/NandhaKishorM/laya) and [laya-coreml](https://github.com/mizorewww/laya-coreml).
