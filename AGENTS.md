# AGENTS.md

Guidance for automated agents working in NanoJev.

## Architecture & System Overview

NanoJev is a 0.6B parallel decision model based on Qwen3-0.6B with custom decision heads (`choice`, `boolean`, `score`). It outputs full probability distributions in one forward pass with zero token autoregressive decoding.

- **Dual-language codebase**: Python 3.11+ (model training, evaluation, data synthesis, local HTTP serving) and Node.js >=22 (Vercel AI SDK integration, teacher probing, browser capture/replay).
- **Core modules** live flat in `scripts/`:
  - `predict_toy_decisions.py` / `train_pipeline_decisions.py`: Model architecture, decision heads, loss functions, token packing, and training loop.
  - `calibrated_objectives.py`: Proper scoring rules (Brier loss, paired proper-reward learning).
  - `scaled_maze.py` / `snake_game.py`: Environment generators, deterministic state transitions, and rule solvers.
  - `serve_decisions.py`: Persistent local HTTP inference server exposing `POST /api/evaluate`.
  - `teachers.mjs` / `jev_probe.mjs`: Node.js wrapper around Vercel AI SDK / TypeSafe gateway.
- **Contract nuances**:
  - Question types: `choice` (2–255 options), `boolean` (single proposition or true/false criteria), `score` (2–10 ordered levels).
  - Schema uses `boolean`, NOT direct TypeSafe API's `noul`.
  - Input isolation: Each candidate/option is tokenized independently with state + instruction; no cross-question attention.
  - Labels use `gold_probs` (probability distributions) or `gold` (hard targets).

## Developer Commands

### Environment & Setup
- **Python**: requires Python 3.11+; ML packages (`torch==2.14.0`, `transformers==5.17.0`, `safetensors`, `numpy`) in `requirements-toy.txt`. Note: standard-library scripts run without `torch`.
- **Node.js**: Node >=22. Run `npm install` for dependencies (`ai`).
- **Env**: Copy `.env.example` to `.env` (`AI_GATEWAY_API_KEY`) only when calling external teacher models or AI gateways.

### Running Tests
All unit tests live in `scripts/` with `test_*.py` or `test_*.mjs`. Because `scripts/` is not a packaged directory, import paths expect `scripts` on the Python path or runner targeting:

```bash
# 1. Standard-library unit tests (no PyTorch / GPU / network required)
python3 -m unittest discover -s scripts -p "test_*.py"

# Or run individual Python test files directly:
python3 scripts/test_question_contract.py
python3 scripts/test_snake_game.py
python3 scripts/test_scaled_maze.py
python3 scripts/test_game_outcomes.py
python3 scripts/test_scaled_pipeline.py
python3 scripts/test_composed_snake.py
python3 scripts/test_composed_maze.py
python3 scripts/test_freeze_scaled_labels.py
python3 scripts/test_build_local_maze_data.py
python3 scripts/test_model_edges_maze.py

# 2. PyTorch objective tests (requires torch)
python3 scripts/test_calibrated_objectives.py

# 3. Node.js distribution probe tests (offline, no API key needed)
node scripts/test_probe_jev_distributions.mjs
```

### Data Pipeline & Verification
```bash
# Dataset generation
npm run data:workflows   # python3 scripts/build_workflow_decisions.py
npm run data:games       # python3 scripts/build_game_decisions.py

# Validate JSONL dataset schema without training
python3 scripts/train_pipeline_decisions.py --input <path-to-jsonl> --validate-only
```

### Serving & Web Replay
```bash
# Web demo replay (local static server)
npm run demo:replay      # python3 -m http.server 8080 --bind 127.0.0.1 --directory web
# Access at http://127.0.0.1:8080/side-by-side.html or /arcade.html

# Persistent model serving on Apple Silicon macOS (MLX Native)
# Exposes both TypeSafe official `POST /v1/systemone` and batch `POST /api/evaluate`
python3 scripts/serve_mlx_decisions.py --checkpoint-dir checkpoints/NanoJev --web-root web --port 8765

# Persistent model serving on CUDA (PyTorch)
python3 scripts/serve_decisions.py --checkpoint-dir checkpoints/NanoJev --web-root web --port 8765
```

### Apple Silicon (MLX) Workflow
NanoJev includes native MLX implementations for zero-PyTorch execution on macOS:
- Requirements: `pip install -r requirements-mlx.txt`
- Unit tests: `python3 scripts/test_calibrated_objectives_mlx.py`, `python3 scripts/test_mlx_pipeline.py`, `python3 scripts/test_typesafe_api.py`
- MLX Inference: `python3 scripts/predict_mlx_decisions.py --checkpoint-dir <dir> --input <file>`
- MLX Training: `python3 scripts/train_mlx_decisions.py --input <data.jsonl> --base-checkpoint <dir> --output-dir checkpoints/mlx_run`
- TypeSafe Wire Format: `scripts/typesafe_adapter.py` enables standard TypeSafe API wire format (`POST /v1/systemone`) compatibility.

## Critical Gotchas & Constraints

- **Python Path in Tests**: Do NOT run `python3 -m unittest scripts/test_foo.py` without `PYTHONPATH=scripts` because test files import sister modules in `scripts/` directly (e.g. `import scaled_maze`). Use `python3 scripts/test_foo.py` or `python3 -m unittest discover -s scripts -p "test_foo.py"`.
- **Torch Dependency Separation**: `test_calibrated_objectives.py` requires `torch`. Other `test_*.py` files use standard library and mocks; they can and should run without PyTorch.
- **Do Not Guess API Keys / Endpoints**: Generating data programmatically requires no keys. Only `teachers.mjs` and live probing require `AI_GATEWAY_API_KEY`.
- **Model Checkpoints**: `.safetensors`, `checkpoints/`, and large datasets are gitignored. Do not commit checkpoint files or large generated datasets.
