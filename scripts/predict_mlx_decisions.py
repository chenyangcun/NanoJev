"""MLX-based decision prediction engine for NanoJev.

Provides:
- MLXDecisionPredictor: loads Qwen3 backbone via mlx-lm or Hugging Face config + safetensors,
  executes DecisionHeads natively on Apple Silicon GPU/ANE.
- predict(): drop-in compatible interface with predict_toy_decisions.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer

from mlx_decision_model import MLXDecisionModel
from predict_toy_decisions import (
    answer_from_probabilities,
    complete_question_batches,
    local_checkpoint_files,
    prepare_examples,
    read_json,
    validate_request,
)


def load_mlx_decision_model(checkpoint_dir: str):
    """Load backbone and decision heads weights into MLXDecisionModel."""
    root, paths = local_checkpoint_files(checkpoint_dir)
    run_config = read_json(paths["run_config"])

    tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Qwen3 backbone via mlx_lm architecture
    from mlx_lm.models import qwen3
    cfg = read_json(paths["body_config"] / "config.json")
    if "rope_theta" not in cfg and "rope_parameters" in cfg:
        cfg["rope_theta"] = cfg["rope_parameters"].get("rope_theta", 1000000.0)

    model_args = qwen3.ModelArgs.from_dict(cfg)
    backbone = qwen3.Model(model_args)

    # If checkpoint is quantized, apply quantize structure before loading weights
    quant_cfg = cfg.get("quantization") or run_config.get("quantization")
    if quant_cfg:
        def predicate(path, module):
            if "heads" in path or isinstance(module, (nn.Embedding, nn.RMSNorm, nn.LayerNorm)):
                return False
            return isinstance(module, nn.Linear)
        nn.quantize(backbone, group_size=quant_cfg["group_size"], bits=quant_cfg["bits"], class_predicate=predicate)

    model = MLXDecisionModel(backbone, set_head=run_config.get("set_head", "none"))

    from mlx_deep_heads import DeepDecisionHeads
    from mlx_multi_head_registry import MultiHeadRegistry

    hidden_size = backbone.model.embed_tokens.weight.shape[1]
    registry = MultiHeadRegistry(hidden_size=hidden_size, default_head_name="router")

    # Load weights from best.safetensors into model
    weights_path = str(paths["weights"])
    backbone_weights = {}
    head_weights_by_name = {}

    with safe_open(weights_path, framework="numpy") as f:
        for k in f.keys():
            tensor = mx.array(f.get_tensor(k))
            if k.startswith("backbone."):
                clean_k = "model." + k[len("backbone.") :]
                backbone_weights[clean_k] = tensor
            else:
                # Check for namespaced heads: heads.<name>.<param> vs heads.<param>
                if k.startswith("heads."):
                    sub = k[len("heads.") :]
                    parts = sub.split(".", 1)
                    if len(parts) == 2 and parts[0] not in ("fc1", "fc2", "norm", "scalar", "set_attention", "set_project", "set_output"):
                        h_name, p_name = parts[0], parts[1]
                    else:
                        h_name, p_name = "router", sub
                else:
                    h_name, p_name = "router", k

                if h_name not in head_weights_by_name:
                    head_weights_by_name[h_name] = {}
                head_weights_by_name[h_name][p_name] = tensor

    # Load backbone
    if backbone_weights:
        model.backbone.load_weights(list(backbone_weights.items()), strict=False)

    # Instantiate and register heads from main checkpoint
    for h_name, w_dict in head_weights_by_name.items():
        h_mod = DeepDecisionHeads(hidden_size=hidden_size, set_head=run_config.get("set_head", "none"))
        # Map parameters: fc1.weight -> fc1.weight, norm.weight -> norm.weight
        mapped = [(k if not k.startswith("heads.") else k[6:], v) for k, v in w_dict.items()]
        h_mod.load_weights(mapped, strict=False)
        registry.register_head(h_name, h_mod)

    # Ensure router and default always exist
    if "router" in registry.heads and "default" not in registry.heads:
        registry.register_head("default", registry.heads["router"])
    elif "default" in registry.heads and "router" not in registry.heads:
        registry.register_head("router", registry.heads["default"])

    # 4. Check for external pluggable heads in <checkpoint_dir>/heads/*.safetensors
    heads_dir = root / "heads"
    if heads_dir.is_dir():
        for sf in heads_dir.glob("*.safetensors"):
            h_name = sf.stem.replace("_head", "")
            try:
                plug_weights = {}
                with safe_open(str(sf), framework="numpy") as pf:
                    for pk in pf.keys():
                        t = mx.array(pf.get_tensor(pk))
                        clean_pk = pk
                        if clean_pk.startswith(f"heads.{h_name}."):
                            clean_pk = clean_pk[len(f"heads.{h_name}.") :]
                        elif clean_pk.startswith("heads."):
                            clean_pk = clean_pk[len("heads.") :]
                        plug_weights[clean_pk] = t
                plug_head = DeepDecisionHeads(hidden_size=hidden_size, set_head=run_config.get("set_head", "none"))
                plug_head.load_weights(list(plug_weights.items()), strict=False)
                registry.register_head(h_name, plug_head)
                print(f"[MultiHeadRegistry] Loaded pluggable head '{h_name}' from {sf.name}", flush=True)
            except Exception as e:
                print(f"[MultiHeadRegistry] Failed to load pluggable head from {sf}: {e}", flush=True)

    model.heads = registry
    return model, tokenizer, root, run_config


class MLXDecisionPredictor:
    """Persistent inference predictor on Apple Silicon using MLX with optional Prefix Sharing."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_length: Optional[int] = None,
        enable_prefix_sharing: bool = True,
        enable_adaptive_temp: bool = True,
        early_exit_layer: int = 0,
        early_exit_confidence: float = 0.98,
    ):
        model, tokenizer, root, run_config = load_mlx_decision_model(checkpoint_dir)
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.run_config = run_config
        self.limit = limit
        self.enable_prefix_sharing = enable_prefix_sharing
        self.enable_adaptive_temp = enable_adaptive_temp
        self.early_exit_layer = early_exit_layer
        self.early_exit_confidence = early_exit_confidence
        self.inference_calls = 0

        # Initialize pre-allocated in-place StaticKVCachePool if prefix sharing enabled
        if self.enable_prefix_sharing:
            from cross_question_sharing_engine import StaticKVCachePool
            self.cache_pool = StaticKVCachePool(self.model.backbone.model)
        else:
            self.cache_pool = None

    def predict(self, payload: dict, batch_questions: int = 0, temperature: float = 1.0) -> dict:
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")

        self.inference_calls += 1

        # Highest-performance path: Cross-Question Hierarchical State Sharing Engine
        if self.enable_prefix_sharing:
            from cross_question_sharing_engine import evaluate_state_cross_question_sharing
            outputs = {}
            total_tokens = 0
            candidate_paths = 0
            total_questions = 0

            for st in states:
                st_id = st["id"]
                answers, tok_cnt = evaluate_state_cross_question_sharing(
                    self.model,
                    self.tokenizer,
                    self.cache_pool,
                    state_id=st_id,
                    state_val=st["state"],
                    questions_dict=st["questions"],
                    temperature=temperature,
                    enable_adaptive_temp=self.enable_adaptive_temp,
                    early_exit_layer=self.early_exit_layer,
                    early_exit_confidence=self.early_exit_confidence,
                    max_length=self.limit,
                )
                outputs[st_id] = {"id": st_id, "answers": answers}
                total_tokens += tok_cnt
                total_questions += len(st["questions"])
                for q in st["questions"].values():
                    candidate_paths += 1 if q["type"] == "boolean" else len(q["criteria"])

            return {
                "schema_version": "openjev-mlx-inference-v1",
                "checkpoint": {
                    "directory": str(self.root),
                    "base_model": self.run_config.get("model"),
                    "set_head": self.run_config.get("set_head", "none"),
                },
                "temperature": {"value": float(temperature), "adaptive": self.enable_adaptive_temp},
                "execution": {
                    "engine": "mlx-cross-question-tree-sharing",
                    "device": str(mx.default_device()),
                    "states": len(states),
                    "questions": total_questions,
                    "candidate_paths": candidate_paths,
                    "total_input_tokens": total_tokens,
                    "forward_passes": total_questions + len(states),
                    "early_exit_layer": self.early_exit_layer,
                    "autoregressive_decode_steps": 0,
                    "inference_call_index": self.inference_calls,
                },
                "states": list(outputs.values()),
            }

        # Legacy concatenated batch path
        examples = prepare_examples(payload, self.tokenizer, self.limit)
        batches = complete_question_batches(examples, batch_questions)

        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}

        for batch in batches:
            logits, valid = self.model(batch, self.tokenizer.pad_token_id)
            mx.eval(logits)

            for example, values in zip(batch, logits):
                k = len(example["candidate_ids"])
                scores = values[:k]
                if not mx.all(mx.isfinite(scores)).item():
                    raise ValueError("模型产生非有限logits，未返回部分预测")
                probs = mx.softmax(scores / temperature, axis=-1).tolist()
                outputs[example["state_id"]]["answers"][example["qid"]] = answer_from_probabilities(
                    example, probs
                )

        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

        total_tokens = sum(len(leaf) for ex in examples for leaf in ex["leaf_tokens"])
        return {
            "schema_version": "openjev-mlx-inference-v1",
            "checkpoint": {
                "directory": str(self.root),
                "base_model": self.run_config.get("model"),
                "set_head": self.run_config.get("set_head", "none"),
            },
            "temperature": {"value": float(temperature)},
            "execution": {
                "engine": "mlx",
                "device": str(mx.default_device()),
                "states": len(states),
                "questions": len(examples),
                "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
                "total_input_tokens": total_tokens,
                "forward_passes": len(batches),
                "autoregressive_decode_steps": 0,
                "inference_call_index": self.inference_calls,
            },
            "states": list(outputs.values()),
        }


def predict(payload, checkpoint_dir, temperature=1.0, batch_questions=0, max_length=None):
    engine = MLXDecisionPredictor(checkpoint_dir, max_length=max_length)
    return engine.predict(payload, batch_questions=batch_questions, temperature=temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True, help="Path to JSON file containing {'states': [...]}")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    payload = read_json(args.input)
    result = predict(
        payload,
        checkpoint_dir=args.checkpoint_dir,
        temperature=args.temperature,
        batch_questions=args.batch_questions,
        max_length=args.max_length,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
