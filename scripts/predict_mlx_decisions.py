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

    # Check if checkpoint uses DeepDecisionHeads (2-layer MLP)
    if run_config.get("heads_architecture") == "deep_mlp":
        from mlx_deep_heads import DeepDecisionHeads
        model.heads = DeepDecisionHeads(hidden_size=model.heads.hidden_size, set_head=run_config.get("set_head", "none"))

    # Load weights from best.safetensors into model
    weights_path = str(paths["weights"])
    backbone_weights = {}
    head_weights = {}

    with safe_open(weights_path, framework="numpy") as f:
        for k in f.keys():
            tensor = mx.array(f.get_tensor(k))
            if k.startswith("backbone."):
                clean_k = "model." + k[len("backbone.") :]
                backbone_weights[clean_k] = tensor
            elif k.startswith("heads."):
                head_weights[k] = tensor
            elif k.startswith("norm."):
                head_weights["heads." + k] = tensor
            elif k.startswith("scalar."):
                head_weights["heads." + k] = tensor
            elif k.startswith("fc1.") or k.startswith("fc2."):
                head_weights["heads." + k] = tensor
            elif k.startswith("set_project."):
                head_weights["heads." + k] = tensor
            elif k.startswith("set_output."):
                head_weights["heads." + k] = tensor
            elif k.startswith("set_attention."):
                head_weights["heads." + k] = tensor

    if backbone_weights:
        model.backbone.load_weights(list(backbone_weights.items()), strict=False)

    if head_weights:
        model.load_weights(list(head_weights.items()), strict=False)

    return model, tokenizer, root, run_config


class MLXDecisionPredictor:
    """Persistent inference predictor on Apple Silicon using MLX."""

    def __init__(self, checkpoint_dir: str, max_length: Optional[int] = None):
        model, tokenizer, root, run_config = load_mlx_decision_model(checkpoint_dir)
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.run_config = run_config
        self.limit = limit
        self.inference_calls = 0

    def predict(self, payload: dict, batch_questions: int = 0, temperature: float = 1.0) -> dict:
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")

        examples = prepare_examples(payload, self.tokenizer, self.limit)
        batches = complete_question_batches(examples, batch_questions)
        self.inference_calls += 1

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

        # Clear Metal memory cache immediately after forward evaluation to release GPU buffer
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
