#!/usr/bin/env python3
"""Structural Layer Pruning for NanoJev in Apple MLX.

Prunes Qwen3-0.6B backbone from 28 layers down to N layers (e.g. 14 layers, a 50% depth reduction).
Pruning strategies:
- 'uniform': evenly sample layers (e.g., [0, 2, 4, 6, ..., 26])
- 'skip_middle': preserve early (syntactic/lexical) and late (task semantic) layers, drop redundant middle blocks
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def prune_checkpoint(source_dir: str, target_dir: str, target_layers: int = 14, strategy: str = "uniform"):
    src = Path(source_dir).resolve()
    dst = Path(target_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    # 1. Copy tokenizer
    tok_src = src / "tokenizer"
    tok_dst = dst / "tokenizer"
    if tok_dst.exists():
        shutil.rmtree(tok_dst)
    shutil.copytree(tok_src, tok_dst)

    # 2. Modify backbone_config
    cfg_src = src / "backbone_config" / "config.json"
    cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
    orig_layers = cfg["num_hidden_layers"]
    print(f"Original layers: {orig_layers} -> Target layers: {target_layers} (Strategy: {strategy})")

    if strategy == "uniform":
        # Evenly spaced layer selection
        selected_layers = [int(round(i * (orig_layers - 1) / (target_layers - 1))) for i in range(target_layers)]
    elif strategy == "skip_middle":
        # Keep first 4, last 4, uniformly sample middle
        keep_ends = min(4, target_layers // 3)
        middle_slots = target_layers - 2 * keep_ends
        first = list(range(keep_ends))
        last = list(range(orig_layers - keep_ends, orig_layers))
        middle_orig = list(range(keep_ends, orig_layers - keep_ends))
        step = len(middle_orig) / middle_slots
        sampled_middle = [middle_orig[int(i * step)] for i in range(middle_slots)]
        selected_layers = sorted(list(set(first + sampled_middle + last)))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Ensure strictly target_layers unique sorted indices
    selected_layers = sorted(list(set(selected_layers)))
    while len(selected_layers) < target_layers:
        for i in range(orig_layers):
            if i not in selected_layers:
                selected_layers.append(i)
                break
        selected_layers.sort()
    selected_layers = selected_layers[:target_layers]

    print(f"Selected layer indices from original backbone:\n  {selected_layers}")

    cfg_dst_dir = dst / "backbone_config"
    cfg_dst_dir.mkdir(parents=True, exist_ok=True)
    cfg["num_hidden_layers"] = target_layers
    if "layer_types" in cfg and isinstance(cfg["layer_types"], list):
        cfg["layer_types"] = [cfg["layer_types"][i] for i in selected_layers]
    (cfg_dst_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 3. Modify run config.json
    run_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    run_cfg["num_hidden_layers"] = target_layers
    run_cfg["pruned_layers"] = selected_layers
    run_cfg["pruning_strategy"] = strategy
    (dst / "config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 4. Prune safetensors weights
    weights_path = src / "best.safetensors"
    layer_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_layers)}

    pruned_weights = {}
    orig_keys_count = 0
    orig_params = 0
    with safe_open(str(weights_path), framework="numpy") as f:
        orig_keys = list(f.keys())
        orig_keys_count = len(orig_keys)
        for k in orig_keys:
            tensor = f.get_tensor(k)
            orig_params += tensor.size
            parts = k.split(".")
            if "layers" in parts:
                idx_pos = parts.index("layers") + 1
                try:
                    old_layer_idx = int(parts[idx_pos])
                    if old_layer_idx in layer_map:
                        new_layer_idx = layer_map[old_layer_idx]
                        parts[idx_pos] = str(new_layer_idx)
                        new_k = ".".join(parts)
                        pruned_weights[new_k] = tensor
                except (ValueError, IndexError):
                    pruned_weights[k] = tensor
            else:
                pruned_weights[k] = tensor

    pruned_params = sum(v.size for v in pruned_weights.values())
    print(f"Original tensor keys: {orig_keys_count} -> Pruned tensor keys: {len(pruned_weights)}")
    print(f"Original total params: {orig_params / 1e6:.1f}M -> Pruned: {pruned_params / 1e6:.1f}M ({pruned_params/orig_params*100:.1f}%)")

    save_file(pruned_weights, str(dst / "best.safetensors"))
    print(f"Pruned checkpoint successfully written to: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="checkpoints/router_realistic_mlp")
    parser.add_argument("--target-dir", default="checkpoints/router_pruned_14l")
    parser.add_argument("--target-layers", type=int, default=14)
    parser.add_argument("--strategy", choices=["uniform", "skip_middle"], default="uniform")
    args = parser.parse_args()
    prune_checkpoint(args.source_dir, args.target_dir, args.target_layers, args.strategy)
