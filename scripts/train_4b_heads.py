#!/usr/bin/env python3
"""Train lightweight 0-Token Decision Heads on top of Qwen3.5-4B backbone in MLX.

Architecture:
- Frozen Backbone: Qwen3.5-4B (oQ4e-FP16)
- Features: 2560-dim representation from last token of Evidence prefix
- Heads:
    - complexity: 2560 -> 256 -> 4 (bounded, standard, complex, exceptional)
    - effort: 2560 -> 256 -> 5 (low, medium, high, xhigh, max)
    - high_risk: 2560 -> 256 -> 2 (no, yes)
    - independent: 2560 -> 256 -> 2 (no, yes)
    - model_change_required: 2560 -> 256 -> 2 (no, yes)
    - context_mode: 2560 -> 256 -> 2 (task, chat)

Inference Cost:
- 1 single prefill of State (0 tokens decoded)
- Heads computation: <0.01ms (single matrix multiply)
- In hot cache state: latency is <5ms!
"""
import argparse
import glob
import json
import os
from pathlib import Path
import random
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import numpy as np
from safetensors.numpy import save_file

DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

LABEL_MAPS = {
    "complexity": ["bounded", "standard", "complex", "exceptional"],
    "effort": ["low", "medium", "high", "xhigh", "max"],
    "high_risk": ["no", "yes"],
    "independent": ["no", "yes"],
    "model_change_required": ["no", "yes"],
    "context_mode": ["task", "chat"],
}


class DecisionHead(nn.Module):
    def __init__(self, in_features=2560, hidden_dim=256, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def __call__(self, x):
        return self.fc2(self.norm(nn.silu(self.fc1(x))))


class RouterMultiHeads(nn.Module):
    def __init__(self, in_features=2560, hidden_dim=256):
        super().__init__()
        self.complexity = DecisionHead(in_features, hidden_dim, 4)
        self.effort = DecisionHead(in_features, hidden_dim, 5)
        self.high_risk = DecisionHead(in_features, hidden_dim, 2)
        self.independent = DecisionHead(in_features, hidden_dim, 2)
        self.model_change_required = DecisionHead(in_features, hidden_dim, 2)
        self.context_mode = DecisionHead(in_features, hidden_dim, 2)

    def __call__(self, x):
        return {
            "complexity": self.complexity(x),
            "effort": self.effort(x),
            "high_risk": self.high_risk(x),
            "independent": self.independent(x),
            "model_change_required": self.model_change_required(x),
            "context_mode": self.context_mode(x),
        }


def load_dataset(log_dir, extra_files=None, max_samples=None):
    samples = []
    seen_states = set()

    # 1. From real router logs
    log_files = sorted(glob.glob(f"{log_dir}/jev-2026-*.jsonl"))
    for fp in log_files:
        if not os.path.isfile(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    req = d.get("request", {})
                    resp = d.get("response", {})
                    st = req.get("state")
                    ans = resp.get("answers", {})
                    if not st or not ans:
                        continue
                    st_str = json.dumps(st, ensure_ascii=False) if isinstance(st, (dict, list)) else str(st)
                    if st_str in seen_states:
                        continue
                    seen_states.add(st_str)

                    # Extract labels
                    labels = {}
                    for qk, qmap in LABEL_MAPS.items():
                        if qk in ans:
                            val = ans[qk].get("choice", ans[qk].get("noul", ans[qk].get("score")))
                            if isinstance(val, (int, float)):
                                val_str = "yes" if val >= 0.5 else "no"
                            else:
                                val_str = str(val).strip().lower()
                            if val_str in qmap:
                                labels[qk] = qmap.index(val_str)
                    if labels:
                        samples.append({"state": st_str, "labels": labels})
                except Exception:
                    continue

    # 2. From extra jsonl datasets if provided
    if extra_files:
        for fp in extra_files:
            if not os.path.isfile(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        st = d.get("state")
                        st_str = json.dumps(st, ensure_ascii=False) if isinstance(st, (dict, list)) else str(st)
                        if st_str in seen_states:
                            continue
                        seen_states.add(st_str)

                        # Check gold_probs or answers
                        labels = {}
                        gp = d.get("gold_probs", {})
                        for qk, qmap in LABEL_MAPS.items():
                            if qk in gp:
                                top_k = max(gp[qk].keys(), key=lambda k: gp[qk][k]).lower()
                                if top_k in ("true", "1"):
                                    top_k = "yes"
                                elif top_k in ("false", "0"):
                                    top_k = "no"
                                if top_k in qmap:
                                    labels[qk] = qmap.index(top_k)
                        if labels:
                            samples.append({"state": st_str, "labels": labels})
                    except Exception:
                        continue

    random.seed(42)
    random.shuffle(samples)
    if max_samples:
        samples = samples[:max_samples]
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints/qwen35_4b_oq4e_fp16")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--data-dir", default="../NanoJev/data")
    parser.add_argument("--cache-file", default="heads_cache_4b.npz")
    parser.add_argument("--output-file", default="checkpoints/qwen35_4b_oq4e_fp16/heads/router_heads.safetensors")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    extra_files = [
        f"{args.data_dir}/exact_router_dataset.jsonl",
        f"{args.data_dir}/router_calibrated_full.jsonl",
        f"{args.data_dir}/exact_production_audit_train.jsonl",
    ]

    print("=" * 75)
    print("NanoJev-4B: Training 0-Token Lightweight Decision Heads")
    print("=" * 75)

    samples = load_dataset(args.log_dir, extra_files=extra_files, max_samples=args.limit)
    print(f"Loaded {len(samples)} distinct state samples with ground-truth decision labels.")

    cache_path = Path(args.cache_file)
    if cache_path.exists():
        print(f"Loading cached 4B backbone representations from {cache_path}...")
        npz = np.load(cache_path, allow_pickle=True)
        X = mx.array(npz["X"])
        labels_json = json.loads(str(npz["labels"]))
        print(f"Loaded {X.shape[0]} feature vectors of dim {X.shape[1]}.")
    else:
        print(f"Loading Qwen3.5-4B model from {args.checkpoint_dir} to extract representations...")
        t0 = time.time()
        model, tokenizer = load(args.checkpoint_dir, tokenizer_config={"trust_remote_code": False})
        inner = model.language_model.model
        print(f"Model loaded in {time.time() - t0:.2f}s.")

        print("Extracting last-token representations across samples (this runs once)...")
        features = []
        valid_labels = []
        t0 = time.time()

        for idx, item in enumerate(samples, 1):
            st_text = item["state"]
            prefix_text = f"<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\nEvidence:\n{st_text}\n\n"
            toks = tokenizer.encode(prefix_text, add_special_tokens=False)
            if len(toks) > 3500:
                toks = toks[:500] + toks[-3000:]

            h = inner(mx.array([toks], dtype=mx.int32))
            last_vec = h[0, -1, :]
            mx.eval(last_vec)

            features.append(np.array(last_vec.tolist(), dtype=np.float32))
            valid_labels.append(item["labels"])

            if idx % 20 == 0 or idx == len(samples):
                elapsed = time.time() - t0
                speed = idx / elapsed
                rem = (len(samples) - idx) / speed if speed > 0 else 0
                print(f"  Processed [{idx:>3}/{len(samples)}] in {elapsed:.1f}s ({speed:.1f} smp/s, ETA: {rem:.1f}s)")

        X_np = np.stack(features)
        X = mx.array(X_np)
        labels_json = valid_labels
        np.savez_compressed(cache_path, X=X_np, labels=json.dumps(labels_json))
        print(f"Cached extracted representations to {cache_path} ({X.shape[0]} vectors).")

    # Split train/val (85% train, 15% val)
    N = len(labels_json)
    indices = list(range(N))
    random.seed(42)
    random.shuffle(indices)
    n_train = int(N * 0.85)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    print(f"\nTrain samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    heads = RouterMultiHeads(in_features=2560, hidden_dim=256)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def loss_fn(model, x, target_labels):
        out = model(x)
        total_loss = mx.array(0.0)
        n_tasks = 0

        for qk in LABEL_MAPS.keys():
            if qk in target_labels:
                t = target_labels[qk]
                mask = t >= 0
                if mx.sum(mask).item() > 0:
                    logits = out[qk]
                    # Cross entropy
                    ce = nn.losses.cross_entropy(logits, t, reduction="none")
                    total_loss = total_loss + mx.mean(ce * mask)
                    n_tasks += 1

        return total_loss / max(1, n_tasks)

    loss_and_grad_fn = nn.value_and_grad(heads, loss_fn)

    print("\nTraining Multi-Head Decision Adapter...")
    t0_train = time.time()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_idx)
        epoch_losses = []

        for b_start in range(0, len(train_idx), args.batch_size):
            b_indices = train_idx[b_start : b_start + args.batch_size]
            b_x = X[mx.array(b_indices)]

            # Build batch targets
            b_targets = {}
            for qk in LABEL_MAPS.keys():
                vals = [labels_json[i].get(qk, -1) for i in b_indices]
                b_targets[qk] = mx.array(vals, dtype=mx.int32)

            loss, grads = loss_and_grad_fn(heads, b_x, b_targets)
            optimizer.update(heads, grads)
            mx.eval(heads.parameters(), optimizer.state)
            epoch_losses.append(loss.item())

        if epoch % 5 == 0 or epoch == args.epochs:
            # Evaluate Validation Set
            val_x = X[mx.array(val_idx)]
            val_preds = heads(val_x)
            mx.eval(val_preds)

            acc_strs = []
            for qk in LABEL_MAPS.keys():
                actual = [labels_json[i].get(qk) for i in val_idx]
                valid_pairs = [(actual[j], val_preds[qk][j].tolist()) for j in range(len(actual)) if actual[j] is not None]
                if valid_pairs:
                    correct = sum(1 for act, p_logits in valid_pairs if act == p_logits.index(max(p_logits)))
                    acc = correct / len(valid_pairs) * 100
                    acc_strs.append(f"{qk}: {acc:.1f}%")

            mean_l = statistics.mean(epoch_losses)
            print(f"Epoch {epoch:>2}/{args.epochs} | Loss: {mean_l:.4f} | Val Acc -> {' | '.join(acc_strs)}")

    train_dt = time.time() - t0_train
    print(f"\nTraining completed in {train_dt:.2f}s!")

    # Save trained heads weights as safetensors
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from mlx.utils import tree_flatten
    weights_dict = dict(tree_flatten(heads.parameters()))
    mx.save_safetensors(str(out_path), weights_dict)
    print(f"Saved trained 0-Token Decision Heads to: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
