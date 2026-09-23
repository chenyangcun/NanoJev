#!/usr/bin/env python3
"""Train 0-Token Decision Heads for Webpage Memory Quality Review on top of Qwen3.5-4B backbone in MLX.

Heads (Browser Memory Review):
- page_role: 2560 -> 256 -> 8 (durable_reference, durable_record, navigation_or_search, list_or_feed, login_or_account, boilerplate_or_empty, one_off_or_process, uncertain)
- future_useful: 2560 -> 256 -> 2 (no, yes)
- content_sufficiency: 2560 -> 256 -> 3 (0, 1, 2)

Saves to:
  checkpoints/qwen35_4b_oq4e_fp16/heads/memory_heads.safetensors
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

DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

LABEL_MAPS_MEMORY = {
    "page_role": [
        "durable_reference",
        "durable_record",
        "navigation_or_search",
        "list_or_feed",
        "login_or_account",
        "boilerplate_or_empty",
        "one_off_or_process",
        "uncertain",
    ],
    "future_useful": ["no", "yes"],
    "content_sufficiency": ["0", "1", "2"],
}


class DecisionHead(nn.Module):
    def __init__(self, in_features=2560, hidden_dim=256, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def __call__(self, x):
        return self.fc2(self.norm(nn.silu(self.fc1(x))))


class MemoryMultiHeads(nn.Module):
    def __init__(self, in_features=2560, hidden_dim=256):
        super().__init__()
        self.page_role = DecisionHead(in_features, hidden_dim, 8)
        self.future_useful = DecisionHead(in_features, hidden_dim, 2)
        self.content_sufficiency = DecisionHead(in_features, hidden_dim, 3)

    def __call__(self, x):
        return {
            "page_role": self.page_role(x),
            "future_useful": self.future_useful(x),
            "content_sufficiency": self.content_sufficiency(x),
        }


def load_memory_samples(log_dir):
    files = sorted(glob.glob(f"{log_dir}/jev-2026-*.jsonl"))
    samples = []
    seen = set()

    for fp in files:
        if not os.path.isfile(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if d.get("profile") != "browser_memory_review":
                        continue
                    req = d.get("request", {})
                    resp = d.get("response", {})
                    st = req.get("state")
                    ans = resp.get("answers", {})
                    if not st or not ans:
                        continue
                    st_str = json.dumps(st, ensure_ascii=False) if isinstance(st, (dict, list)) else str(st)
                    if st_str in seen:
                        continue
                    seen.add(st_str)

                    labels = {}
                    # 1. page_role
                    if "page_role" in ans:
                        choice = ans["page_role"].get("choice")
                        if choice in LABEL_MAPS_MEMORY["page_role"]:
                            labels["page_role"] = LABEL_MAPS_MEMORY["page_role"].index(choice)

                    # 2. future_useful
                    if "future_useful" in ans:
                        val = ans["future_useful"].get("noul", 0.5)
                        labels["future_useful"] = 1 if val >= 0.5 else 0

                    # 3. content_sufficiency
                    if "content_sufficiency" in ans:
                        score_val = ans["content_sufficiency"].get("score", 1.0)
                        # round 0..2 to nearest bucket 0, 1, 2
                        idx = int(round(max(0.0, min(2.0, float(score_val)))))
                        labels["content_sufficiency"] = idx

                    if len(labels) == 3:
                        samples.append({"state": st_str, "labels": labels})
                except Exception:
                    continue

    random.seed(42)
    random.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints/qwen35_4b_oq4e_fp16")
    parser.add_argument("--log-dir", default="data/personal_memory_logs")
    parser.add_argument("--cache-file", default="heads_cache_memory_4b.npz")
    parser.add_argument("--output-file", default="checkpoints/qwen35_4b_oq4e_fp16/heads/memory_heads.safetensors")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("=" * 75)
    print("NanoJev-4B: Training 0-Token Decision Heads for Webpage Memory Review")
    print("=" * 75)

    samples = load_memory_samples(args.log_dir)
    print(f"Loaded {len(samples)} distinct webpage memory review samples.")

    cache_path = Path(args.cache_file)
    if cache_path.exists():
        print(f"Loading cached 4B backbone representations from {cache_path}...")
        npz = np.load(cache_path, allow_pickle=True)
        X = mx.array(npz["X"])
        labels_json = json.loads(str(npz["labels"]))
        print(f"Loaded {X.shape[0]} feature vectors of dim {X.shape[1]}.")
    else:
        print(f"Loading Qwen3.5-4B model from {args.checkpoint_dir}...")
        t0 = time.time()
        model, tokenizer = load(args.checkpoint_dir, tokenizer_config={"trust_remote_code": False})
        inner = model.language_model.model
        print(f"Model loaded in {time.time() - t0:.2f}s.")

        print("Extracting webpage state representations across samples...")
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

            if idx % 15 == 0 or idx == len(samples):
                elapsed = time.time() - t0
                speed = idx / elapsed
                rem = (len(samples) - idx) / speed if speed > 0 else 0
                print(f"  Processed [{idx:>3}/{len(samples)}] in {elapsed:.1f}s ({speed:.1f} smp/s, ETA: {rem:.1f}s)")

        X_np = np.stack(features)
        X = mx.array(X_np)
        labels_json = valid_labels
        np.savez_compressed(cache_path, X=X_np, labels=json.dumps(labels_json))
        print(f"Cached extracted representations to {cache_path} ({X.shape[0]} vectors).")

    # Split train/val (80% train, 20% val)
    N = len(labels_json)
    indices = list(range(N))
    random.seed(42)
    random.shuffle(indices)
    n_train = int(N * 0.80)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    print(f"\nTrain samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    heads = MemoryMultiHeads(in_features=2560, hidden_dim=256)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def loss_fn(model, x, target_labels):
        out = model(x)
        total_loss = mx.array(0.0)
        n_tasks = 0

        for qk in LABEL_MAPS_MEMORY.keys():
            if qk in target_labels:
                t = target_labels[qk]
                mask = t >= 0
                if mx.sum(mask).item() > 0:
                    logits = out[qk]
                    ce = nn.losses.cross_entropy(logits, t, reduction="none")
                    total_loss = total_loss + mx.mean(ce * mask)
                    n_tasks += 1

        return total_loss / max(1, n_tasks)

    loss_and_grad_fn = nn.value_and_grad(heads, loss_fn)

    print("\nTraining Webpage Memory Decision Heads...")
    t0_train = time.time()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_idx)
        epoch_losses = []

        for b_start in range(0, len(train_idx), args.batch_size):
            b_indices = train_idx[b_start : b_start + args.batch_size]
            b_x = X[mx.array(b_indices)]

            b_targets = {}
            for qk in LABEL_MAPS_MEMORY.keys():
                vals = [labels_json[i].get(qk, -1) for i in b_indices]
                b_targets[qk] = mx.array(vals, dtype=mx.int32)

            loss, grads = loss_and_grad_fn(heads, b_x, b_targets)
            optimizer.update(heads, grads)
            mx.eval(heads.parameters(), optimizer.state)
            epoch_losses.append(loss.item())

        if epoch % 5 == 0 or epoch == args.epochs:
            val_x = X[mx.array(val_idx)]
            val_preds = heads(val_x)
            mx.eval(val_preds)

            acc_strs = []
            for qk in LABEL_MAPS_MEMORY.keys():
                actual = [labels_json[i].get(qk) for i in val_idx]
                valid_pairs = [(actual[j], val_preds[qk][j].tolist()) for j in range(len(actual)) if actual[j] is not None]
                if valid_pairs:
                    correct = sum(1 for act, p_logits in valid_pairs if act == p_logits.index(max(p_logits)))
                    acc = correct / len(valid_pairs) * 100
                    acc_strs.append(f"{qk}: {acc:.1f}%")

            mean_l = statistics.mean(epoch_losses)
            print(f"Epoch {epoch:>2}/{args.epochs} | Loss: {mean_l:.4f} | Val Acc -> {' | '.join(acc_strs)}")

    print(f"\nTraining completed in {time.time() - t0_train:.2f}s!")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from mlx.utils import tree_flatten
    weights_dict = dict(tree_flatten(heads.parameters()))
    mx.save_safetensors(str(out_path), weights_dict)
    print(f"Saved trained 0-Token Webpage Memory Heads to: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
