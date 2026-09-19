"""Proper scoring objectives for parallel categorical decisions implemented in Apple MLX.

Supports:
- Brier loss (multiclass and soft distributions)
- Cross Entropy loss (hard targets or probability distributions)
- Paired Brier Policy Gradient (proper-reward learning for observed outcomes)
"""
import math
import mlx.core as mx


def log_softmax(logits, axis=-1):
    return logits - mx.logsumexp(logits, axis=axis, keepdims=True)


def _validate(logits, outcome):
    if logits.ndim != 1 or logits.size < 2:
        raise ValueError("Expected one complete question with at least two logits")
    if not mx.all(mx.isfinite(logits)).item():
        raise ValueError("Logits must be finite; exclude padded candidates")
    if isinstance(outcome, bool) or not isinstance(outcome, int) or not (0 <= outcome < logits.size):
        raise ValueError("Outcome must be an offered candidate index")


def brier_loss(logits, outcome):
    """Multiclass Brier loss; accepts integer class index or target distribution."""
    _validate(logits, outcome)
    p = mx.softmax(logits, axis=-1)
    target = mx.zeros_like(p)
    # MLX allows index assignment target[outcome] = 1.0 or one_hot
    target = mx.array([1.0 if i == outcome else 0.0 for i in range(p.shape[0])], dtype=p.dtype)
    return mx.sum(mx.square(p - target))


def brier_loss_distribution(logits, target_probs):
    """Brier loss against a known probability distribution."""
    if logits.ndim != 1 or logits.size < 2:
        raise ValueError("Expected one complete question with at least two logits")
    p = mx.softmax(logits, axis=-1)
    t = mx.array(target_probs, dtype=p.dtype)
    if not mx.all(mx.isfinite(p)).item() or not mx.all(mx.isfinite(t)).item():
        raise ValueError("Nonfinite probabilities or logits")
    return mx.sum(mx.square(p - t))


def cross_entropy_distribution(logits, target_probs):
    """Cross-entropy loss against a target distribution (deterministic one-hot or soft)."""
    if logits.ndim != 1 or logits.size < 2:
        raise ValueError("Expected one complete question with at least two logits")
    logp = log_softmax(logits, axis=-1)
    t = mx.array(target_probs, dtype=logits.dtype)
    if not mx.all(mx.isfinite(logp)).item() or not mx.all(mx.isfinite(t)).item():
        raise ValueError("Nonfinite probabilities or logits")
    return -mx.sum(t * logp)


def paired_brier_policy_loss(logits, outcome, samples=32, baseline=True, key=None):
    """Unbiased score-function gradient of negative expected proper reward in MLX.

    R = 2/M sum_i 1[A_i=Y] - sum_{i!=j} 1[A_i=A_j]/(M*(M-1)).
    E[R] = 2*p.q - ||p||^2.
    """
    _validate(logits, outcome)
    if type(samples) is not int or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if key is None:
        key = mx.random.key(0)

    logp = log_softmax(logits, axis=-1)
    p = mx.stop_gradient(mx.exp(logp))

    # Categorical sampling in MLX via Gumbel-max or logits sampling:
    # logits + Gumbel(0,1) sampling or mx.random.categorical
    # mx.random.categorical(logits, num_samples=samples, key=key)
    actions = mx.random.categorical(logp, num_samples=samples, key=key)
    
    # Count occurrences of each action:
    k = logits.size
    one_hots = mx.eye(k)[actions]  # [samples, k]
    counts = mx.sum(one_hots, axis=0)  # [k]

    hit = (actions == outcome).astype(logits.dtype)
    coefficient = 2.0 / (samples * (samples - 1))
    action_counts = counts[actions]
    local_reward = (2.0 / samples) * hit - coefficient * (action_counts - 1)

    if baseline:
        other_probability_sum = mx.sum(p[actions]) - p[actions]
        control = (2.0 / samples) * p[outcome] - coefficient * other_probability_sum
        advantage = local_reward - control
    else:
        advantage = local_reward

    loss = -mx.sum(mx.stop_gradient(advantage) * logp[actions])
    reward = 2 * mx.mean(hit) - mx.sum(counts * (counts - 1)) / (samples * (samples - 1))

    return loss, {
        "reward": float(reward.item()),
        "samples": samples,
        "baseline": bool(baseline),
        "outcome": outcome,
        "independent_sampling_with_replacement": True,
    }


def grouped_calibrated_loss_mlx(logits, examples, objective, loss_kind, samples=32, key=None):
    """Equal weight per complete question in MLX."""
    from train_pipeline_decisions import target_for
    if loss_kind not in {"ce", "brier", "paired_brier_pg"}:
        raise ValueError("Unknown calibrated loss")
    if loss_kind == "paired_brier_pg" and objective != "observed_outcome":
        raise ValueError("The sampled reward requires observed outcomes, not a policy or API target")

    losses = []
    for values, example in zip(logits, examples):
        k = len(example["candidate_ids"])
        z = values[:k]
        target = target_for(example, objective)
        if target is None:
            raise ValueError("Missing eligible target")
        if loss_kind == "paired_brier_pg":
            if key is not None:
                key, subkey = mx.random.split(key)
            else:
                subkey = None
            loss, _ = paired_brier_policy_loss(z, example["gold_index"], samples=samples, key=subkey)
        elif loss_kind == "ce":
            loss = cross_entropy_distribution(z, target)
        elif loss_kind == "brier":
            loss = brier_loss_distribution(z, target)
        losses.append(loss)
    return mx.stack(losses)
