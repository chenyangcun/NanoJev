"""RLCD (Reinforcement Learning for Calibrated Decisions) Loss in Apple MLX.

Faithfully implements the Dohnuts / Laya specification:
1. Logit Gaussian perturbation exploration (M=4, sigma=0.3).
2. Proper scoring rule composite reward (Log-Score + 0.75 * Spherical-Score - RPS penalty for score).
3. Group-mean Advantage normalization and policy gradient.
4. Auxiliary cross-entropy loss (weight=1.0).
"""

from dataclasses import dataclass
import mlx.core as mx


@dataclass(frozen=True)
class RLCDConfig:
    samples: int = 4
    sigma: float = 0.3
    log_weight: float = 1.0
    spherical_weight: float = 0.75
    ordinal_weight: float = 1.0
    ce_weight: float = 1.0
    log_floor: float = -9.21


def rlcd_loss_single(
    logits: mx.array,
    target_probs: mx.array,
    qtype: str = "choice",
    config: RLCDConfig = None,
    key: mx.array = None,
):
    """Compute joint RLCD and cross-entropy loss for a single decision.

    Args:
        logits: 1D MLX array of unnormalized logits [K].
        target_probs: 1D MLX array of target distribution or one-hot target [K].
        qtype: "choice", "boolean", or "score".
        config: RLCDConfig hyperparameter set.
        key: PRNG key for Gaussian exploration sampling.
    """
    if config is None:
        config = RLCDConfig()
    if key is None:
        key = mx.random.key(0)

    k = logits.size
    mean = logits.astype(mx.float32)

    # 1. Sample perturbations
    noise = mx.random.normal(shape=(config.samples, k), key=key) * config.sigma
    # Zero-sum projection
    noise = noise - mx.mean(noise, axis=-1, keepdims=True)
    actions = mx.stop_gradient(mean[None, :] + noise)

    # 2. Distribution rewards
    log_probs = actions - mx.logsumexp(actions, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)

    # Log score (clamped)
    log_p_clamped = mx.clip(mx.log(mx.maximum(probs, 1e-12)), a_min=config.log_floor, a_max=0.0)
    log_score = mx.sum(target_probs[None, :] * log_p_clamped, axis=-1)

    # Spherical score
    target_prob = mx.sum(probs * target_probs[None, :], axis=-1)
    spherical = target_prob / mx.maximum(mx.linalg.norm(probs, axis=-1), 1e-9)

    reward = config.log_weight * log_score + config.spherical_weight * spherical

    # RPS penalty for ordinal scores
    if qtype == "score" and k > 1:
        target_cdf = mx.cumsum(target_probs, axis=-1)
        probs_cdf = mx.cumsum(probs, axis=-1)
        rps = mx.sum(mx.square(probs_cdf[:, :-1] - target_cdf[None, :-1]), axis=-1) / max(1, k - 1)
        reward = reward - config.ordinal_weight * rps

    # 3. Advantage normalization
    adv = reward - mx.mean(reward)
    adv_rms = mx.sqrt(mx.mean(mx.square(adv))) + 1e-6
    adv = adv / adv_rms

    # 4. Policy loss
    log_density = -0.5 * mx.sum(mx.square((actions - mean[None, :]) / config.sigma), axis=-1)
    policy_loss = -mx.mean(mx.stop_gradient(adv) * log_density)

    # 5. Auxiliary cross-entropy
    logp = mean - mx.logsumexp(mean, axis=-1, keepdims=True)
    ce_loss = -mx.sum(target_probs * logp)

    total_loss = policy_loss + config.ce_weight * ce_loss

    stats = {
        "reward_mean": mx.mean(reward),
        "policy_loss": policy_loss,
        "ce_loss": ce_loss,
        "adv_rms": adv_rms,
    }
    return total_loss, stats


def grouped_rlcd_loss_mlx(
    logits: mx.array,
    examples: list,
    objective: str = "gold",
    config: RLCDConfig = None,
    key: mx.array = None,
):
    """Compute grouped RLCD loss over a batch of examples."""
    from train_pipeline_decisions import target_for

    if config is None:
        config = RLCDConfig()
    if key is None:
        key = mx.random.key(0)

    losses = []
    reward_means = []
    policy_losses = []
    ce_losses = []

    for values, example in zip(logits, examples):
        k = len(example["candidate_ids"])
        z = values[:k]
        target = None
        if "gold_distribution_probs" in example:
            target = target_for(example, "gold_distribution")
        elif "gold_index" in example:
            target = target_for(example, "gold")
        elif "gold_probs" in example:
            target = example["gold_probs"]

        if target is None:
            raise ValueError(f"Missing target for question {example['id']}")
        t_arr = mx.array(target, dtype=mx.float32)

        key, subkey = mx.random.split(key)
        loss_val, stats = rlcd_loss_single(
            z,
            t_arr,
            qtype=example.get("type", "choice"),
            config=config,
            key=subkey,
        )
        losses.append(loss_val)
        reward_means.append(stats["reward_mean"])
        policy_losses.append(stats["policy_loss"])
        ce_losses.append(stats["ce_loss"])

    stacked_losses = mx.stack(losses)
    total_loss = mx.mean(stacked_losses)

    telemetry = {
        "reward_mean": mx.mean(mx.stack(reward_means)),
        "policy_loss": mx.mean(mx.stack(policy_losses)),
        "ce_loss": mx.mean(mx.stack(ce_losses)),
    }
    return total_loss, telemetry
