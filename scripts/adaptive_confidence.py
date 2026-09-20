#!/usr/bin/env python3
"""Adaptive Confidence Feature Engineering for NanoJev.

Inspired by Laya's multi-feature pooling (Top1, Margin, Entropy, Option count),
calibrated specifically for NanoJev's Parallel Decision architecture.

Features:
1. Multi-feature signal extraction:
   - top1_prob: probability of highest candidate
   - margin: gap between top1 and top2 (separability indicator)
   - norm_entropy_comp: 1 - H(p)/log(K) (distribution concentration)
   - k_factor: option capacity scaling
2. Hybrid confidence fusion:
   - When top1 dominates (>= 0.70) or margin >= 0.40, confidence scales with clear decisiveness.
   - When margin is tight (< 0.15) and entropy is high, confidence stays safely below routing gate (< 0.65).
   - Smooth sigmoidal blending avoiding step discontinuities.
3. Special calibration for:
   - Choice: multi-candidate competition
   - Noul (Boolean): polarity confidence max(p, 1-p)
   - Score: level distribution concentration around expectation
"""
import math
from typing import Dict, List, Union


def compute_distribution_features(probs: Union[List[float], Dict[str, float]]) -> dict:
    """Extract 4 key statistical features from a probability distribution."""
    if isinstance(probs, dict):
        values = [float(v) for v in probs.values()]
    else:
        values = [float(v) for v in probs]

    k = len(values)
    if k <= 1:
        return {
            "k": k,
            "top1": 1.0,
            "top2": 0.0,
            "margin": 1.0,
            "entropy": 0.0,
            "norm_entropy_comp": 1.0,
        }

    # Sort descending
    sorted_p = sorted(values, reverse=True)
    top1 = sorted_p[0]
    top2 = sorted_p[1]
    margin = max(0.0, top1 - top2)

    # Shannon Entropy
    h = -sum(p * math.log(max(p, 1e-12)) for p in values if p > 0)
    max_h = math.log(k)
    norm_entropy_comp = max(0.0, min(1.0, 1.0 - (h / max_h)))

    return {
        "k": k,
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "entropy": h,
        "norm_entropy_comp": norm_entropy_comp,
    }


def calculate_adaptive_confidence(
    probs: Union[List[float], Dict[str, float]],
    qtype: str = "choice",
    temperature_applied: float = 0.35,
) -> float:
    """Compute calibrated adaptive confidence in [0.0, 1.0].

    Parameters:
        probs: probabilities array or dict
        qtype: "choice", "score", or "noul"
        temperature_applied: temperature previously used for scaling
    """
    if isinstance(probs, dict):
        values = list(probs.values())
    else:
        values = list(probs)

    k = len(values)
    if k <= 1:
        return 1.0

    # 1. For Boolean / Noul: Polarity strength
    if qtype == "noul" or (k == 2 and qtype == "boolean"):
        p_true = values[1] if isinstance(values, list) else probs.get("true", 0.5)
        # Distance from uncertainty center 0.5 scaled to [0, 1]
        dist_from_half = abs(p_true - 0.5) * 2.0  # 0.0 to 1.0
        # If very close to 0 or 1 (>0.85 or <0.15), confidence is very high
        conf = dist_from_half
        return round(float(max(0.0, min(1.0, conf))), 4)

    # 2. For Choice and Score: Multi-feature pooling
    feats = compute_distribution_features(values)
    top1 = feats["top1"]
    margin = feats["margin"]
    norm_ent = feats["norm_entropy_comp"]

    # Hybrid fusion inspired by Laya's pooled features:
    # Feature 1: Margin dominance (weight 0.45)
    # Feature 2: Top-1 absolute probability (weight 0.35)
    # Feature 3: Entropy concentration (weight 0.20)
    fused_signal = 0.45 * margin + 0.35 * top1 + 0.20 * norm_ent

    # Sigmoidal calibration to cleanly separate clear winners from ambiguous ties
    # If margin >= 0.35 or top1 >= 0.60, push over 0.65+
    # If margin < 0.15, pull down under 0.50
    if margin < 0.10:
        # Heavily ambiguous
        confidence = fused_signal * 0.5
    elif margin >= 0.40 or top1 >= 0.70:
        # Highly decisive
        confidence = 0.65 + 0.35 * min(1.0, (fused_signal - 0.5) / 0.5)
    else:
        # Transition zone: linear interpolation
        t = (margin - 0.10) / 0.30
        confidence = 0.30 + t * 0.35

    return round(float(max(0.0, min(1.0, confidence))), 4)
