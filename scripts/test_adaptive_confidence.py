#!/usr/bin/env python3
"""Unit tests for Adaptive Confidence Feature Engineering in NanoJev."""
import unittest

from adaptive_confidence import (
    calculate_adaptive_confidence,
    compute_distribution_features,
)


class TestAdaptiveConfidence(unittest.TestCase):
    def test_feature_extraction(self):
        # Clear winner: [0.80, 0.10, 0.05, 0.05]
        probs = {"a": 0.80, "b": 0.10, "c": 0.05, "d": 0.05}
        feats = compute_distribution_features(probs)
        self.assertEqual(feats["k"], 4)
        self.assertAlmostEqual(feats["top1"], 0.80)
        self.assertAlmostEqual(feats["top2"], 0.10)
        self.assertAlmostEqual(feats["margin"], 0.70)
        self.assertTrue(0.0 <= feats["norm_entropy_comp"] <= 1.0)

    def test_decisive_choice_high_confidence(self):
        # Dominant decision should easily pass router gate (>= 0.65)
        probs_decisive = {"bounded": 0.95, "standard": 0.03, "complex": 0.01, "exceptional": 0.01}
        conf = calculate_adaptive_confidence(probs_decisive, qtype="choice")
        self.assertGreaterEqual(conf, 0.85)

        probs_strong = {"standard": 0.75, "bounded": 0.15, "complex": 0.05, "exceptional": 0.05}
        conf_strong = calculate_adaptive_confidence(probs_strong, qtype="choice")
        self.assertGreaterEqual(conf_strong, 0.65)

    def test_ambiguous_choice_safe_low_confidence(self):
        # Close tie between two candidates should stay safely below gate (< 0.50)
        # to trigger router fallback to default model
        probs_ambiguous = {"bounded": 0.46, "standard": 0.44, "complex": 0.05, "exceptional": 0.05}
        conf = calculate_adaptive_confidence(probs_ambiguous, qtype="choice")
        self.assertLess(conf, 0.50)

        # Equal distribution
        probs_equal = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        conf_eq = calculate_adaptive_confidence(probs_equal, qtype="choice")
        self.assertLess(conf_eq, 0.20)

    def test_noul_polarity_calibration(self):
        # Extreme probability -> very high confidence
        self.assertGreaterEqual(calculate_adaptive_confidence({"false": 0.01, "true": 0.99}, qtype="noul"), 0.95)
        self.assertGreaterEqual(calculate_adaptive_confidence({"false": 0.98, "true": 0.02}, qtype="noul"), 0.95)

        # Pure uncertainty (0.50) -> zero confidence
        self.assertAlmostEqual(calculate_adaptive_confidence({"false": 0.50, "true": 0.50}, qtype="noul"), 0.0, places=3)

        # Mild probability (0.60) -> moderate confidence
        self.assertAlmostEqual(calculate_adaptive_confidence({"false": 0.40, "true": 0.60}, qtype="noul"), 0.20, places=2)


if __name__ == "__main__":
    unittest.main()
