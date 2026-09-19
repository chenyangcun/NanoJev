#!/usr/bin/env python3
"""Test MLX implementation of calibrated objectives without PyTorch."""
import unittest
import mlx.core as mx
import numpy as np

from calibrated_objectives_mlx import (
    brier_loss,
    brier_loss_distribution,
    cross_entropy_distribution,
    paired_brier_policy_loss,
    log_softmax,
)


class TestCalibratedObjectivesMLX(unittest.TestCase):
    def test_brier_loss_basic(self):
        # 3 logits: [2.0, 1.0, 0.1]
        logits = mx.array([2.0, 1.0, 0.1])
        loss = brier_loss(logits, 0)
        p = mx.softmax(logits).tolist()
        expected = (p[0] - 1.0) ** 2 + (p[1] - 0.0) ** 2 + (p[2] - 0.0) ** 2
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_brier_distribution(self):
        logits = mx.array([0.0, 0.0])
        probs = [0.75, 0.25]
        loss = brier_loss_distribution(logits, probs)
        # Softmax of [0, 0] is [0.5, 0.5]
        # (0.5 - 0.75)^2 + (0.5 - 0.25)^2 = 0.0625 + 0.0625 = 0.125
        self.assertAlmostEqual(loss.item(), 0.125, places=5)

    def test_cross_entropy_distribution(self):
        logits = mx.array([1.0, -1.0])
        probs = [0.8, 0.2]
        loss = cross_entropy_distribution(logits, probs)
        logp = log_softmax(logits).tolist()
        expected = -(probs[0] * logp[0] + probs[1] * logp[1])
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_paired_brier_policy_gradient(self):
        logits = mx.array([1.5, 0.5, -0.5])
        loss, meta = paired_brier_policy_loss(logits, outcome=0, samples=64, key=mx.random.key(42))
        self.assertEqual(meta["samples"], 64)
        self.assertEqual(meta["outcome"], 0)
        self.assertTrue(np.isfinite(loss.item()))
        self.assertTrue(np.isfinite(meta["reward"]))

    def test_gradient_computation(self):
        # Verify gradient can be computed via mx.grad
        def func(logits):
            return brier_loss(logits, 1)

        grad_fn = mx.grad(func)
        logits = mx.array([1.0, 2.0, 0.5])
        grads = grad_fn(logits)
        self.assertEqual(grads.shape, logits.shape)
        self.assertTrue(mx.all(mx.isfinite(grads)).item())


if __name__ == "__main__":
    unittest.main()
