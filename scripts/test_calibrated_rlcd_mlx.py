#!/usr/bin/env python3
import unittest
import mlx.core as mx
from calibrated_rlcd_mlx import RLCDConfig, rlcd_loss_single, grouped_rlcd_loss_mlx


class TestCalibratedRLCDMLX(unittest.TestCase):
    def test_rlcd_loss_single_finite(self):
        logits = mx.array([2.0, -1.0, 0.5])
        target = mx.array([1.0, 0.0, 0.0])
        config = RLCDConfig(samples=4, sigma=0.3)
        loss, stats = rlcd_loss_single(logits, target, qtype="choice", config=config, key=mx.random.key(123))
        mx.eval(loss, stats["reward_mean"])
        self.assertTrue(mx.isfinite(loss).item())
        self.assertTrue(mx.isfinite(stats["reward_mean"]).item())

    def test_rlcd_score_type_finite(self):
        logits = mx.array([0.1, 0.2, 0.5, 0.8, 1.2])
        target = mx.array([0.0, 0.0, 0.0, 0.0, 1.0])
        config = RLCDConfig(samples=4, sigma=0.3)
        loss, stats = rlcd_loss_single(logits, target, qtype="score", config=config, key=mx.random.key(456))
        mx.eval(loss)
        self.assertTrue(mx.isfinite(loss).item())

    def test_grouped_rlcd_loss(self):
        logits = mx.array([
            [1.0, 0.0, -1.0, 0.0],
            [-0.5, 1.5, 0.0, 0.0]
        ])
        examples = [
            {"id": "q1", "type": "choice", "candidate_ids": ["a", "b", "c"], "gold_probs": [0.8, 0.1, 0.1]},
            {"id": "q2", "type": "boolean", "candidate_ids": ["false", "true"], "gold_probs": [0.1, 0.9]},
        ]
        loss, tele = grouped_rlcd_loss_mlx(logits, examples, key=mx.random.key(789))
        mx.eval(loss, tele["reward_mean"])
        self.assertTrue(mx.isfinite(loss).item())
        self.assertTrue(mx.isfinite(tele["reward_mean"]).item())


if __name__ == "__main__":
    unittest.main()
