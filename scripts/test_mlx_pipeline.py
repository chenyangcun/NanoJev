#!/usr/bin/env python3
"""End-to-end unit tests for MLX Decision Model forward pass and loss computation."""
import unittest
import mlx.core as mx
import mlx.nn as nn

from mlx_decision_model import DecisionHeads
from calibrated_objectives_mlx import grouped_calibrated_loss_mlx


class TestMLXPipeline(unittest.TestCase):
    def test_heads_forward_and_loss(self):
        hidden_size = 64
        heads = DecisionHeads(hidden_size=hidden_size, set_head="attention")

        examples = [
            {
                "type": "choice",
                "candidate_ids": ["north", "south", "east", "west"],
                "gold_distribution_probs": [0.1, 0.7, 0.1, 0.1],
                "leaf_tokens": [[1, 2], [1, 3], [1, 4], [1, 5]],
            },
            {
                "type": "boolean",
                "candidate_ids": ["false", "true"],
                "gold_distribution_probs": [0.2, 0.8],
                "leaf_tokens": [[6, 7]],
            },
        ]

        total_paths = 4 + 1
        leaves = mx.random.normal((total_paths, hidden_size))
        logits, valid = heads(leaves, examples, kmax=4)

        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(valid.shape, (2, 4))

        # Check masked logits
        self.assertTrue(logits[1, 2].item() < -1e8)
        self.assertTrue(logits[1, 3].item() < -1e8)

        # Compute Brier and CE losses
        loss_ce = grouped_calibrated_loss_mlx(logits, examples, objective="gold_distribution", loss_kind="ce")
        self.assertEqual(loss_ce.shape, (2,))
        self.assertTrue(mx.all(mx.isfinite(loss_ce)).item())

        loss_brier = grouped_calibrated_loss_mlx(logits, examples, objective="gold_distribution", loss_kind="brier")
        self.assertEqual(loss_brier.shape, (2,))
        self.assertTrue(mx.all(mx.isfinite(loss_brier)).item())

    def test_heads_gradient(self):
        hidden_size = 32
        heads = DecisionHeads(hidden_size=hidden_size, set_head="attention")

        examples = [
            {
                "type": "choice",
                "candidate_ids": ["a", "b"],
                "gold_distribution_probs": [0.5, 0.5],
                "leaf_tokens": [[1, 2], [1, 3]],
            }
        ]

        def compute_loss(h_model, leaves):
            logits, _ = h_model(leaves, examples, kmax=2)
            losses = grouped_calibrated_loss_mlx(logits, examples, objective="gold_distribution", loss_kind="brier")
            return mx.mean(losses)

        loss_and_grad = nn.value_and_grad(heads, compute_loss)
        leaves = mx.random.normal((2, hidden_size))
        loss, grads = loss_and_grad(heads, leaves)

        self.assertTrue(mx.isfinite(loss).item())
        self.assertIn("scalar", grads)
        self.assertIn("weight", grads["scalar"])


if __name__ == "__main__":
    unittest.main()
