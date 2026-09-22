#!/usr/bin/env python3
"""Unit tests for MLX-Native PointerHead:
1. Forward computation and shape validation
2. Large candidate set testing (K=32, K=64)
3. Permutation Invariance: candidate scores remain identical regardless of option ordering
4. Integration with MultiHeadRegistry
"""
import copy
import math
import unittest

import mlx.core as mx
from mlx_pointer_head import PointerHead
from mlx_multi_head_registry import MultiHeadRegistry


class TestPointerHead(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 128
        self.pointer_dim = 64
        self.pointer_head = PointerHead(hidden_size=self.hidden_size, pointer_dim=self.pointer_dim)

    def test_forward_shapes_and_values(self):
        # 4 candidates
        k = 4
        h_decide = mx.random.normal((self.hidden_size,))
        h_opts = mx.random.normal((k, self.hidden_size))

        logits = self.pointer_head.compute_logits(h_decide, h_opts)
        self.assertEqual(logits.shape, (k,))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_large_candidate_pool(self):
        # Test on K=64 large candidate pool (e.g. 64 skills)
        k_large = 64
        h_decide = mx.random.normal((self.hidden_size,))
        h_opts = mx.random.normal((k_large, self.hidden_size))

        logits = self.pointer_head.compute_logits(h_decide, h_opts)
        self.assertEqual(logits.shape, (k_large,))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

        # Test softmax over large pool
        probs = mx.softmax(logits, axis=-1)
        self.assertAlmostEqual(mx.sum(probs).item(), 1.0, places=5)

    def test_permutation_invariance(self):
        # In a decision model, if candidates are reordered, their respective scores should remain identical
        k = 5
        h_decide = mx.random.normal((self.hidden_size,))
        h_opts = mx.random.normal((k, self.hidden_size))

        logits_orig = self.pointer_head.compute_logits(h_decide, h_opts).tolist()

        # Permute candidates: reverse order
        perm = list(range(k))[::-1]
        h_opts_perm = h_opts[mx.array(perm)]
        logits_perm = self.pointer_head.compute_logits(h_decide, h_opts_perm).tolist()

        # Check each candidate score matches its original score
        for orig_idx, perm_idx in enumerate(perm):
            self.assertAlmostEqual(logits_orig[orig_idx], logits_perm[perm_idx], places=5)

    def test_multi_head_registry_integration(self):
        registry = MultiHeadRegistry(hidden_size=self.hidden_size, default_head_name="router")
        registry.register_head("skill", self.pointer_head)

        q_skill = {
            "type": "choice",
            "instructions": "Pick appropriate skill",
            "criteria": {f"skill_{i}": f"description {i}" for i in range(8)},
        }
        head, name, reason = registry.resolve_head("skill_selector", q_skill)
        self.assertEqual(name, "skill")
        self.assertIsInstance(head, PointerHead)


if __name__ == "__main__":
    unittest.main()
