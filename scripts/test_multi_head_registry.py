#!/usr/bin/env python3
"""Unit tests for MultiHeadRegistry, dynamic pluggable heads, and heuristic audit logging."""
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors.numpy import save_file
import numpy as np

from mlx_deep_heads import DeepDecisionHeads
from mlx_multi_head_registry import MultiHeadRegistry


class TestMultiHeadRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "test_dispatch.jsonl"
        self.registry = MultiHeadRegistry(hidden_size=64, default_head_name="router")

        # Register two mock heads: router and skill
        self.router_head = DeepDecisionHeads(hidden_size=64, set_head="none")
        self.skill_head = DeepDecisionHeads(hidden_size=64, set_head="none")
        self.registry.register_head("router", self.router_head)
        self.registry.register_head("skill", self.skill_head)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_explicit_routing_no_audit_log(self):
        q = {
            "type": "choice",
            "head": "skill",
            "instructions": "Pick skill",
            "criteria": {"a": "desc A", "b": "desc B"},
        }
        head, name, reason = self.registry.resolve_head("custom_q", q, log_path=self.log_path)
        self.assertEqual(name, "skill")
        self.assertEqual(reason, "explicit")
        # Explicit routing should not trigger heuristic audit log
        self.assertFalse(self.log_path.exists())

    def test_heuristic_pattern_routing_with_audit_log(self):
        # 1. Matches skill keywords
        q_skill = {
            "type": "choice",
            "instructions": "Which plugin or skill to use?",
            "criteria": {"drawio": "draw diagrams", "none": "no tools"},
        }
        head, name, reason = self.registry.resolve_head("skill_choice", q_skill, log_path=self.log_path)
        self.assertEqual(name, "skill")
        self.assertTrue(reason.startswith("heuristic:pattern:skill_keywords"))

        # Verify log written
        self.assertTrue(self.log_path.exists())
        lines = [json.loads(line) for line in self.log_path.read_text().splitlines() if line]
        self.assertEqual(len(lines), 1)
        event = lines[0]
        self.assertEqual(event["qid"], "skill_choice")
        self.assertEqual(event["resolved_head"], "skill")
        self.assertEqual(event["routing_mode"], "heuristic")
        self.assertEqual(event["candidate_count"], 2)

    def test_fallback_routing_with_audit_log(self):
        # Unknown question id
        q_unknown = {
            "type": "boolean",
            "instructions": "Is today Sunday?",
        }
        head, name, reason = self.registry.resolve_head("random_check", q_unknown, log_path=self.log_path)
        self.assertEqual(name, "router")
        self.assertEqual(reason, "fallback:default")

        lines = [json.loads(line) for line in self.log_path.read_text().splitlines() if line]
        self.assertEqual(len(lines), 1)
        event = lines[0]
        self.assertEqual(event["qid"], "random_check")
        self.assertEqual(event["resolved_head"], "router")
        self.assertEqual(event["routing_mode"], "fallback")

    def test_standalone_head_weights_loading(self):
        # Create a mock standalone agent head file
        heads_dir = Path(self.temp_dir.name) / "heads"
        heads_dir.mkdir(parents=True, exist_ok=True)
        agent_file = heads_dir / "agent.safetensors"

        mock_agent_head = DeepDecisionHeads(hidden_size=64, set_head="none")
        np_weights = {
            "heads.norm.weight": np.ones((64,), dtype=np.float32),
            "heads.norm.bias": np.zeros((64,), dtype=np.float32),
            "heads.fc1.weight": np.ones((512, 64), dtype=np.float32),
            "heads.fc1.bias": np.zeros((512,), dtype=np.float32),
            "heads.fc2.weight": np.ones((1, 512), dtype=np.float32),
            "heads.fc2.bias": np.zeros((1,), dtype=np.float32),
        }
        save_file(np_weights, str(agent_file))

        # Dynamically load into registry
        from safetensors import safe_open
        plug_weights = {}
        with safe_open(str(agent_file), framework="numpy") as pf:
            for pk in pf.keys():
                clean_pk = pk[6:] if pk.startswith("heads.") else pk
                plug_weights[clean_pk] = mx.array(pf.get_tensor(pk))

        loaded_head = DeepDecisionHeads(hidden_size=64, set_head="none")
        loaded_head.load_weights(list(plug_weights.items()), strict=False)
        self.registry.register_head("agent", loaded_head)

        # Query agent question
        q_agent = {
            "type": "choice",
            "instructions": "Pick subagent",
            "criteria": {"codex": "fast", "opencode": "thorough"},
        }
        head, name, reason = self.registry.resolve_head("subagent_dispatch", q_agent, log_path=self.log_path)
        self.assertEqual(name, "agent")
        self.assertEqual(head, loaded_head)

    def test_multi_head_forward_call(self):
        # Forward through router head vs skill head
        leaves = mx.random.normal((2, 64))
        mock_ex = [{"type": "boolean", "candidate_ids": ["false", "true"], "leaf_tokens": [[1]]}]
        logits_router, valid_r = self.registry(leaves, mock_ex, kmax=2, head_name="router")
        logits_skill, valid_s = self.registry(leaves, mock_ex, kmax=2, head_name="skill")
        self.assertEqual(logits_router.shape, (1, 2))
        self.assertEqual(logits_skill.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
