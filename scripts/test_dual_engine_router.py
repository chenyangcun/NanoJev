#!/usr/bin/env python3
"""Benchmark and verify DualEngineRouter (ANE Fast-Lane vs MLX GPU Main-Lane)."""
import time
import unittest

from dual_engine_router import DualEngineRouter


class TestDualEngineRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Setup] Initializing DualEngineRouter (ANE + MLX GPU)...")
        t0 = time.time()
        cls.router = DualEngineRouter(
            gpu_checkpoint_dir="checkpoints/NanoJev",
            ane_checkpoint_dir="checkpoints/laya_multilingual_ane",
            enable_ane=True,
        )
        print(f"[Setup] DualEngineRouter ready in {time.time() - t0:.2f}s!\n")

    def test_ane_fast_lane_short_decision(self):
        # Ultra-short, latency-critical proposition: Should route to ANE!
        payload = {
            "states": [
                {
                    "id": "s_fast",
                    "state": "The user asks for money refund.",
                    "questions": {
                        "is_refund": {
                            "type": "boolean",
                            "instructions": "Does the user request a refund?",
                        }
                    },
                }
            ]
        }

        # Warmup
        self.router.predict(payload)

        # Benchmark 10 calls
        timings = []
        for _ in range(10):
            t0 = time.perf_counter()
            res = self.router.predict(payload)
            timings.append((time.perf_counter() - t0) * 1000)

        lane = res["checkpoint"].get("engine_lane")
        print(f"Short Task Decision -> Lane: {lane}, Latency: P50={sorted(timings)[5]:.2f}ms, Min={min(timings):.2f}ms")
        self.assertIn("Neural Engine", lane)
        self.assertTrue(timings[0] < 35.0)  # ANE is around 5ms ~ 20ms

        ans = res["states"][0]["answers"]["is_refund"]
        self.assertIn("p_true", ans)
        print("ANE Answer:", ans)

    def test_gpu_lane_rich_agent_task(self):
        # Long, structured multi-turn conversation: Should route to MLX GPU!
        payload = {
            "states": [
                {
                    "id": "s_agent",
                    "state": {
                        "user_task": "Investigate an intermittent race between request routing and audit logging, then propose a minimal fix.",
                        "previous_assistant": "I have analyzed the database logs and noticed a deadlock between workers.",
                        "recent_tool_calls": ["rg -i 'deadlock' src/"],
                        "user_turn_count": 3,
                    },
                    "questions": {
                        "complexity": {
                            "type": "choice",
                            "instructions": "Choose complexity",
                            "criteria": {"bounded": "small", "complex": "deep architecture"},
                        }
                    },
                }
            ]
        }

        t0 = time.perf_counter()
        res = self.router.predict(payload)
        dt = (time.perf_counter() - t0) * 1000

        lane = res["checkpoint"].get("engine_lane")
        print(f"Rich Agent Task -> Lane: {lane}, Latency: {dt:.2f}ms")
        self.assertIn("Metal GPU", lane)

        ans = res["states"][0]["answers"]["complexity"]
        print("GPU Answer:", ans)


if __name__ == "__main__":
    unittest.main()
