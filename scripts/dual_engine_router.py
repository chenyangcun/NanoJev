"""Dual Engine Hybrid Router for NanoJev on Apple Silicon.

Dispatches incoming TypeSafe System One requests intelligently across:
1. Apple Neural Engine (ANE / Core ML):
   - Handles short, latency-critical, bounded/boolean decisions (< 90 tokens)
   - Extremely fast (~5ms P50) and highly energy-efficient (~0.15 J/decision)
2. Apple Metal GPU (MLX Qwen3-0.6B 8-bit):
   - Handles long-context complex agent tasks, deep multi-turn sessions, and delicate code reviews (up to 4096 tokens)
   - Features Hierarchical State Prefill, Adaptive Temperature, and Pluggable Decision Heads
"""
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("DualEngineRouter")


class DualEngineRouter:
    def __init__(
        self,
        gpu_checkpoint_dir: str = "checkpoints/router_quant_8bit",
        ane_checkpoint_dir: Optional[str] = "checkpoints/laya_multilingual_ane",
        ane_token_limit: int = 90,
        enable_ane: bool = True,
        max_length: int = 4096,
        default_temperature: float = 0.35,
    ):
        self.ane_token_limit = ane_token_limit
        self.enable_ane = enable_ane
        self.default_temperature = default_temperature

        # 1. Initialize MLX GPU Engine
        from predict_mlx_decisions import MLXDecisionPredictor
        logger.info(f"[DualEngine] Initializing MLX GPU Engine from: {gpu_checkpoint_dir}")
        self.gpu_engine = MLXDecisionPredictor(
            gpu_checkpoint_dir,
            max_length=max_length,
            enable_prefix_sharing=True,
            enable_adaptive_temp=True,
        )

        # 2. Optionally Initialize Core ML ANE Engine
        self.ane_agent = None
        if enable_ane and ane_checkpoint_dir:
            ane_path = Path(ane_checkpoint_dir).resolve()
            if ane_path.exists():
                try:
                    # Dynamically find laya-coreml in common directories
                    search_dirs = [
                        Path("/Users/chenyc/Documents/study/laya-coreml").resolve(),
                        Path.home() / "work" / "laya-coreml",
                        Path(__file__).resolve().parents[1].parent / "laya-coreml",
                    ]
                    for d in search_dirs:
                        if d.exists() and str(d) not in sys.path:
                            sys.path.insert(0, str(d))

                    import laya_coreml.ane as ane
                    logger.info(f"[DualEngine] Initializing Core ML ANE Engine from: {ane_path}")
                    self.ane_agent = ane.ANEAgent(str(ane_path), compute_units="cpu_ne")
                    logger.info("[DualEngine] ANE Engine loaded successfully!")
                except Exception as e:
                    logger.warning(f"[DualEngine] Could not initialize ANE Engine: {e}. Falling back to 100% GPU.")
                    self.ane_agent = None

    def can_route_to_ane(self, state_text: str, questions: dict) -> Tuple[bool, str]:
        """Check if request strictly conforms to ANE limitations."""
        if not self.ane_agent:
            return False, "ane_disabled_or_unavailable"

        # 1. Check questions count and candidate count
        # ANE bundle fixed max options is 32, batch_size=1
        if len(questions) > 3:
            return False, "too_many_questions_for_ane"

        for qid, q in questions.items():
            if q.get("head"):
                return False, f"explicit_head_requested_{q['head']}"
            qid_lower = qid.lower()
            if any(k in qid_lower for k in ("complex", "risk", "indep", "tier", "effort")):
                return False, "router_code_decision_requires_gpu"
            if any(k in qid_lower for k in ("skill", "tool", "plugin", "shortlist", "winner")) or qid_lower.startswith("verify_"):
                return False, "skill_selection_requires_gpu"
            qtype = q.get("type")
            if qtype not in ("choice", "noul", "boolean", "score"):
                return False, f"unsupported_qtype_{qtype}"
            crit = q.get("criteria")
            if qtype == "choice" and isinstance(crit, dict) and len(crit) > 16:
                return False, "choice_options_exceed_ane_budget"

        # 2. Quick token budget check using ANE tokenizer
        try:
            tok = self.ane_agent.tok
            # Rough estimate: if state text length in chars > 150, it will almost certainly exceed 90 tokens
            if len(state_text) > 160:
                return False, f"state_length_exceeds_budget_{len(state_text)}_chars"

            # Strict sequence check
            for qid, q in questions.items():
                q_copy = dict(q)
                if q_copy["type"] == "boolean":
                    q_copy["type"] = "noul"
                internal_q = self.ane_agent._to_internal(q_copy)
                from laya_coreml.common import build_sequence
                ids, markers = build_sequence(tok, state_text, internal_q, max_len=96, head_max_len=64)
                if len(ids) > self.ane_token_limit:
                    return False, f"token_length_{len(ids)}_exceeds_ane_limit_{self.ane_token_limit}"
        except Exception as e:
            return False, f"ane_check_error_{e}"

        return True, "fits_ane_fast_lane"

    def predict(self, payload: dict, temperature: float = 0.35) -> dict:
        """Route request to either ANE or GPU engine."""
        states = payload.get("states", [])
        if not states:
            from predict_toy_decisions import validate_request
            states = validate_request(payload)

        # Single state check for fast-lane ANE routing
        if len(states) == 1 and self.ane_agent:
            st = states[0]
            st_text = st["state"]
            if isinstance(st_text, (dict, list)):
                # If structured dictionary has history/tool calls, it is definitely a rich agent task -> GPU
                if any(k in st_text for k in ("previous_assistant", "recent_tool_calls", "prior_user_context", "task_origin")):
                    can_ane, reason = False, "rich_context_agent_task"
                else:
                    st_repr = json.dumps(st_text, ensure_ascii=False)
                    can_ane, reason = self.can_route_to_ane(st_repr, st["questions"])
            else:
                st_repr = str(st_text)
                can_ane, reason = self.can_route_to_ane(st_repr, st["questions"])

            if can_ane:
                t0 = time.perf_counter()
                try:
                    # Prepare questions for ANE
                    ane_questions = {}
                    for qid, q in st["questions"].items():
                        q_copy = dict(q)
                        if q_copy["type"] == "boolean":
                            q_copy["type"] = "noul"
                        ane_questions[qid] = q_copy

                    ane_res = self.ane_agent.predict(st_repr, ane_questions)
                    dt = time.perf_counter() - t0

                    # Convert ANE answers to standard format
                    answers = {}
                    for qid, ans in ane_res.get("answers", {}).items():
                        norm_ans = copy.deepcopy(ans)
                        if "action" in norm_ans:
                            del norm_ans["action"]
                        if norm_ans.get("type") == "noul":
                            norm_ans["p_true"] = norm_ans.get("noul", 0.5)
                        answers[qid] = norm_ans

                    return {
                        "schema_version": "openjev-dual-engine-v1",
                        "checkpoint": {
                            "directory": "checkpoints/laya_multilingual_ane",
                            "engine_lane": "Apple Neural Engine (ANE)",
                            "routing_reason": reason,
                        },
                        "temperature": {"value": float(temperature), "lane": "ane-calibrated"},
                        "execution": {
                            "engine": "ane-coreml-fastlane",
                            "device": "Neural Engine (NPU)",
                            "states": 1,
                            "questions": len(st["questions"]),
                            "latency_seconds": dt,
                            "speed_lane": "5ms-ultra-fast",
                        },
                        "states": [{"id": st["id"], "answers": answers}],
                    }
                except Exception as e:
                    logger.warning(f"[DualEngine] ANE execution failed ({e}), falling back to GPU lane.", exc_info=True)

        # Main Lane: High-Capacity MLX GPU Engine
        t0 = time.perf_counter()
        gpu_res = self.gpu_engine.predict(payload, temperature=temperature)
        gpu_res["checkpoint"]["engine_lane"] = "Apple Metal GPU (MLX 8-bit)"
        return gpu_res
