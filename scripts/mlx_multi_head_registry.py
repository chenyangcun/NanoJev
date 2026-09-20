#!/usr/bin/env python3
"""Multi-Head Pluggable Registry and Heuristic Routing Dispatcher for NanoJev.

Features:
1. MultiHeadRegistry: Holds multiple task-specific decision heads on top of a single Qwen3 backbone.
   - e.g. "router" (complexity, risk, independence)
   - "skill" (task-based skill & tool selection)
   - "agent" (subagent & delegation selection)
   - "news" (editorial & value classification)
2. Smart Question-to-Head Dispatcher:
   - Explicit: question specifies "head": "<name>"
   - Heuristic: pattern-matches qid / question intent (logs dispatch event for offline review & tuning)
   - Fallback: gracefully falls back to default/router head (logs fallback event)
3. Structured Logging for Heuristic Dispatches:
   - Appends JSON events to head_dispatch.jsonl for analysis of routing behavior & dataset gathering.
"""
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_deep_heads import DeepDecisionHeads


# Default log path for heuristic dispatch audit
DEFAULT_LOG_PATH = Path(
    os.environ.get("NANOJEV_DISPATCH_LOG", Path.home() / "Library" / "Logs" / "NanoJev" / "head_dispatch.jsonl")
)


def log_heuristic_dispatch(
    qid: str,
    resolved_head: str,
    mode: str,
    reason: str,
    question_dict: dict,
    log_path: Path = DEFAULT_LOG_PATH,
):
    """Log an audit event whenever a question is dispatched heuristically or via fallback."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        crit = question_dict.get("criteria")
        cand_cnt = len(crit) if isinstance(crit, (dict, list)) else (2 if question_dict.get("type") == "boolean" else None)
        inst = str(question_dict.get("instructions", ""))[:120]

        event = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "qid": qid,
            "resolved_head": resolved_head,
            "routing_mode": mode,
            "rule": reason,
            "question_type": question_dict.get("type"),
            "candidate_count": cand_cnt,
            "instructions_preview": inst,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Non-blocking: Logging failure should never interrupt model inference
        pass


class MultiHeadRegistry(nn.Module):
    def __init__(self, hidden_size: int = 1024, default_head_name: str = "router"):
        super().__init__()
        self.hidden_size = hidden_size
        self.default_head_name = default_head_name
        self.heads: Dict[str, nn.Module] = {}

    def register_head(self, name: str, head_module: nn.Module):
        """Register a decision head under a unique domain name."""
        self.heads[name] = head_module

    def get_head(self, name: str) -> Optional[nn.Module]:
        return self.heads.get(name)

    def resolve_head(
        self,
        qid: str,
        question_dict: dict,
        log_path: Path = DEFAULT_LOG_PATH,
    ) -> Tuple[nn.Module, str, str]:
        """Resolve the appropriate head for a given question.

        Returns:
            (head_module, head_name, routing_reason)
        """
        # 1. Explicit Routing: Question specifies "head": "<name>"
        explicit_name = question_dict.get("head")
        if explicit_name and explicit_name in self.heads:
            return self.heads[explicit_name], explicit_name, "explicit"

        # 2. Heuristic Pattern Matching
        qid_lower = qid.lower()
        matched_name = None
        matched_rule = None

        if any(k in qid_lower for k in ("complex", "risk", "indep", "route", "tier")):
            matched_name = "router"
            matched_rule = "pattern:router_keywords"
        elif (
            any(k in qid_lower for k in ("skill", "tool", "plugin", "action", "ability", "shortlist"))
            or qid_lower.startswith("verify_")
            or "skill" in str(question_dict.get("instructions", "")).lower()
        ):
            matched_name = "skill"
            matched_rule = "pattern:skill_keywords"
        elif any(k in qid_lower for k in ("agent", "subagent", "worker", "delegat")):
            matched_name = "agent"
            matched_rule = "pattern:agent_keywords"
        elif any(k in qid_lower for k in ("news", "value", "sentiment", "editorial")):
            matched_name = "news"
            matched_rule = "pattern:news_keywords"

        # If a heuristic rule matched and the head exists:
        if matched_name and matched_name in self.heads:
            log_heuristic_dispatch(qid, matched_name, "heuristic", matched_rule, question_dict, log_path)
            return self.heads[matched_name], matched_name, f"heuristic:{matched_rule}"

        # 3. Fallback: Default Head
        fallback_name = self.default_head_name if self.default_head_name in self.heads else next(iter(self.heads.keys()))
        reason = "fallback:default"
        log_heuristic_dispatch(qid, fallback_name, "fallback", reason, question_dict, log_path)
        return self.heads[fallback_name], fallback_name, reason

    def __call__(self, leaves: mx.array, examples: list, kmax: int, head_name: Optional[str] = None):
        """Forward through a specific head (or default head)."""
        target_name = head_name or self.default_head_name
        head = self.heads.get(target_name) or next(iter(self.heads.values()))
        return head(leaves, examples, kmax)
