"""TypeSafe AI official API adapter for NanoJev.

Maps between official TypeSafe systemone wire format (https://docs.typesafe.ai/api)
and NanoJev internal evaluation format.

Key differences handled:
- Top-level request:
  TypeSafe: {"state": ..., "model": "jev-latest", "questions": {...}}
  NanoJev: {"states": [{"id": ..., "state": ..., "questions": {...}}]}
- Question types:
  TypeSafe "noul" <-> NanoJev "boolean"
  TypeSafe "choice" criteria with null descriptions -> default to option key
- Answers format:
  TypeSafe "noul" returns {"type": "noul", "noul": <float>}
  TypeSafe "choice" returns {"type": "choice", "choice": <str>, "probabilities": {...}, "confidence": <float>}
  TypeSafe "score" returns {"type": "score", "score": <float>, "legend": {...}, "probabilities": {...}, "confidence": <float>}
- Confidence calculation:
  Normalized difference or normalized entropy from probability distribution.
"""
import json
import math
from typing import Any, Dict, Tuple


def calculate_confidence(probs: Dict[str, float], qtype: str = "choice") -> float:
    """Calculate calibrated adaptive confidence in [0, 1] using multi-feature pooling.

    Integrates Top1, Margin gap (Top1 - Top2), and Normalized Entropy complement.
    """
    try:
        from adaptive_confidence import calculate_adaptive_confidence
        return calculate_adaptive_confidence(probs, qtype=qtype)
    except Exception:
        # Fallback to standard Shannon entropy complement
        values = list(probs.values())
        k = len(values)
        if k <= 1:
            return 1.0
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in values if p > 0)
        max_entropy = math.log(k)
        confidence = max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))
        return round(confidence, 4)


def typesafe_request_to_nanojev(ts_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert TypeSafe POST /v1/systemone request to NanoJev internal format.

    Returns:
        (nanojev_payload, request_metadata)
    """
    if not isinstance(ts_payload, dict):
        raise ValueError("Request body must be a JSON object")
    if "state" not in ts_payload or "questions" not in ts_payload:
        raise ValueError("Request body must contain 'state' and 'questions'")

    state = ts_payload["state"]
    # If state is dict, preserve as dict if it contains image or structured fields
    if isinstance(state, dict):
        state_repr = dict(state)
    elif isinstance(state, list):
        state_repr = list(state)
    elif isinstance(state, str):
        state_repr = state
    else:
        raise ValueError("Field 'state' must be string, object, or array")

    raw_questions = ts_payload["questions"]
    if not isinstance(raw_questions, dict) or not raw_questions:
        raise ValueError("Field 'questions' must be a non-empty object")

    req_model = str(ts_payload.get("model", "")).strip().lower()
    req_head = ts_payload.get("head")
    # If model is explicitly named after a registered head (e.g. "router", "general", "skill", "agent", "subagent", "memory", "news")
    if not req_head and req_model in ("router", "general", "skill", "agent", "subagent", "memory", "news"):
        req_head = req_model

    nj_questions = {}
    score_legends = {}

    for qid, q in raw_questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"Question '{qid}' must be an object")
        q_type = q.get("type")
        instructions = q.get("instructions")
        if not instructions:
            raise ValueError(f"Question '{qid}' missing 'instructions'")

        if isinstance(instructions, (dict, list)):
            instr_str = json.dumps(instructions, ensure_ascii=False)
        else:
            instr_str = str(instructions)

        head_for_q = q.get("head") or req_head

        if q_type == "noul":
            nj_q = {"type": "boolean", "instructions": instr_str}
            if "criteria" in q and isinstance(q["criteria"], dict):
                nj_q["criteria"] = {
                    k: str(v) for k, v in q["criteria"].items() if k in ("true", "false") and v
                }
            if head_for_q:
                nj_q["head"] = str(head_for_q).strip()
            nj_questions[qid] = nj_q

        elif q_type == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"Choice question '{qid}' requires criteria map with >= 2 options")
            cleaned_criteria = {}
            for opt, desc in criteria.items():
                cleaned_criteria[opt] = desc if desc is not None else opt
            nj_q = {
                "type": "choice",
                "instructions": instr_str,
                "criteria": cleaned_criteria,
            }
            if head_for_q:
                nj_q["head"] = str(head_for_q).strip()
            nj_questions[qid] = nj_q

        elif q_type == "score":
            criteria = q.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(f"Score question '{qid}' requires criteria array with >= 2 levels")
            str_levels = [str(lvl) for lvl in criteria]
            score_legends[qid] = {str(i): lvl for i, lvl in enumerate(str_levels)}
            nj_q = {
                "type": "score",
                "instructions": instr_str,
                "criteria": str_levels,
            }
            if head_for_q:
                nj_q["head"] = str(head_for_q).strip()
            nj_questions[qid] = nj_q
        else:
            raise ValueError(f"Unsupported question type: '{q_type}'")

    state_id = "req_0"
    nj_payload = {
        "states": [
            {
                "id": state_id,
                "state": state_repr,
                "questions": nj_questions,
            }
        ]
    }

    meta = {
        "model": ts_payload.get("model", "nanojev"),
        "raw_questions": raw_questions,
        "score_legends": score_legends,
    }
    return nj_payload, meta


def nanojev_response_to_typesafe(nj_output: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NanoJev internal evaluation response to official TypeSafe response."""
    state_result = nj_output["states"][0]
    answers = {}

    for qid, raw_q in meta["raw_questions"].items():
        q_type = raw_q.get("type")
        nj_ans = state_result["answers"].get(qid)
        if not nj_ans:
            continue

        if q_type == "noul":
            # Return {"type": "noul", "noul": <p_true>}
            p_true = nj_ans.get("p_true")
            if p_true is None:
                p_true = nj_ans["probabilities"].get("true", 0.5)
            answers[qid] = {
                "type": "noul",
                "noul": round(p_true, 4),
            }

        elif q_type == "choice":
            probs = {k: round(v, 4) for k, v in nj_ans["probabilities"].items()}
            answers[qid] = {
                "type": "choice",
                "choice": nj_ans["choice"],
                "probabilities": probs,
                "confidence": calculate_confidence(probs, qtype="choice"),
            }

        elif q_type == "score":
            probs = {k: round(v, 4) for k, v in nj_ans["probabilities"].items()}
            legend = meta["score_legends"].get(qid, {})
            answers[qid] = {
                "type": "score",
                "score": round(nj_ans["score"], 2),
                "legend": legend,
                "probabilities": probs,
                "confidence": calculate_confidence(probs, qtype="score"),
            }

    total_input_tokens = sum(len(ex["leaf_tokens"]) for ex in nj_output.get("examples", []))
    if not total_input_tokens:
        total_input_tokens = nj_output.get("execution", {}).get("total_input_tokens", 0)
    if not total_input_tokens:
        total_input_tokens = nj_output.get("execution", {}).get("candidate_paths", 0) * 150

    return {
        "model": meta["model"],
        "answers": answers,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": 0,  # Zero output tokens in NanoJev/Jev!
        },
    }
