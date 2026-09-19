#!/usr/bin/env python3
"""Compare NanoJev predictions against captured Jev official audit logs."""
import argparse
import json
import urllib.request
from typing import Dict, Any


def compare_cases(audit_path: str, server_url: str, limit: int = 5):
    endpoint = f"{server_url.rstrip('/')}/v1/systemone"
    cases = []
    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "jev_request" in d and d.get("judgments"):
                cases.append({
                    "request": d["jev_request"],
                    "jev_judgments": d["judgments"],
                    "chosen_model": d.get("chosen_model"),
                    "effort": d.get("effort"),
                    "at": d.get("at"),
                })
                if limit and len(cases) >= limit:
                    break

    print(f"Comparing {len(cases)} cases against live NanoJev at {endpoint}...\n")
    print("=" * 80)

    for i, c in enumerate(cases):
        req = c["request"]
        body = json.dumps(req).encode("utf-8")
        http_req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(http_req) as resp:
            nano_res = json.loads(resp.read().decode("utf-8"))

        nano_answers = nano_res.get("answers", {})
        jev_answers = c["jev_judgments"]

        print(f"Case #{i+1} ({c.get('at')})")
        print(f"  Task state preview: {str(req.get('state', {}))[:120]}...")
        print("-" * 80)

        # Compare questions
        for qid in req.get("questions", {}):
            qtype = req["questions"][qid]["type"]
            nano_ans = nano_answers.get(qid, {})
            jev_ans = jev_answers.get(qid, {})

            if qtype == "noul":
                n_val = nano_ans.get("noul")
                j_val = jev_ans.get("noul")
                diff = abs(n_val - j_val) if (n_val is not None and j_val is not None) else None
                diff_str = f"{diff:+.4f}" if diff is not None else "N/A"
                print(f"  [{qid} - noul]")
                print(f"    Official Jev:  {j_val}")
                print(f"    NanoJev (MLX): {n_val}  (abs diff: {diff_str})")

            elif qtype == "choice":
                n_choice = nano_ans.get("choice")
                j_choice = jev_ans.get("choice")
                match = "MATCH ✅" if n_choice == j_choice else "DIFF ❌"
                print(f"  [{qid} - choice]  {match}")
                print(f"    Official Jev:  choice='{j_choice}', confidence={jev_ans.get('confidence')}")
                print(f"                   probs={jev_ans.get('probabilities')}")
                print(f"    NanoJev (MLX): choice='{n_choice}', confidence={nano_ans.get('confidence')}")
                print(f"                   probs={nano_ans.get('probabilities')}")

            elif qtype == "score":
                print(f"  [{qid} - score]")
                print(f"    Official Jev:  score={jev_ans.get('score')}")
                print(f"    NanoJev (MLX): score={nano_ans.get('score')}")

        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-log", default="/Users/chenyc/Downloads/router/jev-router-audit.jsonl")
    parser.add_argument("--server-url", default="http://127.0.0.1:8769")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    compare_cases(args.audit_log, args.server_url, args.limit)
