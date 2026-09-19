#!/usr/bin/env python3
"""Extract TypeSafe requests from router audit log and test against NanoJev service.

Usage:
  # 1. Extract test cases only:
  python3 scripts/test_router_audit.py --audit-log /Users/chenyc/Downloads/router/jev-router-audit.jsonl --export-only /tmp/router_cases.json

  # 2. Test against a running server:
  python3 scripts/test_router_audit.py --audit-log /Users/chenyc/Downloads/router/jev-router-audit.jsonl --server-url http://127.0.0.1:8765

  # 3. Test local conversion and mock response (offline self-check):
  python3 scripts/test_router_audit.py --audit-log /Users/chenyc/Downloads/router/jev-router-audit.jsonl --offline-dry-run
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typesafe_adapter import (
    calculate_confidence,
    nanojev_response_to_typesafe,
    typesafe_request_to_nanojev,
)


def extract_audit_cases(audit_path: str, limit: int = 50):
    cases = []
    with open(audit_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            if "jev_request" in data:
                cases.append({
                    "line": line_num,
                    "at": data.get("at"),
                    "request": data["jev_request"],
                    "original_judgments": data.get("judgments"),
                    "chosen_model": data.get("chosen_model"),
                    "effort": data.get("effort"),
                })
                if limit and len(cases) >= limit:
                    break
    return cases


def run_offline_dry_run(cases):
    print(f"\n[Offline Dry Run] Verifying {len(cases)} captured TypeSafe requests through adapter...")
    passed = 0
    total = len(cases)

    for idx, c in enumerate(cases):
        req = c["request"]
        orig_judg = c["original_judgments"]

        # 1. Convert TypeSafe request to NanoJev format
        try:
            nj_payload, meta = typesafe_request_to_nanojev(req)
        except Exception as e:
            print(f"❌ Case {idx+1} (Line {c['line']}) conversion to NanoJev failed: {e}")
            continue

        # Check NanoJev format integrity
        states = nj_payload.get("states", [])
        assert len(states) == 1
        st = states[0]
        assert "state" in st and "questions" in st
        for qid, q in st["questions"].items():
            assert q["type"] in ("boolean", "choice", "score")

        # 2. Simulate response conversion
        mock_nj_answers = {}
        for qid, raw_q in meta["raw_questions"].items():
            if raw_q["type"] == "noul":
                mock_nj_answers[qid] = {
                    "type": "boolean",
                    "p_true": 0.75,
                    "probabilities": {"false": 0.25, "true": 0.75},
                    "value": True,
                }
            elif raw_q["type"] == "choice":
                opts = list(raw_q["criteria"].keys())
                mock_nj_answers[qid] = {
                    "type": "choice",
                    "choice": opts[0],
                    "value": opts[0],
                    "probabilities": {opt: 1.0 / len(opts) for opt in opts},
                }
            elif raw_q["type"] == "score":
                levels = raw_q["criteria"]
                mock_nj_answers[qid] = {
                    "type": "score",
                    "score": 1.5,
                    "level": 1,
                    "probabilities": {str(i): 1.0 / len(levels) for i in range(len(levels))},
                }

        mock_nj_res = {
            "execution": {"candidate_paths": 10},
            "states": [{"id": "req_0", "answers": mock_nj_answers}],
        }

        # 3. Convert back to official TypeSafe response
        try:
            ts_res = nanojev_response_to_typesafe(mock_nj_res, meta)
            assert "answers" in ts_res
            assert ts_res["model"] == req.get("model")
            for qid in meta["raw_questions"]:
                assert qid in ts_res["answers"]
                ans = ts_res["answers"][qid]
                if meta["raw_questions"][qid]["type"] == "noul":
                    assert ans["type"] == "noul" and "noul" in ans
                elif meta["raw_questions"][qid]["type"] == "choice":
                    assert ans["type"] == "choice" and "choice" in ans and "confidence" in ans
            passed += 1
        except Exception as e:
            print(f"❌ Case {idx+1} response conversion failed: {e}")

    print(f"✅ Offline dry-run passed: {passed}/{total} cases structurally valid.")
    return passed == total


def test_against_live_server(cases, server_url: str):
    print(f"\n[Live Test] Sending requests to TypeSafe endpoint: {server_url}/v1/systemone ...")
    endpoint = f"{server_url.rstrip('/')}/v1/systemone"

    for idx, c in enumerate(cases):
        req = c["request"]
        body = json.dumps(req).encode("utf-8")
        http_req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(http_req, timeout=10) as resp:
                status = resp.status
                res_data = json.loads(resp.read().decode("utf-8"))
                dt = (time.time() - t0) * 1000
                print(f"[{idx+1}/{len(cases)}] Status {status} in {dt:.1f}ms")
                print(f"  Response answers: {list(res_data.get('answers', {}).keys())}")
                if c.get("original_judgments"):
                    print(f"  Captured Jev was: {c['original_judgments']}")
        except urllib.error.HTTPError as e:
            print(f"❌ Request failed with HTTP {e.code}: {e.read().decode('utf-8')}")
        except Exception as e:
            print(f"❌ Connection error: {e}")
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-log", required=True, help="Path to jev-router-audit.jsonl")
    parser.add_argument("--limit", type=int, default=20, help="Number of audit cases to extract")
    parser.add_argument("--export-only", help="Export extracted cases to a JSON file and exit")
    parser.add_argument("--server-url", help="URL of running MLX server (e.g. http://127.0.0.1:8765)")
    parser.add_argument("--offline-dry-run", action="store_true", help="Run offline validation through adapter")
    args = parser.parse_args()

    cases = extract_audit_cases(args.audit_log, limit=args.limit)
    print(f"Extracted {len(cases)} live TypeSafe requests from {args.audit_log}")

    if args.export_only:
        Path(args.export_only).write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Exported to {args.export_only}")
        return

    if args.offline_dry_run or not args.server_url:
        run_offline_dry_run(cases)

    if args.server_url:
        test_against_live_server(cases, args.server_url)


if __name__ == "__main__":
    main()
