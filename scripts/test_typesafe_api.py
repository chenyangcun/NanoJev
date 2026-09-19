#!/usr/bin/env python3
"""Test TypeSafe AI standard wire format adapter."""
import unittest

from typesafe_adapter import (
    calculate_confidence,
    nanojev_response_to_typesafe,
    typesafe_request_to_nanojev,
)


class TestTypeSafeAdapter(unittest.TestCase):
    def test_typesafe_request_conversion(self):
        ts_req = {
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency"},
                },
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments, refunds",
                        "technical": "Bugs, outages",
                        "sales": None,
                    },
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"],
                },
            },
        }

        nj_payload, meta = typesafe_request_to_nanojev(ts_req)
        self.assertIn("states", nj_payload)
        self.assertEqual(len(nj_payload["states"]), 1)
        st = nj_payload["states"][0]
        self.assertEqual(st["state"], ts_req["state"])

        # Check types translated correctly
        self.assertEqual(st["questions"]["is_urgent"]["type"], "boolean")
        self.assertEqual(st["questions"]["department"]["type"], "choice")
        self.assertEqual(st["questions"]["department"]["criteria"]["sales"], "sales")  # null handled
        self.assertEqual(st["questions"]["frustration"]["type"], "score")

        # Mock NanoJev response
        nj_res = {
            "execution": {"candidate_paths": 7},
            "states": [
                {
                    "id": "req_0",
                    "answers": {
                        "is_urgent": {"type": "boolean", "p_true": 0.92, "probabilities": {"false": 0.08, "true": 0.92}},
                        "department": {
                            "type": "choice",
                            "choice": "technical",
                            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
                        },
                        "frustration": {
                            "type": "score",
                            "score": 1.6,
                            "level": 2,
                            "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
                        },
                    },
                }
            ],
        }

        ts_res = nanojev_response_to_typesafe(nj_res, meta)
        self.assertEqual(ts_res["model"], "jev-latest")
        self.assertIn("answers", ts_res)
        answers = ts_res["answers"]

        # 1. Noul answer
        self.assertEqual(answers["is_urgent"]["type"], "noul")
        self.assertEqual(answers["is_urgent"]["noul"], 0.92)

        # 2. Choice answer
        self.assertEqual(answers["department"]["type"], "choice")
        self.assertEqual(answers["department"]["choice"], "technical")
        self.assertIn("confidence", answers["department"])
        self.assertTrue(0.0 <= answers["department"]["confidence"] <= 1.0)

        # 3. Score answer
        self.assertEqual(answers["frustration"]["type"], "score")
        self.assertEqual(answers["frustration"]["score"], 1.6)
        self.assertEqual(answers["frustration"]["legend"]["0"], "Calm")
        self.assertEqual(answers["frustration"]["legend"]["1"], "Frustrated")
        self.assertEqual(answers["frustration"]["legend"]["2"], "Very angry")
        self.assertIn("confidence", answers["frustration"])

        # 4. Usage
        self.assertIn("usage", ts_res)
        self.assertEqual(ts_res["usage"]["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
