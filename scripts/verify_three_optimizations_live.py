import json
import time
import sys
from pathlib import Path

# Add router directory
sys.path.insert(0, "/Users/chenyc/Documents/study/jev-cliproxy-router")
from http_transport import post_systemone_json

URL = "http://192.168.123.88:8770/v1/systemone"
SESSION_ID = f"test-session-{int(time.time())}"

# Common questions
QUESTIONS = {
    "complexity": {
        "type": "choice",
        "instructions": "Choose complexity",
        "criteria": {"bounded": "A", "standard": "B", "complex": "C", "exceptional": "D"}
    },
    "high_risk": {
        "type": "boolean",
        "instructions": "Is it high risk?",
        "criteria": {"true": "Yes", "false": "No"}
    },
    "independent": {
        "type": "boolean",
        "instructions": "Is it independent?",
        "criteria": {"true": "Yes", "false": "No"}
    }
}

headers = {
    "Content-Type": "application/json",
    "X-Decision-Head": "router",
    "X-Session-ID": SESSION_ID,
}

print("=" * 70)
print(f"Testing End-to-End Coordinated Multi-Turn Optimization with Session: {SESSION_ID}")
print("=" * 70)

# Turn 1: Cold start of session (Inherits from Static Pre-baked System Prompt)
state_t1 = {
    "user_task": "Task: Refactor database schema and implement optimistic locking mechanism in PostgreSQL.",
    "has_image": False,
    "user_turn_count": 1,
    "is_new_user_turn": True
}
payload_t1 = {"state": state_t1, "questions": QUESTIONS}

t0 = time.perf_counter()
resp1 = post_systemone_json(URL, payload_t1, headers, timeout=10.0)
dt1 = (time.perf_counter() - t0) * 1000
print(f"Turn 1 (Cold Session -> Pre-baked System Cache): status={resp1.status} | latency={dt1:.1f}ms")
print(f"       Answers: {resp1.payload.get('answers')}")

# Turn 2: Second turn in same session (Inherits from Turn 1 via Radix / Session-Affinity)
# New state appends previous tool execution results
state_t2 = dict(state_t1)
state_t2["previous_assistant"] = "I investigated the schema in db/schema.prisma and prepared migration script."
state_t2["recent_tool_calls"] = [{"name": "read_file", "operation": "read", "risk_tags": []}]
state_t2["user_turn_count"] = 2
payload_t2 = {"state": state_t2, "questions": QUESTIONS}

t0 = time.perf_counter()
resp2 = post_systemone_json(URL, payload_t2, headers, timeout=10.0)
dt2 = (time.perf_counter() - t0) * 1000
print(f"\nTurn 2 (Delta Tokens -> Radix Prefix Extension): status={resp2.status} | latency={dt2:.1f}ms!")
print(f"       Answers: {resp2.payload.get('answers')}")

# Turn 3: Exact State Match (e.g. repeated verification or cached query)
t0 = time.perf_counter()
resp3 = post_systemone_json(URL, payload_t2, headers, timeout=10.0)
dt3 = (time.perf_counter() - t0) * 1000
print(f"\nTurn 3 (Exact State Match -> Full Cache Hit):    status={resp3.status} | latency={dt3:.2f}ms!")
print(f"       Answers: {resp3.payload.get('answers')}")

print("\n" + "=" * 70)
print(f"⚡ Performance Evolution across Turns:")
print(f"  ● Turn 1 (Pre-baked System Cache):  {dt1:>6.1f} ms")
print(f"  ● Turn 2 (Radix Prefix Extension):   {dt2:>6.1f} ms (⚡ {dt1/dt2:.2f}x speedup!)")
print(f"  ● Turn 3 (Exact Full Match):         {dt3:>6.2f} ms (🚀 {dt1/dt3:.1f}x instant response!)")
print("=" * 70)
