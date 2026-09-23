import http.client
import json
import os
import sys
import time

jb_path = os.path.expanduser("~/work/NanoJev/data/jevbench_v130")
if os.path.exists(jb_path):
    sys.path.insert(0, jb_path)

from jevbench.tasks import load_jsonl
from jevbench.metrics import ece_top_label, brier_score
from jevbench import composite_v13 as score

def main():
    easy = load_jsonl(os.path.join(jb_path, "datasets/public/easy.jsonl"))
    orig = load_jsonl(os.path.join(jb_path, "datasets/public/original.jsonl"))
    hard = load_jsonl(os.path.join(jb_path, "datasets/public/hard.jsonl"))
    tasks = easy + orig + hard
    print(f"Loaded {len(tasks)} tasks (easy={len(easy)}, standard={len(orig)}, hard={len(hard)}).")

    temperatures = [0.35, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

    headers = {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection("127.0.0.1", 8769, timeout=30)

    print("=" * 96)
    print(f"{'Temp':<6} {'Acc(%)':<8} {'MeanConf':<10} {'Conf>95%':<10} {'ECE(%)':<8} {'Brier':<8} {'CalibScore':<12} {'JB 1.3 Score'}")
    print("=" * 96)

    best_temp = 0.35
    best_jb_score = 0.0

    for temp in temperatures:
        pairs = []
        briers = []
        easy_corr = 0
        std_corr = 0
        hard_corr = 0
        latencies = []

        t0 = time.time()
        for t in tasks:
            q = {"type": t.question["type"], "instructions": t.question["instructions"]}
            if t.question.get("criteria") is not None:
                q["criteria"] = t.question["criteria"]
            body = {
                "state": t.state,
                "model": "general",
                "temperature": temp,
                "questions": {"decision": q}
            }
            body_bytes = json.dumps(body).encode("utf-8")
            
            t_req = time.perf_counter()
            conn.request("POST", "/v1/systemone", body=body_bytes, headers=headers)
            resp = conn.getresponse()
            data = json.loads(resp.read().decode("utf-8"))
            latencies.append(time.perf_counter() - t_req)

            ans = data["answers"]["decision"]
            qtype = t.question["type"]
            if qtype == "noul":
                p_yes = ans["noul"]
                probs = {"yes": p_yes, "no": 1.0 - p_yes}
                pred = "yes" if p_yes >= 0.5 else "no"
            else:
                probs = ans["probabilities"]
                pred = max(probs.keys(), key=lambda k: probs[k])

            gold = str(t.expected)
            is_corr = (pred == gold)
            if is_corr:
                if t.id.startswith("easy-"):
                    easy_corr += 1
                elif t.id.startswith("original-"):
                    std_corr += 1
                elif t.id.startswith("hard-"):
                    hard_corr += 1

            conf = max(probs.values())
            pairs.append((conf, is_corr))
            bs = brier_score(probs, gold, t.labels)
            briers.append(bs)

        tot_corr = easy_corr + std_corr + hard_corr
        acc = tot_corr / len(tasks)
        mean_conf = sum(p[0] for p in pairs) / len(pairs)
        high_conf_cnt = sum(1 for p in pairs if p[0] > 0.95)

        ece_res = ece_top_label(pairs)
        ece_val = ece_res["ece"]
        calib = score.calibration(ece_val)
        mean_brier = sum(briers) / len(briers)

        tiers = {
            "easy": easy_corr / len(easy),
            "standard": std_corr / len(orig),
            "hard": hard_corr / len(hard)
        }
        intel = score.intelligence(tiers)

        # Speed score
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        spd = score.speed(p50, p95, endpoint_kind="gpu")

        # Cost score (reference $0.00614 per 1k)
        cst = score.cost(0.006137)

        axes = {
            "intelligence": intel,
            "calibration": calib,
            "speed": spd,
            "cost": cst
        }
        jb_score = score.jevbench_score(axes)

        if jb_score > best_jb_score:
            best_jb_score = jb_score
            best_temp = temp

        print(f"{temp:<6.2f} {acc*100:<8.2f} {mean_conf*100:<10.2f} {high_conf_cnt:<10d} {ece_val*100:<8.2f} {mean_brier:<8.4f} {calib:<12.2f} {jb_score:<8.2f}")

    conn.close()
    print("=" * 96)
    print(f"Optimal Temperature: T = {best_temp} -> Peak JevBench 1.3 Score = {best_jb_score:.2f}")

if __name__ == "__main__":
    main()
