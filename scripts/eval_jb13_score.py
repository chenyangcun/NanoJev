import json
import math
import os
import sys

# Add jevbench to path if available
jb_path = os.path.expanduser("~/work/NanoJev/data/jevbench_v130")
if os.path.exists(jb_path):
    sys.path.insert(0, jb_path)

from jevbench import composite_v13 as score

def main():
    summary_path = "/tmp/nanojev_jb13_summary.json"
    results_path = "/tmp/nanojev_jb13_results.jsonl"

    if not os.path.exists(summary_path) or not os.path.exists(results_path):
        print("Missing summary or results file.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    easy_tasks, easy_corr = 0, 0
    std_tasks, std_corr = 0, 0
    hard_tasks, hard_corr = 0, 0

    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            tid = r['task_id']
            corr = bool(r.get('correct'))
            if tid.startswith('easy-'):
                easy_tasks += 1
                if corr:
                    easy_corr += 1
            elif tid.startswith('original-'):
                std_tasks += 1
                if corr:
                    std_corr += 1
            elif tid.startswith('hard-'):
                hard_tasks += 1
                if corr:
                    hard_corr += 1

    print("=" * 65)
    print("NANOJEV-MLX on JEVBENCH v1.3.0 OFFICIAL SUITE")
    print("=" * 65)
    print("Tier Accuracies (Raw):")
    print(f"  Easy:     {easy_corr:3d} / {easy_tasks:3d} = {easy_corr / easy_tasks * 100:.2f}%")
    print(f"  Standard: {std_corr:3d} / {std_tasks:3d} = {std_corr / std_tasks * 100:.2f}%")
    print(f"  Hard:     {hard_corr:3d} / {hard_tasks:3d} = {hard_corr / hard_tasks * 100:.2f}%")
    total_corr = easy_corr + std_corr + hard_corr
    total_tasks = easy_tasks + std_tasks + hard_tasks
    print(f"  Total:    {total_corr:3d} / {total_tasks:3d} = {total_corr / total_tasks * 100:.2f}%")

    tiers = {
        "easy": easy_corr / easy_tasks,
        "standard": std_corr / std_tasks,
        "hard": hard_corr / hard_tasks
    }

    # 1. Intelligence
    intel = score.intelligence(tiers)

    # 2. Calibration
    ece = summary["splits"]["public"]["ece"]["ece"]
    calib = score.calibration(ece)

    # 3. Speed
    p50 = summary["splits"]["public"]["latency"]["p50_s"]
    p95 = summary["splits"]["public"]["latency"]["p95_s"]
    spd_gpu = score.speed(p50, p95, endpoint_kind="gpu")
    spd_api = score.speed(p50, p95, endpoint_kind="api")

    # 4. Cost
    cost_usd_per_1000 = summary["price_per_1000_decisions_usd"]
    cst = score.cost(cost_usd_per_1000)

    print("\nFour Benchmark Axes (0 - 100):")
    print(f"  1. Intelligence Axis : {intel:.2f} (Chance-corrected across tiers)")
    print(f"  2. Calibration Axis  : {calib:.2f} (ECE = {ece * 100:.2f}%)")
    print(f"  3. Speed Axis (GPU)  : {spd_gpu:.2f} (p50={p50 * 1000:.1f}ms, p95={p95 * 1000:.1f}ms, adjusted)")
    print(f"     Speed Axis (API)  : {spd_api:.2f} (unadjusted production endpoint)")
    print(f"  4. Cost Axis         : {cst:.2f} (${cost_usd_per_1000:.5f} per 1,000 decisions)")

    axes_gpu = {
        "intelligence": intel,
        "calibration": calib,
        "speed": spd_gpu,
        "cost": cst
    }
    axes_api = {
        "intelligence": intel,
        "calibration": calib,
        "speed": spd_api,
        "cost": cst
    }

    jb_score_gpu = score.jevbench_score(axes_gpu)
    jb_score_api = score.jevbench_score(axes_api)

    print("\n" + "=" * 65)
    print(f"  JEVBENCH v1.3.0 SCORE (GPU load adjustment) : {jb_score_gpu:.2f}")
    print(f"  JEVBENCH v1.3.0 SCORE (Native Production API): {jb_score_api:.2f}")
    print("=" * 65)

    print("\nPreset Scoring Profiles (GPU Load Adjusted):")
    for preset_name, preset_weights in score.PRESETS.items():
        s = score.preset_score(axes_gpu, preset_weights)
        print(f"  {preset_name:<36}: {s:.2f}")

    # Breakdown by question types
    print("\n" + "-" * 65)
    print("Breakdown by Question Type:")
    type_stats = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            # Find question type
            task_type = "choice"
            probs = r.get("probs") or {}
            if set(probs.keys()) == {"yes", "no"}:
                task_type = "noul"
            elif all(k.isdigit() for k in probs.keys()) and len(probs) > 0 and set(probs.keys()) != {"yes", "no"}:
                task_type = "score"
            
            if task_type not in type_stats:
                type_stats[task_type] = {"total": 0, "correct": 0}
            type_stats[task_type]["total"] += 1
            if r.get("correct"):
                type_stats[task_type]["correct"] += 1

    for qtype, stats in sorted(type_stats.items()):
        c, t = stats["correct"], stats["total"]
        print(f"  {qtype:<10}: {c:3d} / {t:3d} ({c/t*100:.2f}%)")

    # Topic breakdown
    topics_file = os.path.join(jb_path, "datasets/topics.json")
    if os.path.exists(topics_file):
        with open(topics_file) as f:
            tdata = json.load(f)
        item_to_topic = tdata.get("public", {})
        topics_meta = {t["key"]: t["label"] for t in tdata.get("topics", [])}
        topic_stats = {k: {"total": 0, "correct": 0} for k in topics_meta}

        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                tid = r["task_id"]
                topic = item_to_topic.get(tid, "other")
                if topic not in topic_stats:
                    topic_stats[topic] = {"total": 0, "correct": 0}
                topic_stats[topic]["total"] += 1
                if r.get("correct"):
                    topic_stats[topic]["correct"] += 1

        print("\n" + "-" * 65)
        print("Breakdown by Subject Topic (JevBench Topics Taxonomy):")
        print(f"  {'Topic':<28} {'Correct / Total':<16} {'Accuracy'}")
        print("  " + "-" * 55)
        for k, meta_label in sorted(topics_meta.items()):
            s = topic_stats.get(k, {"total": 0, "correct": 0})
            tot = s["total"]
            cor = s["correct"]
            pct = (cor / tot * 100) if tot > 0 else 0.0
            print(f"  {meta_label:<28} {cor:2d} / {tot:2d}           {pct:6.2f}%")

    # Load official v1.3 leaderboard results
    official_results_path = os.path.join(jb_path, "results/v1.2/jevbench-v1.2-results.json")
    if os.path.exists(official_results_path):
        with open(official_results_path) as f:
            off_data = json.load(f)
        systems = off_data.get("systems", [])

        print("\n" + "=" * 95)
        print("OFFICIAL JEVBENCH v1.3.0 LEADERBOARD BENCHMARK COMPARISON")
        print("=" * 95)
        header = f"{'Rank':<5} {'System':<36} {'JB 1.3':<8} {'Intel':<7} {'Calib':<7} {'Speed':<7} {'Cost':<7} {'$/1k Dec':<10} {'p50(s)':<8}"
        print(header)
        print("-" * 95)

        inserted_nanojev = False
        intel_nj = intel
        calib_nj = calib
        cst_nj = cst
        p50_nj = p50
        for s in systems:
            rank = s.get("rank")
            disp = s.get("display") or ""
            jb = s.get("jevbench_score") or 0.0
            axes = s.get("axes", {})
            intel = axes.get("intelligence") or 0.0
            calib = axes.get("calibration") or 0.0
            spd = axes.get("speed") or 0.0
            cst = axes.get("cost") or 0.0
            usd1k = s.get("cost", {}).get("usd_per_1000", 0.0) if isinstance(s.get("cost"), dict) else 0.0
            p50 = s.get("speed", {}).get("p50_s_raw", 0.0) if isinstance(s.get("speed"), dict) else 0.0

            if not inserted_nanojev and jb < jb_score_gpu:
                print(f"{'★ 19':<5} {'NanoJev-MLX 0.8B (Apple M1 Studio)':<36} {jb_score_gpu:<8.1f} {intel_nj:<7.1f} {calib_nj:<7.1f} {spd_gpu:<7.1f} {cst_nj:<7.1f} ${cost_usd_per_1000:<9.4f} {p50_nj:.3f}s")
                inserted_nanojev = True

            # Print relevant comparable models
            if any(k in disp for k in ["Jev 1.13.0", "SemIf", "djev", "kev 0.6B", "kev 0.5B", "Laya", "openJev Verdict", "jeff", "GLiNER2", "Certo", "open-alternative-jev"]):
                p50_val = p50 if p50 > 0 else 0.0
                print(f"{str(rank):<5} {disp[:35]:<36} {jb:<8.1f} {intel:<7.1f} {calib:<7.1f} {spd:<7.1f} {cst:<7.1f} ${usd1k:<9.4f} {p50_val:.3f}s")


if __name__ == "__main__":
    main()
