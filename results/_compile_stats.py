"""Parse all IASCIS logs + profiles to find successful task runs and compile stats."""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

LOGS_DIR = Path("logs")
results = []

for log_file in sorted(LOGS_DIR.glob("IASCIS_*.log")):
    content = log_file.read_text(encoding="utf-8", errors="replace")

    plan_match = re.search(r"Executing plan: (.+)", content)
    if not plan_match:
        continue
    query = plan_match.group(1).strip()

    ts_match = re.search(r"IASCIS_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", log_file.name)
    timestamp = ts_match.group(1) if ts_match else ""

    completed_match = re.search(r"Completed in ([\d.]+)ms", content)
    total_time_ms = float(completed_match.group(1)) if completed_match else None

    plan_steps_match = re.search(r"EXECUTION PLAN: (\d+) steps", content)
    num_steps = int(plan_steps_match.group(1)) if plan_steps_match else None

    tool_plan_matches = re.findall(r"Tool: (\w+)", content)
    tools_used = tool_plan_matches if tool_plan_matches else []

    step_completions = re.findall(r"Step (\d+): Completed in ([\d.]+)s", content)
    steps_completed = len(step_completions)

    # Find matching profile file
    profile_data = None
    log_ts = timestamp.replace("-", "").replace("_", "")
    best_profile = None
    best_diff = float("inf")
    for pf in LOGS_DIR.glob("profiles_*.json"):
        pf_ts = pf.stem.replace("profiles_", "").replace("_", "")
        try:
            diff = abs(int(pf_ts) - int(log_ts[: len(pf_ts)]))
            if diff < best_diff and diff < 500:
                best_diff = diff
                best_profile = pf
        except ValueError:
            continue

    if best_profile:
        try:
            profile_data = json.loads(best_profile.read_text(encoding="utf-8"))
        except:
            pass

    tool_profiles = []
    if profile_data and "summary" in profile_data:
        for tool_name, stats in profile_data["summary"].items():
            tool_profiles.append(
                {
                    "tool": tool_name,
                    "avg_time_ms": round(stats.get("avg_time_ms", 0), 2),
                    "peak_mem_mb": round(stats.get("max_memory_mb", 0), 4),
                    "grade": stats.get("last_grade", "?"),
                    "calls": stats.get("call_count", 1),
                }
            )

    all_tools_success = True
    if profile_data and "raw_profiles" in profile_data:
        for tool_name, runs in profile_data["raw_profiles"].items():
            for run in runs:
                if not run.get("success", True):
                    all_tools_success = False

    all_steps_done = (num_steps is not None and steps_completed >= num_steps) or (
        num_steps is None and steps_completed > 0
    )
    is_successful = all_steps_done and total_time_ms is not None and all_tools_success

    results.append(
        {
            "timestamp": timestamp,
            "log_file": log_file.name,
            "query": query,
            "num_steps": num_steps or steps_completed,
            "steps_completed": steps_completed,
            "tools": tools_used,
            "total_time_ms": total_time_ms,
            "total_time_s": round(total_time_ms / 1000, 2) if total_time_ms else None,
            "step_timings": {s: float(t) for s, t in step_completions},
            "tool_profiles": tool_profiles,
            "is_successful": is_successful,
        }
    )

successful = [r for r in results if r["is_successful"]]

# Deduplicate by query - keep fastest run
seen_queries = {}
for r in successful:
    q = r["query"].lower().strip()
    q_key = re.sub(r"[^a-z0-9 ]", "", q)[:80]
    if q_key not in seen_queries or (
        r["total_time_ms"]
        and r["total_time_ms"] < (seen_queries[q_key]["total_time_ms"] or float("inf"))
    ):
        seen_queries[q_key] = r

unique_successful = sorted(seen_queries.values(), key=lambda x: x["timestamp"])

print(f"\n{'=' * 80}")
print("  IASCIS - SUCCESSFUL TASK RUNS COMPILATION")
print(f"{'=' * 80}")
print(f"  Total log files scanned:  {len(results)}")
print(f"  Successful runs found:    {len(successful)}")
print(f"  Unique successful tasks:  {len(unique_successful)}")
print(f"{'=' * 80}\n")

for i, r in enumerate(unique_successful, 1):
    total_s = r["total_time_s"] or 0
    tools_str = ", ".join(r["tools"]) if r["tools"] else "LLM-only"

    print(f"  {'-' * 70}")
    print(f"  #{i:02d} | {r['query']}")
    print(
        f"      | Time: {total_s}s | Steps: {r['num_steps']} | Date: {r['timestamp']}"
    )
    print(f"      | Tools: {tools_str}")

    if r["tool_profiles"]:
        for tp in r["tool_profiles"]:
            grade_icon = {
                "fast": "++",
                "moderate": "+ ",
                "slow": "- ",
                "critical": "--",
            }.get(tp["grade"], "??")
            print(
                f"      |   [{grade_icon}] {tp['tool']}: {tp['avg_time_ms']}ms, {tp['peak_mem_mb']}MB peak ({tp['grade']})"
            )

    if r["step_timings"]:
        timings = " -> ".join([f"S{s}:{t:.1f}s" for s, t in r["step_timings"].items()])
        print(f"      | Pipeline: {timings}")

    print()

print(f"{'=' * 80}")
print("  AGGREGATE STATISTICS")
print(f"{'=' * 80}")

all_times = [r["total_time_ms"] for r in unique_successful if r["total_time_ms"]]
if all_times:
    print(f"  Avg total time:   {sum(all_times) / len(all_times) / 1000:.2f}s")
    print(f"  Min time:         {min(all_times) / 1000:.2f}s")
    print(f"  Max time:         {max(all_times) / 1000:.2f}s")

tool_counts = {}
for r in unique_successful:
    for t in r["tools"]:
        tool_counts[t] = tool_counts.get(t, 0) + 1

if tool_counts:
    print("\n  Tool Usage Frequency:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"    {tool}: {count} tasks")

step_counts = {}
for r in unique_successful:
    n = r["num_steps"]
    step_counts[n] = step_counts.get(n, 0) + 1

if step_counts:
    print("\n  Pipeline Complexity:")
    for steps, count in sorted(step_counts.items()):
        print(f"    {steps}-step tasks: {count}")

print()

output = {
    "scan_date": datetime.now().isoformat(),
    "total_logs": len(results),
    "successful_runs": len(successful),
    "unique_tasks": len(unique_successful),
    "aggregate": {
        "avg_time_s": round(sum(all_times) / len(all_times) / 1000, 2)
        if all_times
        else 0,
        "min_time_s": round(min(all_times) / 1000, 2) if all_times else 0,
        "max_time_s": round(max(all_times) / 1000, 2) if all_times else 0,
        "tool_frequency": tool_counts,
        "step_distribution": {str(k): v for k, v in step_counts.items()},
    },
    "tasks": unique_successful,
}
with open("logs/successful_runs_summary.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)

print("  JSON summary saved to logs/successful_runs_summary.json")
