import time
import json
from dotenv import load_dotenv
from litellm import completion
from main import IASCIS

# Load environment variables
load_dotenv()

TASK_DESCRIPTION = (
    "Write a Python function that calculates the fibonacci sequence up to n terms"
)


def run_system_benchmark():
    print("\n[System] Running task via IASCIS system...")
    start_time = time.perf_counter()
    with IASCIS(enable_profiling=False) as system:
        result = system.run(TASK_DESCRIPTION)

    duration = (time.perf_counter() - start_time) * 1000
    print(f"[System] Completed in {duration:.2f}ms")
    return result["result"], duration


def run_direct_benchmark():
    print("\n[Direct] Running task via direct Gemini 3 Flash execution...")
    start_time = time.perf_counter()

    messages = [{"role": "user", "content": TASK_DESCRIPTION}]

    try:
        # Direct call using litellm as requested
        response = completion(
            model="gemini/gemini-3-flash-preview", messages=messages, temperature=0.0
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"Error: {e}"

    duration = (time.perf_counter() - start_time) * 1000
    print(f"[Direct] Completed in {duration:.2f}ms")
    return content, duration


def main():
    print(f"Benchmark Task: {TASK_DESCRIPTION}")
    print("-" * 50)

    # Run System
    try:
        system_output, system_time = run_system_benchmark()
    except Exception as e:
        print(f"[System] Failed: {e}")
        system_output = str(e)
        system_time = 0

    # Run Direct
    try:
        direct_output, direct_time = run_direct_benchmark()
    except Exception as e:
        print(f"[Direct] Failed: {e}")
        direct_output = str(e)
        direct_time = 0

    # Compare
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"IASCIS System Time : {system_time:.2f} ms")
    print(f"Direct Gemini Time : {direct_time:.2f} ms")

    overhead = 0
    overhead_pct = 0
    if direct_time > 0:
        overhead = system_time - direct_time
        overhead_pct = (overhead / direct_time) * 100
        print(f"System Overhead    : {overhead:.2f} ms ({overhead_pct:.1f}%)")

    print("\n[System Output Preview]")
    print(
        str(system_output)[:200] + "..."
        if len(str(system_output)) > 200
        else str(system_output)
    )

    print("\n[Direct Output Preview]")
    print(
        str(direct_output)[:200] + "..."
        if len(str(direct_output)) > 200
        else str(direct_output)
    )

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_description": TASK_DESCRIPTION,
        "iascis_system_time_ms": system_time,
        "direct_gemini_time_ms": direct_time,
        "overhead_ms": overhead,
        "overhead_percentage": overhead_pct,
        "system_output": str(system_output),
        "direct_output": str(direct_output),
    }

    output_file = "performance_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[Results] Saved to {output_file}")


if __name__ == "__main__":
    main()
