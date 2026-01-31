# IASCIS Benchmark Metrics Documentation

> **Complete reference for all metrics collected by the IASCIS benchmarking system.**
> Use this document as a reference for your research paper.

---

## Table of Contents

- [IASCIS Benchmark Metrics Documentation](#iascis-benchmark-metrics-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Performance Metrics](#1-performance-metrics)
    - [Summary Statistics Provided](#summary-statistics-provided)
  - [2. LLM/Model Metrics](#2-llmmodel-metrics)
    - [Pricing Table (per 1M tokens)](#pricing-table-per-1m-tokens)
  - [3. Tool Usage Metrics](#3-tool-usage-metrics)
    - [Per-Tool Breakdown Example](#per-tool-breakdown-example)
  - [4. Privacy \& Routing Metrics](#4-privacy--routing-metrics)
    - [Privacy Guarantees](#privacy-guarantees)
  - [5. Resource Utilization Metrics](#5-resource-utilization-metrics)
    - [Resource Snapshots](#resource-snapshots)
  - [6. Self-Correction \& Error Handling Metrics](#6-self-correction--error-handling-metrics)
  - [7. Quality \& Correctness Metrics](#7-quality--correctness-metrics)
  - [8. Comparative Metrics](#8-comparative-metrics)
    - [Per-Model Breakdown (`per_model_breakdown`)](#per-model-breakdown-per_model_breakdown)
    - [Per-Zone Breakdown (`per_zone_breakdown`)](#per-zone-breakdown-per_zone_breakdown)
  - [9. Scalability Metrics](#9-scalability-metrics)
  - [10. Toolsmith-Specific Metrics](#10-toolsmith-specific-metrics)
  - [How to Run Benchmarks](#how-to-run-benchmarks)
    - [Quick Test (2 tasks)](#quick-test-2-tasks)
    - [Full Benchmark Suite](#full-benchmark-suite)
    - [With Multiple Runs (for statistical significance)](#with-multiple-runs-for-statistical-significance)
    - [Specific Category Only](#specific-category-only)
    - [Custom Output Directory](#custom-output-directory)
  - [Output Format](#output-format)
    - [File Structure](#file-structure)
    - [Task Results JSON Structure](#task-results-json-structure)
    - [Summary JSON Structure](#summary-json-structure)
  - [Statistical Analysis Recommendations](#statistical-analysis-recommendations)
  - [Citation](#citation)

---

## 1. Performance Metrics

Timing-based metrics that measure the speed and efficiency of task execution.

| Metric                         | Field Name                   | Unit         | Description                               | How It's Measured                                 |
| ------------------------------ | ---------------------------- | ------------ | ----------------------------------------- | ------------------------------------------------- |
| **End-to-End Latency**         | `total_duration_ms`          | milliseconds | Total time from task start to completion  | `time.perf_counter()` around entire task          |
| **Time-to-First-Token (TTFT)** | `time_to_first_token_ms`     | milliseconds | Time until LLM starts generating output   | First response timestamp - request timestamp      |
| **Planning Time**              | `planning_time_ms`           | milliseconds | Time spent in Orchestrator planning phase | `time.perf_counter()` around `orchestrator.run()` |
| **Execution Time**             | `execution_time_ms`          | milliseconds | Time spent in Agent execution loop        | `time.perf_counter()` around `agent.run()`        |
| **Dispatcher Routing Latency** | `dispatcher_routing_time_ms` | milliseconds | Time to classify task as public/private   | `time.perf_counter()` around `dispatcher.route()` |
| **Per-Turn Latency**           | `turns[].latency_ms`         | milliseconds | Average time per reasoning turn           | Individual timing per LLM call                    |
| **Tasks per Minute**           | `tasks_per_minute`           | count/min    | Throughput of the system                  | `total_tasks / (total_time_minutes)`              |

### Summary Statistics Provided
- Mean, Median, Min, Max, Standard Deviation for all timing metrics

---

## 2. LLM/Model Metrics

Metrics related to LLM usage, token consumption, and cost.

| Metric                        | Field Name                   | Unit    | Description                       | How It's Measured                       |
| ----------------------------- | ---------------------------- | ------- | --------------------------------- | --------------------------------------- |
| **Input Tokens**              | `total_tokens.input_tokens`  | count   | Total prompt tokens consumed      | From `response.usage.prompt_tokens`     |
| **Output Tokens**             | `total_tokens.output_tokens` | count   | Total completion tokens generated | From `response.usage.completion_tokens` |
| **Total Tokens**              | `total_tokens.total_tokens`  | count   | Sum of input + output tokens      | `input_tokens + output_tokens`          |
| **Cost per Task**             | `total_tokens.cost_usd`      | USD     | Estimated API cost                | `(tokens × rate_per_million)`           |
| **Number of Reasoning Turns** | `total_turns`                | count   | LLM loops before task completion  | Counter in `send_message()` loop        |
| **Tool Calls per Task**       | `total_tool_calls`           | count   | Number of tool invocations        | Count of `response.tool_calls`          |
| **Retry Count**               | `retry_count`                | count   | Retries due to 429/503 errors     | Counter in retry logic                  |
| **Max Turns Reached**         | `max_turns_reached`          | boolean | Whether MAX_TURNS limit was hit   | `total_turns >= 10`                     |

### Pricing Table (per 1M tokens)
| Model                           | Input Rate | Output Rate |
| ------------------------------- | ---------- | ----------- |
| `gemini/gemini-1.5-flash`       | $0.075     | $0.30       |
| `gemini/gemini-1.5-pro`         | $1.25      | $5.00       |
| `gemini/gemini-3-flash-preview` | $0.10      | $0.40       |
| `ollama/qwen2.5-coder:7b`       | $0.00      | $0.00       |

---

## 3. Tool Usage Metrics

Metrics about tool invocations, success rates, and execution times.

| Metric                        | Field Name                          | Unit         | Description                     | How It's Measured                         |
| ----------------------------- | ----------------------------------- | ------------ | ------------------------------- | ----------------------------------------- |
| **Tool Call Frequency**       | `tool_call_distribution`            | count        | Which tools are used most often | Counter per tool name                     |
| **Tool Call Distribution**    | `tool_call_distribution_percent`    | percentage   | % breakdown of tool usage       | `calls[tool] / total_calls × 100`         |
| **Tool Execution Time**       | `tool_execution_times_ms`           | milliseconds | Time spent executing each tool  | `time.perf_counter()` around `tool.run()` |
| **Tool Success Rate**         | `tool_success_rates`                | percentage   | % of tool calls that succeeded  | `successful / total × 100`                |
| **Overall Tool Success Rate** | `overall_tool_success_rate_percent` | percentage   | Global tool success rate        | All successful / all calls                |
| **Tool Error Distribution**   | `tool_errors`                       | list         | Types of errors per tool        | Logged from exceptions                    |
| **Average Arguments Size**    | `average_arguments_size_bytes`      | bytes        | Size of parameters passed       | `len(json.dumps(args))`                   |

### Per-Tool Breakdown Example
```json
{
  "write_file": {
    "success_rate_percent": 98.5,
    "total_calls": 45,
    "mean_execution_time_ms": 12.3
  },
  "run_command": {
    "success_rate_percent": 87.2,
    "total_calls": 89,
    "mean_execution_time_ms": 1250.6
  }
}
```

---

## 4. Privacy & Routing Metrics

Metrics about the Dispatcher's routing decisions and privacy zone usage.

| Metric                    | Field Name                   | Unit         | Description                   | How It's Measured                |
| ------------------------- | ---------------------------- | ------------ | ----------------------------- | -------------------------------- |
| **Routing Accuracy**      | `routing_accuracy_percent`   | percentage   | Correctly classified tasks    | Compare with `expected_zone`     |
| **Private Zone Usage**    | `private_zone_percent`       | percentage   | % routed to local Ollama      | `private_tasks / total × 100`    |
| **Public Zone Usage**     | `public_zone_percent`        | percentage   | % routed to Gemini cloud      | `public_tasks / total × 100`     |
| **Routing Decision Time** | `dispatcher_routing_time_ms` | milliseconds | Time to make routing decision | Timed `dispatcher.route()`       |
| **Correct Routings**      | `correct_routings`           | count        | Number correctly routed       | Count where `routed == expected` |
| **Incorrect Routings**    | `incorrect_routings`         | count        | Number incorrectly routed     | Count where `routed != expected` |

### Privacy Guarantees
- **Data Leakage Incidents**: Should always be 0
- Measured by: Log inspection + regex validation for sensitive patterns

---

## 5. Resource Utilization Metrics

System resource consumption during task execution.

| Metric                      | Field Name            | Unit       | Description                 | How It's Measured                    |
| --------------------------- | --------------------- | ---------- | --------------------------- | ------------------------------------ |
| **Peak RAM Usage**          | `peak_ram_mb`         | megabytes  | Maximum memory consumption  | `psutil.Process().memory_info().rss` |
| **Average RAM Usage**       | `average_ram_mb`      | megabytes  | Mean memory across duration | Periodic sampling every 500ms        |
| **Peak CPU Utilization**    | `peak_cpu_percent`    | percentage | Maximum CPU usage           | `psutil.cpu_percent()`               |
| **Average CPU Utilization** | `average_cpu_percent` | percentage | Mean CPU across duration    | Periodic sampling                    |
| **GPU/VRAM Usage**          | `gpu_memory_mb`       | megabytes  | For local Ollama model      | `pynvml` library (if available)      |
| **Disk Read**               | `disk_read_mb`        | megabytes  | Total bytes read            | `psutil.disk_io_counters()`          |
| **Disk Write**              | `disk_write_mb`       | megabytes  | Total bytes written         | `psutil.disk_io_counters()`          |

### Resource Snapshots
Collected at 500ms intervals with full `ResourceSnapshot` objects containing all metrics.

---

## 6. Self-Correction & Error Handling Metrics

Metrics about the system's ability to recover from errors autonomously.

| Metric                        | Field Name                          | Unit         | Description                      | How It's Measured                       |
| ----------------------------- | ----------------------------------- | ------------ | -------------------------------- | --------------------------------------- |
| **Error Recovery Rate**       | `error_recovery_rate_percent`       | percentage   | % of errors auto-recovered       | `recovered / (recovered + unrecovered)` |
| **Commands Re-executed**      | `commands_reexecuted`               | count        | Commands run again after failure | Counter on failed `run_command`         |
| **Code Regeneration Count**   | `code_regenerations`                | count        | Times code was rewritten         | Counter when code recreated             |
| **Backoff/Sleep Time**        | `backoff_time_total_ms`             | milliseconds | Time in exponential backoff      | Accumulated delay values                |
| **Total Retries**             | `retry_count`                       | count        | Total retry attempts             | Counter in retry logic                  |
| **Average Backoff per Retry** | `average_backoff_time_per_retry_ms` | milliseconds | Mean backoff duration            | `total_backoff / total_retries`         |

---

## 7. Quality & Correctness Metrics

Metrics about the quality and correctness of agent outputs.

| Metric                     | Field Name                        | Unit       | Description                       | How It's Measured              |
| -------------------------- | --------------------------------- | ---------- | --------------------------------- | ------------------------------ |
| **Task Completion Rate**   | `task_completion_rate_percent`    | percentage | % of tasks successfully finished  | `completed / total × 100`      |
| **Output Correctness**     | `output_correctness_rate_percent` | percentage | Whether output matches expected   | Manual evaluation / test cases |
| **Execution Success Rate** | `execution_success_rate_percent`  | percentage | % of code that runs without error | Track `run_command` exit codes |
| **Hallucination Rate**     | `hallucinations_detected`         | count      | References to non-existent items  | Log analysis for unknown tools |
| **Tasks Completed**        | `tasks_completed`                 | count      | Number of completed tasks         | Counter                        |
| **Tasks Incomplete**       | `tasks_incomplete`                | count      | Number of failed tasks            | Counter                        |

---

## 8. Comparative Metrics

Breakdown metrics for comparing different models and zones.

### Per-Model Breakdown (`per_model_breakdown`)
```json
{
  "gemini/gemini-3-flash-preview": {
    "task_count": 25,
    "average_duration_ms": 4523.5,
    "total_tokens": 125000,
    "total_cost_usd": 0.0125,
    "completion_rate_percent": 96.0
  },
  "ollama/qwen2.5-coder:7b": {
    "task_count": 15,
    "average_duration_ms": 8750.2,
    "total_tokens": 85000,
    "total_cost_usd": 0.0,
    "completion_rate_percent": 86.7
  }
}
```

### Per-Zone Breakdown (`per_zone_breakdown`)
```json
{
  "public": {
    "task_count": 25,
    "average_duration_ms": 4523.5,
    "total_tokens": 125000,
    "completion_rate_percent": 96.0
  },
  "private": {
    "task_count": 15,
    "average_duration_ms": 8750.2,
    "total_tokens": 85000,
    "completion_rate_percent": 86.7
  }
}
```

---

## 9. Scalability Metrics

Metrics for measuring system scalability.

| Metric               | Field Name           | Unit      | Description          | How It's Measured             |
| -------------------- | -------------------- | --------- | -------------------- | ----------------------------- |
| **Tasks per Minute** | `tasks_per_minute`   | count/min | System throughput    | `tasks / elapsed_minutes`     |
| **Rate Limit Hits**  | `retry_count` (429s) | count     | Number of 429 errors | Counter for rate limit errors |

---

## 10. Toolsmith-Specific Metrics

Metrics for the dynamic tool generation component (if enabled).

| Metric                          | Field Name | Unit         | Description                    | How It's Measured                 |
| ------------------------------- | ---------- | ------------ | ------------------------------ | --------------------------------- |
| **Tool Generation Time**        | TBD        | milliseconds | Time to create a new tool      | Measure `toolsmith.create_tool()` |
| **Generated Tool Success Rate** | TBD        | percentage   | % of generated tools that work | Test generated tools              |
| **Tool Reuse Rate**             | TBD        | count        | How often tools are reused     | Track usage over time             |
| **Tool TTL**                    | TBD        | days         | Average lifespan of tools      | From `tool_metadata.json`         |

---

## How to Run Benchmarks

### Quick Test (2 tasks)
```bash
cd agentic_system
uv run python -m benchmark.benchmark --quick
```

### Full Benchmark Suite
```bash
uv run python -m benchmark.benchmark
```

### With Multiple Runs (for statistical significance)
```bash
uv run python -m benchmark.benchmark --runs 5
```

### Specific Category Only
```bash
uv run python -m benchmark.benchmark --category private
uv run python -m benchmark.benchmark --category public
```

### Custom Output Directory
```bash
uv run python -m benchmark.benchmark --output my_results --runs 3
```

---

## Output Format

### File Structure
```
benchmark_results/
├── benchmark_tasks_20260131_095300.json    # Individual task results
└── benchmark_summary_20260131_095300.json  # Aggregate summary
```

### Task Results JSON Structure
```json
{
  "task_id": "pub_001_run1",
  "task_description": "Write a Python function...",
  "task_category": "public",
  "complexity_level": "simple",
  "start_time": "2026-01-31T09:50:00",
  "end_time": "2026-01-31T09:50:15",
  "total_duration_ms": 15234.5,
  "routed_zone": "public",
  "model_used": "gemini/gemini-3-flash-preview",
  "total_turns": 3,
  "total_tokens": {
    "input_tokens": 1250,
    "output_tokens": 890,
    "total_tokens": 2140,
    "cost_usd": 0.000481
  },
  "tool_call_distribution": {
    "write_file": 1,
    "run_command": 2
  },
  "task_completed": true,
  "execution_success": true
}
```

### Summary JSON Structure
```json
{
  "benchmark_info": {
    "total_tasks": 24,
    "timestamp": "2026-01-31T10:15:00"
  },
  "performance_metrics": {
    "total_duration": {"mean_ms": 12500, "median_ms": 11200, ...},
    "planning_time": {"mean_ms": 2500, ...},
    "tasks_per_minute": 4.8
  },
  "llm_metrics": {
    "total_tokens": 125000,
    "total_cost_usd": 0.0125,
    "reasoning_turns": {"mean": 3.2, "median": 3, ...}
  },
  "tool_metrics": {...},
  "routing_metrics": {...},
  "resource_metrics": {...},
  "self_correction_metrics": {...},
  "quality_metrics": {...},
  "per_model_breakdown": {...},
  "per_zone_breakdown": {...}
}
```

---

## Statistical Analysis Recommendations

For your research paper, consider:

1. **Run each task 3-5 times** to account for variance
2. **Calculate confidence intervals** (95% CI) for key metrics
3. **Use appropriate statistical tests**:
   - Paired t-test for comparing public vs private zone
   - ANOVA for comparing across complexity levels
4. **Report standard deviation** alongside means
5. **Create visualizations**:
   - Box plots for latency distributions
   - Bar charts for tool usage frequency
   - Time series for resource utilization

---

## Citation

If using this benchmark in your research, please cite:

```bibtex
@software{iascis_benchmark,
  title = {IASCIS Benchmark Suite},
  author = {Your Name},
  year = {2026},
  description = {Comprehensive benchmarking for Independent Autonomous Self-Correcting Intelligent Systems}
}
```
