# Profiler Agent Documentation

> **Complete reference for the IASCIS Profiler Agent**
> Use this document to understand and integrate the profiler into your workflow.

---

## Table of Contents

- [Profiler Agent Documentation](#profiler-agent-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Quick Start](#2-quick-start)
  - [3. Profiling Modes](#3-profiling-modes)
  - [4. Core Components](#4-core-components)
    - [4.1 ProfileResult](#41-profileresult)
    - [4.2 ProfilingContext](#42-profilingcontext)
    - [4.3 Profiler](#43-profiler)
  - [5. Metrics Collected](#5-metrics-collected)
    - [5.1 Timing Metrics](#51-timing-metrics)
    - [5.2 Memory Metrics](#52-memory-metrics)
    - [5.3 CPU Metrics](#53-cpu-metrics)
    - [5.4 I/O Metrics](#54-io-metrics)
    - [5.5 GPU Metrics](#55-gpu-metrics)
    - [5.6 cProfile Bottlenecks](#56-cprofile-bottlenecks)
  - [6. Latency Grading](#6-latency-grading)
  - [7. Efficiency Score](#7-efficiency-score)
  - [8. Usage Examples](#8-usage-examples)
    - [8.1 Basic Function Profiling](#81-basic-function-profiling)
    - [8.2 Tool Profiling](#82-tool-profiling)
    - [8.3 Context Manager](#83-context-manager)
    - [8.4 Decorator](#84-decorator)
    - [8.5 Full Profiling with Bottlenecks](#85-full-profiling-with-bottlenecks)
  - [9. Integration Guide](#9-integration-guide)
    - [9.1 With ToolMetrics](#91-with-toolmetrics)
    - [9.2 With Registry.json](#92-with-registryjson)
    - [9.3 With Benchmark System](#93-with-benchmark-system)
  - [10. API Reference](#10-api-reference)
  - [11. Dependencies](#11-dependencies)
  - [12. Self-Test](#12-self-test)

---

## 1. Overview

The **Profiler Agent** is a performance analysis system for IASCIS that:

1. **Measures Tool Performance** - Execution time, memory usage, CPU utilization
2. **Detects Bottlenecks** - Using cProfile to find slow functions
3. **Grades Latency** - Fast/Moderate/Slow/Critical classification
4. **Calculates Efficiency** - Composite score for tool quality
5. **Supports GPU** - CUDA profiling for ML/AI tools

### Key Features

| Feature | Description |
|---------|-------------|
| **Multiple Modes** | OFF, LIGHTWEIGHT, STANDARD, FULL, GPU |
| **Low Overhead** | Lightweight mode < 1% overhead |
| **Thread-Safe** | Safe for concurrent profiling |
| **History Tracking** | Keeps history of profiles for statistics |
| **Flexible API** | Functions, context managers, decorators |

---

## 2. Quick Start

```python
from architecture.profiler import Profiler, ProfilingMode

# Create profiler
profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)

# Profile a function
def my_function(n):
    return sum(range(n))

result, profile = profiler.profile(my_function, 10000)

print(f"Result: {result}")
print(profile.summary())
```

Output:
```
Result: 49995000
═══ Profile: my_function ═══
  Mode: lightweight
  Time: 0.42ms 🟢 fast
  Memory: 0.15MB peak (Δ+0.02MB)
  Efficiency: 99.98%
```

---

## 3. Profiling Modes

| Mode | Overhead | What It Measures |
|------|----------|------------------|
| **OFF** | 0% | Nothing (passthrough) |
| **LIGHTWEIGHT** | <1% | Time + Peak Memory |
| **STANDARD** | ~5% | + CPU + I/O counters |
| **FULL** | ~15% | + cProfile bottlenecks |
| **GPU** | ~10% | + CUDA memory/timing |

### When to Use Each Mode

```python
# Production - minimal overhead
profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)

# Development - more detail
profiler = Profiler(mode=ProfilingMode.STANDARD)

# Debugging performance issues
profiler = Profiler(mode=ProfilingMode.FULL)

# ML/AI tools
profiler = Profiler(mode=ProfilingMode.GPU)
```

---

## 4. Core Components

### 4.1 ProfileResult

The main output of profiling. Contains all collected metrics.

```python
@dataclass
class ProfileResult:
    # Identification
    tool_name: str
    timestamp: datetime
    profiling_mode: ProfilingMode
    
    # Timing
    execution_time_ms: float
    wall_time_ms: float
    
    # Memory
    peak_memory_mb: float
    memory_delta_mb: float
    
    # CPU
    cpu_percent: float
    cpu_time_user_ms: float
    cpu_time_system_ms: float
    
    # I/O
    io_metrics: IOMetrics
    
    # GPU
    gpu_metrics: GPUMetrics
    
    # Bottlenecks
    bottlenecks: List[BottleneckInfo]
    recursion_depth: int
    total_function_calls: int
    
    # Scores
    efficiency_score: float  # 0.0 - 1.0
    latency_grade: LatencyGrade
    
    # Execution
    success: bool
    error_message: str
```

### 4.2 ProfilingContext

Context manager for profiling code blocks.

```python
with ProfilingContext(mode=ProfilingMode.STANDARD, tool_name="my_tool") as ctx:
    # Your code here
    result = do_something()

profile = ctx.result
print(profile.execution_time_ms)
```

### 4.3 Profiler

Main class for profiling operations.

```python
profiler = Profiler(
    mode=ProfilingMode.LIGHTWEIGHT,
    history_size=100,
    auto_detect_gpu=True,
)
```

---

## 5. Metrics Collected

### 5.1 Timing Metrics

| Metric | Field | Description |
|--------|-------|-------------|
| Execution Time | `execution_time_ms` | High-precision CPU time |
| Wall Time | `wall_time_ms` | Real-world clock time |

### 5.2 Memory Metrics

| Metric | Field | Description |
|--------|-------|-------------|
| Peak Memory | `peak_memory_mb` | Maximum memory allocated |
| Memory Delta | `memory_delta_mb` | Change from start to end |
| Start Memory | `memory_start_mb` | Memory at start |
| End Memory | `memory_end_mb` | Memory at end |

### 5.3 CPU Metrics

| Metric | Field | Description |
|--------|-------|-------------|
| CPU Percent | `cpu_percent` | CPU utilization |
| User Time | `cpu_time_user_ms` | User-mode CPU time |
| System Time | `cpu_time_system_ms` | Kernel-mode CPU time |

### 5.4 I/O Metrics

| Metric | Field | Description |
|--------|-------|-------------|
| Disk Read | `io_metrics.disk_read_bytes` | Bytes read from disk |
| Disk Write | `io_metrics.disk_write_bytes` | Bytes written to disk |
| Read Count | `io_metrics.disk_read_count` | Number of read operations |
| Write Count | `io_metrics.disk_write_count` | Number of write operations |

### 5.5 GPU Metrics

| Metric | Field | Description |
|--------|-------|-------------|
| GPU Available | `gpu_metrics.gpu_available` | Whether GPU profiling worked |
| GPU Name | `gpu_metrics.gpu_name` | Name of the GPU |
| Memory Allocated | `gpu_metrics.gpu_memory_allocated_mb` | VRAM in use |
| Memory Reserved | `gpu_metrics.gpu_memory_reserved_mb` | VRAM reserved |
| Utilization | `gpu_metrics.gpu_utilization_percent` | GPU compute usage |
| CUDA Time | `gpu_metrics.cuda_time_ms` | CUDA kernel time |

### 5.6 cProfile Bottlenecks

| Field | Description |
|-------|-------------|
| `bottlenecks[].function_name` | Name of the slow function |
| `bottlenecks[].filename` | Source file |
| `bottlenecks[].line_number` | Line number |
| `bottlenecks[].time_percent` | % of total time |
| `bottlenecks[].call_count` | Number of calls |

---

## 6. Latency Grading

Tools are automatically graded based on execution time:

| Grade | Execution Time | Color | Meaning |
|-------|---------------|-------|---------|
| **FAST** | < 10ms | 🟢 | Excellent |
| **MODERATE** | 10-100ms | 🟡 | Acceptable |
| **SLOW** | 100-1000ms | 🟠 | Needs optimization |
| **CRITICAL** | > 1000ms | 🔴 | Performance issue |

```python
if profile.latency_grade == LatencyGrade.CRITICAL:
    print("⚠️ This tool needs optimization!")
```

---

## 7. Efficiency Score

Composite score from 0.0 to 1.0 based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Latency** | 40% | Faster = better |
| **Memory** | 30% | Lower peak = better |
| **CPU** | 20% | Lower usage = better |
| **Success** | 10% | Success = 1.0, Failure = 0.0 |

```python
if profile.efficiency_score < 0.5:
    print("Tool needs improvement")
elif profile.efficiency_score > 0.9:
    print("Tool is highly efficient")
```

---

## 8. Usage Examples

### 8.1 Basic Function Profiling

```python
from architecture.profiler import Profiler, ProfilingMode

profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)

def calculate_factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

result, profile = profiler.profile(calculate_factorial, 1000)

print(f"Result: {result}")
print(f"Time: {profile.execution_time_ms:.2f}ms")
print(f"Memory: {profile.peak_memory_mb:.2f}MB")
print(f"Grade: {profile.latency_grade.value}")
```

### 8.2 Tool Profiling

```python
from architecture.profiler import Profiler

profiler = Profiler()

class MyTool:
    name = "factorial_tool"
    
    def run(self, n: int) -> int:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

tool = MyTool()
result, profile = profiler.profile_tool(tool, {"n": 100})

print(profile.summary())
```

### 8.3 Context Manager

```python
from architecture.profiler import Profiler, ProfilingMode

profiler = Profiler(mode=ProfilingMode.STANDARD)

with profiler.context(name="data_processing") as ctx:
    data = list(range(100000))
    sorted_data = sorted(data, reverse=True)
    filtered = [x for x in sorted_data if x % 2 == 0]

print(ctx.result.summary())
print(f"I/O: {ctx.result.io_metrics.to_dict()}")
```

### 8.4 Decorator

```python
from architecture.profiler import Profiler, ProfilingMode

profiler = Profiler()

@profiler.wrap
def my_function(n):
    return sum(range(n))

# Call the function
result = my_function(100000)

# Access the profile
print(my_function._last_profile.summary())
```

With custom settings:

```python
@profiler.wrap(name="custom_name", mode=ProfilingMode.FULL)
def another_function():
    # Complex logic here
    pass
```

### 8.5 Full Profiling with Bottlenecks

```python
from architecture.profiler import Profiler, ProfilingMode

profiler = Profiler()

def recursive_fibonacci(n):
    if n <= 1:
        return n
    return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)

result, profile = profiler.profile(
    recursive_fibonacci, 25,
    _profile_mode=ProfilingMode.FULL
)

print(f"Fibonacci(25) = {result}")
print(f"Total function calls: {profile.total_function_calls}")
print(f"\nTop Bottlenecks:")
for b in profile.bottlenecks[:3]:
    print(f"  {b.function_name}: {b.time_percent:.1f}% ({b.call_count} calls)")
```

---

## 9. Integration Guide

### 9.1 With ToolMetrics

```python
from architecture.profiler import profile_to_metrics_update

# After profiling a tool
updates = profile_to_metrics_update(profile)
# Returns: {
#     "execution_time_ms": 1.5,
#     "peak_memory_mb": 2.3,
#     "efficiency_score": 0.95,
#     "latency_grade": "fast",
#     ...
# }

# Use to update ToolMetrics
tool_metrics.update(**updates)
```

### 9.2 With Registry.json

```python
from architecture.profiler import profile_to_registry_update

# Get update for registry
update = profile_to_registry_update(profile)
# Returns: {
#     "performance": {
#         "avg_execution_time_ms": 1.5,
#         "peak_memory_mb": 2.3,
#         "efficiency_score": 0.95,
#         "latency_grade": "fast",
#         "last_profiled": "2026-02-04T21:00:00Z"
#     }
# }

# Update registry.json
registry[tool_name].update(update)
```

### 9.3 With Benchmark System

```python
# In benchmark.py
from architecture.profiler import Profiler, ProfilingMode

profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)

# Profile each tool execution
result, profile = profiler.profile_tool(tool, args)

# Add to metrics
metrics_collector.record_tool_profile(
    tool_name=tool.name,
    execution_time_ms=profile.execution_time_ms,
    peak_memory_mb=profile.peak_memory_mb,
    efficiency_score=profile.efficiency_score,
    latency_grade=profile.latency_grade.value,
)
```

---

## 10. API Reference

### Profiler

```python
class Profiler:
    def __init__(
        self,
        mode: ProfilingMode = ProfilingMode.LIGHTWEIGHT,
        history_size: int = 100,
        auto_detect_gpu: bool = True,
    ): ...
    
    def profile(
        self,
        func: Callable,
        *args,
        _profile_name: str = "",
        _profile_mode: Optional[ProfilingMode] = None,
        **kwargs
    ) -> Tuple[Any, ProfileResult]: ...
    
    def profile_tool(
        self,
        tool: ToolProtocol,
        args: Dict[str, Any],
        mode: Optional[ProfilingMode] = None,
    ) -> Tuple[Any, ProfileResult]: ...
    
    def context(
        self,
        name: str = "",
        mode: Optional[ProfilingMode] = None,
    ) -> ProfilingContext: ...
    
    def wrap(
        self,
        func: Optional[Callable] = None,
        *,
        name: str = "",
        mode: Optional[ProfilingMode] = None,
    ) -> Callable: ...
    
    @property
    def history(self) -> List[ProfileResult]: ...
    
    def clear_history(self) -> None: ...
    
    def get_statistics(self) -> Dict[str, Any]: ...
    
    def get_tool_statistics(self, tool_name: str) -> Dict[str, Any]: ...
```

### Convenience Functions

```python
# Get/set default profiler
def get_profiler() -> Profiler: ...
def set_profiler(profiler: Profiler) -> None: ...

# Quick profiling with default profiler
def profile(func: Callable, *args, **kwargs) -> Tuple[Any, ProfileResult]: ...

# Decorator with default profiler
@profiled
def my_function(): ...

@profiled(name="custom", mode=ProfilingMode.FULL)
def another(): ...
```

---

## 11. Dependencies

### Required
- Python 3.9+
- `tracemalloc` (built-in)
- `cProfile` (built-in)

### Optional (for enhanced features)
| Package | Feature | Install |
|---------|---------|---------|
| `psutil` | CPU + I/O metrics | `pip install psutil` |
| `torch` | GPU profiling | `pip install torch` |
| `pynvml` | GPU utilization | `pip install nvidia-ml-py3` |

The profiler gracefully degrades if optional dependencies are missing.

---

## 12. Self-Test

Run the profiler's self-test to verify installation:

```bash
cd agentic_system
uv run python -m architecture.profiler
```

Expected output:
```
============================================================
PROFILER SELF-TEST
============================================================

Test 1: Factorial(100) = 9332621544394415...
═══ Profile: test_function ═══
  Mode: standard
  Time: 0.05ms 🟢 fast
  Memory: 0.02MB peak (Δ+0.00MB)
  CPU: 0.0%
  Efficiency: 100.00%
...

============================================================
SELF-TEST COMPLETE
============================================================
```

---

*Document created: February 2026*
*Version: 1.0.0*
