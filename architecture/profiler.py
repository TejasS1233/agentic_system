"""
Profiler Agent for IASCIS

Dynamically profiles tool executions to provide performance metadata for:
- Tool Decay Algorithm (better decay scores based on actual performance)
- Vector Store / Knowledge Graph (enriched metadata for weighted retrieval)
- Self-Correction (identify slow or inefficient tools for regeneration)
- Latency-Aware Program Synthesis (research metrics)

Profiling Modes:
- OFF: No profiling (production batches)
- LIGHTWEIGHT: Time + Peak Memory only (<1% overhead)
- STANDARD: + CPU + I/O (~5% overhead)
- FULL: + cProfile bottlenecks (~15% overhead)
- GPU: + CUDA metrics for ML tools (~10% overhead)

Usage:
    from architecture.profiler import Profiler, ProfilingMode

    profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)

    # Profile a function
    result, profile = profiler.profile(my_func, arg1, arg2, kwarg1=value)

    # Profile a callable (tool)
    result, profile = profiler.profile_tool(my_tool, {"n": 100})

    # Context manager
    with profiler.context() as ctx:
        result = my_func()
    profile = ctx.result

"""

import os
import time
import cProfile
import pstats
import io
import tracemalloc
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Protocol,
    runtime_checkable,
)
from functools import wraps

# Try to import psutil for CPU/IO metrics
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Try to import GPU libraries
try:
    import torch

    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
except (ImportError, Exception):
    HAS_PYNVML = False

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
#                            ENUMS & CONSTANTS
# =============================================================================


class ProfilingMode(Enum):
    """Profiling intensity levels."""

    OFF = "off"  # No profiling
    LIGHTWEIGHT = "lightweight"  # Time + memory only
    STANDARD = "standard"  # + CPU + I/O
    FULL = "full"  # + cProfile bottlenecks
    GPU = "gpu"  # + CUDA metrics


class LatencyGrade(Enum):
    """Tool latency classification."""

    FAST = "fast"  # < 10ms
    MODERATE = "moderate"  # 10-100ms
    SLOW = "slow"  # 100-1000ms
    CRITICAL = "critical"  # > 1000ms
    UNKNOWN = "unknown"  # Not yet profiled


# Latency thresholds in milliseconds
LATENCY_THRESHOLDS = {
    LatencyGrade.FAST: 10.0,
    LatencyGrade.MODERATE: 100.0,
    LatencyGrade.SLOW: 1000.0,
    # Anything above 1000ms is CRITICAL
}

# Memory thresholds in MB
MEMORY_THRESHOLDS = {
    "low": 10.0,
    "moderate": 50.0,
    "high": 100.0,
    "critical": 500.0,
}


# =============================================================================
#                            DATA CLASSES
# =============================================================================


@dataclass
class BottleneckInfo:
    """Information about a performance bottleneck."""

    function_name: str
    filename: str
    line_number: int
    total_time_seconds: float
    cumulative_time_seconds: float
    call_count: int
    time_percent: float  # Percentage of total execution time

    def to_dict(self) -> dict:
        return {
            "function": self.function_name,
            "file": self.filename,
            "line": self.line_number,
            "total_time_s": round(self.total_time_seconds, 6),
            "cumulative_time_s": round(self.cumulative_time_seconds, 6),
            "calls": self.call_count,
            "percent": round(self.time_percent, 2),
        }


@dataclass
class IOMetrics:
    """I/O operation metrics."""

    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    disk_read_count: int = 0
    disk_write_count: int = 0

    def to_dict(self) -> dict:
        return {
            "disk_read_bytes": self.disk_read_bytes,
            "disk_write_bytes": self.disk_write_bytes,
            "disk_read_mb": round(self.disk_read_bytes / (1024 * 1024), 3),
            "disk_write_mb": round(self.disk_write_bytes / (1024 * 1024), 3),
            "disk_read_count": self.disk_read_count,
            "disk_write_count": self.disk_write_count,
        }


@dataclass
class GPUMetrics:
    """GPU profiling metrics."""

    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cuda_time_ms: float = 0.0

    def to_dict(self) -> dict:
        if not self.gpu_available:
            return {"gpu_available": False}
        return {
            "gpu_available": True,
            "gpu_name": self.gpu_name,
            "memory_allocated_mb": round(self.gpu_memory_allocated_mb, 2),
            "memory_reserved_mb": round(self.gpu_memory_reserved_mb, 2),
            "memory_total_mb": round(self.gpu_memory_total_mb, 2),
            "memory_used_percent": round(
                (self.gpu_memory_allocated_mb / self.gpu_memory_total_mb * 100)
                if self.gpu_memory_total_mb > 0
                else 0,
                2,
            ),
            "utilization_percent": round(self.gpu_utilization_percent, 2),
            "cuda_time_ms": round(self.cuda_time_ms, 3),
        }


@dataclass
class ProfileResult:
    """
    Complete result of profiling a tool or function execution.

    This is the primary output of the Profiler, containing all collected
    metrics organized by category.
    """

    # Identification
    tool_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    profiling_mode: ProfilingMode = ProfilingMode.LIGHTWEIGHT

    # Core timing metrics
    execution_time_ms: float = 0.0
    wall_time_ms: float = 0.0  # Wall clock time (may differ due to threading)

    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0  # Change from start to end
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_time_user_ms: float = 0.0
    cpu_time_system_ms: float = 0.0

    # I/O metrics
    io_metrics: IOMetrics = field(default_factory=IOMetrics)

    # GPU metrics
    gpu_metrics: GPUMetrics = field(default_factory=GPUMetrics)

    # cProfile results
    bottlenecks: List[BottleneckInfo] = field(default_factory=list)
    recursion_depth: int = 0
    total_function_calls: int = 0

    # Computed scores
    efficiency_score: float = 1.0  # 0.0 - 1.0 (higher is better)
    latency_grade: LatencyGrade = LatencyGrade.UNKNOWN

    # Execution info
    success: bool = True
    error_message: str = ""
    return_value_type: str = ""

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_latency_grade()
        self._calculate_efficiency_score()

    def _calculate_latency_grade(self) -> None:
        """Assign latency grade based on execution time."""
        if self.execution_time_ms < LATENCY_THRESHOLDS[LatencyGrade.FAST]:
            self.latency_grade = LatencyGrade.FAST
        elif self.execution_time_ms < LATENCY_THRESHOLDS[LatencyGrade.MODERATE]:
            self.latency_grade = LatencyGrade.MODERATE
        elif self.execution_time_ms < LATENCY_THRESHOLDS[LatencyGrade.SLOW]:
            self.latency_grade = LatencyGrade.SLOW
        else:
            self.latency_grade = LatencyGrade.CRITICAL

    def _calculate_efficiency_score(self) -> None:
        """
        Calculate overall efficiency score (0.0 - 1.0).

        Factors:
        - Latency (40%): Faster is better
        - Memory (30%): Lower peak memory is better
        - CPU (20%): Lower CPU usage is better
        - Success (10%): Success = 1.0, Failure = 0.0
        """
        # Latency score (inversely proportional, capped)
        # 1ms = 1.0, 1000ms = 0.1, >10000ms = 0.0
        if self.execution_time_ms <= 1:
            latency_score = 1.0
        elif self.execution_time_ms >= 10000:
            latency_score = 0.0
        else:
            latency_score = 1.0 - (self.execution_time_ms / 10000)

        # Memory score (inversely proportional)
        # <10MB = 1.0, 100MB = 0.5, >500MB = 0.0
        if self.peak_memory_mb <= MEMORY_THRESHOLDS["low"]:
            memory_score = 1.0
        elif self.peak_memory_mb >= MEMORY_THRESHOLDS["critical"]:
            memory_score = 0.0
        else:
            memory_score = 1.0 - (self.peak_memory_mb / MEMORY_THRESHOLDS["critical"])

        # CPU score (lower is better, but some CPU usage is expected)
        # <10% = 1.0, 100% = 0.5
        cpu_score = max(0.0, 1.0 - (self.cpu_percent / 200))

        # Success score
        success_score = 1.0 if self.success else 0.0

        # Weighted combination
        self.efficiency_score = (
            latency_score * 0.4
            + memory_score * 0.3
            + cpu_score * 0.2
            + success_score * 0.1
        )

    def to_dict(self) -> dict:
        """Export to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat(),
            "profiling_mode": self.profiling_mode.value,
            "timing": {
                "execution_time_ms": round(self.execution_time_ms, 3),
                "wall_time_ms": round(self.wall_time_ms, 3),
            },
            "memory": {
                "peak_mb": round(self.peak_memory_mb, 3),
                "delta_mb": round(self.memory_delta_mb, 3),
                "start_mb": round(self.memory_start_mb, 3),
                "end_mb": round(self.memory_end_mb, 3),
            },
            "cpu": {
                "percent": round(self.cpu_percent, 2),
                "user_time_ms": round(self.cpu_time_user_ms, 3),
                "system_time_ms": round(self.cpu_time_system_ms, 3),
            },
            "io": self.io_metrics.to_dict(),
            "gpu": self.gpu_metrics.to_dict(),
            "bottlenecks": [b.to_dict() for b in self.bottlenecks[:5]],  # Top 5
            "recursion_depth": self.recursion_depth,
            "total_function_calls": self.total_function_calls,
            "scores": {
                "efficiency": round(self.efficiency_score, 4),
                "latency_grade": self.latency_grade.value,
            },
            "execution": {
                "success": self.success,
                "error_message": self.error_message,
                "return_value_type": self.return_value_type,
            },
        }

    def summary(self) -> str:
        lines = [
            f"═══ Profile: {self.tool_name or 'Anonymous'} ═══",
            f"  Mode: {self.profiling_mode.value}",
            f"  Time: {self.execution_time_ms:.2f}ms {self.latency_grade.value}",
            f"  Memory: {self.peak_memory_mb:.2f}MB peak (Δ{self.memory_delta_mb:+.2f}MB)",
        ]

        if self.cpu_percent > 0:
            lines.append(f"  CPU: {self.cpu_percent:.1f}%")

        if self.io_metrics.disk_read_bytes > 0 or self.io_metrics.disk_write_bytes > 0:
            lines.append(
                f"  I/O: R:{self.io_metrics.disk_read_bytes / 1024:.1f}KB "
                f"W:{self.io_metrics.disk_write_bytes / 1024:.1f}KB"
            )

        if self.gpu_metrics.gpu_available:
            lines.append(
                f"  GPU: {self.gpu_metrics.gpu_memory_allocated_mb:.1f}MB "
                f"({self.gpu_metrics.gpu_utilization_percent:.1f}%)"
            )

        if self.bottlenecks:
            lines.append(f"  Bottlenecks: {len(self.bottlenecks)} found")
            for b in self.bottlenecks[:3]:
                lines.append(f"    - {b.function_name}: {b.time_percent:.1f}%")

        lines.append(f"  Efficiency: {self.efficiency_score:.2%}")

        if not self.success:
            lines.append(f"  ⚠️ Error: {self.error_message[:50]}...")

        return "\n".join(lines)


# =============================================================================
#                            PROFILING CONTEXT
# =============================================================================


class ProfilingContext:
    """
    Context manager for profiling code blocks.

    Usage:
        with ProfilingContext(mode=ProfilingMode.STANDARD) as ctx:
            # Your code here
            result = do_something()

        profile = ctx.result
        print(profile.summary())
    """

    def __init__(
        self,
        mode: ProfilingMode = ProfilingMode.LIGHTWEIGHT,
        tool_name: str = "",
    ):
        self.mode = mode
        self.tool_name = tool_name
        self._result: Optional[ProfileResult] = None

        # Timing
        self._start_time: float = 0.0
        self._start_perf: float = 0.0

        # Memory
        self._memory_snapshot_start: Optional[Any] = None

        # CPU (psutil)
        self._process: Optional[Any] = None
        self._cpu_times_start: Optional[Any] = None
        self._io_counters_start: Optional[Any] = None

        # cProfile
        self._profiler: Optional[cProfile.Profile] = None

        # GPU
        self._cuda_start_event: Optional[Any] = None
        self._cuda_end_event: Optional[Any] = None
        self._gpu_memory_start: float = 0.0

    def __enter__(self) -> "ProfilingContext":
        self._start_profiling()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_profiling(
            success=exc_type is None, error_message=str(exc_val) if exc_val else ""
        )

    @property
    def result(self) -> ProfileResult:
        """Get the profiling result. Available after context exits."""
        if self._result is None:
            raise RuntimeError("Profiling not yet complete. Use within 'with' block.")
        return self._result

    def _start_profiling(self) -> None:
        """Initialize all profiling mechanisms."""
        # Always: timing
        self._start_time = time.time()
        self._start_perf = time.perf_counter()

        # Lightweight+: memory
        if self.mode != ProfilingMode.OFF:
            tracemalloc.start()
            self._memory_snapshot_start = tracemalloc.take_snapshot()

        # Standard+: CPU and I/O
        if self.mode in (ProfilingMode.STANDARD, ProfilingMode.FULL, ProfilingMode.GPU):
            if HAS_PSUTIL:
                self._process = psutil.Process()
                self._cpu_times_start = self._process.cpu_times()
                try:
                    self._io_counters_start = self._process.io_counters()
                except (AttributeError, psutil.Error):
                    self._io_counters_start = None

        # Full: cProfile
        if self.mode == ProfilingMode.FULL:
            self._profiler = cProfile.Profile()
            self._profiler.enable()

        # GPU: CUDA events
        if self.mode == ProfilingMode.GPU and HAS_TORCH:
            torch.cuda.synchronize()
            self._cuda_start_event = torch.cuda.Event(enable_timing=True)
            self._cuda_end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_start_event.record()
            self._gpu_memory_start = torch.cuda.memory_allocated() / (1024 * 1024)

    def _stop_profiling(self, success: bool = True, error_message: str = "") -> None:
        """Stop profiling and collect results."""
        # Timing
        execution_time_ms = (time.perf_counter() - self._start_perf) * 1000
        wall_time_ms = (time.time() - self._start_time) * 1000

        # Initialize result
        self._result = ProfileResult(
            tool_name=self.tool_name,
            profiling_mode=self.mode,
            execution_time_ms=execution_time_ms,
            wall_time_ms=wall_time_ms,
            success=success,
            error_message=error_message,
        )

        if self.mode == ProfilingMode.OFF:
            return

        # Memory
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self._result.peak_memory_mb = peak / (1024 * 1024)
            self._result.memory_end_mb = current / (1024 * 1024)

            # Calculate delta if we have start snapshot
            if self._memory_snapshot_start:
                snapshot_end = (
                    tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
                )
                # Approximate start memory from peak - delta
                self._result.memory_start_mb = max(
                    0, self._result.peak_memory_mb - 10
                )  # Estimate

            self._result.memory_delta_mb = (
                self._result.memory_end_mb - self._result.memory_start_mb
            )

        # CPU and I/O (Standard+)
        if self._process and HAS_PSUTIL:
            # CPU times
            cpu_times_end = self._process.cpu_times()
            if self._cpu_times_start:
                self._result.cpu_time_user_ms = (
                    cpu_times_end.user - self._cpu_times_start.user
                ) * 1000
                self._result.cpu_time_system_ms = (
                    cpu_times_end.system - self._cpu_times_start.system
                ) * 1000

            # CPU percent
            try:
                self._result.cpu_percent = self._process.cpu_percent()
            except psutil.Error:
                pass

            # I/O counters
            if self._io_counters_start:
                try:
                    io_end = self._process.io_counters()
                    self._result.io_metrics = IOMetrics(
                        disk_read_bytes=io_end.read_bytes
                        - self._io_counters_start.read_bytes,
                        disk_write_bytes=io_end.write_bytes
                        - self._io_counters_start.write_bytes,
                        disk_read_count=io_end.read_count
                        - self._io_counters_start.read_count,
                        disk_write_count=io_end.write_count
                        - self._io_counters_start.write_count,
                    )
                except (AttributeError, psutil.Error):
                    pass

        # cProfile (Full)
        if self._profiler:
            self._profiler.disable()
            self._analyze_cprofile()

        # GPU
        if self.mode == ProfilingMode.GPU and HAS_TORCH:
            self._cuda_end_event.record()
            torch.cuda.synchronize()

            cuda_time = self._cuda_start_event.elapsed_time(self._cuda_end_event)
            gpu_memory_end = torch.cuda.memory_allocated() / (1024 * 1024)

            self._result.gpu_metrics = GPUMetrics(
                gpu_available=True,
                gpu_name=torch.cuda.get_device_name(0),
                gpu_memory_allocated_mb=gpu_memory_end,
                gpu_memory_reserved_mb=torch.cuda.memory_reserved() / (1024 * 1024),
                gpu_memory_total_mb=torch.cuda.get_device_properties(0).total_memory
                / (1024 * 1024),
                cuda_time_ms=cuda_time,
            )

            # Get GPU utilization via pynvml if available
            if HAS_PYNVML:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._result.gpu_metrics.gpu_utilization_percent = util.gpu
                except Exception:
                    pass

        # Recalculate derived metrics
        self._result._calculate_latency_grade()
        self._result._calculate_efficiency_score()

    def _analyze_cprofile(self) -> None:
        """Analyze cProfile results to find bottlenecks."""
        if not self._profiler:
            return

        # Get stats
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats("cumulative")

        # Extract bottleneck info
        bottlenecks = []
        total_time = 0.0

        # Get raw stats
        for (filename, line, func_name), (
            cc,
            nc,
            tt,
            ct,
            callers,
        ) in stats.stats.items():
            total_time += tt

        if total_time == 0:
            total_time = 1e-9  # Avoid division by zero

        # Find top bottlenecks
        for (filename, line, func_name), (
            cc,
            nc,
            tt,
            ct,
            callers,
        ) in stats.stats.items():
            time_percent = (tt / total_time) * 100

            # Skip very fast functions
            if time_percent < 1.0:
                continue

            bottlenecks.append(
                BottleneckInfo(
                    function_name=func_name,
                    filename=os.path.basename(filename),
                    line_number=line,
                    total_time_seconds=tt,
                    cumulative_time_seconds=ct,
                    call_count=nc,
                    time_percent=time_percent,
                )
            )

        # Sort by time percent
        bottlenecks.sort(key=lambda x: x.time_percent, reverse=True)
        self._result.bottlenecks = bottlenecks[:10]  # Top 10

        # Total function calls
        self._result.total_function_calls = stats.total_calls

        # Calculate recursion depth (approximate from call counts vs unique functions)
        unique_funcs = len(stats.stats)
        if unique_funcs > 0:
            self._result.recursion_depth = max(1, stats.total_calls // unique_funcs)


# =============================================================================
#                            MAIN PROFILER CLASS
# =============================================================================


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for tool-like objects."""

    name: str

    def run(self, **kwargs) -> Any: ...


T = TypeVar("T")


class Profiler:
    """
    Main Profiler class for profiling tool and function executions.

    Features:
    - Multiple profiling modes (OFF, LIGHTWEIGHT, STANDARD, FULL, GPU)
    - Automatic GPU detection
    - Thread-safe profiling
    - History tracking
    - Aggregation statistics

    Usage:
        profiler = Profiler(mode=ProfilingMode.STANDARD)

        # Profile a function
        result, profile = profiler.profile(my_func, arg1, arg2)

        # Profile a tool
        result, profile = profiler.profile_tool(my_tool, {"n": 100})

        # Use as decorator
        @profiler.wrap
        def my_function():
            pass
    """

    def __init__(
        self,
        mode: ProfilingMode = ProfilingMode.LIGHTWEIGHT,
        history_size: int = 100,
        auto_detect_gpu: bool = True,
    ):
        """
        Initialize the Profiler.

        Args:
            mode: Default profiling mode
            history_size: Maximum number of profiles to keep in history
            auto_detect_gpu: If True, automatically use GPU mode when CUDA available
        """
        self.mode = mode
        self.history_size = history_size
        self._history: List[ProfileResult] = []
        self._lock = threading.Lock()

        # Auto-detect GPU
        if auto_detect_gpu and HAS_TORCH and mode != ProfilingMode.OFF:
            logger.info("GPU detected, GPU profiling available")

        # Log capabilities
        logger.info(
            f"Profiler initialized (mode={mode.value}, "
            f"psutil={HAS_PSUTIL}, torch={HAS_TORCH}, pynvml={HAS_PYNVML})"
        )

    def profile(
        self,
        func: Callable[..., T],
        *args,
        _profile_name: str = "",
        _profile_mode: Optional[ProfilingMode] = None,
        **kwargs,
    ) -> Tuple[T, ProfileResult]:
        """
        Profile a function execution.

        Args:
            func: The function to profile
            *args: Positional arguments for the function
            _profile_name: Optional name for the profile
            _profile_mode: Override the default profiling mode
            **kwargs: Keyword arguments for the function

        Returns:
            Tuple of (function result, ProfileResult)
        """
        mode = _profile_mode or self.mode
        name = _profile_name or getattr(func, "__name__", "anonymous")

        result = None
        exc_to_raise = None

        try:
            with ProfilingContext(mode=mode, tool_name=name) as ctx:
                result = func(*args, **kwargs)
        except Exception as e:
            exc_to_raise = e

        # Set return value type after context exits (when _result is populated)
        profile = ctx.result
        profile.return_value_type = (
            type(result).__name__ if result is not None else "None"
        )
        self._add_to_history(profile)

        # Re-raise if there was an exception
        if exc_to_raise is not None:
            raise exc_to_raise

        return result, profile

    def profile_tool(
        self,
        tool: ToolProtocol,
        args: Dict[str, Any],
        mode: Optional[ProfilingMode] = None,
    ) -> Tuple[Any, ProfileResult]:
        """
        Profile a tool execution.

        Args:
            tool: The tool object (must have .name and .run() method)
            args: Arguments to pass to tool.run()
            mode: Override the default profiling mode

        Returns:
            Tuple of (tool result, ProfileResult)
        """
        mode = mode or self.mode

        result = None
        with ProfilingContext(mode=mode, tool_name=tool.name) as ctx:
            result = tool.run(**args)

        # Set return value type after context exits (when _result is populated)
        profile = ctx.result
        profile.return_value_type = (
            type(result).__name__ if result is not None else "None"
        )
        self._add_to_history(profile)

        return result, profile

    def context(
        self,
        name: str = "",
        mode: Optional[ProfilingMode] = None,
    ) -> ProfilingContext:
        """
        Get a profiling context manager.

        Args:
            name: Name for this profiling session
            mode: Override the default profiling mode

        Returns:
            ProfilingContext that can be used with 'with' statement
        """
        return ProfilingContext(mode=mode or self.mode, tool_name=name)

    def wrap(
        self,
        func: Optional[Callable] = None,
        *,
        name: str = "",
        mode: Optional[ProfilingMode] = None,
    ) -> Callable:
        """
        Decorator to profile a function every time it's called.

        Usage:
            @profiler.wrap
            def my_function():
                pass

            @profiler.wrap(name="custom_name", mode=ProfilingMode.FULL)
            def another_function():
                pass
        """

        def decorator(fn: Callable) -> Callable:
            fn_name = name or getattr(fn, "__name__", "anonymous")

            @wraps(fn)
            def wrapper(*args, **kwargs):
                result, profile = self.profile(
                    fn, *args, _profile_name=fn_name, _profile_mode=mode, **kwargs
                )
                # Store profile as attribute on the function
                wrapper._last_profile = profile
                return result

            wrapper._last_profile = None
            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def _add_to_history(self, profile: ProfileResult) -> None:
        """Add a profile to history (thread-safe)."""
        with self._lock:
            self._history.append(profile)
            if len(self._history) > self.history_size:
                self._history.pop(0)

    @property
    def history(self) -> List[ProfileResult]:
        """Get profiling history (copy to prevent modification)."""
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        """Clear profiling history."""
        with self._lock:
            self._history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all profiled executions.

        Returns:
            Dictionary with min, max, mean, median for key metrics
        """
        with self._lock:
            if not self._history:
                return {"count": 0}

            times = [p.execution_time_ms for p in self._history]
            memories = [p.peak_memory_mb for p in self._history]
            efficiencies = [p.efficiency_score for p in self._history]

            grades = {}
            for p in self._history:
                grade = p.latency_grade.value
                grades[grade] = grades.get(grade, 0) + 1

            return {
                "count": len(self._history),
                "execution_time_ms": {
                    "min": min(times),
                    "max": max(times),
                    "mean": sum(times) / len(times),
                    "median": sorted(times)[len(times) // 2],
                },
                "peak_memory_mb": {
                    "min": min(memories),
                    "max": max(memories),
                    "mean": sum(memories) / len(memories),
                },
                "efficiency_score": {
                    "min": min(efficiencies),
                    "max": max(efficiencies),
                    "mean": sum(efficiencies) / len(efficiencies),
                },
                "latency_grade_distribution": grades,
                "success_rate": sum(1 for p in self._history if p.success)
                / len(self._history),
            }

    def get_tool_statistics(self, tool_name: str) -> Dict[str, Any]:
        """Get statistics for a specific tool."""
        with self._lock:
            tool_profiles = [p for p in self._history if p.tool_name == tool_name]

            if not tool_profiles:
                return {"tool_name": tool_name, "count": 0}

            times = [p.execution_time_ms for p in tool_profiles]
            memories = [p.peak_memory_mb for p in tool_profiles]

            return {
                "tool_name": tool_name,
                "count": len(tool_profiles),
                "execution_time_ms": {
                    "min": min(times),
                    "max": max(times),
                    "mean": sum(times) / len(times),
                },
                "peak_memory_mb": {
                    "mean": sum(memories) / len(memories),
                    "max": max(memories),
                },
                "success_rate": sum(1 for p in tool_profiles if p.success)
                / len(tool_profiles),
                "last_profiled": max(p.timestamp for p in tool_profiles).isoformat(),
            }


# =============================================================================
#                            CONVENIENCE FUNCTIONS
# =============================================================================

# Global default profiler instance
_default_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get or create the default profiler instance."""
    global _default_profiler
    if _default_profiler is None:
        _default_profiler = Profiler()
    return _default_profiler


def set_profiler(profiler: Profiler) -> None:
    """Set the default profiler instance."""
    global _default_profiler
    _default_profiler = profiler


def profile(func: Callable[..., T], *args, **kwargs) -> Tuple[T, ProfileResult]:
    """Profile a function using the default profiler."""
    return get_profiler().profile(func, *args, **kwargs)


def profiled(
    func: Optional[Callable] = None,
    *,
    name: str = "",
    mode: Optional[ProfilingMode] = None,
) -> Callable:
    """
    Decorator to profile a function using the default profiler.

    Usage:
        @profiled
        def my_function():
            pass

        @profiled(name="custom", mode=ProfilingMode.FULL)
        def another():
            pass
    """
    return get_profiler().wrap(func, name=name, mode=mode)


# =============================================================================
#                            INTEGRATION HELPERS
# =============================================================================


def profile_to_metrics_update(profile: ProfileResult) -> Dict[str, Any]:
    """
    Convert a ProfileResult to a format suitable for updating ToolMetrics.

    This helper is for integration with the existing tool decay system.

    Returns:
        Dictionary with keys that can be used to update ToolMetrics
    """
    return {
        "execution_time_ms": profile.execution_time_ms,
        "peak_memory_mb": profile.peak_memory_mb,
        "cpu_percent": profile.cpu_percent,
        "efficiency_score": profile.efficiency_score,
        "latency_grade": profile.latency_grade.value,
        "success": profile.success,
    }


def profile_to_registry_update(profile: ProfileResult) -> Dict[str, Any]:
    """
    Convert a ProfileResult to a format suitable for registry.json updates.

    Returns:
        Dictionary for the 'performance' field in registry.json
    """
    return {
        "performance": {
            "avg_execution_time_ms": profile.execution_time_ms,
            "peak_memory_mb": profile.peak_memory_mb,
            "efficiency_score": round(profile.efficiency_score, 4),
            "latency_grade": profile.latency_grade.value,
            "last_profiled": profile.timestamp.isoformat(),
        }
    }


# =============================================================================
#                            SELF-TEST
# =============================================================================

if __name__ == "__main__":
    # Quick self-test
    print("=" * 60)
    print("PROFILER SELF-TEST")
    print("=" * 60)

    # Test 1: Simple function profiling
    def test_function(n: int) -> int:
        """Compute factorial."""
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    profiler = Profiler(mode=ProfilingMode.STANDARD)

    result, profile = profiler.profile(test_function, 100)
    print(f"\nTest 1: Factorial(100) = {result}")
    print(profile.summary())

    # Test 2: Context manager
    print("\n" + "-" * 40)
    with profiler.context(name="context_test") as ctx:
        # Simulate some work
        data = list(range(10000))
        sorted_data = sorted(data, reverse=True)

    print("Test 2: Context manager")
    print(ctx.result.summary())

    # Test 3: Decorator
    print("\n" + "-" * 40)

    @profiler.wrap
    def memory_test():
        """Allocate some memory."""
        return [i**2 for i in range(50000)]

    result = memory_test()
    print(f"Test 3: Memory test - created {len(result)} items")
    print(memory_test._last_profile.summary())

    # Test 4: Statistics
    print("\n" + "-" * 40)
    print("Test 4: Aggregate Statistics")
    stats = profiler.get_statistics()
    print(f"  Total profiles: {stats['count']}")
    print(f"  Avg execution time: {stats['execution_time_ms']['mean']:.2f}ms")
    print(f"  Avg memory: {stats['peak_memory_mb']['mean']:.2f}MB")
    print(f"  Success rate: {stats['success_rate']:.0%}")

    # Test 5: Full profiling
    print("\n" + "-" * 40)
    print("Test 5: Full profiling with cProfile")

    def recursive_fib(n: int) -> int:
        if n <= 1:
            return n
        return recursive_fib(n - 1) + recursive_fib(n - 2)

    result, profile = profiler.profile(
        recursive_fib, 20, _profile_mode=ProfilingMode.FULL
    )
    print(f"Fibonacci(20) = {result}")
    print(profile.summary())

    print("\n" + "=" * 60)
    print("SELF-TEST COMPLETE")
    print("=" * 60)


# =============================================================================
#                     PROFILE METRICS FROM LOGS
# =============================================================================


def get_weighted_metrics_from_logs(logs_dir: str = None) -> Dict[str, Dict[str, float]]:
    """
    Load the latest profiles JSON from logs folder and calculate scaled weighted metrics.

    Process:
    1. Sum raw metrics for each tool across all calls in an iteration
    2. Divide by call count to get average raw metrics per tool
    3. Scale the averaged values using MinMax
    4. Apply formula to calculate final weighted score

    Args:
        logs_dir: Path to logs directory. Defaults to project's logs folder.

    Returns:
        Dictionary with tool names as keys and their scaled weighted metrics as values.
    """
    import glob
    from pathlib import Path

    # Find logs directory
    if logs_dir is None:
        # Default to project root/logs
        current_dir = Path(__file__).parent.parent
        logs_dir = current_dir / "logs"
    else:
        logs_dir = Path(logs_dir)

    if not logs_dir.exists():
        logger.warning(f"Logs directory not found: {logs_dir}")
        return {}

    # Find latest profile JSON
    profile_files = sorted(glob.glob(str(logs_dir / "profiles_*.json")))
    if not profile_files:
        logger.warning("No profile JSON files found in logs directory")
        return {}

    latest_file = profile_files[-1]  # Most recent by filename (timestamp-sorted)
    logger.info(f"Loading profiles from: {latest_file}")

    # Load the JSON
    import json

    with open(latest_file, "r") as f:
        data = json.load(f)

    raw_profiles = data.get("raw_profiles", {})

    # Step 1: For each tool, sum raw metrics and calculate averages
    tool_avg_metrics: Dict[str, Dict[str, float]] = {}

    for tool_name, profiles in raw_profiles.items():
        # Initialize sums
        sum_execution_time = 0.0
        sum_peak_memory = 0.0
        sum_memory_delta = 0.0
        sum_efficiency = 0.0
        sum_success = 0.0
        call_count = len(profiles)

        # Sum all metrics for this tool
        for profile in profiles:
            sum_execution_time += profile.get("execution_time_ms", 0)
            sum_peak_memory += profile.get("peak_memory_mb", 0)
            sum_memory_delta += profile.get("memory_delta_mb", 0)
            sum_efficiency += profile.get("efficiency_score", 0)
            sum_success += 1.0 if profile.get("success", False) else 0.0

        # Calculate average raw metrics for this tool
        if call_count > 0:
            tool_avg_metrics[tool_name] = {
                "execution_time": sum_execution_time / call_count,
                "peak_memory": sum_peak_memory / call_count,
                "memory_delta": sum_memory_delta / call_count,
                "efficiency": sum_efficiency / call_count,
                "success": sum_success / call_count,
                "call_count": call_count,
            }
            logger.info(
                f"{tool_name} avg raw metrics (calls={call_count}): "
                f"time={tool_avg_metrics[tool_name]['execution_time']:.4f}ms, "
                f"mem={tool_avg_metrics[tool_name]['peak_memory']:.4f}MB"
            )

    # Step 2: Calculate min/max for MinMax scaling using averaged values
    def get_min_max(values):
        if not values:
            return 0, 1
        min_val = min(values)
        max_val = max(values)
        return min_val, max_val

    def minmax_scale(value, min_val, max_val):
        if max_val == min_val:
            # No variance - all tools have the same value
            # Return the actual normalized value (capped to 0-1)
            # This handles cases like: all tools succeeded (1.0) → return 1.0
            # Or all tools have same time → return 0.5 (neutral)
            if min_val == 0:
                return 0.5  # Neutral for zero values
            return min(max(value, 0.0), 1.0)  # Return actual value, clamped to 0-1
        return (value - min_val) / (max_val - min_val)

    # Collect all averaged metrics for scaling
    all_avg_metrics = {
        "execution_time": [m["execution_time"] for m in tool_avg_metrics.values()],
        "peak_memory": [m["peak_memory"] for m in tool_avg_metrics.values()],
        "memory_delta": [m["memory_delta"] for m in tool_avg_metrics.values()],
        "efficiency": [m["efficiency"] for m in tool_avg_metrics.values()],
        "success": [m["success"] for m in tool_avg_metrics.values()],
    }

    # Get scaling parameters from averaged values
    scales = {}
    for metric_name, values in all_avg_metrics.items():
        scales[metric_name] = get_min_max(values)
        logger.info(
            f"Scaler for avg_{metric_name}: min={scales[metric_name][0]:.4f}, max={scales[metric_name][1]:.4f}"
        )

    # Step 3 & 4: Scale the averaged values, invert cost metrics, and apply weighted average
    #
    # IMPORTANT: For cost metrics (time, memory), LOWER is BETTER.
    # After MinMax scaling, the slowest/heaviest tool gets 1.0 and fastest/lightest gets 0.0.
    # We INVERT these so that 1.0 = BEST (fastest/lightest) and 0.0 = WORST (slowest/heaviest).
    #
    # For benefit metrics (efficiency, success), HIGHER is BETTER - no inversion needed.
    #
    # Weights (must sum to 1.0):
    #   - Execution Time: 40% (most critical for tool performance)
    #   - Peak Memory: 20%
    #   - Memory Delta: 10% (less important than absolute memory)
    #   - Success Rate: 30% (critical - failed tools are useless)
    #
    # Note: We exclude the pre-calculated 'efficiency' metric to avoid double-counting
    # since it already incorporates time, memory, and success internally.

    WEIGHT_TIME = 0.40
    WEIGHT_PEAK_MEM = 0.20
    WEIGHT_MEM_DELTA = 0.10
    WEIGHT_SUCCESS = 0.30

    tool_totals: Dict[str, Dict[str, float]] = {}

    for tool_name, avg in tool_avg_metrics.items():
        # Scale each averaged metric to 0-1 range
        scaled_exec = minmax_scale(avg["execution_time"], *scales["execution_time"])
        scaled_peak = minmax_scale(avg["peak_memory"], *scales["peak_memory"])
        scaled_delta = minmax_scale(avg["memory_delta"], *scales["memory_delta"])
        scaled_eff = minmax_scale(avg["efficiency"], *scales["efficiency"])
        scaled_succ = minmax_scale(avg["success"], *scales["success"])

        # INVERT cost metrics: 1.0 - scaled_value
        # After inversion: 1.0 = fastest/lightest (BEST), 0.0 = slowest/heaviest (WORST)
        score_time = 1.0 - scaled_exec
        score_peak_mem = 1.0 - scaled_peak
        score_mem_delta = 1.0 - scaled_delta

        # Benefit metrics remain as-is (higher = better)
        score_success = scaled_succ

        # Apply weighted average formula (result is 0-1, then scale to 0-100)
        final_score = (
            (score_time * WEIGHT_TIME)
            + (score_peak_mem * WEIGHT_PEAK_MEM)
            + (score_mem_delta * WEIGHT_MEM_DELTA)
            + (score_success * WEIGHT_SUCCESS)
        ) * 100  # Scale to 0-100

        tool_totals[tool_name] = {
            "avg_execution_time": round(avg["execution_time"], 4),
            "avg_peak_memory": round(avg["peak_memory"], 4),
            "avg_memory_delta": round(avg["memory_delta"], 4),
            "avg_efficiency": round(avg["efficiency"], 4),
            "avg_success": round(avg["success"], 4),
            "scaled_execution_time": round(scaled_exec, 4),
            "scaled_peak_memory": round(scaled_peak, 4),
            "scaled_memory_delta": round(scaled_delta, 4),
            "scaled_efficiency": round(scaled_eff, 4),
            "scaled_success": round(scaled_succ, 4),
            "score_time": round(score_time, 4),
            "score_peak_mem": round(score_peak_mem, 4),
            "score_mem_delta": round(score_mem_delta, 4),
            "score_success": round(score_success, 4),
            "call_count": avg["call_count"],
            "combined_score": round(final_score, 4),  # 0-100 scale
        }

        logger.info(
            f"Tool '{tool_name}': time_score={score_time:.4f}, "
            f"mem_score={score_peak_mem:.4f}, success_score={score_success:.4f} "
            f"→ final={final_score:.2f}/100"
        )

    return tool_totals


def get_all_tool_metrics_list(logs_dir: str = None) -> List[Dict[str, Any]]:
    """
    Get a flat list of all tool metrics from latest profiles.

    Returns:
        List of dictionaries, each containing tool name and its weighted metrics.
    """
    tool_totals = get_weighted_metrics_from_logs(logs_dir)

    metrics_list = []
    for tool_name, metrics in tool_totals.items():
        entry = {"tool_name": tool_name, **metrics}
        metrics_list.append(entry)

    return metrics_list
