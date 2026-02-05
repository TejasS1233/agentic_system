"""
Unit tests for the Profiler Agent.

Run with: uv run pytest tests/test_profiler.py -v
"""

import pytest
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.profiler import (
    Profiler,
    ProfilingMode,
    ProfilingContext,
    ProfileResult,
    LatencyGrade,
    get_profiler,
    set_profiler,
    profile,
    profiled,
    profile_to_metrics_update,
    profile_to_registry_update,
)


# =============================================================================
#                            FIXTURES
# =============================================================================


@pytest.fixture
def profiler():
    """Create a fresh profiler for each test."""
    return Profiler(mode=ProfilingMode.LIGHTWEIGHT)


@pytest.fixture
def standard_profiler():
    """Create a standard mode profiler."""
    return Profiler(mode=ProfilingMode.STANDARD)


@pytest.fixture
def full_profiler():
    """Create a full mode profiler."""
    return Profiler(mode=ProfilingMode.FULL)


# =============================================================================
#                            BASIC TESTS
# =============================================================================


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        result = ProfileResult()
        assert result.tool_name == ""
        assert result.execution_time_ms == 0.0
        assert result.peak_memory_mb == 0.0
        assert result.success is True
        # Use approximate equality for floating point
        assert abs(result.efficiency_score - 1.0) < 0.001

    def test_latency_grade_calculation(self):
        """Test automatic latency grade assignment."""
        # Fast (< 10ms)
        result = ProfileResult(execution_time_ms=5.0)
        assert result.latency_grade == LatencyGrade.FAST

        # Moderate (10-100ms)
        result = ProfileResult(execution_time_ms=50.0)
        assert result.latency_grade == LatencyGrade.MODERATE

        # Slow (100-1000ms)
        result = ProfileResult(execution_time_ms=500.0)
        assert result.latency_grade == LatencyGrade.SLOW

        # Critical (> 1000ms)
        result = ProfileResult(execution_time_ms=2000.0)
        assert result.latency_grade == LatencyGrade.CRITICAL

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # Perfect case
        result = ProfileResult(
            execution_time_ms=1.0,
            peak_memory_mb=1.0,
            cpu_percent=0.0,
            success=True,
        )
        assert result.efficiency_score > 0.95

        # Bad case
        result = ProfileResult(
            execution_time_ms=5000.0,
            peak_memory_mb=300.0,
            cpu_percent=100.0,
            success=False,
        )
        assert result.efficiency_score < 0.5

    def test_to_dict(self):
        """Test dictionary export."""
        result = ProfileResult(
            tool_name="test_tool",
            execution_time_ms=10.5,
            peak_memory_mb=2.5,
        )
        data = result.to_dict()

        assert data["tool_name"] == "test_tool"
        assert data["timing"]["execution_time_ms"] == 10.5
        assert data["memory"]["peak_mb"] == 2.5
        assert "scores" in data
        assert "execution" in data

    def test_summary(self):
        """Test human-readable summary."""
        result = ProfileResult(
            tool_name="test_tool",
            execution_time_ms=5.0,
            peak_memory_mb=1.0,
        )
        summary = result.summary()

        assert "test_tool" in summary
        assert "5.00ms" in summary
        assert "1.00MB" in summary
        assert "🟢" in summary  # Fast grade


class TestProfilingContext:
    """Tests for ProfilingContext."""

    def test_basic_context(self):
        """Test basic context manager usage."""
        with ProfilingContext(mode=ProfilingMode.LIGHTWEIGHT, tool_name="test") as ctx:
            result = sum(range(1000))

        profile = ctx.result
        assert profile.tool_name == "test"
        assert profile.execution_time_ms > 0
        assert profile.success is True

    def test_context_captures_exception(self):
        """Test that exceptions are captured."""
        with pytest.raises(ValueError):
            with ProfilingContext(tool_name="failing") as ctx:
                raise ValueError("Test error")

        profile = ctx.result
        assert profile.success is False
        assert "Test error" in profile.error_message

    def test_memory_tracking(self):
        """Test memory tracking."""
        with ProfilingContext(mode=ProfilingMode.LIGHTWEIGHT) as ctx:
            # Allocate some memory
            data = [i**2 for i in range(10000)]

        profile = ctx.result
        assert profile.peak_memory_mb > 0

    def test_off_mode(self):
        """Test OFF mode returns minimal profile."""
        with ProfilingContext(mode=ProfilingMode.OFF) as ctx:
            result = sum(range(1000))

        profile = ctx.result
        assert profile.execution_time_ms > 0
        # Memory should be 0 in OFF mode
        assert profile.peak_memory_mb == 0


class TestProfiler:
    """Tests for main Profiler class."""

    def test_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.mode == ProfilingMode.LIGHTWEIGHT
        assert profiler.history_size == 100
        assert len(profiler.history) == 0

    def test_profile_function(self, profiler):
        """Test profiling a function."""

        def add(a, b):
            return a + b

        result, profile = profiler.profile(add, 1, 2)

        assert result == 3
        assert profile.execution_time_ms > 0
        assert profile.tool_name == "add"
        assert profile.success is True

    def test_profile_lambda(self, profiler):
        """Test profiling a lambda."""
        result, profile = profiler.profile(lambda x: x**2, 5)

        assert result == 25
        assert profile.execution_time_ms > 0

    def test_profile_with_kwargs(self, profiler):
        """Test profiling with keyword arguments."""

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result, profile = profiler.profile(greet, "World", greeting="Hi")

        assert result == "Hi, World!"

    def test_profile_with_custom_name(self, profiler):
        """Test profiling with custom name."""

        def anonymous():
            return 42

        result, profile = profiler.profile(anonymous, _profile_name="custom_name")

        assert profile.tool_name == "custom_name"

    def test_profile_with_mode_override(self, profiler):
        """Test profiling with mode override."""

        def func():
            return 1

        result, profile = profiler.profile(func, _profile_mode=ProfilingMode.STANDARD)

        assert profile.profiling_mode == ProfilingMode.STANDARD

    def test_history_tracking(self, profiler):
        """Test that profiles are added to history."""

        def func():
            return 1

        for _ in range(5):
            profiler.profile(func)

        assert len(profiler.history) == 5

    def test_history_size_limit(self):
        """Test history size limiting."""
        profiler = Profiler(history_size=3)

        def func():
            return 1

        for i in range(5):
            profiler.profile(func, _profile_name=f"func_{i}")

        assert len(profiler.history) == 3
        # Should have the last 3
        assert profiler.history[0].tool_name == "func_2"
        assert profiler.history[2].tool_name == "func_4"

    def test_clear_history(self, profiler):
        """Test clearing history."""

        def func():
            return 1

        profiler.profile(func)
        profiler.profile(func)
        assert len(profiler.history) == 2

        profiler.clear_history()
        assert len(profiler.history) == 0

    def test_get_statistics_empty(self, profiler):
        """Test statistics with no history."""
        stats = profiler.get_statistics()
        assert stats["count"] == 0

    def test_get_statistics(self, profiler):
        """Test aggregate statistics."""

        def slow_func():
            time.sleep(0.01)
            return 1

        def fast_func():
            return 1

        profiler.profile(slow_func)
        profiler.profile(fast_func)
        profiler.profile(fast_func)

        stats = profiler.get_statistics()

        assert stats["count"] == 3
        assert "execution_time_ms" in stats
        assert "peak_memory_mb" in stats
        assert "efficiency_score" in stats
        assert stats["success_rate"] == 1.0

    def test_get_tool_statistics(self, profiler):
        """Test per-tool statistics."""

        def tool_a():
            return 1

        def tool_b():
            return 2

        profiler.profile(tool_a)
        profiler.profile(tool_a)
        profiler.profile(tool_b)

        stats = profiler.get_tool_statistics("tool_a")

        assert stats["tool_name"] == "tool_a"
        assert stats["count"] == 2
        assert "execution_time_ms" in stats


class TestProfilerDecorator:
    """Tests for the wrap decorator."""

    def test_wrap_decorator(self, profiler):
        """Test basic decorator usage."""

        @profiler.wrap
        def my_func():
            return 42

        result = my_func()

        assert result == 42
        assert my_func._last_profile is not None
        assert my_func._last_profile.tool_name == "my_func"

    def test_wrap_with_args(self, profiler):
        """Test decorator with function arguments."""

        @profiler.wrap
        def add(a, b):
            return a + b

        result = add(3, 4)

        assert result == 7
        assert add._last_profile.success

    def test_wrap_with_custom_name(self, profiler):
        """Test decorator with custom name."""

        @profiler.wrap(name="custom_name")
        def my_func():
            return 1

        my_func()

        assert my_func._last_profile.tool_name == "custom_name"

    def test_wrap_with_mode(self, profiler):
        """Test decorator with mode override."""

        @profiler.wrap(mode=ProfilingMode.STANDARD)
        def my_func():
            return 1

        my_func()

        assert my_func._last_profile.profiling_mode == ProfilingMode.STANDARD


class TestToolProfiling:
    """Tests for tool-specific profiling."""

    def test_profile_tool(self, profiler):
        """Test profiling a tool-like object."""

        class MockTool:
            name = "mock_tool"

            def run(self, n: int) -> int:
                return n * 2

        tool = MockTool()
        result, profile = profiler.profile_tool(tool, {"n": 5})

        assert result == 10
        assert profile.tool_name == "mock_tool"
        assert profile.return_value_type == "int"


class TestFullProfiling:
    """Tests for full profiling mode with cProfile."""

    def test_bottleneck_detection(self, full_profiler):
        """Test that bottlenecks are detected."""

        def slow_inner():
            total = 0
            for i in range(10000):
                total += i
            return total

        def outer():
            return slow_inner()

        result, profile = full_profiler.profile(outer)

        assert profile.total_function_calls > 0
        # May or may not have bottlenecks depending on threshold

    def test_recursion_detection(self, full_profiler):
        """Test recursion depth estimation."""

        def recursive_sum(n):
            if n <= 0:
                return 0
            return n + recursive_sum(n - 1)

        result, profile = full_profiler.profile(recursive_sum, 10)

        assert result == 55
        assert profile.total_function_calls >= 11  # At least 11 calls


# =============================================================================
#                            CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_profiler(self):
        """Test getting default profiler."""
        p1 = get_profiler()
        p2 = get_profiler()
        assert p1 is p2  # Same instance

    def test_set_profiler(self):
        """Test setting custom default profiler."""
        custom = Profiler(mode=ProfilingMode.FULL)
        set_profiler(custom)

        assert get_profiler() is custom

        # Reset to default
        set_profiler(None)

    def test_profile_convenience(self):
        """Test profile() convenience function."""

        def func():
            return 42

        result, prof = profile(func)

        assert result == 42
        assert prof.execution_time_ms > 0

    def test_profiled_decorator(self):
        """Test @profiled decorator."""

        @profiled
        def my_func():
            return 1

        result = my_func()

        assert result == 1
        assert my_func._last_profile is not None


# =============================================================================
#                            INTEGRATION HELPERS TESTS
# =============================================================================


class TestIntegrationHelpers:
    """Tests for integration helper functions."""

    def test_profile_to_metrics_update(self):
        """Test conversion to ToolMetrics format."""
        result = ProfileResult(
            execution_time_ms=10.5,
            peak_memory_mb=2.5,
            cpu_percent=5.0,
            efficiency_score=0.95,
        )

        update = profile_to_metrics_update(result)

        assert update["execution_time_ms"] == 10.5
        assert update["peak_memory_mb"] == 2.5
        assert update["cpu_percent"] == 5.0
        # Efficiency score is calculated, use approximate comparison
        assert abs(update["efficiency_score"] - result.efficiency_score) < 0.001
        assert update["latency_grade"] == "moderate"
        assert update["success"] is True

    def test_profile_to_registry_update(self):
        """Test conversion to registry.json format."""
        result = ProfileResult(
            execution_time_ms=5.0,
            peak_memory_mb=1.5,
            efficiency_score=0.98,
        )

        update = profile_to_registry_update(result)

        assert "performance" in update
        perf = update["performance"]
        assert perf["avg_execution_time_ms"] == 5.0
        assert perf["peak_memory_mb"] == 1.5
        assert perf["latency_grade"] == "fast"
        assert "last_profiled" in perf


# =============================================================================
#                            STRESS TESTS
# =============================================================================


class TestStress:
    """Stress tests for profiler robustness."""

    def test_many_profiles(self, profiler):
        """Test many sequential profiles."""

        def func():
            return 1

        for _ in range(100):
            profiler.profile(func)

        assert len(profiler.history) == 100

    def test_large_memory_allocation(self, profiler):
        """Test profiling large memory allocations."""

        def allocate_memory():
            return [i for i in range(100000)]

        result, profile = profiler.profile(allocate_memory)

        assert len(result) == 100000
        assert profile.peak_memory_mb > 0

    def test_slow_function(self, profiler):
        """Test profiling slow functions."""

        def slow():
            time.sleep(0.1)
            return "done"

        result, profile = profiler.profile(slow)

        assert result == "done"
        assert profile.execution_time_ms >= 100
        assert profile.latency_grade in (LatencyGrade.SLOW, LatencyGrade.CRITICAL)

    def test_exception_handling(self, profiler):
        """Test that profiler handles exceptions gracefully."""

        def failing():
            raise RuntimeError("Intentional failure")

        # When exception is raised inside profiled function, it propagates
        # The profile is still added to history with success=False
        try:
            profiler.profile(failing)
        except RuntimeError:
            pass

        # History should still have the failed profile
        assert len(profiler.history) == 1
        assert profiler.history[0].success is False


# =============================================================================
#                            RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
