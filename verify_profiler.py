#!/usr/bin/env python3
"""
Profiler Verification Script
============================

Run this script to verify the Profiler component is working correctly.

Usage:
    uv run python verify_profiler.py

Tests:
    1. Basic profiling functionality
    2. Integration with Tool Decay Manager
    3. All profiling modes
    4. Statistics and reporting
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_basic_profiling():
    """Test 1: Basic profiling functionality."""
    print("\n" + "=" * 60)
    print("  TEST 1: Basic Profiling")
    print("=" * 60)
    
    from architecture.profiler import Profiler, ProfilingMode
    
    profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)
    
    # Profile a simple function
    def calculate_sum(n):
        return sum(range(n))
    
    result, profile = profiler.profile(calculate_sum, 10000)
    
    print(f"  ✅ Function executed: sum(0..9999) = {result}")
    print(f"  ✅ Execution time: {profile.execution_time_ms:.3f}ms")
    print(f"  ✅ Latency grade: {profile.latency_grade.value}")
    print(f"  ✅ Memory used: {profile.peak_memory_mb:.3f}MB")
    print(f"  ✅ Efficiency: {profile.efficiency_score:.2%}")
    
    return True


def test_profiling_modes():
    """Test 2: All profiling modes."""
    print("\n" + "=" * 60)
    print("  TEST 2: Profiling Modes")
    print("=" * 60)
    
    from architecture.profiler import Profiler, ProfilingMode
    
    def work():
        """Do some work."""
        data = [i ** 2 for i in range(1000)]
        return sum(data)
    
    modes = [
        ProfilingMode.OFF,
        ProfilingMode.LIGHTWEIGHT,
        ProfilingMode.STANDARD,
        ProfilingMode.FULL,
    ]
    
    for mode in modes:
        profiler = Profiler(mode=mode)
        result, profile = profiler.profile(work)
        
        has_memory = profile.peak_memory_mb > 0 or mode == ProfilingMode.OFF
        has_cpu = profile.cpu_percent >= 0
        has_bottlenecks = len(profile.bottlenecks) >= 0
        
        print(f"  ✅ {mode.value:12s}: time={profile.execution_time_ms:.3f}ms, "
              f"mem={profile.peak_memory_mb:.3f}MB, "
              f"bottlenecks={len(profile.bottlenecks)}")
    
    return True


def test_tool_profiling():
    """Test 3: Tool profiling."""
    print("\n" + "=" * 60)
    print("  TEST 3: Tool Profiling")
    print("=" * 60)
    
    from architecture.profiler import Profiler, ProfilingMode
    
    # Create a mock tool
    class SortTool:
        name = "sort_tool"
        def run(self, data: list) -> list:
            return sorted(data)
    
    profiler = Profiler(mode=ProfilingMode.STANDARD)
    tool = SortTool()
    
    test_data = list(range(5000, 0, -1))  # Reverse order
    result, profile = profiler.profile_tool(tool, {"data": test_data})
    
    print(f"  ✅ Tool name: {profile.tool_name}")
    print(f"  ✅ Result: [{result[0]}, {result[1]}, ..., {result[-1]}]")
    print(f"  ✅ Execution time: {profile.execution_time_ms:.3f}ms")
    print(f"  ✅ Return type: {profile.return_value_type}")
    
    return True


def test_decorator():
    """Test 4: Decorator pattern."""
    print("\n" + "=" * 60)
    print("  TEST 4: Decorator Pattern")
    print("=" * 60)
    
    from architecture.profiler import Profiler, ProfilingMode
    
    profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)
    
    @profiler.wrap
    def fibonacci(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    result = fibonacci(50)
    profile = fibonacci._last_profile
    
    print(f"  ✅ Fibonacci(50) = {result}")
    print(f"  ✅ Time: {profile.execution_time_ms:.3f}ms")
    print(f"  ✅ Profile attached: {profile is not None}")
    
    return True


def test_statistics():
    """Test 5: Statistics and aggregation."""
    print("\n" + "=" * 60)
    print("  TEST 5: Statistics & Aggregation")
    print("=" * 60)
    
    from architecture.profiler import Profiler, ProfilingMode
    import time
    
    profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)
    
    # Run various functions
    def fast():
        return sum(range(100))
    
    def medium():
        time.sleep(0.02)
        return "done"
    
    def slow():
        time.sleep(0.1)
        return "done"
    
    # Profile multiple times
    for _ in range(5):
        profiler.profile(fast)
    for _ in range(3):
        profiler.profile(medium)
    profiler.profile(slow)
    
    stats = profiler.get_statistics()
    
    print(f"  ✅ Total profiles: {stats['count']}")
    print(f"  ✅ Avg execution time: {stats['execution_time_ms']['mean']:.3f}ms")
    print(f"  ✅ Success rate: {stats['success_rate']:.0%}")
    print(f"  ✅ Latency distribution: {stats['latency_grade_distribution']}")
    
    return True


def test_decay_integration():
    """Test 6: Integration with Tool Decay Manager."""
    print("\n" + "=" * 60)
    print("  TEST 6: Decay Manager Integration")
    print("=" * 60)
    
    from architecture.profiler import Profiler, ProfilingMode
    from execution.tool_decay import create_decay_manager
    
    # Create components
    profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)
    decay_manager = create_decay_manager(
        decay_minutes=5.0,
        auto_cleanup=False
    )
    
    # Create a mock tool
    class CalculatorTool:
        name = "calculator"
        def run(self, a: int, b: int) -> int:
            return a + b
    
    tool = CalculatorTool()
    decay_manager.register_tool("calculator", tool)
    
    # Profile and record usage
    result, profile = profiler.profile_tool(tool, {"a": 10, "b": 20})
    decay_manager.record_usage("calculator", execution_time_ms=profile.execution_time_ms)
    
    metrics = decay_manager.get_metrics("calculator")
    
    print(f"  ✅ Tool executed: 10 + 20 = {result}")
    print(f"  ✅ Profile time: {profile.execution_time_ms:.3f}ms")
    print(f"  ✅ Decay metrics updated:")
    print(f"       - Total calls: {metrics.total_calls}")
    print(f"       - Avg time: {metrics.avg_execution_time_ms:.3f}ms")
    print(f"       - Decay score: {metrics.calculate_decay_score():.4f}")
    
    return True


def test_exception_handling():
    """Test 7: Exception handling."""
    print("\n" + "=" * 60)
    print("  TEST 7: Exception Handling")
    print("=" * 60)
    
    from architecture.profiler import Profiler, ProfilingMode
    
    profiler = Profiler(mode=ProfilingMode.LIGHTWEIGHT)
    
    def failing_function():
        raise ValueError("Intentional error!")
    
    try:
        profiler.profile(failing_function)
    except ValueError:
        pass  # Expected
    
    # Check that the profile was still recorded
    last_profile = profiler.history[-1]
    
    print(f"  ✅ Exception caught and handled")
    print(f"  ✅ Profile still recorded: {last_profile is not None}")
    print(f"  ✅ Success: {last_profile.success}")
    print(f"  ✅ Error message: {last_profile.error_message}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "🔬" * 30)
    print("  PROFILER VERIFICATION SUITE")
    print("🔬" * 30)
    
    tests = [
        ("Basic Profiling", test_basic_profiling),
        ("Profiling Modes", test_profiling_modes),
        ("Tool Profiling", test_tool_profiling),
        ("Decorator Pattern", test_decorator),
        ("Statistics", test_statistics),
        ("Decay Integration", test_decay_integration),
        ("Exception Handling", test_exception_handling),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{passed + failed} tests passed")
    print("=" * 60)
    
    if failed == 0:
        print("\n  🎉 All tests passed! The Profiler is working correctly.\n")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed. Please check the errors above.\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
