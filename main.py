"""
IASCIS - Independent Autonomous Self-Correcting Intelligent System

Main entry point integrating:
- Sandbox: Persistent Docker container for tool execution
- Orchestrator: Domain-based decomposition + tool retrieval
- Executor: Plan execution with dependency ordering
- Toolsmith: Dynamic tool generation
- Gatekeeper: Safety validation
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from architecture.dispatcher import Dispatcher
from architecture.gatekeeper import Gatekeeper, ValidationResult
from architecture.orchestrator import Orchestrator
from architecture.executor import ExecutorAgent
from architecture.sandbox import Sandbox
from architecture.toolsmith import Toolsmith
from architecture.reflector import ExecutionResult, Reflector
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

WORKSPACE_PATH = Path(os.getcwd()) / "workspace"
TOOLS_DIR = WORKSPACE_PATH / "tools"


class IASCIS:
    """Main system orchestrating all components."""

    def __init__(
        self,
        workspace_path: Path = None,
        public_model: str = "groq/llama-3.3-70b-versatile",
        private_model: str = "groq/llama-3.3-70b-versatile",
        safe_mode: bool = True,
        enable_profiling: bool = True,
        profiling_mode: str = "lightweight",
    ):
        self.workspace_path = Path(workspace_path) if workspace_path else WORKSPACE_PATH
        self.tools_dir = self.workspace_path / "tools"
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        self.public_model = public_model
        self.private_model = private_model
        self._enable_profiling = enable_profiling
        self._profiling_mode = profiling_mode

        # Initialize components
        self.gatekeeper = Gatekeeper(strict_mode=safe_mode, stateful=True)
        self.toolsmith = Toolsmith(safe_mode=safe_mode, gatekeeper=self.gatekeeper)
        self.dispatcher = Dispatcher()
        self.reflector = Reflector(max_retries=3)

        # Sandbox and Executor initialized in context manager
        self.sandbox = None
        self.executor = None
        self.orchestrator = None

        logger.info(f"IASCIS initialized (workspace={self.workspace_path}, profiling={enable_profiling})")

    def _initialize_runtime(self):
        """Initialize Sandbox, Executor, and Orchestrator."""
        self.sandbox = Sandbox(tools_dir=self.tools_dir)
        self.sandbox.start()

        self.executor = ExecutorAgent(
            workspace_path=str(self.workspace_path),
            tools_dir=str(self.tools_dir),
            sandbox=self.sandbox,
            enable_profiling=self._enable_profiling,
            profiling_mode=self._profiling_mode,
        )

        self.orchestrator = Orchestrator(
            toolsmith=self.toolsmith,
            executor=self.executor,
            registry_path=str(self.tools_dir / "registry.json"),
        )

        logger.info("Runtime initialized with Sandbox")

    def _shutdown_runtime(self):
        """Stop Sandbox and cleanup."""
        if self.sandbox:
            self.sandbox.stop()
            self.sandbox = None
        self.executor = None
        self.orchestrator = None
        logger.info("Runtime shutdown complete")

    def __enter__(self):
        self._initialize_runtime()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shutdown_runtime()
        return False

    def run(self, task: str, file_context: list[str] = None) -> dict:
        """Execute a task through the orchestrator-executor pipeline."""
        if not self.orchestrator:
            raise RuntimeError(
                "IASCIS must be used as context manager: 'with IASCIS() as system:'"
            )

        file_context = file_context or []
        start_time = time.perf_counter()

        logger.info(f"Task received: {task[:100]}...")

        zone = self.dispatcher.route(task, file_context)
        model = self.private_model if zone == "private" else self.public_model
        logger.info(f"Routed to {zone} zone, model: {model}")

        try:
            result = self.orchestrator.run(task)
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            result = f"Error: {e}"

        duration_ms = (time.perf_counter() - start_time) * 1000

        return {
            "task": task,
            "zone": zone,
            "model": model,
            "result": result,
            "duration_ms": duration_ms,
        }

    def run_with_reflection(self, task: str, max_attempts: int = 3) -> dict:
        """Execute with self-correction loop."""
        result = None
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1}/{max_attempts}")

            result = self.run(task)

            if "Error" not in str(result.get("result", "")):
                return result

            exec_result = ExecutionResult(
                success=False,
                error=str(result.get("result")),
                exit_code=1,
            )

            reflection = self.reflector.reflect(exec_result)
            logger.warning(
                f"Attempt {attempt + 1} failed: {reflection.diagnosis.category.value}"
            )

            if not reflection.should_retry:
                logger.error("No retry recommended")
                break

            task = reflection.corrective_prompt

        return result

    def validate_code(self, code: str) -> ValidationResult:
        """Validate code through Gatekeeper."""
        return self.gatekeeper.validate(code)

    def create_tool(self, requirement: str) -> str:
        """Create a new tool via Toolsmith."""
        return self.toolsmith.create_tool(requirement)

    def reset(self):
        """Reset stateful components for new session."""
        self.gatekeeper.reset_state()
        self.reflector.reset()

    # ========== Profiling Methods ==========

    def get_profiler_statistics(self) -> dict:
        """Get aggregate profiling statistics from the executor."""
        if self.executor is None:
            return {"error": "Runtime not initialized"}
        return self.executor.get_profiler_statistics()

    def get_tool_performance_summary(self) -> dict:
        """Get performance summary per tool."""
        if self.executor is None:
            return {"error": "Runtime not initialized"}
        return self.executor.get_tool_performance_summary()

    def get_execution_profiles(self, tool_name: str = None) -> list:
        """Get execution profiles, optionally filtered by tool name."""
        if self.executor is None:
            return []
        return self.executor.get_execution_profiles(tool_name)


def main():
    """Main entry point."""
    task = input("Enter your task (or press Enter for demo): ").strip()
    if not task:
        task = "Calculate the square root of 144, then multiply by 2"

    logger.info("Starting IASCIS with Sandbox and Profiling")

    with IASCIS(enable_profiling=True) as system:
        result = system.run(task)

        logger.info(f"Completed in {result['duration_ms']:.2f}ms")
        logger.info(f"Zone: {result['zone']}")
        print(f"\n{'=' * 50}")
        print(f"Result:\n{result['result']}")
        
        # Display profiling information
        print(f"\n{'=' * 50}")
        print("PROFILING SUMMARY")
        print('=' * 50)
        
        perf_summary = system.get_tool_performance_summary()
        if perf_summary:
            for tool_name, stats in perf_summary.items():
                print(f"\n  {tool_name}:")
                print(f"    Calls: {stats['call_count']}")
                print(f"    Avg Time: {stats['avg_time_ms']:.2f}ms")
                print(f"    Max Time: {stats['max_time_ms']:.2f}ms")
                print(f"    Avg Memory: {stats['avg_memory_mb']:.4f}MB")
                print(f"    Memory Delta: {stats.get('avg_memory_delta_mb', 0):.4f}MB")
                print(f"    Efficiency: {stats.get('avg_efficiency', 0):.2%}")
                print(f"    Grade: {stats['last_grade']}")
        else:
            print("  No tools were executed.")
        
        # Overall stats
        profiler_stats = system.get_profiler_statistics()
        if profiler_stats.get("profiling_enabled"):
            print(f"\n  Total Profiles: {profiler_stats.get('count', 0)}")
            if profiler_stats.get('count', 0) > 0:
                print(f"  Success Rate: {profiler_stats.get('success_rate', 0):.0%}")
        
        # Export profiles to JSON
        if system.executor:
            export_path = system.executor.export_profiles()
            print(f"\n  Profiles exported to: {export_path}")
            
            # Update registry.json with calculated performance metrics
            updated_metrics = system.toolsmith.update_metrics_from_profiles()
            if updated_metrics:
                print(f"  Registry updated with metrics for {len(updated_metrics)} tools")


if __name__ == "__main__":
    main()
