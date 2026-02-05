"""Executor Agent - Runs tools via persistent Docker container with profiling."""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

from architecture.schemas import ExecutionPlan, ExecutionStep
from architecture.sandbox import Sandbox
from architecture.llm_manager import get_llm_manager
from utils.logger import get_logger

# Import profiler (optional)
try:
    from architecture.profiler import (
        Profiler,
        ProfilingMode,
        ProfileResult,
        profile_to_registry_update,
    )
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    Profiler = None
    ProfilingMode = None

logger = get_logger(__name__)


class ExecutorAgent:
    """Executes ExecutionPlan by running tools in a persistent Docker container with profiling."""

    def __init__(
        self, 
        workspace_path: str, 
        tools_dir: str, 
        sandbox: Sandbox,
        enable_profiling: bool = True,
        profiling_mode: str = "standard",
    ):
        self.workspace = Path(workspace_path)
        self.tools_dir = Path(tools_dir)
        self.registry_path = self.tools_dir / "registry.json"
        self.sandbox = sandbox
        self._definitions = {}
        self._execution_profiles: Dict[str, list] = {}  # Store profiles per tool
        
        # Initialize profiler
        self._enable_profiling = enable_profiling and HAS_PROFILER
        if self._enable_profiling:
            mode_map = {
                "off": ProfilingMode.OFF,
                "lightweight": ProfilingMode.LIGHTWEIGHT,
                "standard": ProfilingMode.STANDARD,
                "full": ProfilingMode.FULL,
            }
            self._profiler = Profiler(
                mode=mode_map.get(profiling_mode, ProfilingMode.STANDARD),
                history_size=500,
            )
            logger.info(f"ExecutorAgent profiling enabled (mode={profiling_mode})")
        else:
            self._profiler = None
            if enable_profiling and not HAS_PROFILER:
                logger.warning("Profiling requested but profiler module not available")
        
        self.load_registry()
        self.llm = get_llm_manager()
        logger.info("ExecutorAgent initialized with Sandbox")

    def load_registry(self) -> None:
        """Load tool definitions from registry.json."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self._definitions = json.load(f)
            logger.info(f"Loaded {len(self._definitions)} tool definitions")
        else:
            logger.warning(f"Registry not found: {self.registry_path}")

    def execute(self, plan: ExecutionPlan) -> str:
        """Execute the ExecutionPlan respecting dependencies."""
        logger.info(f"Executing plan: {plan.original_query}")
        logger.info(f"Total steps: {len(plan.steps)}")

        results = {}

        while not plan.is_complete() and not plan.has_failed():
            ready = plan.get_ready_steps()

            if not ready:
                logger.error("Deadlock: No ready steps but plan not complete")
                break

            for step in ready:
                result = self._execute_step(step, results)
                results[step.step_number] = result

        final_step = max(plan.steps, key=lambda s: s.step_number)

        return json.dumps(
            {
                "query": plan.original_query,
                "success": plan.is_complete(),
                "result": final_step.result,
                "step_results": results,
            },
            indent=2,
        )

    def _execute_step(self, step: ExecutionStep, previous_results: dict) -> str:
        """Execute a single step using the Sandbox with profiling."""
        logger.info(
            f"Step {step.step_number}: {step.description} using {step.tool_name}"
        )
        step.status = "running"

        if not step.tool_name:
            step.status = "failed"
            step.result = "Error: No tool assigned"
            return step.result

        self.load_registry()

        if step.tool_name not in self._definitions:
            step.status = "failed"
            step.result = f"Error: Tool {step.tool_name} not in registry"
            logger.error(f"Step {step.step_number}: Tool not in registry")
            return step.result

        definition = self._definitions[step.tool_name]
        tool_file = definition["file"]
        tool_path = self.tools_dir / tool_file

        if not tool_path.exists():
            step.status = "failed"
            step.result = f"Error: Tool file not found: {tool_file}"
            logger.error(f"Step {step.step_number}: Tool file missing")
            return step.result

        if step.depends_on:
            input_data = previous_results.get(step.depends_on[-1], "")
        else:
            input_data = step.description

        args = self._infer_args(step, input_data, definition)
        logger.info(f"DEBUG: Executor inferred args = {args}")
        logger.info(f"DEBUG: Args type = {type(args)}")

        # Execute tool in sandbox (profiling happens inside container)
        start_time = time.time()
        result = self.sandbox.execute_with_args(step.tool_name, tool_file, args)
        host_latency = time.time() - start_time

        # Extract profile from sandbox result (profiling done inside container)
        sandbox_profile = result.get("profile")
        
        if sandbox_profile:
            # Store the profile data
            if step.tool_name not in self._execution_profiles:
                self._execution_profiles[step.tool_name] = []
            self._execution_profiles[step.tool_name].append(sandbox_profile)
            
            # Log profile from INSIDE the container
            logger.info(
                f"Step {step.step_number} SANDBOX Profile: "
                f"time={sandbox_profile['execution_time_ms']:.3f}ms, "
                f"memory={sandbox_profile['peak_memory_mb']:.4f}MB, "
                f"grade={sandbox_profile['latency_grade']}"
            )
            
            # Also log total round-trip time for comparison
            logger.info(
                f"   (Total round-trip: {host_latency * 1000:.2f}ms)"
            )

        if result["success"]:
            step.result = result["output"]
            step.status = "completed"
            logger.info(f"Step {step.step_number}: Completed in {host_latency:.2f}s")
        else:
            step.status = "failed"
            step.result = f"Error: {result['error']}"
            logger.error(
                f"Step {step.step_number}: Failed after {host_latency:.2f}s - {result['error']}"
            )

        return step.result

    def _execute_with_profiling(
        self, tool_name: str, tool_file: str, args: dict
    ) -> tuple:
        """Execute a tool with profiling wrapper."""
        # Create a wrapper function for the sandbox execution
        def sandbox_execution():
            return self.sandbox.execute_with_args(tool_name, tool_file, args)
        
        # Profile the execution
        result, profile = self._profiler.profile(
            sandbox_execution, 
            _profile_name=tool_name
        )
        
        return result, profile

    def get_execution_profiles(self, tool_name: str = None) -> list:
        """
        Get execution profiles for tools.
        
        Args:
            tool_name: Optional tool name to filter by. If None, returns all.
        
        Returns:
            List of profile dicts from sandbox execution
        """
        if tool_name:
            return self._execution_profiles.get(tool_name, [])
        
        # Return all profiles flattened
        all_profiles = []
        for profiles in self._execution_profiles.values():
            all_profiles.extend(profiles)
        return all_profiles

    def get_profiler_statistics(self) -> dict:
        """Get aggregate profiling statistics from sandbox profiles."""
        all_profiles = self.get_execution_profiles()
        
        if not all_profiles:
            return {
                "profiling_enabled": True,
                "source": "sandbox",
                "count": 0,
                "tools_profiled": list(self._execution_profiles.keys()),
            }
        
        times = [p["execution_time_ms"] for p in all_profiles if p]
        memories = [p["peak_memory_mb"] for p in all_profiles if p]
        successes = [p.get("success", True) for p in all_profiles if p]
        
        return {
            "profiling_enabled": True,
            "source": "sandbox",
            "count": len(all_profiles),
            "tools_profiled": list(self._execution_profiles.keys()),
            "execution_time_ms": {
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "mean": sum(times) / len(times) if times else 0,
            },
            "peak_memory_mb": {
                "min": min(memories) if memories else 0,
                "max": max(memories) if memories else 0,
                "mean": sum(memories) / len(memories) if memories else 0,
            },
            "success_rate": sum(successes) / len(successes) if successes else 0,
        }

    def get_tool_performance_summary(self) -> dict:
        """Get a summary of performance metrics per tool from sandbox profiles."""
        summary = {}
        
        for tool_name, profiles in self._execution_profiles.items():
            if not profiles:
                continue
            
            # Profiles are now dicts from sandbox, not ProfileResult objects
            times = [p["execution_time_ms"] for p in profiles if p]
            memories = [p["peak_memory_mb"] for p in profiles if p]
            
            summary[tool_name] = {
                "call_count": len(profiles),
                "avg_time_ms": sum(times) / len(times) if times else 0,
                "max_time_ms": max(times) if times else 0,
                "min_time_ms": min(times) if times else 0,
                "avg_memory_mb": sum(memories) / len(memories) if memories else 0,
                "max_memory_mb": max(memories) if memories else 0,
                "last_grade": profiles[-1].get("latency_grade", "unknown") if profiles else "unknown",
            }
        
        return summary

    def _infer_args(
        self, step: ExecutionStep, input_data: str, definition: dict
    ) -> dict:
        """Use LLM to extract argument values from task description."""
        tool_file = self.tools_dir / definition.get("file", "")
        arg_names = self._get_arg_names_from_file(tool_file)

        if not arg_names:
            return {"input": input_data}

        # Always use LLM to extract actual values (not raw description)
        prompt = f"""Extract the actual values for these arguments from the task.

Arguments needed: {arg_names}
Task description: {step.description}
Previous step result (if any): {str(input_data)[:500] if input_data else "None"}

IMPORTANT: Return the ACTUAL VALUES, not the description text.
- For numeric arguments, return the number (e.g., 144, not "calculate 144")
- For text arguments, return the actual text value

Return ONLY a valid JSON object.
Example for ['number']: {{"number": 144}}
Example for ['text', 'count']: {{"text": "hello", "count": 5}}
"""

        response = self.llm.generate_json(
            messages=[{"role": "user", "content": prompt}], max_tokens=256
        )

        if response.get("error"):
            logger.error(f"Arg inference failed: {response['error']}")
            return {arg_names[0]: input_data}

        try:
            args = json.loads(response["content"])
            logger.info(f"LLM extracted args: {args}")
            return {k: v for k, v in args.items() if k in arg_names}
        except Exception as e:
            logger.error(f"Failed to parse LLM args response: {e}")
            return {arg_names[0]: input_data}

    def _get_arg_names_from_file(self, tool_file: Path) -> list:
        """Extract argument names from the tool's Args class using AST."""
        import ast

        try:
            with open(tool_file, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Find the Args class (ends with "Args" or "ToolArgs")
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and ("Args" in node.name):
                    args = []
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            args.append(item.target.id)
                    if args:
                        return args

            # Fallback: check run method signature
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "run":
                    args = []
                    for arg in node.args.args:
                        if arg.arg != "self":
                            args.append(arg.arg)
                    return args

            return []
        except Exception as e:
            logger.error(f"Failed to parse args from {tool_file}: {e}")
            return []
