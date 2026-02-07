"""Executor Agent - Runs tools via persistent Docker container with profiling."""

import ast
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from architecture.llm_manager import get_llm_manager
from architecture.pipeline_state import DataTransformer, StepResult
from architecture.prompts import (
    get_arg_extraction_prompt,
    get_chart_args_prompt,
    get_pipeline_aware_arg_prompt,
    get_response_synthesis_prompt,
)
from architecture.reflector import ExecutionResult, Reflector
from architecture.sandbox import Sandbox
from architecture.schemas import ExecutionPlan, ExecutionStep
from utils.logger import get_logger

try:
    from architecture.profiler import (
        Profiler,
        ProfilingMode,
    )

    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    Profiler = None
    ProfilingMode = None

logger = get_logger(__name__)


class ExecutorAgent:
    """Executes ExecutionPlan by running tools in a persistent Docker container with profiling."""

    ARG_ERROR_PATTERNS = [
        r"missing.*argument",
        r"KeyError",
        r"takes.*positional",
        r"got.*unexpected.*argument",
        r"required field",
        r"validation error",
    ]

    def __init__(
        self,
        workspace_path: str,
        tools_dir: str,
        sandbox: Sandbox,
        toolsmith=None,
        enable_profiling: bool = True,
        profiling_mode: str = "standard",
        max_retries: int = 2,
    ):
        self.workspace = Path(workspace_path)
        self.tools_dir = Path(tools_dir)
        self.registry_path = self.tools_dir / "registry.json"
        self.sandbox = sandbox
        self.toolsmith = toolsmith
        self.reflector = Reflector(max_retries=max_retries)
        self.max_retries = max_retries
        self._definitions = {}
        self._execution_profiles: Dict[str, list] = {}

        self._data_transformer = DataTransformer()

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
        synthesized = self._synthesize_response(plan.original_query, final_step.result)

        return json.dumps(
            {
                "query": plan.original_query,
                "success": plan.is_complete(),
                "result": synthesized,
                "raw_data": final_step.result,
                "step_results": results,
            },
            indent=2,
        )

    def _synthesize_response(self, query: str, raw_result: str) -> str:
        """Use LLM to convert raw tool output into a human-readable response."""
        try:
            try:
                data = (
                    json.loads(raw_result)
                    if isinstance(raw_result, str)
                    else raw_result
                )
            except json.JSONDecodeError:
                data = raw_result

            if isinstance(data, dict) and "error" in data:
                return f"Error: {data['error']}"

            data_str = (
                json.dumps(data, indent=2) if isinstance(data, dict) else str(data)
            )
            prompt = get_response_synthesis_prompt(query, data_str)

            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            if response.get("error"):
                logger.warning(f"Synthesis failed: {response['error']}")
                return raw_result

            return response.get("content", raw_result)

        except Exception as e:
            logger.warning(f"Response synthesis error: {e}")
            return raw_result

    def _execute_step(self, step: ExecutionStep, previous_results: dict) -> str:
        """Execute a single step using the Sandbox with profiling."""
        step_type = getattr(step, "step_type", "tool")

        if step_type == "transform":
            return self._execute_transform_step(step, previous_results)

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

        if step.input_from and step.input_from in previous_results:
            input_data = previous_results[step.input_from]
            logger.info(
                f"Step {step.step_number}: Using input from step {step.input_from}"
            )
        elif step.depends_on:
            input_data = previous_results.get(step.depends_on[-1], "")
            logger.info(
                f"Step {step.step_number}: Using output from dependency step {step.depends_on[-1]}"
            )
        else:
            input_data = step.description
            logger.info(f"Step {step.step_number}: Using description as input")

        args = self._infer_args(step, input_data, definition)
        logger.debug(f"Executor inferred args = {args}")
        logger.debug(f"Args type = {type(args)}")

        start_time = time.time()
        result = self.sandbox.execute_with_args(step.tool_name, tool_file, args)
        host_latency = time.time() - start_time

        sandbox_profile = result.get("profile")

        if sandbox_profile:
            if step.tool_name not in self._execution_profiles:
                self._execution_profiles[step.tool_name] = []
            self._execution_profiles[step.tool_name].append(sandbox_profile)

            logger.info(
                f"Step {step.step_number} SANDBOX Profile: "
                f"time={sandbox_profile['execution_time_ms']:.3f}ms, "
                f"peak_mem={sandbox_profile['peak_memory_mb']:.4f}MB, "
                f"delta_mem={sandbox_profile.get('memory_delta_mb', 0):.4f}MB, "
                f"efficiency={sandbox_profile.get('efficiency_score', 0):.2%}, "
                f"grade={sandbox_profile['latency_grade']}"
            )
            logger.info(f"   (Total round-trip: {host_latency * 1000:.2f}ms)")

        if result["success"]:
            step.result = result["output"]
            step.status = "completed"
            logger.info(f"Step {step.step_number}: Completed in {host_latency:.2f}s")
            return step.result

        error_msg = result.get("error", "Unknown error")
        logger.error(f"Step {step.step_number}: Failed - {error_msg}")

        retry_result = self._handle_failure(
            step, error_msg, input_data, definition, tool_file, previous_results
        )
        if retry_result is not None:
            return retry_result

        step.status = "failed"
        step.result = f"Error: {error_msg}"
        return step.result

    def _execute_transform_step(
        self, step: ExecutionStep, previous_results: dict
    ) -> str:
        """Execute a transform step using LLM to process data."""
        logger.info(f"Step {step.step_number}: TRANSFORM - {step.description}")
        step.status = "running"

        input_data = ""
        if step.input_from and step.input_from in previous_results:
            input_data = previous_results[step.input_from]
            logger.info(
                f"Step {step.step_number}: Using input from step {step.input_from}"
            )

        if not input_data:
            step.status = "failed"
            step.result = "Error: Transform step has no input data"
            return step.result

        prompt = f"""Process this data according to the instruction.

INSTRUCTION: {step.description}

INPUT DATA:
{input_data[:8000]}

OUTPUT: Return ONLY the transformed data as valid JSON. No explanations."""

        response = self.llm.generate_json(
            messages=[{"role": "user", "content": prompt}], max_tokens=2048
        )

        if response.get("error"):
            step.status = "failed"
            step.result = f"Transform failed: {response['error']}"
            return step.result

        result = response.get("content", "{}")
        step.status = "completed"
        step.result = result
        logger.info(f"Step {step.step_number}: Transform completed")
        return result

    def _classify_failure(self, error: str) -> str:
        """Classify failure as arg_extraction or tool_code error."""
        for pattern in self.ARG_ERROR_PATTERNS:
            if re.search(pattern, error, re.IGNORECASE):
                return "arg_extraction"
        return "tool_code"

    def _handle_failure(
        self,
        step: ExecutionStep,
        error: str,
        input_data: str,
        definition: dict,
        tool_file: str,
        previous_results: dict,
    ) -> str | None:
        """Route failure to appropriate retry strategy."""
        exec_result = ExecutionResult(success=False, error=error, exit_code=1)
        reflection = self.reflector.reflect(exec_result)

        if not reflection.should_retry:
            logger.warning(f"Step {step.step_number}: No retry (max retries reached)")
            return None

        failure_type = self._classify_failure(error)
        logger.info(
            f"Step {step.step_number}: Failure type={failure_type}, retrying..."
        )

        if failure_type == "arg_extraction":
            return self._retry_with_new_args(
                step,
                error,
                input_data,
                definition,
                tool_file,
                reflection.corrective_prompt,
            )
        else:
            return self._retry_with_regenerated_tool(
                step, error, input_data, previous_results
            )

    def _retry_with_new_args(
        self,
        step: ExecutionStep,
        error: str,
        input_data: str,
        definition: dict,
        tool_file: str,
        corrective_prompt: str,
    ) -> str | None:
        """Re-extract arguments using error context and retry execution."""
        logger.info(f"Step {step.step_number}: Re-extracting args with error context")

        enhanced_input = f"{input_data}\n\nPrevious error: {error}\n{corrective_prompt}"
        new_args = self._infer_args(step, enhanced_input, definition)
        logger.info(f"Step {step.step_number}: New args = {new_args}")

        result = self.sandbox.execute_with_args(step.tool_name, tool_file, new_args)

        if result["success"]:
            step.result = result["output"]
            step.status = "completed"
            self.reflector.record_outcome(
                error, self.reflector.classifier.classify(error), True
            )
            logger.info(f"Step {step.step_number}: Retry succeeded with new args")
            return step.result

        logger.warning(f"Step {step.step_number}: Retry with new args failed")
        return None

    def _retry_with_regenerated_tool(
        self,
        step: ExecutionStep,
        error: str,
        input_data: str,
        previous_results: dict,
    ) -> str | None:
        """Regenerate tool with error context and retry execution."""
        if not self.toolsmith:
            logger.warning("No toolsmith available for tool regeneration")
            return None

        logger.info(f"Step {step.step_number}: Regenerating tool with error context")

        input_sample = str(input_data)[:500] if input_data else "No input"
        regeneration_prompt = (
            f"Task: {step.description}\n\n"
            f"Input data sample:\n{input_sample}\n\n"
            f"Previous implementation failed with error:\n{error}\n\n"
            f"Create a tool that correctly handles this input format and fixes the error."
        )

        result_msg = self.toolsmith.create_tool(regeneration_prompt)
        logger.info(f"Step {step.step_number}: Toolsmith result = {result_msg}")

        if "Error" in result_msg or "failed" in result_msg.lower():
            logger.warning(f"Step {step.step_number}: Tool regeneration failed")
            return None

        new_tool_name = self._extract_tool_name(result_msg)
        if not new_tool_name:
            logger.warning(
                f"Step {step.step_number}: Could not extract tool name from result"
            )
            return None

        self.load_registry()

        if new_tool_name not in self._definitions:
            logger.warning(
                f"Step {step.step_number}: New tool {new_tool_name} not in registry"
            )
            return None

        logger.info(f"Step {step.step_number}: Using regenerated tool {new_tool_name}")
        step.tool_name = new_tool_name

        definition = self._definitions[new_tool_name]
        tool_file = definition["file"]
        new_args = self._infer_args(step, input_data, definition)

        result = self.sandbox.execute_with_args(new_tool_name, tool_file, new_args)

        if result["success"]:
            step.result = result["output"]
            step.status = "completed"
            logger.info(
                f"Step {step.step_number}: Retry succeeded with {new_tool_name}"
            )
            return step.result

        logger.warning(
            f"Step {step.step_number}: Retry with {new_tool_name} failed - {result.get('error')}"
        )
        return None

    def _extract_tool_name(self, result_msg: str) -> str | None:
        """Extract tool name from Toolsmith result message."""
        match = re.search(r"Created (\w+)", result_msg)
        if match:
            return match.group(1)
        return None

    def _execute_with_profiling(
        self, tool_name: str, tool_file: str, args: dict
    ) -> tuple:
        """Execute a tool with profiling wrapper."""

        def sandbox_execution():
            return self.sandbox.execute_with_args(tool_name, tool_file, args)

        result, profile = self._profiler.profile(
            sandbox_execution, _profile_name=tool_name
        )
        return result, profile

    def get_execution_profiles(self, tool_name: str = None) -> list:
        """Get execution profiles for tools, optionally filtered by name."""
        if tool_name:
            return self._execution_profiles.get(tool_name, [])

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

            times = [p["execution_time_ms"] for p in profiles if p]
            memories = [p["peak_memory_mb"] for p in profiles if p]
            memory_deltas = [p.get("memory_delta_mb", 0) for p in profiles if p]
            efficiencies = [p.get("efficiency_score", 0) for p in profiles if p]

            summary[tool_name] = {
                "call_count": len(profiles),
                "avg_time_ms": sum(times) / len(times) if times else 0,
                "max_time_ms": max(times) if times else 0,
                "min_time_ms": min(times) if times else 0,
                "avg_memory_mb": sum(memories) / len(memories) if memories else 0,
                "max_memory_mb": max(memories) if memories else 0,
                "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas)
                if memory_deltas
                else 0,
                "avg_efficiency": sum(efficiencies) / len(efficiencies)
                if efficiencies
                else 0,
                "last_grade": profiles[-1].get("latency_grade", "unknown")
                if profiles
                else "unknown",
            }

        return summary

    def export_profiles(self, filepath: str = None) -> str:
        """Export all profiles to a JSON file."""
        if filepath is None:
            logs_dir = self.workspace.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(logs_dir / f"profiles_{timestamp}.json")

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_tool_performance_summary(),
            "statistics": self.get_profiler_statistics(),
            "raw_profiles": self._execution_profiles,
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Profiles exported to: {filepath}")
        return filepath

    def _infer_args(
        self, step: ExecutionStep, input_data: str, definition: dict
    ) -> dict:
        """Use DataTransformer + LLM to extract argument values with pipeline context."""
        tool_file = self.tools_dir / definition.get("file", "")
        arg_names = self._get_arg_names_from_file(tool_file)
        arg_types = self._get_arg_types_from_file(tool_file)

        if not arg_names:
            return {"input": input_data}

        is_pipeline_step = input_data and input_data != step.description
        source_schema = {}

        if is_pipeline_step:
            try:
                parsed = (
                    json.loads(input_data)
                    if isinstance(input_data, str)
                    else input_data
                )
                source_schema = StepResult._infer_schema(parsed)

                transformed_args = {}
                for arg_name in arg_names:
                    transformed = self._data_transformer.transform(
                        parsed, source_schema, arg_name, arg_types.get(arg_name)
                    )
                    if transformed is not None:
                        transformed_args[arg_name] = transformed
                        logger.info(
                            f"DataTransformer: {arg_name} <- extracted from pipeline"
                        )

                if transformed_args:
                    remaining_args = [a for a in arg_names if a not in transformed_args]
                    if remaining_args:
                        prompt = get_chart_args_prompt(remaining_args, step.description)
                        response = self.llm.generate_json(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=256,
                        )
                        if not response.get("error"):
                            try:
                                other_values = json.loads(response["content"])
                                for k, v in other_values.items():
                                    if k in remaining_args:
                                        transformed_args[k] = v
                            except Exception:
                                pass

                    logger.info(
                        f"Pipeline args via DataTransformer: {list(transformed_args.keys())}"
                    )
                    return transformed_args

                if (
                    "data" in arg_names
                    and isinstance(parsed, dict)
                    and any(
                        k in parsed
                        for k in ["repos", "results", "items", "data", "labels"]
                    )
                ):
                    logger.info(
                        "Direct pass-through: Using structured JSON for 'data' arg"
                    )
                    return self._infer_args_with_direct_data(
                        step, parsed, arg_names, definition
                    )

            except (json.JSONDecodeError, TypeError):
                pass

        available_urls = ""
        url_args = ["url", "target_url", "source_url", "webpage"]
        if any(arg in url_args for arg in arg_names):
            try:
                from architecture.api_registry import search_urls

                urls = search_urls(step.description, limit=3)
                if urls:
                    available_urls = "\n".join(
                        [f"- {u['name']}: {u['url']}" for u in urls]
                    )
            except Exception as e:
                logger.warning(f"Could not search for URLs: {e}")

        if is_pipeline_step and source_schema:
            prompt = get_pipeline_aware_arg_prompt(
                arg_names, arg_types, step.description, input_data, source_schema
            )
        else:
            prompt = get_arg_extraction_prompt(
                arg_names, step.description, input_data, available_urls, arg_types
            )

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

    def _infer_args_with_direct_data(
        self, step: ExecutionStep, parsed_data: dict, arg_names: list, definition: dict
    ) -> dict:
        """Build args with direct data pass-through, using LLM only for other args."""
        args = {"data": parsed_data}

        other_args = [a for a in arg_names if a != "data"]
        if other_args:
            prompt = get_chart_args_prompt(other_args, step.description)

            response = self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}], max_tokens=256
            )

            if not response.get("error"):
                try:
                    other_values = json.loads(response["content"])
                    for k, v in other_values.items():
                        if k in other_args:
                            args[k] = v
                except Exception:
                    pass

        logger.info(f"Direct data pass-through args: {list(args.keys())}")
        return args

    def _get_arg_names_from_file(self, tool_file: Path) -> list:
        """Extract argument names from the tool's Args class using AST."""
        try:
            with open(tool_file, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

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

    def _get_arg_types_from_file(self, tool_file: Path) -> dict:
        """Extract argument types and constraints from the tool's Args class using AST."""
        try:
            with open(tool_file, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and ("Args" in node.name):
                    arg_types = {}
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            arg_name = item.target.id
                            if isinstance(item.annotation, ast.Name):
                                arg_types[arg_name] = item.annotation.id
                            elif isinstance(item.annotation, ast.Subscript):
                                if isinstance(item.annotation.value, ast.Name):
                                    base_type = item.annotation.value.id
                                    if base_type == "Literal":
                                        literal_values = self._extract_literal_values(
                                            item.annotation
                                        )
                                        arg_types[arg_name] = (
                                            f"Literal[{', '.join(literal_values)}]"
                                        )
                                    else:
                                        arg_types[arg_name] = base_type
                            else:
                                arg_types[arg_name] = "Any"
                    if arg_types:
                        return arg_types

            return {}
        except Exception as e:
            logger.error(f"Failed to parse arg types from {tool_file}: {e}")
            return {}

    def _extract_literal_values(self, subscript_node) -> list:
        """Extract values from a Literal type annotation."""
        values = []
        try:
            if isinstance(subscript_node.slice, ast.Tuple):
                for elt in subscript_node.slice.elts:
                    if isinstance(elt, ast.Constant):
                        values.append(repr(elt.value))
            elif isinstance(subscript_node.slice, ast.Constant):
                values.append(repr(subscript_node.slice.value))
        except Exception:
            pass
        return values
