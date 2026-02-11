"""Executor Agent - Runs tools via persistent Docker container with profiling."""

import re
import time
import json
from pathlib import Path
from typing import Dict, Optional

from architecture.schemas import ExecutionPlan, ExecutionStep
from architecture.sandbox import Sandbox
from architecture.llm_manager import get_llm_manager
from architecture.reflector import Reflector, ExecutionResult
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
            with open(self.registry_path, encoding="utf-8") as f:
                self._definitions = json.load(f)
            logger.info(f"Loaded {len(self._definitions)} tool definitions")
        else:
            logger.warning(f"Registry not found: {self.registry_path}")

    def _translate_args(self, args: dict) -> dict:
        """Translate file paths in arguments to container paths."""
        translated = args.copy()
        for key, value in translated.items():
            if isinstance(value, str):
                # Only try to translate if it looks like a path or filename
                # (contains separators or extension)
                if "/" in value or "\\" in value or "." in value:
                    translated[key] = self.sandbox.translate_path_for_container(value)
        return translated

    def execute(self, plan: ExecutionPlan, document_context: str = None) -> str:
        """Execute the ExecutionPlan respecting dependencies.

        Args:
            plan: The execution plan to execute
            document_context: Optional pre-loaded document content to use as context
        """
        logger.info(f"Executing plan: {plan.original_query}")
        logger.info(f"Total steps: {len(plan.steps)}")

        # Store document context for use in argument inference
        self._document_context = document_context
        if document_context:
            logger.info(
                f"Document context available for execution ({len(document_context)} chars)"
            )

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

        # Synthesize human-readable response from raw results
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
            # Try to parse as JSON for structured data
            try:
                data = (
                    json.loads(raw_result)
                    if isinstance(raw_result, str)
                    else raw_result
                )
            except json.JSONDecodeError:
                data = raw_result

            # If it's an error, just return it
            if isinstance(data, dict) and "error" in data:
                return f"Error: {data['error']}"

            prompt = f"""You are a helpful assistant. The user asked: "{query}"

Here is the raw data retrieved:
{json.dumps(data, indent=2) if isinstance(data, dict) else data}

Based on this data, write a clear, concise, and helpful response that directly answers the user's question.
- Use natural language, not JSON
- Highlight the most relevant information first
- Format nicely with bullet points if listing multiple items
- Keep it concise (2-4 paragraphs max)
- Don't mention "the data shows" or "according to the results" - just answer naturally"""

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

        # Determine input data for this step
        # Priority: 1) input_from (explicit data source), 2) last dependency, 3) description
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

        # Translate paths for container
        translated_args = self._translate_args(args)

        logger.info(f"DEBUG: Executor inferred args = {args}")
        logger.info(f"DEBUG: Translated args = {translated_args}")
        logger.info(f"DEBUG: Args type = {type(args)}")

        # Execute tool in sandbox (profiling happens inside container)
        start_time = time.time()
        result = self.sandbox.execute_with_args(
            step.tool_name, tool_file, translated_args
        )
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
                f"peak_mem={sandbox_profile['peak_memory_mb']:.4f}MB, "
                f"delta_mem={sandbox_profile.get('memory_delta_mb', 0):.4f}MB, "
                f"efficiency={sandbox_profile.get('efficiency_score', 0):.2%}, "
                f"grade={sandbox_profile['latency_grade']}"
            )

            # Also log total round-trip time for comparison
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
        translated_args = self._translate_args(new_args)
        logger.info(
            f"Step {step.step_number}: New args = {new_args} (Translated: {translated_args})"
        )

        result = self.sandbox.execute_with_args(
            step.tool_name, tool_file, translated_args
        )

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
        translated_args = self._translate_args(new_args)
        logger.info(
            f"Step {step.step_number}: Translated regenerated tool args: {translated_args}"
        )

        result = self.sandbox.execute_with_args(
            new_tool_name, tool_file, translated_args
        )

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

        # Create a wrapper function for the sandbox execution
        def sandbox_execution():
            return self.sandbox.execute_with_args(tool_name, tool_file, args)

        # Profile the execution
        result, profile = self._profiler.profile(
            sandbox_execution, _profile_name=tool_name
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
        """
        Export all profiles to a JSON file.

        Args:
            filepath: Optional path. Defaults to logs/profiles_<timestamp>.json

        Returns:
            Path to the exported file
        """
        from datetime import datetime

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
        """Use LLM to extract argument values from task description.

        When document context is available (from InputLoader), it takes priority
        over attempting online searches.
        """
        tool_file = self.tools_dir / definition.get("file", "")
        arg_names = self._get_arg_names_from_file(tool_file)

        # Prioritize document context if available
        if hasattr(self, "_document_context") and self._document_context:
            logger.info("Using document context for argument inference")
            input_data = self._document_context

            # Direct mapping: if tool has a content-type arg, assign document context directly
            content_arg_names = [
                "html_content",
                "content",
                "text",
                "text_content",
                "input",
                "document",
                "raw_text",
                "source_text",
            ]
            for content_arg in content_arg_names:
                if content_arg in arg_names:
                    logger.info(
                        f"Direct mapping document context to '{content_arg}' argument"
                    )
                    # Return with all other args as defaults or inferred
                    result = {content_arg: self._document_context}
                    # Set other required args to sensible defaults
                    for arg in arg_names:
                        if arg != content_arg and arg not in result:
                            result[arg] = ""  # Default empty for other args
                    return result

        if not arg_names:
            return {"input": input_data}

        # If 'data' arg is needed and input_data looks like structured JSON, pass it directly
        if "data" in arg_names and input_data and input_data != step.description:
            # Try JSON first
            try:
                parsed = (
                    json.loads(input_data)
                    if isinstance(input_data, str)
                    else input_data
                )
                if isinstance(parsed, dict) and len(parsed) > 0:
                    # Check for known structured data patterns
                    structured_keys = [
                        "repos",
                        "results",
                        "items",
                        "data",
                        "labels",  # generic
                        "posts",
                        "subreddit",
                        "subreddits",  # reddit
                        "sentiments",
                        "by_subreddit",
                        "aggregate",  # sentiment
                        "comparison",
                        "ranking",  # comparison
                        "symbols",
                        "dates",
                        "values",  # financial/chart
                        "post_count",
                        "total_posts",  # counts
                    ]
                    if any(k in parsed for k in structured_keys):
                        logger.info(
                            f"Direct pass-through: Using structured JSON for 'data' arg (keys: {list(parsed.keys())[:5]})"
                        )
                        return self._infer_args_with_direct_data(
                            step, parsed, arg_names, definition
                        )
            except (json.JSONDecodeError, TypeError):
                pass

            # Try CSV: if input_data has comma-separated lines with a header row
            if isinstance(input_data, str) and "," in input_data:
                csv_parsed = self._try_parse_csv_to_data(input_data, step.description)
                if csv_parsed is not None:
                    logger.info(
                        "Direct pass-through: Parsed CSV document context into structured data"
                    )
                    return self._infer_args_with_direct_data(
                        step, csv_parsed, arg_names, definition
                    )

        # Check if URL arguments are needed - if so, search for relevant URLs
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

        # Always use LLM to extract actual values (not raw description)
        from architecture.prompts import get_arg_extraction_prompt

        prompt = get_arg_extraction_prompt(
            arg_names, step.description, input_data, available_urls
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
            # Filter out None values so tool defaults are used instead
            return {k: v for k, v in args.items() if k in arg_names and v is not None}
        except Exception as e:
            logger.error(f"Failed to parse LLM args response: {e}")
            return {arg_names[0]: input_data}

    def _infer_args_with_direct_data(
        self, step: ExecutionStep, parsed_data: dict, arg_names: list, definition: dict
    ) -> dict:
        """Build args with direct data pass-through, using LLM only for other args."""
        # Start with direct data
        args = {"data": parsed_data}

        # Build a compact data summary so the LLM can generate contextual labels
        data_summary = ""
        try:
            if isinstance(parsed_data, dict):
                # Use embedded summary if available (e.g. from StockPriceTool)
                if "summary" in parsed_data and parsed_data["summary"]:
                    data_summary = str(parsed_data["summary"])[:500]
                else:
                    # Build a quick summary from top-level keys and structure
                    summary_parts = []
                    if "symbols" in parsed_data:
                        summary_parts.append(f"Symbols: {parsed_data['symbols']}")
                    if "comparison" in parsed_data and isinstance(
                        parsed_data["comparison"], dict
                    ):
                        keys = [
                            k for k in parsed_data["comparison"].keys() if k != "dates"
                        ]
                        if keys:
                            summary_parts.append(f"Series: {keys}")
                    # Fallback: show top-level keys
                    if not summary_parts:
                        summary_parts.append(
                            f"Data keys: {list(parsed_data.keys())[:10]}"
                        )
                    data_summary = "; ".join(summary_parts)
        except Exception:
            data_summary = ""

        # Get other args from LLM (title, xlabel, ylabel, etc.) but with much simpler prompt
        other_args = [a for a in arg_names if a != "data"]
        if other_args:
            # Detect if this is a chart/visualization tool
            chart_args = {
                "chart_type",
                "xlabel",
                "ylabel",
                "figsize",
                "colors",
                "legend_labels",
                "theme",
            }
            is_chart = bool(chart_args.intersection(set(other_args)))

            data_context = f"\nData context: {data_summary}\n" if data_summary else ""

            if is_chart:
                prompt = f"""For a chart visualization, provide values for these arguments:
{other_args}

Task: {step.description}
{data_context}
Use the data context to generate accurate, descriptive labels (title, xlabel, ylabel, legend_labels, etc.).
Return ONLY a JSON object. Example: {{"chart_type": "bar", "title": "My Chart", "xlabel": "X", "ylabel": "Y"}}
Use sensible defaults for any optional args (empty string is fine)."""
            else:
                prompt = f"""Extract values for these arguments from the task context:
{other_args}

Task: {step.description}
{data_context}
Return ONLY a JSON object. Omit any argument you cannot determine a value for."""

            response = self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}], max_tokens=256
            )

            if not response.get("error"):
                try:
                    other_values = json.loads(response["content"])
                    for k, v in other_values.items():
                        if k in other_args and v is not None:
                            args[k] = v
                except Exception:
                    pass

        logger.info(f"Direct data pass-through args: {list(args.keys())}")
        return args

    def _try_parse_csv_to_data(self, text: str, description: str) -> Optional[dict]:
        """Try to parse CSV text into a structured dict suitable for chart tools.

        Handles both raw CSV and CSV wrapped in document context markers
        (e.g., --- FILE: data.csv --- ... --- END: data.csv ---).
        Uses task description to pick the right columns, then aggregates.
        Returns dict like {'labels': [...], 'values': [...]} or None.
        """
        import csv
        import io
        import re

        try:
            # Extract CSV content from document context wrapper if present
            csv_text = text
            csv_match = re.search(
                r"--- FILE: .+\.csv ---\n(.*?)\n--- END: .+\.csv ---", text, re.DOTALL
            )
            if csv_match:
                csv_text = csv_match.group(1).strip()
                logger.info("Extracted CSV content from document context wrapper")

            lines = [l.strip() for l in csv_text.strip().splitlines() if l.strip()]
            if len(lines) < 2:
                return None

            reader = csv.DictReader(io.StringIO(csv_text.strip()))
            fieldnames = reader.fieldnames
            if not fieldnames or len(fieldnames) < 2:
                return None

            rows = list(reader)
            if not rows:
                return None

            # Convert numeric strings to numbers
            for row in rows:
                for key in row:
                    try:
                        row[key] = int(row[key])
                    except (ValueError, TypeError):
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            pass

            # Identify column types
            str_cols = [k for k in fieldnames if isinstance(rows[0].get(k), str)]
            num_cols = [
                k for k in fieldnames if isinstance(rows[0].get(k), (int, float))
            ]

            if not num_cols:
                return None

            # Try to match columns mentioned in the description
            desc_lower = description.lower()
            label_col = None
            value_col = None

            for col in str_cols:
                if col.lower() in desc_lower:
                    label_col = col
                    break

            for col in num_cols:
                if col.lower() in desc_lower:
                    value_col = col
                    break

            # Fallbacks
            if not label_col and str_cols:
                label_col = str_cols[0]
            if not value_col:
                value_col = num_cols[0]

            if label_col:
                # Aggregate: sum values by label
                aggregated = {}
                for row in rows:
                    key = str(row.get(label_col, ""))
                    val = row.get(value_col, 0)
                    if isinstance(val, (int, float)):
                        aggregated[key] = aggregated.get(key, 0) + val
                result = {
                    "labels": list(aggregated.keys()),
                    "values": list(aggregated.values()),
                }
            else:
                result = {"values": [row.get(value_col, 0) for row in rows]}

            logger.info(
                f"CSV parsed: {len(rows)} rows, columns: {fieldnames}, "
                f"using label={label_col}, value={value_col}"
            )
            return result
        except Exception as e:
            logger.debug(f"CSV parse failed: {e}")
            return None

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
