"""Executor Agent with plan-based DAG execution."""

import time
import json
import importlib.util
from typing import Protocol
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


class Tool(Protocol):
    """Protocol for tool interface."""

    name: str
    description: str

    def run(self, input_data: str) -> str: ...


@dataclass
class Step:
    """A single step in the execution plan."""

    step: int
    subtask: str
    tool: str
    depends_on: list[int]
    result: str | None = None
    status: str = "pending"


@dataclass
class ExecutionPlan:
    """DAG-based execution plan from Orchestrator."""

    original_query: str
    steps: list[Step]

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        steps = [Step(**s) for s in data["steps"]]
        return cls(original_query=data["original_query"], steps=steps)

    def get_ready_steps(self) -> list[Step]:
        """Return steps whose dependencies are all completed."""
        completed = {s.step for s in self.steps if s.status == "completed"}
        return [
            s
            for s in self.steps
            if s.status == "pending" and all(d in completed for d in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status == "completed" for s in self.steps)

    def has_failed(self) -> bool:
        return any(s.status == "failed" for s in self.steps)


@dataclass
class ToolRegistry:
    """Registry that loads tools from registry.json."""

    registry_path: Path
    tools_dir: Path
    _tools: dict[str, Tool] = field(default_factory=dict)
    _definitions: dict[str, dict] = field(default_factory=dict)

    def load_registry(self) -> None:
        """Load tool definitions from registry.json."""
        with open(self.registry_path) as f:
            self._definitions = json.load(f)
        logger.info(f"Loaded {len(self._definitions)} tool definitions")

    def load_tool(self, name: str) -> Tool | None:
        """Dynamically load a tool by name."""
        if name in self._tools:
            return self._tools[name]

        if name not in self._definitions:
            logger.error(f"Tool {name} not found in registry")
            return None

        definition = self._definitions[name]
        tool_file = self.tools_dir / definition["file"]

        if not tool_file.exists():
            logger.error(f"Tool file not found: {tool_file}")
            return None

        try:
            spec = importlib.util.spec_from_file_location(name, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            tool_class = getattr(module, name, None)
            if tool_class:
                tool = tool_class()
                self._tools[name] = tool
                logger.info(f"Loaded tool: {name}")
                return tool
        except Exception as e:
            logger.error(f"Failed to load tool {name}: {e}")

        return None

    def get_definition(self, name: str) -> dict | None:
        """Get tool definition (file, description, tags, input/output types, domain)."""
        return self._definitions.get(name)


class ExecutorAgent:
    """Executes plans by running tools in dependency order."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        logger.info("ExecutorAgent initialized")

    def execute_plan(self, plan: ExecutionPlan) -> dict:
        """Execute all steps respecting dependencies."""
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
                results[step.step] = result

        final_step = max(plan.steps, key=lambda s: s.step)

        return {
            "query": plan.original_query,
            "success": plan.is_complete(),
            "result": final_step.result,
            "step_results": results,
        }

    def _execute_step(self, step: Step, previous_results: dict) -> str:
        """Execute a single step."""
        logger.info(f"Step {step.step}: {step.subtask} using {step.tool}")
        step.status = "running"

        tool = self.registry.load_tool(step.tool)
        if not tool:
            step.status = "failed"
            step.result = f"Error: Tool {step.tool} not available"
            logger.error(f"Step {step.step}: Tool not available")
            return step.result

        if step.depends_on:
            input_data = previous_results.get(step.depends_on[-1], "")
        else:
            input_data = ""

        start_time = time.time()
        try:
            result = tool.run(input_data)
            latency = time.time() - start_time
            step.result = result
            step.status = "completed"
            logger.info(f"Step {step.step}: Completed in {latency:.2f}s")
        except Exception as e:
            latency = time.time() - start_time
            step.status = "failed"
            step.result = f"Error: {e}"
            logger.error(f"Step {step.step}: Failed after {latency:.2f}s - {e}")

        return step.result
