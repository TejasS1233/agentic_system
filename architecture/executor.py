"""Executor Agent - Runs tools via persistent Docker container."""

import time
import json
from pathlib import Path

from architecture.schemas import ExecutionPlan, ExecutionStep
from architecture.sandbox import Sandbox
from architecture.llm_manager import get_llm_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutorAgent:
    """Executes ExecutionPlan by running tools in a persistent Docker container."""

    def __init__(self, workspace_path: str, tools_dir: str, sandbox: Sandbox):
        self.workspace = Path(workspace_path)
        self.tools_dir = Path(tools_dir)
        self.registry_path = self.tools_dir / "registry.json"
        self.sandbox = sandbox
        self._definitions = {}
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
        """Execute a single step using the Sandbox."""
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

        start_time = time.time()
        logger.info(
            f"DEBUG: Calling sandbox.execute_with_args({step.tool_name}, {tool_file}, {args})"
        )
        result = self.sandbox.execute_with_args(step.tool_name, tool_file, args)
        latency = time.time() - start_time

        if result["success"]:
            step.result = result["output"]
            step.status = "completed"
            logger.info(f"Step {step.step_number}: Completed in {latency:.2f}s")
        else:
            step.status = "failed"
            step.result = f"Error: {result['error']}"
            logger.error(
                f"Step {step.step_number}: Failed after {latency:.2f}s - {result['error']}"
            )

        return step.result

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
