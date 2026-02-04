"""Executor Agent - Runs tools via Docker containers."""

import os
import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from architecture.schemas import ExecutionPlan, ExecutionStep
from architecture.llm_manager import get_llm_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutorAgent:
    """Executes ExecutionPlan by running tools in Docker containers."""

    def __init__(self, workspace_path: str, tools_dir: str):
        self.workspace = Path(workspace_path)
        self.tools_dir = Path(tools_dir)
        self.registry_path = self.tools_dir / "registry.json"
        self._definitions = {}
        self.load_registry()
        self.llm = get_llm_manager()
        logger.info("ExecutorAgent initialized")

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

        results = {}  # step_number -> result

        while not plan.is_complete() and not plan.has_failed():
            ready = plan.get_ready_steps()

            if not ready:
                logger.error("Deadlock: No ready steps but plan not complete")
                break

            for step in ready:
                result = self._execute_step(step, results)
                results[step.step_number] = result

        final_step = max(plan.steps, key=lambda s: s.step_number)
        
        return json.dumps({
            "query": plan.original_query,
            "success": plan.is_complete(),
            "result": final_step.result,
            "step_results": results,
        }, indent=2)

    def _execute_step(self, step: ExecutionStep, previous_results: dict) -> str:
        """Execute a single step using Docker."""
        logger.info(f"Step {step.step_number}: {step.description} using {step.tool_name}")
        step.status = "running"

        if not step.tool_name:
            step.status = "failed"
            step.result = "Error: No tool assigned"
            return step.result

        # Reload registry to get latest tools
        self.load_registry()

        if step.tool_name not in self._definitions:
            step.status = "failed"
            step.result = f"Error: Tool {step.tool_name} not in registry"
            logger.error(f"Step {step.step_number}: Tool not in registry")
            return step.result

        definition = self._definitions[step.tool_name]
        tool_file = self.tools_dir / definition["file"]

        if not tool_file.exists():
            step.status = "failed"
            step.result = f"Error: Tool file not found: {tool_file}"
            logger.error(f"Step {step.step_number}: Tool file missing")
            return step.result

        # Get input from dependencies
        if step.depends_on:
            input_data = previous_results.get(step.depends_on[-1], "")
        else:
            input_data = step.description

        # Infer arguments
        args = self._infer_args(step, input_data, definition)

        start_time = time.time()
        try:
            # Run tool in Docker
            result = self._run_in_docker(step.tool_name, tool_file, definition, args)
            latency = time.time() - start_time
            step.result = result
            step.status = "completed"
            logger.info(f"Step {step.step_number}: Completed in {latency:.2f}s")
        except Exception as e:
            latency = time.time() - start_time
            step.status = "failed"
            step.result = f"Error: {e}"
            logger.error(f"Step {step.step_number}: Failed after {latency:.2f}s - {e}")

        return step.result

    def _run_in_docker(self, tool_name: str, tool_file: Path, definition: dict, args: dict) -> str:
        """Run a tool inside a Docker container."""
        # Get explicit dependencies from registry
        dependencies = set(definition.get("dependencies", []))
        
        # Always include pydantic (used by all tools)
        dependencies.add("pydantic")
        
        # Parse tool file for additional imports
        additional_deps = self._parse_imports(tool_file)
        dependencies.update(additional_deps)
        
        dependencies_list = list(dependencies)
        logger.info(f"Dependencies for {tool_name}: {dependencies_list}")
        
        # Create a runner script
        runner_code = f'''
import sys
import json
sys.path.insert(0, '/workspace')

# JSON compatibility
null = None
true = True
false = False

# Import and run the tool
from {tool_file.stem} import *

# Find the tool class
tool_class = None
for name, obj in list(globals().items()):
    if isinstance(obj, type) and 'Tool' in name and name != 'Tool':
        tool_class = obj
        break

if tool_class is None:
    print(json.dumps({{"error": "No Tool class found"}}))
    sys.exit(1)

try:
    tool = tool_class()
    args = {json.dumps(args)}
    result = tool.run(**args)
    print(json.dumps({{"success": True, "result": str(result)}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''
        
        # Write runner to temp file
        runner_path = self.tools_dir / f"_runner_{tool_name}.py"
        with open(runner_path, "w") as f:
            f.write(runner_code)

        # Build pip install command
        pip_install = ""
        if dependencies_list:
            pip_install = f"pip install --quiet {' '.join(dependencies_list)} && "

        # Docker command
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.tools_dir}:/workspace",
            "-w", "/workspace",
            "python:3.11-slim",
            "bash", "-c",
            f"{pip_install}python /workspace/_runner_{tool_name}.py"
        ]

        logger.info(f"Running Docker: {' '.join(docker_cmd[:6])}...")

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Clean up runner
            if runner_path.exists():
                runner_path.unlink()

            if result.returncode != 0:
                error = result.stderr or result.stdout
                logger.error(f"Docker failed: {error[:200]}")
                return f"Error: {error[:500]}"

            # Parse output
            output = result.stdout.strip()
            try:
                data = json.loads(output.split('\n')[-1])  # Last line is JSON
                if "error" in data:
                    return f"Error: {data['error']}"
                return data.get("result", output)
            except json.JSONDecodeError:
                return output

        except subprocess.TimeoutExpired:
            return "Error: Docker execution timed out"
        except Exception as e:
            return f"Error running Docker: {e}"

    def _parse_imports(self, tool_file: Path) -> set:
        """Parse a Python file to extract third-party imports."""
        # Map common imports to PyPI package names
        import_to_package = {
            "bs4": "beautifulsoup4",
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "pyyaml",
            "dotenv": "python-dotenv",
        }
        
        # Standard library modules to skip
        stdlib = {
            "sys", "os", "json", "re", "time", "datetime", "pathlib", "typing",
            "collections", "itertools", "functools", "math", "random", "string",
            "io", "tempfile", "subprocess", "urllib", "http", "abc", "dataclasses",
            "enum", "copy", "hashlib", "base64", "textwrap", "shutil", "glob"
        }
        
        deps = set()
        try:
            with open(tool_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            import re
            # Match: import X or from X import Y
            imports = re.findall(r'^(?:from|import)\s+(\w+)', content, re.MULTILINE)
            
            for imp in imports:
                if imp in stdlib:
                    continue
                # Map to PyPI name if needed
                package = import_to_package.get(imp, imp)
                deps.add(package)
                
        except Exception as e:
            logger.error(f"Failed to parse imports from {tool_file}: {e}")
        
        return deps

    def _infer_args(self, step: ExecutionStep, input_data: str, definition: dict) -> dict:
        """Determine tool arguments by parsing the tool file."""
        # Parse actual argument names from the tool file
        tool_file = self.tools_dir / definition.get("file", "")
        arg_names = self._get_arg_names_from_file(tool_file)
        
        if not arg_names:
            # Fallback: single input argument
            return {"input": input_data}
        
        # If only one argument, just pass the input
        if len(arg_names) == 1:
            return {arg_names[0]: input_data}
        
        # For multiple args, use LLM
        prompt = f"""Map the input to these EXACT argument names: {arg_names}

Task: {step.description}
Input data: {input_data[:500] if input_data else 'None'}

Return JSON with ONLY these keys: {arg_names}
Example: {{"{arg_names[0]}": "value1"{', "' + arg_names[1] + '": "value2"' if len(arg_names) > 1 else ''}}}
"""

        response = self.llm.generate_json(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )

        if response.get("error"):
            logger.error(f"Arg inference failed: {response['error']}")
            return {arg_names[0]: input_data}

        try:
            args = json.loads(response["content"])
            # Validate that returned keys match expected
            return {k: v for k, v in args.items() if k in arg_names}
        except Exception:
            return {arg_names[0]: input_data}

    def _get_arg_names_from_file(self, tool_file: Path) -> list:
        """Extract argument names from the tool's Args class."""
        try:
            with open(tool_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            import re
            # Find Args class fields: name: Type = Field(...) or name: Type
            matches = re.findall(r'^\s+(\w+)\s*:\s*\w+', content, re.MULTILINE)
            # Filter out common class attributes
            skip = {'name', 'description', 'args_schema'}
            return [m for m in matches if m not in skip]
        except Exception as e:
            logger.error(f"Failed to parse args from {tool_file}: {e}")
            return []

