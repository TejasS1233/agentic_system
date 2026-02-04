"""Sandbox for isolated tool execution in Docker container."""

import ast
from pathlib import Path

import docker
from docker.errors import ImageNotFound

from utils.logger import get_logger

logger = get_logger(__name__)

STDLIB_MODULES = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "errno",
    "fnmatch",
    "fractions",
    "functools",
    "glob",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "keyword",
    "logging",
    "math",
    "mimetypes",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "pprint",
    "queue",
    "random",
    "re",
    "shutil",
    "signal",
    "socket",
    "sqlite3",
    "ssl",
    "stat",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "traceback",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "xml",
    "zipfile",
    "zlib",
    "typing_extensions",
    "field",
}

IMPORT_TO_PYPI = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}


def extract_dependencies(tool_path: Path) -> list[str]:
    """Parse tool file and extract pip packages from imports."""
    try:
        with open(tool_path) as f:
            tree = ast.parse(f.read())
    except SyntaxError as e:
        logger.error(f"Syntax error parsing {tool_path}: {e}")
        return []

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    deps = []
    for pkg in imports:
        if pkg in STDLIB_MODULES:
            continue
        pypi_name = IMPORT_TO_PYPI.get(pkg, pkg)
        deps.append(pypi_name)

    return deps


class Sandbox:
    """Docker-based sandbox for isolated tool execution."""

    def __init__(self, tools_dir: Path, image: str = "python:3.11-slim"):
        self.tools_dir = Path(tools_dir).resolve()
        self.image = image
        self.client = None
        self.container = None
        self._installed_deps: set[str] = set()
        logger.info(f"Sandbox initialized with tools_dir={self.tools_dir}")

    def start(self) -> None:
        """Start the sandbox container."""
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise

        try:
            self.client.images.get(self.image)
        except ImageNotFound:
            logger.info(f"Pulling image {self.image}...")
            self.client.images.pull(self.image)

        self.container = self.client.containers.run(
            self.image,
            command="tail -f /dev/null",
            volumes={str(self.tools_dir): {"bind": "/tools", "mode": "ro"}},
            working_dir="/tools",
            detach=True,
            remove=True,
        )
        logger.info(f"Sandbox container started: {self.container.short_id}")

    def stop(self) -> None:
        """Stop and remove the sandbox container."""
        if self.container:
            try:
                self.container.stop(timeout=5)
                logger.info(f"Sandbox container stopped: {self.container.short_id}")
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            self.container = None
        self._installed_deps.clear()

    def _install_deps(self, deps: list[str]) -> None:
        """Install dependencies that haven't been installed yet."""
        new_deps = [d for d in deps if d not in self._installed_deps]
        if not new_deps:
            return

        deps_str = " ".join(new_deps)
        logger.info(f"Installing dependencies: {deps_str}")
        exit_code, output = self.container.exec_run(
            f"pip install -q {deps_str}", stderr=True
        )
        if exit_code != 0:
            logger.warning(f"pip install failed: {output.decode()}")
        else:
            self._installed_deps.update(new_deps)

    def execute(self, tool_file: str, input_data: str) -> dict:
        """Execute a tool inside the sandbox."""
        if not self.container:
            return {"success": False, "error": "Sandbox not started", "output": ""}

        tool_path = self.tools_dir / tool_file
        if not tool_path.exists():
            return {
                "success": False,
                "error": f"Tool file not found: {tool_file}",
                "output": "",
            }

        deps = extract_dependencies(tool_path)
        if deps:
            self._install_deps(deps)

        logger.info(f"Executing tool: {tool_file}")
        try:
            exit_code, output = self.container.exec_run(
                f"python /tools/{tool_file}",
                environment={"INPUT_DATA": input_data},
                stderr=True,
            )
            output_str = output.decode() if output else ""

            if exit_code == 0:
                logger.info(f"Tool {tool_file} executed successfully")
                return {"success": True, "output": output_str, "error": ""}
            else:
                logger.error(f"Tool {tool_file} failed with exit code {exit_code}")
                return {"success": False, "output": "", "error": output_str}

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"success": False, "output": "", "error": str(e)}

    def execute_with_args(self, tool_name: str, tool_file: str, args: dict) -> dict:
        """Execute a tool with specific arguments inside the sandbox with profiling."""
        if not self.container:
            return {"success": False, "error": "Sandbox not started", "output": "", "profile": None}

        tool_path = self.tools_dir / tool_file
        if not tool_path.exists():
            return {
                "success": False,
                "error": f"Tool file not found: {tool_file}",
                "output": "",
                "profile": None,
            }

        # Install dependencies
        deps = extract_dependencies(tool_path)
        deps.append("pydantic")  # Always needed
        self._install_deps(deps)

        # Create runner script with PROFILING
        import json as json_module

        args_json = json_module.dumps(args)

        runner_code = f'''
import sys
import json
import time
import tracemalloc
import traceback

sys.path.insert(0, '/tools')

from {tool_path.stem} import *

# Find tool class via robust search
tool_class = None
for cls_name, obj in list(globals().items()):
    if not isinstance(obj, type):
        continue
    if 'Args' in cls_name or cls_name in ('BaseModel', 'Tool'):
        continue
    # Check if it has the tool contract: name attribute and run method
    if hasattr(obj, 'name') and hasattr(obj, 'run'):
        tool_class = obj
        break
    # Fallback: any non-Args class with a run method
    if hasattr(obj, 'run') and callable(getattr(obj, 'run', None)):
        tool_class = obj
        break

if tool_class is None:
    print(json.dumps({{"error": "No Tool class found", "profile": None}}))
    sys.exit(1)

try:
    # Start profiling
    tracemalloc.start()
    start_time = time.perf_counter()
    start_memory = tracemalloc.get_traced_memory()[0]
    
    # Execute tool
    tool = tool_class()
    args = json.loads('{args_json}')
    result = tool.run(**args)
    
    # Stop profiling
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate metrics
    execution_time_ms = (end_time - start_time) * 1000
    memory_delta_mb = (current - start_memory) / (1024 * 1024)
    peak_memory_mb = peak / (1024 * 1024)
    
    # Determine latency grade
    if execution_time_ms < 10:
        grade = "fast"
    elif execution_time_ms < 100:
        grade = "moderate"
    elif execution_time_ms < 1000:
        grade = "slow"
    else:
        grade = "critical"
    
    # Output result with profile
    output = {{
        "success": True,
        "result": result,  # Return raw result, JSONEncoder will handle serialization
        "profile": {{
            "tool_name": "{tool_name}",
            "execution_time_ms": round(execution_time_ms, 3),
            "peak_memory_mb": round(peak_memory_mb, 4),
            "memory_delta_mb": round(memory_delta_mb, 4),
            "latency_grade": grade,
            "success": True
        }}
    }}
    
    # Custom encoder for non-serializable objects if needed
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                return super().default(obj)
            except TypeError:
                return str(obj)
                
    print(json.dumps(output, cls=CustomEncoder))
    
except Exception as e:
    # Stop profiling even on error
    try:
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        execution_time_ms = (end_time - start_time) * 1000
        peak_memory_mb = peak / (1024 * 1024)
    except:
        execution_time_ms = 0
        peak_memory_mb = 0
    
    output = {{
        "error": str(e),
        "traceback": traceback.format_exc(),
        "profile": {{
            "tool_name": "{tool_name}",
            "execution_time_ms": round(execution_time_ms, 3),
            "peak_memory_mb": round(peak_memory_mb, 4),
            "memory_delta_mb": 0,
            "latency_grade": "critical",
            "success": False,
            "error_message": str(e)
        }}
    }}
    print(json.dumps(output))
'''
        # Write runner to container
        runner_name = f"_runner_{tool_name}.py"
        runner_path = self.tools_dir / runner_name

        logger.info(f"DEBUG: sandbox received args = {args}")
        logger.info(f"DEBUG: args_json = {args_json}")
        logger.info(f"DEBUG: Runner script line: args = json.loads('{args_json}')")

        with open(runner_path, "w") as f:
            f.write(runner_code)

        logger.info(f"Executing tool with args: {tool_name}")
        logger.info(f"Args JSON being passed: {args_json}")
        try:
            exit_code, output = self.container.exec_run(
                f"python /tools/{runner_name}",
                stderr=True,
            )
            output_str = output.decode() if output else ""

            # Cleanup runner
            if runner_path.exists():
                runner_path.unlink()

            # Parse output with profile
            try:
                lines = output_str.strip().split("\n")
                # Look for the last valid JSON line
                json_line = None
                for line in reversed(lines):
                    try:
                        json_module.loads(line)
                        json_line = line
                        break
                    except json_module.JSONDecodeError:
                        continue
                
                if not json_line:
                    raise ValueError("No valid JSON output found")

                data = json_module.loads(json_line)
                
                # Extract profile
                profile = data.get("profile", None)
                
                if "error" in data:
                    return {
                        "success": False, 
                        "error": data["error"], 
                        "output": "",
                        "profile": profile
                    }
                
                # Handle result which might be in "result" field or raw output
                result_val = data.get("result", output_str)
                # Ensure result is distinct from output string if possible
                if isinstance(result_val, (dict, list)):
                    result_val = json_module.dumps(result_val)
                elif not isinstance(result_val, str):
                    result_val = str(result_val)

                return {
                    "success": True,
                    "output": result_val,
                    "error": "",
                    "profile": profile,
                }
            except Exception as e:
                logger.warning(f"Failed to parse sandbox output: {e}, Raw output: {output_str}")
                if exit_code == 0:
                    return {"success": True, "output": output_str, "error": "", "profile": None}
                return {"success": False, "error": output_str, "output": "", "profile": None}

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"success": False, "output": "", "error": str(e), "profile": None}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
