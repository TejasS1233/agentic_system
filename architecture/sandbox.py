"""Sandbox for isolated tool execution in Docker container."""

import ast
import time
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

ESSENTIAL_PACKAGES = {
    "pydantic",
    "numpy",
    "requests",
    "pandas",
    "beautifulsoup4",
}


def extract_dependencies(tool_path: Path) -> set[str]:
    """Parse tool file and extract pip packages from imports."""
    try:
        with open(tool_path) as f:
            tree = ast.parse(f.read())
    except SyntaxError as e:
        logger.error(f"Syntax error parsing {tool_path}: {e}")
        return set()  # Return empty set, not list

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    deps = set()
    for pkg in imports:
        if pkg in STDLIB_MODULES:
            continue
        pypi_name = IMPORT_TO_PYPI.get(pkg, pkg)
        deps.add(pypi_name)

    return deps


class Sandbox:
    """Docker-based sandbox for isolated tool execution with artifact persistence."""

    def __init__(
        self,
        tools_dir: Path,
        output_dir: Path = None,
        image: str = "python:3.11-slim",
    ):
        self.tools_dir = Path(tools_dir).resolve()
        # Default output_dir to workspace/outputs if not specified
        if output_dir is None:
            self.output_dir = self.tools_dir.parent / "outputs"
        else:
            self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.client = None
        self.container = None
        self._installed_deps: set[str] = set()
        logger.info(
            f"Sandbox initialized with tools_dir={self.tools_dir}, output_dir={self.output_dir}"
        )

    def _pre_install_essentials(self):
        """Background task to install essential packages."""
        try:
            time.sleep(1)
            self._install_deps(ESSENTIAL_PACKAGES)
            logger.info("Essential packages pre-installed")
        except Exception as e:
            logger.warning(f"Failed to pre-install essentials: {e}")

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
            volumes={
                str(self.tools_dir): {"bind": "/tools", "mode": "ro"},
                str(self.output_dir): {"bind": "/output", "mode": "rw"},
            },
            working_dir="/tools",
            detach=True,
            remove=True,
            network_mode="bridge",  # Ensure network connectivity
            dns=["8.8.8.8", "8.8.4.4"],  # Google DNS for reliable resolution
        )
        logger.info(f"Sandbox container started: {self.container.short_id}")

        # Async pre-install of essentials
        import threading

        self._pre_install_thread = threading.Thread(
            target=self._pre_install_essentials, daemon=True
        )
        self._pre_install_thread.start()

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

    def _install_deps(self, deps: set[str] | list[str]) -> None:
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

            if "nltk" in new_deps:
                logger.info("Downloading NLTK data...")
                self.container.exec_run(
                    "python -c \"import nltk; nltk.download('punkt_tab', quiet=True); "
                    "nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)\""
                )

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

    def run_tool_test(self, tool_file: str) -> dict:
        """Run the test_tool() function of a generated tool to verify it works.
        
        This executes `python tool_file.py` which should trigger:
        if __name__ == "__main__":
            test_tool()
        
        Returns:
            {"success": bool, "output": str, "error": str}
        """
        if not self.container:
            return {"success": False, "error": "Sandbox not started", "output": ""}

        tool_path = self.tools_dir / tool_file
        if not tool_path.exists():
            return {
                "success": False,
                "error": f"Tool file not found: {tool_file}",
                "output": "",
            }

        # Install dependencies
        deps = extract_dependencies(tool_path)
        deps.add("pydantic")
        self._install_deps(deps)

        logger.info(f"Running test for tool: {tool_file}")
        try:
            exit_code, output = self.container.exec_run(
                f"python /tools/{tool_file}",
                stderr=True,
            )
            output_str = output.decode() if output else ""

            if exit_code == 0:
                logger.info(f"Tool test passed: {tool_file}")
                return {"success": True, "output": output_str, "error": ""}
            else:
                logger.error(f"Tool test failed: {tool_file}")
                return {"success": False, "output": "", "error": output_str}

        except Exception as e:
            logger.error(f"Tool test execution error: {e}")
            return {"success": False, "output": "", "error": str(e)}

    def execute_with_args(self, tool_name: str, tool_file: str, args: dict) -> dict:
        """Execute a tool with specific arguments inside the sandbox with profiling."""
        if not self.container:
            return {
                "success": False,
                "error": "Sandbox not started",
                "output": "",
                "profile": None,
            }

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
        deps.add("pydantic")  # Always needed
        self._install_deps(deps)

        # Create runner script with PROFILING
        import json as json_module
        import base64 as b64_module

        args_json = json_module.dumps(args)
        args_b64 = b64_module.b64encode(args_json.encode()).decode()

        runner_code = f'''
import sys
import json
import time
import tracemalloc
import traceback
import os
import shutil
import base64
from pathlib import Path

sys.path.insert(0, '/tools')

# Track files before execution
files_before = set()
for root, dirs, files in os.walk('/tools'):
    for f in files:
        if not f.startswith('_runner_'):
            files_before.add(os.path.join(root, f))

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
    print(json.dumps({{"error": "No Tool class found", "profile": None, "artifacts": []}}))
    sys.exit(1)

try:
    # Start profiling
    tracemalloc.start()
    start_time = time.perf_counter()
    start_memory = tracemalloc.get_traced_memory()[0]
    
    # Execute tool - decode args from base64 to avoid escaping issues
    args = json.loads(base64.b64decode("{args_b64}").decode())
    tool = tool_class()
    result = tool.run(**args)
    
    # Stop profiling
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Detect new files created during execution
    files_after = set()
    for root, dirs, files in os.walk('/tools'):
        for f in files:
            if not f.startswith('_runner_'):
                files_after.add(os.path.join(root, f))
    
    # Also check current working directory
    cwd = os.getcwd()
    if os.path.exists(cwd):
        for f in os.listdir(cwd):
            full_path = os.path.join(cwd, f)
            if os.path.isfile(full_path) and not f.startswith('_runner_'):
                files_after.add(full_path)
    
    # Find new files
    new_files = files_after - files_before
    
    # Copy artifacts to output directory
    artifacts = []
    for src_path in new_files:
        if os.path.exists(src_path) and os.path.isfile(src_path):
            filename = os.path.basename(src_path)
            dst_path = f'/output/{{filename}}'
            try:
                shutil.copy2(src_path, dst_path)
                artifacts.append(filename)
            except Exception as copy_err:
                pass  # Skip files that can't be copied
    
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
    
    # Calculate efficiency score (0.0 - 1.0)
    latency_score = max(0, 1 - (execution_time_ms / 1000))
    memory_score = max(0, 1 - (peak_memory_mb / 100))
    success_score = 1.0
    efficiency = (latency_score * 0.5) + (memory_score * 0.4) + (success_score * 0.1)
    efficiency = round(min(1.0, max(0.0, efficiency)), 4)
    
    # Output result with profile and artifacts
    output = {{
        "success": True,
        "result": result,
        "artifacts": artifacts,
        "profile": {{
            "tool_name": "{tool_name}",
            "execution_time_ms": round(execution_time_ms, 3),
            "peak_memory_mb": round(peak_memory_mb, 4),
            "memory_delta_mb": round(memory_delta_mb, 4),
            "latency_grade": grade,
            "efficiency_score": efficiency,
            "success": True
        }}
    }}
    
    # Custom encoder for non-serializable objects
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
        "artifacts": [],
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
        
        # Pass API keys from host environment to container
        import os as host_os
        container_env = {}
        for key in ["SERP_API_KEY", "HF_TOKEN", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]:
            if val := host_os.environ.get(key):
                container_env[key] = val
        
        try:
            exit_code, output = self.container.exec_run(
                f"python /tools/{runner_name}",
                environment=container_env,
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
                        "profile": profile,
                        "artifacts": data.get("artifacts", []),
                    }

                # Handle result which might be in "result" field or raw output
                result_val = data.get("result", output_str)
                # Ensure result is distinct from output string if possible
                if isinstance(result_val, (dict, list)):
                    result_val = json_module.dumps(result_val)
                elif not isinstance(result_val, str):
                    result_val = str(result_val)

                # Extract artifacts
                artifacts = data.get("artifacts", [])
                if artifacts:
                    logger.info(f"Artifacts generated: {artifacts}")

                return {
                    "success": True,
                    "output": result_val,
                    "error": "",
                    "profile": profile,
                    "artifacts": artifacts,
                }
            except Exception as e:
                logger.warning(
                    f"Failed to parse sandbox output: {e}, Raw output: {output_str}"
                )
                if exit_code == 0:
                    return {
                        "success": True,
                        "output": output_str,
                        "error": "",
                        "profile": None,
                    }
                return {
                    "success": False,
                    "error": output_str,
                    "output": "",
                    "profile": None,
                }

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"success": False, "output": "", "error": str(e), "profile": None}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False