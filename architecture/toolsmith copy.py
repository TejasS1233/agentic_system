import os
import re
import json
import time
import ast
import subprocess
import tempfile
import importlib.util
from architecture.llm_manager import get_llm_manager

# For PyPI name checking
try:
    import requests
except ImportError:
    requests = None

from architecture.gatekeeper import Gatekeeper


class Toolsmith:
    def __init__(self, safe_mode=True, gatekeeper=None):
        self.safe_mode = safe_mode
        self.gatekeeper = gatekeeper or Gatekeeper(strict_mode=safe_mode)
        self.workspace_root = os.path.join(os.getcwd(), "workspace")
        self.tools_dir = os.path.join(self.workspace_root, "tools")
        self.packages_dir = os.path.join(self.workspace_root, "packages")
        self.registry_path = os.path.join(self.tools_dir, "registry.json")
        self.metrics_path = os.path.join(self.tools_dir, "metrics.json")
        self.tools_source_path = os.path.join(os.getcwd(), "execution", "tools.py")
        
        # PyPI credentials from environment
        self.pypi_username = os.environ.get("PYPI_USERNAME", "__token__")
        self.pypi_token = os.environ.get("PYPI_TOKEN", "")
        
        # Ensure directories exist
        os.makedirs(self.tools_dir, exist_ok=True)
        os.makedirs(self.packages_dir, exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({}, f)
        else:
            # Migrate legacy registry entries to include status fields
            self._migrate_registry_status()

        # Initialize metrics if not present
        if not os.path.exists(self.metrics_path):
            self._log_metrics("init", {"message": "Metrics initialized"})

        # ignore common words in tag matching
        self.stop_words = {
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "and",
            "or",
            "is",
            "it",
            "this",
            "that",
            "tool",
            "can",
            "please",
            "make",
            "create",
            "write",
            "i",
            "need",
            "want",
            "use",
            "help",
            "me",
            "from",
            "into",
        }

    def _migrate_registry_status(self):
        """
        Migrate legacy registry entries to include status and timestamp fields.
        This ensures backward compatibility with older registries.
        """
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            
            modified = False
            current_time = time.time()
            
            for name, meta in data.items():
                # Add status if missing
                if "status" not in meta:
                    meta["status"] = "active"
                    modified = True
                
                # Add created_at if missing (use current time as fallback)
                if "created_at" not in meta:
                    meta["created_at"] = current_time
                    modified = True
                
                # Add updated_at if missing
                if "updated_at" not in meta:
                    meta["updated_at"] = current_time
                    modified = True
                
                # Add use_count if missing (default to 0)
                if "use_count" not in meta:
                    meta["use_count"] = 0
                    modified = True
                
                # Add last_used if missing (default to None)
                if "last_used" not in meta:
                    meta["last_used"] = None
                    modified = True
            
            if modified:
                with open(self.registry_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[Toolsmith] Migrated registry entries to include status fields")
        except Exception as e:
            print(f"[Toolsmith] Registry migration failed: {e}")

    def _log_metrics(self, event_type, details):
        """Logs usage metrics to a JSONL file for paper analysis."""
        entry = {"timestamp": time.time(), "event": event_type, "details": details}
        try:
            # We use append mode for a simpler log, or we could handle a JSON array.
            # For robustness, let's load, append, save.
            data = []
            if os.path.exists(self.metrics_path):
                try:
                    with open(self.metrics_path, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    data = []

            data.append(entry)
            with open(self.metrics_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Toolsmith] Metrics logging failed: {e}")

    def get_tool_creation_timestamps(self) -> dict:
        """
        Get creation timestamps for all tools from metrics.json.
        
        Returns:
            Dict mapping tool names to their creation timestamps (Unix epoch).
            Tools that don't have a recorded creation time are not included.
        """
        timestamps = {}
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r") as f:
                    data = json.load(f)
                
                # Parse metrics to find tool_created events
                for entry in data:
                    if entry.get("event") == "tool_created":
                        details = entry.get("details", {})
                        tool_name = details.get("tool_name")
                        if tool_name:
                            # Use the timestamp from the entry
                            timestamps[tool_name] = entry.get("timestamp")
        except Exception as e:
            print(f"[Toolsmith] Failed to read tool timestamps: {e}")
        
        return timestamps

    def get_tool_last_used_timestamps(self) -> dict:
        """
        Get the last usage timestamp for all tools from metrics.json.
        
        This includes both tool_created events (creation = first "use") and
        deduplication_hit events (tool was matched/used).
        
        Returns:
            Dict mapping tool names to their last used timestamps (Unix epoch).
        """
        last_used = {}
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r") as f:
                    data = json.load(f)
                
                for entry in data:
                    event = entry.get("event")
                    details = entry.get("details", {})
                    timestamp = entry.get("timestamp")
                    
                    if event == "tool_created":
                        tool_name = details.get("tool_name")
                        if tool_name:
                            # Update if this is the most recent event
                            if tool_name not in last_used or timestamp > last_used[tool_name]:
                                last_used[tool_name] = timestamp
                    
                    elif event == "deduplication_hit":
                        tool_name = details.get("matched_tool")
                        if tool_name:
                            if tool_name not in last_used or timestamp > last_used[tool_name]:
                                last_used[tool_name] = timestamp
        except Exception as e:
            print(f"[Toolsmith] Failed to read last used timestamps: {e}")
        
        return last_used

    def _is_safe_code(self, code: str) -> bool:
        """
        Uses the Gatekeeper for static analysis to reject dangerous operations.
        Paper Claim: 'Autonomous Secure Tool Generation'.
        """
        if not self.safe_mode:
            return True

        result = self.gatekeeper.validate(code)

        if not result.is_safe:
            for violation in result.violations:
                print(f"[Toolsmith] Safety Violation: {violation}")

        return result.is_safe

    def _get_existing_tools_context(self):
        try:
            with open(self.tools_source_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"# Could not read tools.py: {e}"

    def _tokenize(self, text):
        # Replace non-alphanumeric with space
        clean_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = set(clean_text.split())
        return tokens - self.stop_words

    def create_tool(self, requirement: str):
        """
        Generates a new python tool based on a requirement using Tag-Based deduplication.
        """
        start_time = time.time()
        print(f"[Toolsmith] Received request: '{requirement}'")

        # 1. Tag-Based Deduplication Check
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)

            req_tokens = self._tokenize(requirement)

            best_match = None
            highest_score = 0.0

            for name, meta in registry.items():
                tool_tags = set(meta.get("tags", []))
                # Fallback to tokenizing description if tags missing
                if not tool_tags:
                    tool_tags = self._tokenize(meta.get("description", ""))
                
                # Also include input_types, output_types, and domain in matching
                input_types = set(meta.get("input_types", []))
                output_types = set(meta.get("output_types", []))
                domain = meta.get("domain", "")
                
                # Combine all matchable terms
                all_tool_terms = tool_tags | input_types | output_types
                if domain:
                    all_tool_terms.add(domain)
                
                # Jaccard-like Overlap Score: Intersection / Request Size
                # (How much of the request is covered by the tool?)
                if not req_tokens:
                    continue
                    
                intersection = req_tokens & all_tool_terms
                
                # Require at least 2 matching keywords to avoid single-word false positives
                if len(intersection) < 2:
                    continue
                
                score = len(intersection) / len(req_tokens)

                if score > highest_score:
                    highest_score = score
                    best_match = name

            # Threshold: 50% overlap implies relevance (raised from 30%)
            if highest_score >= 0.5:
                desc = registry[best_match].get("description", "No description")
                print(
                    f"[Toolsmith] Deduplication HIT. Match: '{best_match}' (Score: {highest_score:.2f})"
                )

                self._log_metrics(
                    "deduplication_hit",
                    {
                        "request": requirement,
                        "matched_tool": best_match,
                        "score": highest_score,
                        "latency": time.time() - start_time,
                    },
                )

                return f"EXISTING TOOL FOUND: '{best_match}' seems to match your request (Score: {highest_score:.2f}).\nDescription: {desc}\nPlease use this tool instead of creating a new one."

            self._log_metrics("generation_start", {"request": requirement})

        except Exception as e:
            print(f"[Toolsmith] Deduplication check failed: {e}")

        # 2. Context Injection
        existing_code = self._get_existing_tools_context()

        # 3. Generate Code (JSON Mode)
        print("[Toolsmith] Tool not found. Generating with LLM...")
        try:
            llm = get_llm_manager()
            
            system_prompt = f"""You are an expert Python Tool Generator. 
You MUST generate a JSON object containing the tool code and metadata.

REFERENCE CODE STYLE:
{existing_code}

OUTPUT FORMAT (JSON ONLY):
{{
  "class_name": "NameOfTool",
  "filename": "name_of_tool.py",
  "tags": ["tag1", "tag2", "tag3"],
  "input_types": ["string", "number", "file", "list", "dict"],
  "output_types": ["string", "number", "file", "image", "json"],
  "domain": "math|text|file|web|visualization|data|system",
  "dependencies": ["library_name_1", "library_name_2"],
  "code": "import ... class NameOfTool(Tool): ..."
}}

RULES:
1. `code` must be a valid, escaped python string. Ensure all newlines are escaped as \\n and quotes are escaped as \\".
2. `tags` should be 3-5 keywords.
3. `dependencies` must list specific PyPI package names required (e.g., ["pandas", "requests"]). If none, use [].
4. `domain` must be ONE of: math, text, file, web, visualization, data, system, conversion, search.
5. CODE STRUCTURE REQUIREMENTS:
   - IMPORTS: Include all necessary standard and third-party imports.
   - ARGS CLASS: Define `class {{ClassName}}Args(BaseModel):` with fields.
   - TOOL CLASS: Define `class {{ClassName}}(Tool):` inheriting from the base Tool class.
   - ATTRIBUTE: Set `args_schema = {{ClassName}}Args` inside the tool class.
   - METHOD: `def run(self, arg1: Type, ...) -> Any:` matching the args.
6. HANDLING CREDENTIALS:
   - Do NOT hardcode API keys. 
   - If an API key is needed, it MUST be passed as an argument to the `run` method (e.g., `api_key: str`).
7. REAL IMPLEMENTATION ONLY:
   - NO MOCK APIS. NO FAKE DATA.
   - Use real logic (e.g., `requests.get`, `math.sqrt`).
   - If the task requires external libraries (like `pandas`, `praw`, `github`), IMPORT them and list them in the `dependencies` key.
8. BASE CLASS DEFINITION:
   - Assume a `Tool` base class exists in the environment that has a `run` method.
   - Do NOT redefine `Tool` unless necessary for standalone testing.
"""

            response = llm.generate_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a tool for: {requirement}"}
                ],
                max_tokens=4096
            )

            if response.get("error"):
                return f"Tool creation failed: {response['error']}"

            content = response["content"]
            tool_data = json.loads(content)

            class_name = tool_data["class_name"]
            file_name = tool_data["filename"]
            tags = tool_data.get("tags", [])
            input_types = tool_data.get("input_types", [])
            output_types = tool_data.get("output_types", [])
            domain = tool_data.get("domain", "")
            tool_code = tool_data["code"]

            # --- SAFETY CHECK ---
            if not self._is_safe_code(tool_code):
                self._log_metrics(
                    "safety_violation",
                    {"request": requirement, "generated_code_snippet": tool_code[:100]},
                )
                return "Error: Generated tool code failed Safety Check (contains banned imports/calls). Request rejected for security."
            # --------------------

            file_path = os.path.join(self.tools_dir, file_name)

            # 4. Save to Disk
            with open(file_path, "w") as f:
                f.write(tool_code)

            print(f"[Toolsmith] Wrote new tool to {file_path}")

            # 5. Detect dependencies and publish to PyPI
            dependencies = self._detect_dependencies(tool_code)
            print(f"[Toolsmith] Detected dependencies: {dependencies}")
            
            pypi_success, pypi_package, pypi_message = self._publish_to_pypi(
                class_name, tool_code, requirement, dependencies, tags, input_types, output_types, domain
            )
            
            if pypi_success:
                print(f"[Toolsmith] PyPI: {pypi_message}")
                
                # If package name differs from class name, update file and registry names
                expected_file = pypi_package.replace("-", "_") + "_tool.py"
                if file_name != expected_file:
                    new_file_path = os.path.join(self.tools_dir, expected_file)
                    os.rename(file_path, new_file_path)
                    file_name = expected_file
                    file_path = new_file_path
                    print(f"[Toolsmith] Renamed tool file to match PyPI: {file_name}")
            else:
                print(f"[Toolsmith] PyPI publishing failed: {pypi_message}")
                pypi_package = ""

            # 6. Update Registry with Tags, Capability Schema, and PyPI package name
            self._update_registry(class_name, file_name, requirement, tags, input_types, output_types, domain, pypi_package)
            
            self._log_metrics("tool_created", {
                "request": requirement,
                "tool_name": class_name,
                "pypi_package": pypi_package,
                "latency": time.time() - start_time
            })
            
            result_msg = f"Successfully created {class_name} with tags {tags}."
            if pypi_package:
                result_msg += f" Published to PyPI as: pip install {pypi_package}"
            return result_msg

        except Exception as e:
            import traceback

            traceback.print_exc()
            self._log_metrics(
                "generation_failed", {"request": requirement, "error": str(e)}
            )
            return f"Tool creation failed: {e}"

    def _update_registry(self, class_name, file_name, description, tags, input_types=None, output_types=None, domain=None, pypi_package=None, status="active"):
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
        
        # Preserve existing fields if tool already exists
        existing_entry = data.get(class_name, {})
        created_at = existing_entry.get("created_at", time.time())
        use_count = existing_entry.get("use_count", 0)  # Preserve use_count
        last_used = existing_entry.get("last_used")  # Preserve last_used
        
        data[class_name] = {
            "file": file_name,
            "description": description,
            "tags": tags,
            "input_types": input_types or [],
            "output_types": output_types or [],
            "domain": domain or "",
            "pypi_package": pypi_package or "",
            "status": status,  # active, inactive, deprecated, failed
            "created_at": created_at,
            "updated_at": time.time(),
            "use_count": use_count,
            "last_used": last_used
        }
        
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def update_tool_status(self, tool_name: str, status: str) -> bool:
        """
        Update the status of a tool in the registry.
        
        Args:
            tool_name: Name of the tool to update
            status: New status - one of: active, inactive, deprecated, failed
            
        Returns:
            True if successful, False if tool not found
        """
        valid_statuses = {"active", "inactive", "deprecated", "failed"}
        if status not in valid_statuses:
            print(f"[Toolsmith] Invalid status: {status}. Must be one of {valid_statuses}")
            return False
        
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            
            if tool_name not in data:
                print(f"[Toolsmith] Tool not found in registry: {tool_name}")
                return False
            
            data[tool_name]["status"] = status
            data[tool_name]["updated_at"] = time.time()
            
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._log_metrics("status_change", {
                "tool_name": tool_name,
                "new_status": status
            })
            
            print(f"[Toolsmith] Updated {tool_name} status to: {status}")
            return True
            
        except Exception as e:
            print(f"[Toolsmith] Failed to update tool status: {e}")
            return False

    def get_tool_status(self, tool_name: str) -> str:
        """
        Get the current status of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Status string or "unknown" if tool not found
        """
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            
            if tool_name in data:
                return data[tool_name].get("status", "active")  # Default to active for legacy entries
            return "unknown"
        except Exception as e:
            print(f"[Toolsmith] Failed to get tool status: {e}")
            return "unknown"

    def get_tools_by_status(self, status: str) -> list:
        """
        Get all tools with a specific status.
        
        Args:
            status: Status to filter by (active, inactive, deprecated, failed)
            
        Returns:
            List of tool names with the specified status
        """
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            
            tools = []
            for name, meta in data.items():
                # Default to "active" for legacy entries without status
                tool_status = meta.get("status", "active")
                if tool_status == status:
                    tools.append(name)
            return tools
        except Exception as e:
            print(f"[Toolsmith] Failed to get tools by status: {e}")
            return []

    def _detect_dependencies(self, code: str) -> list:
        """
        Parse the tool code to detect third-party library imports.
        Returns a list of pip package names.
        """
        # Standard library modules to exclude
        stdlib = {
            "os", "sys", "re", "json", "ast", "time", "math", "datetime", 
            "collections", "itertools", "functools", "typing", "abc",
            "subprocess", "tempfile", "pathlib", "io", "csv", "random",
            "hashlib", "base64", "urllib", "http", "email", "html", "xml",
            "logging", "warnings", "copy", "pickle", "sqlite3", "threading",
            "multiprocessing", "asyncio", "socket", "ssl", "uuid", "platform"
        }
        
        # Known import-to-package mappings
        import_to_package = {
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "sklearn": "scikit-learn",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
        }
        
        dependencies = set()
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in stdlib:
                            pkg = import_to_package.get(module, module)
                            dependencies.add(pkg)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module not in stdlib:
                            pkg = import_to_package.get(module, module)
                            dependencies.add(pkg)
        except:
            pass
        
        # Always include pydantic since our tools use it
        dependencies.add("pydantic")
        
        return list(dependencies)

    def _check_pypi_name(self, name: str) -> bool:
        """
        Check if a package name is available on PyPI.
        Returns True if available, False if taken.
        """
        if not requests:
            print("[Toolsmith] Warning: requests not installed, skipping PyPI name check")
            return True
        
        try:
            response = requests.get(f"https://pypi.org/pypi/{name}/json", timeout=5)
            return response.status_code == 404  # 404 means available
        except:
            return True  # Assume available on error

    def _get_available_pypi_name(self, base_name: str) -> str:
        """
        Find an available PyPI package name, trying variations if needed.
        """
        # Normalize to PyPI naming convention (lowercase, hyphens)
        base_name = re.sub(r'[^a-z0-9]', '-', base_name.lower())
        base_name = re.sub(r'-+', '-', base_name).strip('-')
        
        candidates = [
            base_name,
            f"{base_name}-ts",  # toolsmith suffix
            f"{base_name}-auto",
            f"{base_name}-gen",
            f"{base_name}-{int(time.time()) % 10000}",  # timestamp suffix
        ]
        
        for candidate in candidates:
            if self._check_pypi_name(candidate):
                return candidate
        
        # Last resort: add random suffix
        import random
        return f"{base_name}-{random.randint(1000, 9999)}"

    def _publish_to_pypi(self, class_name: str, tool_code: str, description: str, dependencies: list, tags: list = None, input_types: list = None, output_types: list = None, domain: str = "") -> tuple:
        """
        Build and publish the tool as a minimal PyPI package.
        The package is just the tool code file, installable via pip.
        Returns (success: bool, package_name: str, message: str)
        """
        tags = tags or []
        input_types = input_types or []
        output_types = output_types or []
        if not self.pypi_token:
            return False, "", "PyPI token not configured. Set PYPI_TOKEN in .env"
        
        # Get available package name
        package_name = self._get_available_pypi_name(class_name)
        module_name = package_name.replace("-", "_")
        print(f"[Toolsmith] Publishing as PyPI package: {package_name}")
        
        # Create minimal package directory with src layout
        pkg_dir = os.path.join(self.packages_dir, package_name)
        src_dir = os.path.join(pkg_dir, "src", module_name)
        os.makedirs(src_dir, exist_ok=True)
        
        # Write the tool code as __init__.py
        # Escape triple quotes in tool_code for embedding
        escaped_code_for_init = tool_code.replace('"""', '\\"\\"\\"\\"')
        with open(os.path.join(src_dir, "__init__.py"), "w") as f:
            f.write(f'"""Auto-generated tool: {class_name}"""\n\n')
            f.write(tool_code)
            f.write(f'\n\n__version__ = "0.1.0"\n')
            f.write(f'TOOL_CODE = """{escaped_code_for_init}"""\n')
        
        # Write __main__.py for auto-install capability
        # When user runs: pip install {package} && python -m {module_name}
        # It copies the tool to workspace/tools/ AND updates registry.json
        
        # Escape the tool code for embedding
        escaped_tool_code = tool_code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"\\"')
        escaped_tags = json.dumps(tags)
        escaped_input_types = json.dumps(input_types)
        escaped_output_types = json.dumps(output_types)
        
        main_code = f'''"""Auto-install script for {class_name}"""
import os
import sys
import json

TOOL_CODE = """{tool_code}"""

TOOL_METADATA = {{
    "class_name": "{class_name}",
    "file_name": "{module_name}.py",
    "description": "{description[:200].replace('"', "'")}",
    "tags": {escaped_tags},
    "input_types": {escaped_input_types},
    "output_types": {escaped_output_types},
    "domain": "{domain}",
    "pypi_package": "{package_name}"
}}

def install():
    """Install this tool to workspace/tools/ folder and update registry.json."""
    # Find workspace/tools directory
    cwd = os.getcwd()
    tools_dir = os.path.join(cwd, "workspace", "tools")
    
    if not os.path.exists(tools_dir):
        # Try to find it relative to agentic_system
        for parent in [cwd] + list(os.path.abspath(cwd).split(os.sep)):
            candidate = os.path.join(parent, "workspace", "tools")
            if os.path.exists(candidate):
                tools_dir = candidate
                break
        else:
            os.makedirs(tools_dir, exist_ok=True)
    
    # Write the tool file
    target_file = os.path.join(tools_dir, TOOL_METADATA["file_name"])
    with open(target_file, "w") as f:
        f.write(TOOL_CODE)
    print(f"[{package_name}] Installed tool to: {{target_file}}")
    
    # Update registry.json
    registry_path = os.path.join(tools_dir, "registry.json")
    
    # Load existing registry or create new
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
        except:
            registry = {{}}
    else:
        registry = {{}}
    
    # Check if tool already exists in registry
    class_name = TOOL_METADATA["class_name"]
    if class_name not in registry:
        registry[class_name] = {{
            "file": TOOL_METADATA["file_name"],
            "description": TOOL_METADATA["description"],
            "tags": TOOL_METADATA["tags"],
            "input_types": TOOL_METADATA["input_types"],
            "output_types": TOOL_METADATA["output_types"],
            "domain": TOOL_METADATA["domain"],
            "pypi_package": TOOL_METADATA["pypi_package"]
        }}
        
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"[{package_name}] Added to registry: {{class_name}}")
    else:
        print(f"[{package_name}] Already in registry: {{class_name}}")
    
    return target_file

if __name__ == "__main__":
    install()
'''
        
        with open(os.path.join(src_dir, "__main__.py"), "w") as f:
            f.write(main_code)
        
        # Create pyproject.toml with console script entry point
        pyproject_content = f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
version = "0.1.0"
description = "{description[:100].replace('"', "'")}"
requires-python = ">=3.8"

[project.scripts]
{package_name}-install = "{module_name}.__main__:install"

[tool.setuptools.packages.find]
where = ["src"]
'''
        
        with open(os.path.join(pkg_dir, "pyproject.toml"), "w") as f:
            f.write(pyproject_content)
        
        # Build the package
        print("[Toolsmith] Building package...")
        try:
            build_result = subprocess.run(
                ["python3", "-m", "build"],
                cwd=pkg_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            if build_result.returncode != 0:
                return False, package_name, f"Build failed: {build_result.stderr}"
        except FileNotFoundError:
            return False, package_name, "Build failed: 'build' module not installed. Run: pip install build"
        except subprocess.TimeoutExpired:
            return False, package_name, "Build timed out"
        
        # Upload to PyPI using twine
        print("[Toolsmith] Uploading to PyPI...")
        import glob
        dist_dir = os.path.join(pkg_dir, "dist")
        dist_files = glob.glob(os.path.join(dist_dir, "*"))
        
        if not dist_files:
            return False, package_name, "Build produced no dist files"
        
        try:
            upload_result = subprocess.run(
                [
                    "python3", "-m", "twine", "upload",
                    "--username", self.pypi_username,
                    "--password", self.pypi_token,
                    "--non-interactive"
                ] + dist_files,
                cwd=pkg_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if upload_result.returncode != 0:
                if "already exists" in upload_result.stderr.lower():
                    return True, package_name, "Package already exists on PyPI"
                return False, package_name, f"Upload failed: {upload_result.stderr}"
            
        except FileNotFoundError:
            return False, package_name, "Upload failed: 'twine' not installed. Run: pip install twine"
        except subprocess.TimeoutExpired:
            return False, package_name, "Upload timed out"
        
        print(f"[Toolsmith] Successfully published to PyPI: {package_name}")
        return True, package_name, f"Published as pip install {package_name}"

    def install_tool(self, package_name: str) -> str:
        """
        Install a tool from PyPI into the workspace/tools folder.
        Downloads the package, extracts the .py file, and places it in tools/.
        Also updates the registry.
        
        Args:
            package_name: The PyPI package name (e.g., 'factorial-tool-ts')
        
        Returns:
            Success or error message
        """
        import zipfile
        import tarfile
        
        module_name = package_name.replace("-", "_")
        tool_file = f"{module_name}.py"
        target_path = os.path.join(self.tools_dir, tool_file)
        
        # Check if already installed
        if os.path.exists(target_path):
            return f"Tool already installed: {tool_file}"
        
        print(f"[Toolsmith] Installing {package_name} from PyPI...")
        
        # Create temp directory for download
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the package using pip
            try:
                download_result = subprocess.run(
                    ["python3", "-m", "pip", "download", package_name, "--no-deps", "-d", tmp_dir],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if download_result.returncode != 0:
                    return f"Download failed: {download_result.stderr}"
                    
            except subprocess.TimeoutExpired:
                return "Download timed out"
            except Exception as e:
                return f"Download error: {e}"
            
            # Find the downloaded file
            downloaded_files = os.listdir(tmp_dir)
            if not downloaded_files:
                return "No package found on PyPI"
            
            pkg_file = os.path.join(tmp_dir, downloaded_files[0])
            
            # Extract the .py file from the package
            tool_code = None
            
            if pkg_file.endswith(".whl"):
                # Wheel file (zip format)
                try:
                    with zipfile.ZipFile(pkg_file, 'r') as zf:
                        for name in zf.namelist():
                            if name.endswith(f"{module_name}.py") or name == f"{module_name}.py":
                                tool_code = zf.read(name).decode('utf-8')
                                break
                except Exception as e:
                    return f"Failed to extract wheel: {e}"
                    
            elif pkg_file.endswith(".tar.gz"):
                # Source distribution
                try:
                    with tarfile.open(pkg_file, 'r:gz') as tf:
                        for member in tf.getmembers():
                            if member.name.endswith(f"{module_name}.py"):
                                f = tf.extractfile(member)
                                if f:
                                    tool_code = f.read().decode('utf-8')
                                break
                except Exception as e:
                    return f"Failed to extract tarball: {e}"
            
            if not tool_code:
                return f"Could not find {module_name}.py in package"
            
            # Write to tools directory
            with open(target_path, "w") as f:
                f.write(tool_code)
            
            print(f"[Toolsmith] Installed tool to: {target_path}")
            
            # Try to extract class name from the code
            class_name = module_name.replace("_", " ").title().replace(" ", "") + "Tool"
            try:
                tree = ast.parse(tool_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and "Tool" in node.name:
                        class_name = node.name
                        break
            except:
                pass
            
            # Update registry
            self._update_registry(
                class_name=class_name,
                file_name=tool_file,
                description=f"Installed from PyPI: {package_name}",
                tags=[module_name],
                pypi_package=package_name
            )
            
            return f"Successfully installed {package_name} as {tool_file}"

    def list_available_tools(self) -> list:
        """
        List all tools in the registry with their PyPI package names.
        """
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            tools = []
            for name, meta in registry.items():
                tools.append({
                    "name": name,
                    "file": meta.get("file", ""),
                    "description": meta.get("description", ""),
                    "tags": meta.get("tags", []),
                    "pypi_package": meta.get("pypi_package", ""),
                    "status": meta.get("status", "active"),  # Default to active for legacy entries
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at")
                })
            return tools
        except Exception as e:
            print(f"[Toolsmith] Failed to list tools: {e}")
            return []
