"""Toolsmith - Generates and manages tools for the IASCIS system."""

import os
import re
import json
import time
import ast
import subprocess
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

from architecture.llm_manager import get_llm_manager
from architecture.gatekeeper import Gatekeeper
from architecture.prompts import get_tool_generator_prompt
from architecture.intent_classifier import (
    IntentClassifier,
    validate_domain,
    DOMAIN_KEYWORDS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

STDLIB_MODULES = {
    "os",
    "sys",
    "re",
    "json",
    "ast",
    "time",
    "math",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "typing",
    "abc",
    "subprocess",
    "tempfile",
    "pathlib",
    "io",
    "csv",
    "random",
    "hashlib",
    "base64",
    "urllib",
    "http",
    "email",
    "html",
    "xml",
    "logging",
    "warnings",
    "copy",
    "pickle",
    "sqlite3",
    "threading",
    "multiprocessing",
    "asyncio",
    "socket",
    "ssl",
    "uuid",
    "platform",
}

IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
}

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "it",
    "to",
    "of",
    "for",
    "and",
    "or",
    "that",
    "this",
    "with",
    "from",
    "as",
    "be",
    "by",
    "on",
    "in",
    "at",
    "was",
    "are",
    "i",
    "me",
    "can",
    "will",
    "do",
    "get",
    "make",
    "use",
    "need",
    "want",
    "create",
    "tool",
}


class Toolsmith:
    """Generates, validates, and manages tools with domain-aware deduplication."""

    def __init__(self, safe_mode: bool = False, gatekeeper: Gatekeeper = None):
        self.safe_mode = safe_mode
        self.gatekeeper = gatekeeper or Gatekeeper(strict_mode=safe_mode)

        self.workspace_root = Path.cwd() / "workspace"
        self.tools_dir = self.workspace_root / "tools"
        self.packages_dir = self.workspace_root / "packages"
        self.registry_path = self.tools_dir / "registry.json"
        self.metrics_path = self.tools_dir / "metrics.json"
        self.tools_source_path = Path.cwd() / "execution" / "tools.py"

        self._intent_classifier = None
        self.pypi_username = os.environ.get("PYPI_USERNAME", "__token__")
        self.pypi_token = os.environ.get("PYPI_TOKEN", "")

        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._write_json(self.registry_path, {})
        if not self.metrics_path.exists():
            self._log_metrics("init", {"message": "Metrics initialized"})

        self._migrate_registry_status()

    def _write_json(self, path: Path, data: dict):
        """Write JSON to file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _read_json(self, path: Path) -> dict:
        """Read JSON from file."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _migrate_registry_status(self):
        """Migrate legacy registry entries to include status/timestamp fields."""
        data = self._read_json(self.registry_path)
        modified = False
        current_time = time.time()

        for meta in data.values():
            if "status" not in meta:
                meta["status"] = "active"
                modified = True
            if "created_at" not in meta:
                meta["created_at"] = current_time
                modified = True
            if "last_used" not in meta:
                meta["last_used"] = None
                modified = True
            if "use_count" not in meta:
                meta["use_count"] = 0
                modified = True
            if "metrics" not in meta:
                meta["metrics"] = 0.0
                modified = True

        if modified:
            self._write_json(self.registry_path, data)
            logger.debug("Registry migrated to include status fields")

    def _get_intent_classifier(self) -> IntentClassifier:
        """Lazy load intent classifier."""
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifier()
        return self._intent_classifier

    def _log_metrics(self, event_type: str, details: dict):
        """Log usage metrics to JSONL file."""
        entry = {"timestamp": time.time(), "event": event_type, **details}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _is_safe_code(self, code: str) -> bool:
        """Validate code safety using Gatekeeper."""
        result = self.gatekeeper.validate(code)
        if not result.is_safe:
            for violation in result.violations:
                logger.warning(f"Safety violation: {violation}")
        return result.is_safe

    def _get_existing_tools_context(self) -> str:
        """Get existing tools code for context injection."""
        try:
            return self.tools_source_path.read_text()
        except Exception as e:
            return f"# Could not read tools.py: {e}"

    def _tokenize(self, text: str) -> set:
        """Tokenize text for similarity matching."""
        clean_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = set(clean_text.split())
        return tokens - STOP_WORDS

    def _filter_tags_for_domain(
        self, tags: list, domain: str, max_tags: int = 4
    ) -> list:
        """Filter tags to valid domain keywords."""
        if not domain or domain not in DOMAIN_KEYWORDS:
            return tags[:max_tags] if tags else []

        valid_keywords = set(DOMAIN_KEYWORDS[domain].keys())
        filtered = [
            t.lower().strip() for t in tags if t.lower().strip() in valid_keywords
        ]

        if not filtered:
            domain_kw = DOMAIN_KEYWORDS[domain]
            sorted_kw = sorted(domain_kw.items(), key=lambda x: x[1], reverse=True)
            filtered = [kw for kw, weight in sorted_kw[:max_tags] if weight >= 0.7]

        seen = set()
        unique = []
        for tag in filtered:
            if tag not in seen:
                seen.add(tag)
                unique.append(tag)
                if len(unique) >= max_tags:
                    break
        return unique

    def create_tool(self, requirement: str) -> str:
        """Generate a new tool based on requirement with domain-aware deduplication."""
        start_time = time.time()
        logger.info(f"Request: '{requirement}'")

        # Domain-aware deduplication
        try:
            registry = self._read_json(self.registry_path)
            classifier = self._get_intent_classifier()
            request_domain, method, confidence = classifier.classify(requirement)
            logger.info(
                f"Domain: '{request_domain}' (method={method}, conf={confidence:.2f})"
            )

            req_tokens = self._tokenize(requirement)
            best_match, highest_score = None, 0.0

            for name, meta in registry.items():
                if meta.get("status") != "active":
                    continue

                tool_domain = meta.get("domain", "")
                if tool_domain != request_domain:
                    continue

                tool_tokens = self._tokenize(meta.get("description", ""))
                for tag in meta.get("tags", []):
                    tool_tokens.update(self._tokenize(tag))

                if not req_tokens or not tool_tokens:
                    continue

                intersection = len(req_tokens & tool_tokens)
                union = len(req_tokens | tool_tokens)
                score = intersection / union if union > 0 else 0

                if score > highest_score:
                    highest_score = score
                    best_match = name

            if best_match and highest_score > 0.5:
                logger.info(f"Dedup match: {best_match} (score={highest_score:.2f})")
                self._log_metrics(
                    "deduplication_hit",
                    {
                        "request": requirement,
                        "matched_tool": best_match,
                        "score": highest_score,
                        "domain": request_domain,
                    },
                )

                # Update usage tracking
                registry[best_match]["use_count"] = (
                    registry[best_match].get("use_count", 0) + 1
                )
                registry[best_match]["last_used"] = time.time()
                self._write_json(self.registry_path, registry)

                return (
                    f"Matched existing tool: {best_match} (score: {highest_score:.2f})"
                )

        except Exception as e:
            logger.warning(f"Deduplication check failed: {e}")

        # Generate new tool
        logger.info("No match found. Generating with LLM...")
        try:
            llm = get_llm_manager()
            existing_code = self._get_existing_tools_context()
            system_prompt = get_tool_generator_prompt(existing_code)

            response = llm.generate_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a tool for: {requirement}"},
                ],
                max_tokens=4096,
            )

            if response.get("error"):
                return f"Tool creation failed: {response['error']}"

            tool_data = json.loads(response["content"])
            class_name = tool_data["class_name"]
            file_name = tool_data["filename"]
            tags = tool_data.get("tags", [])
            input_types = tool_data.get("input_types", [])
            output_types = tool_data.get("output_types", [])
            raw_domain = tool_data.get("domain", "")
            tool_code = tool_data["code"]

            # Validate domain - validate_domain returns (domain, is_valid) tuple
            validated = validate_domain(raw_domain)
            domain = validated[0] if validated[0] else request_domain
            logger.info(f"Domain validated: {domain}")

            # Filter tags
            tags = self._filter_tags_for_domain(tags, domain)
            logger.info(f"Tags: {tags}")

            # Safety check
            if self.safe_mode and not self._is_safe_code(tool_code):
                self._log_metrics("safety_rejection", {"request": requirement})
                return "Tool rejected: Safety violation detected"

            # Write tool file
            file_path = self.tools_dir / file_name
            with open(file_path, "w") as f:
                f.write(tool_code)
            logger.info(f"Tool saved: {file_path}")

            # Detect dependencies
            dependencies = self._detect_dependencies(tool_code)
            llm_deps = tool_data.get("dependencies", [])
            dependencies = list(set(dependencies) | set(llm_deps))
            logger.info(f"Dependencies: {dependencies}")

            # PyPI publishing (optional)
            pypi_package = ""
            pypi_success, pkg_name, pypi_msg = self._publish_to_pypi(
                class_name,
                tool_code,
                requirement,
                dependencies,
                tags,
                input_types,
                output_types,
                domain,
            )
            if pypi_success:
                pypi_package = pkg_name
                logger.info(f"PyPI: {pypi_msg}")
            else:
                logger.debug(f"PyPI publishing skipped: {pypi_msg}")

            # Update registry
            self._update_registry(
                class_name,
                file_name,
                requirement,
                tags,
                input_types,
                output_types,
                domain,
                pypi_package,
            )

            self._log_metrics(
                "tool_created",
                {
                    "request": requirement,
                    "tool_name": class_name,
                    "pypi_package": pypi_package,
                    "latency": time.time() - start_time,
                },
            )

            result_msg = f"Created {class_name} with tags {tags}"
            if pypi_package:
                result_msg += f". PyPI: pip install {pypi_package}"
            return result_msg

        except Exception as e:
            logger.error(f"Tool creation failed: {e}", exc_info=True)
            self._log_metrics(
                "generation_failed", {"request": requirement, "error": str(e)}
            )
            return f"Tool creation failed: {e}"

    def _update_registry(
        self,
        class_name: str,
        file_name: str,
        description: str,
        tags: list,
        input_types: list = None,
        output_types: list = None,
        domain: str = None,
        pypi_package: str = None,
        status: str = "active",
    ):
        """Update or create registry entry for a tool."""
        data = self._read_json(self.registry_path)
        existing = data.get(class_name, {})

        data[class_name] = {
            "file": file_name,
            "description": description,
            "tags": tags,
            "input_types": input_types or [],
            "output_types": output_types or [],
            "domain": domain or "",
            "pypi_package": pypi_package or "",
            "status": status,
            "created_at": existing.get("created_at", time.time()),
            "last_used": existing.get("last_used"),
            "use_count": existing.get("use_count", 0),
            "metrics": existing.get("metrics", 0.0),
        }

        self._write_json(self.registry_path, data)
        logger.info(f"Registry updated: {class_name}")

    def update_tool_status(self, tool_name: str, status: str) -> bool:
        """Update tool status (active/inactive/deprecated/failed)."""
        valid_statuses = {"active", "inactive", "deprecated", "failed"}
        if status not in valid_statuses:
            logger.warning(f"Invalid status: {status}")
            return False

        data = self._read_json(self.registry_path)
        if tool_name not in data:
            logger.warning(f"Tool not found: {tool_name}")
            return False

        old_status = data[tool_name].get("status", "unknown")
        data[tool_name]["status"] = status
        self._write_json(self.registry_path, data)

        self._log_metrics(
            "status_change",
            {
                "tool": tool_name,
                "old_status": old_status,
                "new_status": status,
            },
        )
        logger.info(f"Status updated: {tool_name} {old_status} -> {status}")
        return True

    def get_tool_status(self, tool_name: str) -> str:
        """Get current status of a tool."""
        data = self._read_json(self.registry_path)
        return data.get(tool_name, {}).get("status", "unknown")

    def get_tools_by_status(self, status: str) -> list:
        """Get all tools with a specific status."""
        data = self._read_json(self.registry_path)
        return [name for name, meta in data.items() if meta.get("status") == status]

    def _detect_dependencies(self, code: str) -> list:
        """Parse code to detect third-party imports."""
        dependencies = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module not in STDLIB_MODULES:
                            pkg = IMPORT_TO_PACKAGE.get(module, module)
                            dependencies.add(pkg)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if module not in STDLIB_MODULES:
                        pkg = IMPORT_TO_PACKAGE.get(module, module)
                        dependencies.add(pkg)
        except Exception:
            pass

        dependencies.add("pydantic")
        return list(dependencies)

    def _check_pypi_name(self, name: str) -> bool:
        """Check if PyPI package name is available."""
        if not requests:
            return True
        try:
            response = requests.get(f"https://pypi.org/pypi/{name}/json", timeout=5)
            return response.status_code == 404
        except Exception:
            return True

    def _get_available_pypi_name(self, base_name: str) -> str:
        """Find available PyPI package name."""
        name = re.sub(r"[^a-z0-9]+", "-", base_name.lower()).strip("-")
        if self._check_pypi_name(name):
            return name

        for suffix in ["tool", "ai", "llm", "auto"]:
            candidate = f"{name}-{suffix}"
            if self._check_pypi_name(candidate):
                return candidate

        return f"{name}-{int(time.time()) % 10000}"

    def _publish_to_pypi(
        self,
        class_name: str,
        code: str,
        description: str,
        dependencies: list,
        tags: list,
        input_types: list,
        output_types: list,
        domain: str,
    ) -> tuple:
        """Publish tool as PyPI package. Returns (success, package_name, message)."""
        if not self.pypi_token:
            return False, "", "No PyPI token configured"

        package_name = self._get_available_pypi_name(class_name)
        module_name = package_name.replace("-", "_")

        pkg_dir = self.packages_dir / package_name
        src_dir = pkg_dir / "src" / module_name

        try:
            src_dir.mkdir(parents=True, exist_ok=True)

            (src_dir / "__init__.py").write_text(code)
            (src_dir / f"{module_name}.py").write_text(code)

            # pyproject.toml
            pyproject = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{package_name}"
version = "0.1.0"
description = "{description[:100]}"
readme = "README.md"
requires-python = ">=3.8"
dependencies = {json.dumps(dependencies)}
keywords = {json.dumps(tags[:5])}

[project.urls]
Homepage = "https://github.com/iascis/tools"
'''
            (pkg_dir / "pyproject.toml").write_text(pyproject)
            (pkg_dir / "README.md").write_text(f"# {class_name}\n\n{description}\n")

            # Build and upload
            result = subprocess.run(
                ["python", "-m", "build"],
                cwd=str(pkg_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, "", f"Build failed: {result.stderr[:200]}"

            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "twine",
                    "upload",
                    "--skip-existing",
                    "dist/*",
                    "-u",
                    self.pypi_username,
                    "-p",
                    self.pypi_token,
                ],
                cwd=str(pkg_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, "", f"Upload failed: {result.stderr[:200]}"

            return True, package_name, f"Published: {package_name}"

        except Exception as e:
            return False, "", str(e)

    def install_from_pypi(self, package_name: str) -> str:
        """Install a tool from PyPI into the tools directory."""
        import zipfile
        import tarfile

        module_name = package_name.replace("-", "_")
        tool_file = f"{module_name}_tool.py"
        target_path = self.tools_dir / tool_file

        if target_path.exists():
            return f"Tool already exists: {tool_file}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                ["pip", "download", "--no-deps", "-d", tmp_dir, package_name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return f"Download failed: {result.stderr[:200]}"

            downloaded = os.listdir(tmp_dir)
            if not downloaded:
                return "No package found on PyPI"

            pkg_file = os.path.join(tmp_dir, downloaded[0])
            tool_code = None

            if pkg_file.endswith(".whl"):
                try:
                    with zipfile.ZipFile(pkg_file, "r") as zf:
                        for name in zf.namelist():
                            if name.endswith(f"{module_name}.py"):
                                tool_code = zf.read(name).decode("utf-8")
                                break
                except Exception as e:
                    return f"Failed to extract wheel: {e}"

            elif pkg_file.endswith(".tar.gz"):
                try:
                    with tarfile.open(pkg_file, "r:gz") as tf:
                        for member in tf.getmembers():
                            if member.name.endswith(f"{module_name}.py"):
                                f = tf.extractfile(member)
                                if f:
                                    tool_code = f.read().decode("utf-8")
                                break
                except Exception as e:
                    return f"Failed to extract tarball: {e}"

            if not tool_code:
                return f"Could not find {module_name}.py in package"

            target_path.write_text(tool_code)
            logger.info(f"Installed: {target_path}")

            # Extract class name
            class_name = module_name.replace("_", " ").title().replace(" ", "") + "Tool"
            try:
                tree = ast.parse(tool_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and "Tool" in node.name:
                        class_name = node.name
                        break
            except Exception:
                pass

            self._update_registry(
                class_name=class_name,
                file_name=tool_file,
                description=f"Installed from PyPI: {package_name}",
                tags=[module_name],
                pypi_package=package_name,
            )

            return f"Installed {package_name} as {tool_file}"

    def list_available_tools(self) -> list:
        """List all tools in registry."""
        data = self._read_json(self.registry_path)
        return [
            {
                "name": name,
                "file": meta.get("file", ""),
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "status": meta.get("status", "active"),
                "pypi_package": meta.get("pypi_package", ""),
            }
            for name, meta in data.items()
        ]

    def get_tool_creation_timestamps(self) -> dict:
        """Get creation timestamps for all tools."""
        data = self._read_json(self.registry_path)
        return {
            name: meta["created_at"]
            for name, meta in data.items()
            if "created_at" in meta
        }

    def get_tool_last_used_timestamps(self) -> dict:
        """Get last used timestamps for all tools."""
        result = {}
        try:
            with open(self.metrics_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("event") in ("tool_created", "deduplication_hit"):
                        tool_name = entry.get("tool_name") or entry.get("matched_tool")
                        if tool_name:
                            result[tool_name] = max(
                                result.get(tool_name, 0), entry.get("timestamp", 0)
                            )
        except Exception:
            pass
        return result

    def update_metrics_from_profiles(self, logs_dir: str = None) -> dict:
        """Update registry metrics from profile logs."""
        logs_dir = logs_dir or str(Path.cwd() / "logs")
        tool_metrics = {}

        # Find profile files
        profile_files = []
        for root, _, files in os.walk(logs_dir):
            for f in files:
                if f.startswith("profiles_") and f.endswith(".json"):
                    profile_files.append(os.path.join(root, f))

        if not profile_files:
            logger.debug("No profile files found")
            return tool_metrics

        # Aggregate metrics
        tool_raw = {}
        for pf in profile_files:
            try:
                with open(pf, "r") as f:
                    data = json.load(f)
                for profile in data.get("raw_profiles", []):
                    tool_name = profile.get("tool_name")
                    if not tool_name:
                        continue

                    if tool_name not in tool_raw:
                        tool_raw[tool_name] = {"time": [], "memory": [], "success": []}

                    tool_raw[tool_name]["time"].append(
                        profile.get("execution_time_ms", 0)
                    )
                    tool_raw[tool_name]["memory"].append(
                        profile.get("peak_memory_mb", 0)
                    )
                    tool_raw[tool_name]["success"].append(
                        1.0 if profile.get("success") else 0.0
                    )
            except Exception:
                continue

        # Calculate weighted scores
        for tool_name, metrics in tool_raw.items():
            avg_time = (
                sum(metrics["time"]) / len(metrics["time"]) if metrics["time"] else 0
            )
            avg_memory = (
                sum(metrics["memory"]) / len(metrics["memory"])
                if metrics["memory"]
                else 0
            )
            avg_success = (
                sum(metrics["success"]) / len(metrics["success"])
                if metrics["success"]
                else 0
            )

            time_score = max(0, 1 - avg_time / 1000)
            memory_score = max(0, 1 - avg_memory / 100)

            weighted = time_score * 0.4 + memory_score * 0.3 + avg_success * 0.3
            tool_metrics[tool_name] = round(weighted, 4)

        # Update registry
        if tool_metrics:
            registry = self._read_json(self.registry_path)
            for tool_name, score in tool_metrics.items():
                if tool_name in registry:
                    registry[tool_name]["metrics"] = score
            self._write_json(self.registry_path, registry)
            logger.info(f"Metrics updated for {len(tool_metrics)} tools")

        return tool_metrics
