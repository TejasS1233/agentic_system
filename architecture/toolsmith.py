import os
import json
import time
from pathlib import Path
from architecture.llm_manager import get_llm_manager
from architecture.gatekeeper import Gatekeeper
from architecture.prompts import get_tool_generator_prompt
from architecture.api_registry import format_all_sources_for_prompt
from architecture.sandbox import Sandbox
from utils.logger import get_logger

logger = get_logger(__name__)


class Toolsmith:
    """Generates, validates, and manages tools with domain-aware deduplication."""

    def __init__(
        self,
        safe_mode: bool = False,
        gatekeeper: Gatekeeper = None,
        sandbox: Sandbox = None,
        max_retries: int = 2,
    ):
        self.safe_mode = safe_mode
        self.gatekeeper = gatekeeper or Gatekeeper(strict_mode=safe_mode)
        self.max_retries = max_retries

        self.workspace_root = Path.cwd() / "workspace"
        self.tools_dir = self.workspace_root / "tools"
        self.registry_path = self.tools_dir / "registry.json"
        self.metrics_path = self.tools_dir / "metrics.json"
        self.tools_source_path = Path.cwd() / "execution" / "tools.py"

        # Sandbox for verification
        self.sandbox = sandbox

        self.tools_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._write_json(self.registry_path, {})
        if not self.metrics_path.exists():
            self._write_json(self.metrics_path, {})

        self._migrate_registry_status()

    def _write_json(self, path: Path, data: dict):
        """Write JSON to file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _read_json(self, path: Path) -> dict:
        """Read JSON from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
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
            # logger.debug("Registry migrated to include status fields")

    def _log_metrics(self, event_type: str, details: dict):
        """Log usage metrics to JSONL file."""
        entry = {"timestamp": time.time(), "event": event_type, **details}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _is_safe_code(self, code: str) -> bool:
        """Validate code safety using Gatekeeper."""
        result = self.gatekeeper.validate(code)
        if not result.is_safe:
            for violation in result.violations:
                # logger.warning(f"Safety violation: {violation}")
                pass
        return result.is_safe

    def _get_existing_tools_context(self) -> str:
        """Get existing tools code for context injection."""
        try:
            return self.tools_source_path.read_text()
        except Exception as e:
            return f"# Could not read tools.py: {e}"

    def create_tool(self, requirement: str) -> str:
        """Generate a new tool based on requirement with verification."""
        start_time = time.time()
        logger.info(f"Request: '{requirement}'")

        llm = get_llm_manager()
        existing_code = self._get_existing_tools_context()

        # Search for relevant free APIs and scrapable URLs based on the requirement
        available_apis = format_all_sources_for_prompt(
            requirement, api_limit=4, url_limit=2
        )
        system_prompt = get_tool_generator_prompt(
            existing_code, available_apis=available_apis
        )

        last_error = None

        for attempt in range(self.max_retries + 1):
            logger.info(
                f"Generating tool with LLM (attempt {attempt + 1}/{self.max_retries + 1})..."
            )
            # Log APIs for debug
            if attempt == 0:
                logger.info(f"Available APIs context length: {len(available_apis)}")

            try:
                # Build user message with error feedback if retrying
                if attempt == 0:
                    user_message = f"Create a tool for: {requirement}"
                else:
                    user_message = f"""Create a tool for: {requirement}

PREVIOUS ATTEMPT FAILED with error:
{last_error}

Fix the code to resolve this error."""

                response = llm.generate_json(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=4096,
                )

                if response.get("error"):
                    err_msg = response["error"]
                    logger.warning(f"LLM generation error: {err_msg}")

                    # Rate limit backoff
                    if "429" in str(err_msg) or "Rate limit" in str(err_msg):
                        logger.warning("Rate limit hit. Sleeping for 5 seconds...")
                        time.sleep(5)

                    last_error = err_msg
                    continue

                tool_data = json.loads(response["content"])
                class_name = tool_data["class_name"]
                file_name = tool_data["filename"]
                tags = tool_data.get("tags", [])
                input_types = tool_data.get("input_types", [])
                output_types = tool_data.get("output_types", [])
                domain = tool_data.get("domain", "")
                tool_code = tool_data["code"]

                # Safety check
                if self.safe_mode and not self._is_safe_code(tool_code):
                    self._log_metrics("safety_rejection", {"request": requirement})
                    return "Tool rejected: Safety violation detected"

                # Write tool file
                file_path = self.tools_dir / file_name
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(tool_code)
                logger.info(f"Tool saved: {file_path}")

                # Verify tool with sandbox
                if self.sandbox:
                    test_result = self.sandbox.run_tool_test(file_name)
                    if not test_result["success"]:
                        last_error = test_result["error"]
                        logger.warning(
                            f"Tool test failed (attempt {attempt + 1}): {last_error}"
                        )
                        self._log_metrics(
                            "tool_test_failed",
                            {
                                "request": requirement,
                                "tool_name": class_name,
                                "attempt": attempt + 1,
                                "error": last_error[:500],
                            },
                        )
                        continue  # Retry
                    logger.info(f"Tool test passed: {class_name}")

                # Update registry
                self._update_registry(
                    class_name,
                    file_name,
                    requirement,
                    tags,
                    input_types,
                    output_types,
                    domain,
                )

                self._log_metrics(
                    "tool_created",
                    {
                        "request": requirement,
                        "tool_name": class_name,
                        "latency": time.time() - start_time,
                        "attempts": attempt + 1,
                    },
                )

                result_msg = f"Created {class_name} with tags {tags}"
                return result_msg

            except Exception as e:
                logger.error(f"Tool creation failed: {e}", exc_info=True)
                last_error = str(e)
                if "429" in str(e) or "Rate limit" in str(e):
                    logger.warning("Rate limit exception. Sleeping for 5 seconds...")
                    time.sleep(5)
                continue

        # All retries exhausted
        self._log_metrics(
            "generation_failed", {"request": requirement, "error": str(last_error)}
        )
        return (
            f"Tool creation failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _update_registry(
        self,
        class_name: str,
        file_name: str,
        description: str,
        tags: list,
        input_types: list = None,
        output_types: list = None,
        domain: str = None,
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
            with open(self.metrics_path, "r", encoding="utf-8") as f:
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
                with open(pf, "r", encoding="utf-8") as f:
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
