"""Pipeline State Management - Tracks data flow between execution steps."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StepResult:
    """Result from a single execution step with schema information."""

    output: Any
    success: bool
    output_schema: dict = field(default_factory=dict)

    @classmethod
    def from_output(cls, output: Any, success: bool) -> "StepResult":
        """Create StepResult and auto-infer schema from output."""
        schema = cls._infer_schema(output)
        return cls(output=output, success=success, output_schema=schema)

    @staticmethod
    def _infer_schema(output: Any) -> dict:
        """Infer a simple schema from output data."""
        if output is None:
            return {"type": "null"}

        if isinstance(output, str):
            try:
                parsed = json.loads(output)
                return StepResult._infer_schema(parsed)
            except (json.JSONDecodeError, TypeError):
                return {"type": "string"}

        if isinstance(output, list):
            if not output:
                return {"type": "list", "items": "unknown"}
            first_item = output[0]
            if isinstance(first_item, str):
                if first_item.startswith("http"):
                    return {"type": "list", "items": "url", "count": len(output)}
                return {"type": "list", "items": "string", "count": len(output)}
            if isinstance(first_item, dict):
                return {
                    "type": "list",
                    "items": "object",
                    "keys": list(first_item.keys())[:5],
                    "count": len(output),
                }
            return {"type": "list", "items": type(first_item).__name__}

        if isinstance(output, dict):
            return {
                "type": "object",
                "keys": list(output.keys())[:10],
                "has_results": "results" in output,
                "has_urls": any(k in output for k in ["urls", "url", "links", "link"]),
                "has_data": "data" in output,
            }

        return {"type": type(output).__name__}


@dataclass
class PipelineState:
    """Tracks state across pipeline execution."""

    original_query: str
    step_results: dict = field(default_factory=dict)
    current_step: int = 0

    def add_result(self, step_number: int, output: Any, success: bool) -> None:
        """Add a step result with auto-inferred schema."""
        self.step_results[step_number] = StepResult.from_output(output, success)
        self.current_step = step_number

    def get_result(self, step_number: int) -> Optional[StepResult]:
        """Get result from a specific step."""
        return self.step_results.get(step_number)

    def get_previous_output(self, step_number: int) -> tuple[Any, dict]:
        """Get previous step's output and schema for pipeline context."""
        if step_number <= 1:
            return None, {}
        prev_result = self.step_results.get(step_number - 1)
        if prev_result:
            return prev_result.output, prev_result.output_schema
        return None, {}


class DataTransformer:
    """Transforms data between pipeline steps based on schemas."""

    FIELD_MAPPINGS = {
        "url": ["urls", "link", "links", "href", "source_url"],
        "query": ["search_query", "q", "search_term", "keyword"],
        "data": ["results", "items", "records", "response"],
        "text": ["content", "body", "description", "summary"],
    }

    def transform(
        self,
        source_output: Any,
        source_schema: dict,
        target_arg_name: str,
        target_type: str = None,
    ) -> Optional[Any]:
        """Transform source output to match target arg requirements."""
        if source_output is None:
            return None

        parsed = self._parse_output(source_output)
        if parsed is None:
            return None

        if target_arg_name == "data":
            return self._extract_data(parsed, source_schema)

        if target_arg_name in ["url", "link", "source_url", "target_url"]:
            return self._extract_url(parsed, source_schema)

        mapped = self._try_field_mapping(parsed, target_arg_name)
        if mapped is not None:
            return mapped

        if source_schema.get("type") == "list" and target_type != "list":
            if isinstance(parsed, list) and parsed:
                return parsed[0]

        return None

    def _parse_output(self, output: Any) -> Any:
        """Parse output to dict/list if it's a JSON string."""
        if isinstance(output, str):
            try:
                return json.loads(output)
            except (json.JSONDecodeError, TypeError):
                return output
        return output

    def _extract_data(self, parsed: Any, schema: dict) -> Any:
        """Extract data payload from structured output."""
        if isinstance(parsed, dict):
            for key in ["results", "data", "items", "records", "response"]:
                if key in parsed:
                    return parsed[key]
            return parsed
        return parsed

    def _extract_url(self, parsed: Any, schema: dict) -> Optional[str]:
        """Extract a URL from various output formats."""
        if isinstance(parsed, str) and parsed.startswith("http"):
            return parsed

        if isinstance(parsed, list):
            for item in parsed:
                url = self._extract_url(item, {})
                if url:
                    return url

        if isinstance(parsed, dict):
            for key in ["url", "urls", "link", "links", "href", "source"]:
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, str) and val.startswith("http"):
                        return val
                    if isinstance(val, list) and val:
                        first = val[0]
                        if isinstance(first, str) and first.startswith("http"):
                            return first
                        if isinstance(first, dict):
                            return self._extract_url(first, {})

            if "results" in parsed and isinstance(parsed["results"], list):
                for result in parsed["results"]:
                    url = self._extract_url(result, {})
                    if url:
                        return url

        url_pattern = r'https?://[^\s<>"\'}\])]+'
        if isinstance(parsed, str):
            match = re.search(url_pattern, parsed)
            if match:
                return match.group(0)

        return None

    def _try_field_mapping(self, parsed: Any, target_arg: str) -> Optional[Any]:
        """Try to find matching field using known mappings."""
        if not isinstance(parsed, dict):
            return None

        if target_arg in parsed:
            return parsed[target_arg]

        source_fields = self.FIELD_MAPPINGS.get(target_arg, [])
        for source_field in source_fields:
            if source_field in parsed:
                val = parsed[source_field]
                if isinstance(val, list) and val:
                    return val[0]
                return val

        return None
