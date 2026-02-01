"""Reflector for execution log analysis and self-correction feedback."""

import re
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class ErrorCategory(Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    IMPORT = "import"
    TYPE = "type"
    VALUE = "value"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    NETWORK = "network"
    FILE_NOT_FOUND = "file_not_found"
    UNKNOWN = "unknown"


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0


@dataclass
class Diagnosis:
    category: ErrorCategory
    root_cause: str
    suggestion: str
    confidence: float
    context: dict = field(default_factory=dict)


@dataclass
class ReflectionResult:
    diagnosis: Diagnosis
    corrective_prompt: str
    should_retry: bool
    max_retries_remaining: int


class ErrorClassifier:
    """Classifies errors into categories based on patterns."""

    PATTERNS: dict[ErrorCategory, list[str]] = {
        ErrorCategory.SYNTAX: [
            r"SyntaxError",
            r"IndentationError",
            r"TabError",
            r"invalid syntax",
        ],
        ErrorCategory.IMPORT: [
            r"ImportError",
            r"ModuleNotFoundError",
            r"No module named",
        ],
        ErrorCategory.TYPE: [
            r"TypeError",
            r"expected .* but got",
            r"unsupported operand type",
        ],
        ErrorCategory.VALUE: [
            r"ValueError",
            r"invalid literal",
            r"could not convert",
        ],
        ErrorCategory.PERMISSION: [
            r"PermissionError",
            r"Permission denied",
            r"Access is denied",
        ],
        ErrorCategory.TIMEOUT: [
            r"TimeoutError",
            r"timed out",
            r"deadline exceeded",
        ],
        ErrorCategory.MEMORY: [
            r"MemoryError",
            r"out of memory",
            r"Cannot allocate memory",
        ],
        ErrorCategory.NETWORK: [
            r"ConnectionError",
            r"ConnectionRefusedError",
            r"URLError",
            r"socket.error",
            r"Connection refused",
        ],
        ErrorCategory.FILE_NOT_FOUND: [
            r"FileNotFoundError",
            r"No such file or directory",
            r"cannot find the path",
        ],
        ErrorCategory.RUNTIME: [
            r"RuntimeError",
            r"Exception",
            r"Error",
        ],
    }

    def classify(self, error_text: str) -> ErrorCategory:
        """Classify error text into a category."""
        for category, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_text, re.IGNORECASE):
                    return category
        return ErrorCategory.UNKNOWN


class DiagnosisEngine:
    """Generates root cause analysis and correction suggestions."""

    SUGGESTIONS: dict[ErrorCategory, tuple[str, str]] = {
        ErrorCategory.SYNTAX: (
            "Code contains syntax errors",
            "Fix the syntax error at the indicated line. Check for missing colons, parentheses, or incorrect indentation.",
        ),
        ErrorCategory.IMPORT: (
            "Required module is not available",
            "Install the missing module or use an alternative that is available. Check module name spelling.",
        ),
        ErrorCategory.TYPE: (
            "Type mismatch in operation",
            "Check the types of variables being used. Add type conversion if needed.",
        ),
        ErrorCategory.VALUE: (
            "Invalid value provided",
            "Validate input values before using them. Add input validation or use default values.",
        ),
        ErrorCategory.PERMISSION: (
            "Insufficient permissions for operation",
            "Check file/directory permissions. Use a different path or request elevated permissions.",
        ),
        ErrorCategory.TIMEOUT: (
            "Operation exceeded time limit",
            "Optimize the operation or increase timeout. Consider breaking into smaller operations.",
        ),
        ErrorCategory.MEMORY: (
            "Insufficient memory available",
            "Reduce memory usage by processing data in chunks or using generators.",
        ),
        ErrorCategory.NETWORK: (
            "Network connection failed",
            "Check network connectivity. Verify the URL/host is correct. Add retry logic.",
        ),
        ErrorCategory.FILE_NOT_FOUND: (
            "File or directory does not exist",
            "Verify the path exists. Create the file/directory first or use the correct path.",
        ),
        ErrorCategory.RUNTIME: (
            "Runtime error occurred",
            "Review the error message and traceback. Add error handling around the failing code.",
        ),
        ErrorCategory.UNKNOWN: (
            "Unclassified error",
            "Review the full error output. Check documentation for the failing operation.",
        ),
    }

    def diagnose(self, category: ErrorCategory, error_text: str) -> Diagnosis:
        """Generate diagnosis with root cause and suggestion."""
        root_cause, suggestion = self.SUGGESTIONS.get(
            category, self.SUGGESTIONS[ErrorCategory.UNKNOWN]
        )

        line_match = re.search(r"line (\d+)", error_text)
        file_match = re.search(r'File "([^"]+)"', error_text)

        context = {}
        if line_match:
            context["line"] = int(line_match.group(1))
        if file_match:
            context["file"] = file_match.group(1)

        confidence = 0.9 if category != ErrorCategory.UNKNOWN else 0.3

        return Diagnosis(
            category=category,
            root_cause=root_cause,
            suggestion=suggestion,
            confidence=confidence,
            context=context,
        )


class FeedbackMemory:
    """Stores feedback from past failures for learning."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.history: list[tuple[str, Diagnosis, bool]] = []

    def record(self, error_signature: str, diagnosis: Diagnosis, was_resolved: bool):
        """Record a failure and whether it was resolved."""
        self.history.append((error_signature, diagnosis, was_resolved))
        if len(self.history) > self.max_size:
            self.history.pop(0)
        logger.debug(
            f"Recorded feedback: {error_signature[:50]}... resolved={was_resolved}"
        )

    def get_similar_failures(
        self, error_signature: str
    ) -> list[tuple[Diagnosis, bool]]:
        """Find similar past failures."""
        similar = []
        for sig, diag, resolved in self.history:
            if self._is_similar(sig, error_signature):
                similar.append((diag, resolved))
        return similar

    def _is_similar(self, sig1: str, sig2: str) -> bool:
        words1 = set(sig1.lower().split())
        words2 = set(sig2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / len(words1 | words2)
        return overlap > 0.5

    def get_success_rate(self, category: ErrorCategory) -> float:
        """Get resolution success rate for a category."""
        relevant = [(d, r) for _, d, r in self.history if d.category == category]
        if not relevant:
            return 0.0
        return sum(1 for _, r in relevant if r) / len(relevant)


class Reflector:
    """
    Analyzes execution failures and generates corrective feedback.

    Components:
    - ErrorClassifier: Categorizes errors
    - DiagnosisEngine: Generates root cause and suggestions
    - FeedbackMemory: Learns from past failures
    """

    DEFAULT_MAX_RETRIES = 3

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        self.classifier = ErrorClassifier()
        self.diagnosis_engine = DiagnosisEngine()
        self.memory = FeedbackMemory()
        self.retry_counts: dict[str, int] = {}
        self.max_retries = max_retries
        logger.info(f"Reflector initialized (max_retries={max_retries})")

    def reflect(self, result: ExecutionResult, code: str = "") -> ReflectionResult:
        """Analyze execution result and generate correction feedback."""
        if result.success:
            return ReflectionResult(
                diagnosis=Diagnosis(
                    category=ErrorCategory.UNKNOWN,
                    root_cause="No error",
                    suggestion="",
                    confidence=1.0,
                ),
                corrective_prompt="",
                should_retry=False,
                max_retries_remaining=self.max_retries,
            )

        error_text = result.error or result.output
        error_sig = self._get_error_signature(error_text)

        category = self.classifier.classify(error_text)
        diagnosis = self.diagnosis_engine.diagnose(category, error_text)

        similar = self.memory.get_similar_failures(error_sig)
        if similar:
            success_rate = sum(1 for _, r in similar if r) / len(similar)
            diagnosis.confidence *= success_rate if success_rate > 0 else 0.5

        retries_used = self.retry_counts.get(error_sig, 0)
        retries_remaining = max(0, self.max_retries - retries_used)
        self.retry_counts[error_sig] = retries_used + 1

        should_retry = retries_remaining > 0 and category not in [
            ErrorCategory.PERMISSION,
            ErrorCategory.MEMORY,
        ]

        corrective_prompt = self._build_corrective_prompt(diagnosis, error_text, code)

        logger.info(
            f"Reflected: {category.value}, retry={should_retry}, remaining={retries_remaining}"
        )

        return ReflectionResult(
            diagnosis=diagnosis,
            corrective_prompt=corrective_prompt,
            should_retry=should_retry,
            max_retries_remaining=retries_remaining,
        )

    def record_outcome(self, error_text: str, diagnosis: Diagnosis, was_resolved: bool):
        """Record whether a correction attempt succeeded."""
        error_sig = self._get_error_signature(error_text)
        self.memory.record(error_sig, diagnosis, was_resolved)
        if was_resolved:
            self.retry_counts.pop(error_sig, None)

    def _get_error_signature(self, error_text: str) -> str:
        lines = error_text.strip().split("\n")
        if lines:
            return lines[-1][:200]
        return error_text[:200]

    def _build_corrective_prompt(
        self, diagnosis: Diagnosis, error_text: str, code: str
    ) -> str:
        prompt_parts = [
            f"The previous code failed with a {diagnosis.category.value} error.",
            f"Root cause: {diagnosis.root_cause}",
            f"Suggestion: {diagnosis.suggestion}",
        ]

        if diagnosis.context.get("line"):
            prompt_parts.append(f"Error occurred at line {diagnosis.context['line']}.")

        prompt_parts.append(f"\nError output:\n```\n{error_text[:500]}\n```")

        if code:
            prompt_parts.append(f"\nFailing code:\n```python\n{code[:1000]}\n```")

        prompt_parts.append("\nPlease fix the code and try again.")

        return "\n".join(prompt_parts)

    def reset(self):
        """Reset retry counts for new session."""
        self.retry_counts.clear()

    def __call__(self, result: ExecutionResult, code: str = "") -> ReflectionResult:
        return self.reflect(result, code)
