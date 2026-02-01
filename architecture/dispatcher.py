"""Dispatcher for privacy-aware task routing between public and private zones."""

from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class Zone(Enum):
    PUBLIC = "public"
    PRIVATE = "private"


@dataclass
class RoutingDecision:
    zone: Zone
    reason: str
    matched_pattern: str = ""


class Dispatcher:
    """Routes tasks to appropriate execution zones based on data sensitivity."""

    SENSITIVE_KEYWORDS: set[str] = {
        "secret",
        "key",
        "password",
        "private",
        "sensitive",
        "bearer",
        "token",
        "credential",
        "api_key",
        "auth",
        "oauth",
        "jwt",
        "ssn",
        "credit_card",
    }

    SENSITIVE_FILE_PATTERNS: set[str] = {
        ".env",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        ".pem",
        ".key",
        "credentials",
        "secrets",
        ".htpasswd",
        "shadow",
        "passwd",
    }

    def __init__(self):
        logger.info("Dispatcher initialized")

    def route(self, task_description: str, file_context: list[str] = None) -> str:
        """Route task to appropriate zone. Returns 'public' or 'private'."""
        decision = self.analyze(task_description, file_context or [])
        logger.info(f"Routing decision: {decision.zone.value} ({decision.reason})")
        return decision.zone.value

    def analyze(
        self, task_description: str, file_context: list[str]
    ) -> RoutingDecision:
        """Analyze task and return detailed routing decision."""
        task_lower = task_description.lower()

        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in task_lower:
                return RoutingDecision(
                    zone=Zone.PRIVATE,
                    reason="Sensitive keyword in task",
                    matched_pattern=keyword,
                )

        for filepath in file_context:
            filepath_lower = filepath.lower()
            for pattern in self.SENSITIVE_FILE_PATTERNS:
                if pattern in filepath_lower:
                    return RoutingDecision(
                        zone=Zone.PRIVATE,
                        reason="Sensitive file in context",
                        matched_pattern=f"{pattern} in {filepath}",
                    )

        return RoutingDecision(
            zone=Zone.PUBLIC,
            reason="No sensitivity detected",
        )

    def is_sensitive(self, text: str) -> bool:
        """Quick check if text contains sensitive patterns."""
        text_lower = text.lower()
        return any(k in text_lower for k in self.SENSITIVE_KEYWORDS)
