"""Static Gatekeeper implementing three-layer safety: AST, vulnerability scanning, state tracking."""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OperationType(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    NETWORK = "network"


@dataclass
class ValidationResult:
    is_safe: bool
    risk_level: RiskLevel
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    analyzed_imports: list[str] = field(default_factory=list)
    analyzed_calls: list[str] = field(default_factory=list)


@dataclass
class SecurityPattern:
    id: str
    pattern: str
    message: str
    severity: RiskLevel


class VulnerabilityScanner:
    """Pattern-based SAST for common vulnerability detection."""

    PATTERNS: list[SecurityPattern] = [
        SecurityPattern(
            "SQLI-001",
            r'execute\s*\(\s*["\'].*%s.*["\']',
            "SQL injection via string formatting",
            RiskLevel.CRITICAL,
        ),
        SecurityPattern(
            "SQLI-002",
            r'execute\s*\(\s*f["\']',
            "SQL injection via f-string",
            RiskLevel.CRITICAL,
        ),
        SecurityPattern(
            "CMDi-001",
            r"shell\s*=\s*True",
            "Shell injection risk",
            RiskLevel.HIGH,
        ),
        SecurityPattern(
            "SEC-001",
            r'(password|secret|api_key|token|consumer_key|consumer_secret|access_token|access_token_secret|bearer_token|client_secret|client_id)\s*=\s*["\'][^"\']+["\']',
            "Hardcoded credential",
            RiskLevel.CRITICAL,
        ),
        SecurityPattern(
            "SEC-003",
            r'["\']your_(api_key|consumer_key|access_token|token|secret|password)["\']',
            "Placeholder API credential - tool requires authentication",
            RiskLevel.CRITICAL,
        ),
        SecurityPattern(
            "SEC-004",
            r'(OAuthHandler|OAuth1|OAuth2|APIKey|Bearer)\s*\(',
            "OAuth/API authentication - requires credentials",
            RiskLevel.CRITICAL,
        ),
        SecurityPattern(
            "SEC-002",
            r"-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----",
            "Hardcoded private key",
            RiskLevel.CRITICAL,
        ),
        SecurityPattern(
            "DESER-001",
            r"pickle\.loads?\s*\(",
            "Insecure deserialization",
            RiskLevel.HIGH,
        ),
        SecurityPattern(
            "DESER-002",
            r"yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader)",
            "Unsafe YAML load",
            RiskLevel.MEDIUM,
        ),
        SecurityPattern(
            "PATH-001",
            r"open\s*\([^)]*\+[^)]*\)",
            "Path traversal risk",
            RiskLevel.MEDIUM,
        ),
        SecurityPattern(
            "CRYPTO-001",
            r"(md5|sha1)\s*\(",
            "Weak hash algorithm",
            RiskLevel.LOW,
        ),
        SecurityPattern(
            "DEBUG-001",
            r"debug\s*=\s*True",
            "Debug mode enabled",
            RiskLevel.LOW,
        ),
    ]

    def scan(self, code: str) -> list[str]:
        """Scan code for vulnerability patterns. Returns list of findings."""
        findings: list[str] = []
        for p in self.PATTERNS:
            if re.search(p.pattern, code, re.IGNORECASE):
                findings.append(f"[{p.id}] {p.message}")
                logger.warning(f"Vulnerability: {p.id}")
        return findings


class StateGatekeeper:
    """Stateful operation tracking for escalation detection."""

    THRESHOLDS = {
        OperationType.DELETE: 5,
        OperationType.WRITE: 20,
        OperationType.EXECUTE: 10,
        OperationType.NETWORK: 15,
    }

    ESCALATION_PATTERNS = [
        [OperationType.READ, OperationType.WRITE, OperationType.DELETE],
        [OperationType.READ, OperationType.NETWORK],
        [OperationType.WRITE, OperationType.EXECUTE],
    ]

    def __init__(self):
        self.operation_counts: dict[OperationType, int] = {
            op: 0 for op in OperationType
        }
        self.operation_history: list[OperationType] = []

    def record_operation(self, op: OperationType) -> list[str]:
        """Record operation and check for threshold/escalation violations."""
        warnings: list[str] = []

        self.operation_counts[op] += 1
        self.operation_history.append(op)

        if op in self.THRESHOLDS and self.operation_counts[op] >= self.THRESHOLDS[op]:
            warnings.append(
                f"Threshold exceeded: {op.value} ({self.operation_counts[op]})"
            )

        for pattern in self.ESCALATION_PATTERNS:
            if self._matches_pattern(pattern):
                warnings.append(f"Escalation: {' -> '.join(p.value for p in pattern)}")

        return warnings

    def _matches_pattern(self, pattern: list[OperationType]) -> bool:
        if len(self.operation_history) < len(pattern):
            return False
        return self.operation_history[-len(pattern) :] == pattern

    def classify_operation(self, call: str) -> Optional[OperationType]:
        """Classify function call into operation type."""
        c = call.lower()
        if any(k in c for k in ["read", "open", "load", "get", "fetch"]):
            return OperationType.READ
        if any(k in c for k in ["write", "save", "dump", "put", "create"]):
            return OperationType.WRITE
        if any(k in c for k in ["remove", "delete", "unlink", "rmdir", "rmtree"]):
            return OperationType.DELETE
        if any(k in c for k in ["run", "exec", "system", "popen", "call", "spawn"]):
            return OperationType.EXECUTE
        if any(k in c for k in ["request", "urlopen", "socket", "connect", "send"]):
            return OperationType.NETWORK
        return None

    def reset(self):
        """Reset state for new session."""
        self.operation_counts = {op: 0 for op in OperationType}
        self.operation_history = []


class Gatekeeper:
    """
    Three-layer static analysis gatekeeper.

    Layer 1: AST Analysis - Structural validation of imports and calls
    Layer 2: Vulnerability Scanner - Pattern-based SAST
    Layer 3: State Gatekeeper - Stateful operation tracking
    """

    BANNED_IMPORTS: set[str] = {
        "shutil",
        # Auth-requiring libraries (no keys will be provided)
        "tweepy",           # Twitter API
        "twitter",          # Twitter API
        "openai",           # OpenAI API
        "anthropic",        # Anthropic API  
        "google.cloud",     # Google Cloud (requires service account)
        "boto3",            # AWS SDK (requires credentials)
        "stripe",           # Stripe payments
        "twilio",           # Twilio SMS
        "sendgrid",         # SendGrid email
    }
    RISKY_IMPORTS: set[str] = {"subprocess", "multiprocessing", "ctypes", "pickle"}
    BANNED_CALLS: set[str] = {"eval", "exec", "compile", "__import__"}

    BANNED_MODULE_METHODS: dict[str, set[str]] = {
        "os": {"system", "popen", "spawn", "spawnl", "spawnle", "spawnlp"},
    }

    RISKY_MODULE_METHODS: dict[str, set[str]] = {
        "os": {"remove", "rmdir", "unlink", "rename"},
        "pathlib": {"unlink", "rmdir"},
        "subprocess": {"call", "run", "Popen"},
    }

    def __init__(self, strict_mode: bool = False, stateful: bool = True):
        self.strict_mode = strict_mode
        self.vuln_scanner = VulnerabilityScanner()
        self.state_gatekeeper = StateGatekeeper() if stateful else None
        logger.info(
            f"Gatekeeper initialized (strict={strict_mode}, stateful={stateful})"
        )

    def validate(self, code: str) -> ValidationResult:
        """Validate code through all three layers. Returns ValidationResult."""
        violations: list[str] = []
        warnings: list[str] = []
        analyzed_imports: list[str] = []
        analyzed_calls: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                is_safe=False,
                risk_level=RiskLevel.CRITICAL,
                violations=[f"Syntax error: {e}"],
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    analyzed_imports.append(alias.name)
                    self._check_import(module, violations, warnings)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    analyzed_imports.append(node.module)
                    self._check_import(module, violations, warnings)

            elif isinstance(node, ast.Call):
                call_info = self._analyze_call(node, violations, warnings)
                if call_info:
                    analyzed_calls.append(call_info)
                    if self.state_gatekeeper:
                        op_type = self.state_gatekeeper.classify_operation(call_info)
                        if op_type:
                            warnings.extend(
                                self.state_gatekeeper.record_operation(op_type)
                            )

        vulnerabilities = self.vuln_scanner.scan(code)
        risk_level = self._calculate_risk_level(violations, warnings, vulnerabilities)

        if self.strict_mode and warnings:
            violations.extend([f"[STRICT] {w}" for w in warnings])
            warnings = []

        is_safe = len(violations) == 0 and len(vulnerabilities) == 0

        if not is_safe:
            logger.warning(f"Validation failed: {violations + vulnerabilities}")

        return ValidationResult(
            is_safe=is_safe,
            risk_level=risk_level,
            violations=violations,
            warnings=warnings,
            vulnerabilities=vulnerabilities,
            analyzed_imports=analyzed_imports,
            analyzed_calls=analyzed_calls,
        )

    def _check_import(
        self, module: str, violations: list[str], warnings: list[str]
    ) -> None:
        if module in self.BANNED_IMPORTS:
            violations.append(f"Banned import: '{module}'")
        elif module in self.RISKY_IMPORTS:
            warnings.append(f"Risky import: '{module}'")

    def _analyze_call(
        self, node: ast.Call, violations: list[str], warnings: list[str]
    ) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.BANNED_CALLS:
                violations.append(f"Banned call: '{func_name}()'")
            return func_name

        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if isinstance(node.func.value, ast.Name):
                module = node.func.value.id
                full_call = f"{module}.{method}"

                if module in self.BANNED_MODULE_METHODS:
                    if method in self.BANNED_MODULE_METHODS[module]:
                        violations.append(f"Banned method: '{full_call}()'")

                if module in self.RISKY_MODULE_METHODS:
                    if method in self.RISKY_MODULE_METHODS[module]:
                        warnings.append(f"Risky method: '{full_call}()'")

                return full_call
            return method
        return None

    def _calculate_risk_level(
        self, violations: list[str], warnings: list[str], vulnerabilities: list[str]
    ) -> RiskLevel:
        if violations or any("CRITICAL" in v for v in vulnerabilities):
            return RiskLevel.CRITICAL
        if vulnerabilities:
            return RiskLevel.HIGH
        if len(warnings) >= 3:
            return RiskLevel.HIGH
        if len(warnings) >= 1:
            return RiskLevel.MEDIUM
        return RiskLevel.SAFE

    def reset_state(self):
        """Reset stateful tracking for new session."""
        if self.state_gatekeeper:
            self.state_gatekeeper.reset()

    def __call__(self, code: str) -> ValidationResult:
        return self.validate(code)


def is_safe_code(code: str, strict: bool = False) -> bool:
    """Quick safety check. Returns True if code passes all layers."""
    return Gatekeeper(strict_mode=strict).validate(code).is_safe
