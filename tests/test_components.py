"""Test script for Gatekeeper and Reflector components."""

from architecture.gatekeeper import Gatekeeper
from architecture.reflector import Reflector, ExecutionResult
from utils.logger import get_logger

logger = get_logger(__name__)


def test_gatekeeper():
    """Test Gatekeeper with various code samples."""
    logger.info("=== Testing Gatekeeper ===")
    gatekeeper = Gatekeeper(strict_mode=True, stateful=True)

    test_cases = [
        ("Safe code", "def add(a, b):\n    return a + b"),
        ("Dangerous import", "import subprocess\nsubprocess.call(['rm', '-rf', '/'])"),
        ("Eval usage", "user_input = input()\neval(user_input)"),
        ("Exec usage", "code = 'print(1)'\nexec(code)"),
        ("OS system call", "import os\nos.system('whoami')"),
        ("File deletion", "import os\nos.remove('/etc/passwd')"),
        ("Network access", "import socket\ns = socket.socket()"),
        ("Pickle load", "import pickle\npickle.load(open('data.pkl', 'rb'))"),
    ]

    for name, code in test_cases:
        result = gatekeeper.validate(code)
        status = "SAFE" if result.is_safe else "BLOCKED"
        logger.info(f"{name}: {status}")
        if not result.is_safe:
            for v in result.violations[:2]:
                logger.warning(f"  - {v}")

    logger.info("Gatekeeper testing complete\n")


def test_reflector():
    """Test Reflector with various error scenarios."""
    logger.info("=== Testing Reflector ===")
    reflector = Reflector(max_retries=3)

    error_scenarios = [
        ("Import error", "ModuleNotFoundError: No module named 'nonexistent'", 1),
        ("Syntax error", "SyntaxError: invalid syntax at line 5", 1),
        (
            "Type error",
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
            1,
        ),
        (
            "File not found",
            "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'",
            1,
        ),
        (
            "Permission denied",
            "PermissionError: [Errno 13] Permission denied: '/etc/passwd'",
            1,
        ),
        ("Timeout", "TimeoutError: Command timed out after 120 seconds", 1),
        ("Network error", "ConnectionError: Failed to connect to api.example.com", 1),
    ]

    for name, error, exit_code in error_scenarios:
        exec_result = ExecutionResult(
            success=False,
            error=error,
            exit_code=exit_code,
        )
        reflection = reflector.reflect(exec_result)

        logger.info(f"{name}:")
        logger.info(f"  Category: {reflection.diagnosis.category.value}")
        logger.info(f"  Root cause: {reflection.diagnosis.root_cause[:60]}...")
        logger.info(f"  Should retry: {reflection.should_retry}")

    reflector.reset()
    logger.info("Reflector testing complete\n")


def test_integration():
    """Test full integration with IASCIS."""
    logger.info("=== Testing Integration ===")
    from main import IASCIS

    system = IASCIS()

    dangerous_code = """
import os
import subprocess
os.system('rm -rf /')
subprocess.Popen(['curl', 'http://malicious.com'])
eval(input())
"""
    result = system.validate_code(dangerous_code)
    logger.info(
        f"Dangerous code validation: {'BLOCKED' if not result.is_safe else 'ALLOWED'}"
    )
    logger.info(f"Violations found: {len(result.violations)}")
    for v in result.violations[:3]:
        logger.warning(f"  - {v}")

    logger.info("Integration testing complete\n")


if __name__ == "__main__":
    test_gatekeeper()
    test_reflector()
    test_integration()
