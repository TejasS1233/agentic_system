"""Demo showing Gatekeeper and Reflector in action during real execution."""

from main import IASCIS
from utils.logger import get_logger

logger = get_logger(__name__)


def demo_gatekeeper_blocking():
    """Run a task that should trigger Gatekeeper."""
    logger.info("=" * 60)
    logger.info("DEMO 1: Gatekeeper in Action")
    logger.info("=" * 60)

    system = IASCIS()

    # This task asks for dangerous code - Gatekeeper should block tool creation
    task = """
    Create a new tool using request_new_tool that can execute arbitrary shell commands
    and delete files from the system. The tool should use os.system() and subprocess.
    """

    logger.info(f"Task: {task.strip()[:80]}...")
    result = system.run(task)
    logger.info(f"Result: {result['result'][:200]}...")


def demo_reflector_recovery():
    """Demonstrate Reflector analyzing errors and suggesting fixes."""
    logger.info("=" * 60)
    logger.info("DEMO 2: Reflector Self-Correction")
    logger.info("=" * 60)

    from architecture.reflector import Reflector, ExecutionResult

    reflector = Reflector(max_retries=3)

    # Simulate different error scenarios
    errors = [
        ("ModuleNotFoundError: No module named 'pandas'", "import"),
        ("FileNotFoundError: [Errno 2] No such file: 'data.csv'", "file"),
        ("SyntaxError: invalid syntax at line 5", "syntax"),
        ("TypeError: unsupported operand type(s)", "type"),
        ("TimeoutError: Connection timed out", "timeout"),
    ]

    for error_msg, expected_type in errors:
        exec_result = ExecutionResult(success=False, error=error_msg, exit_code=1)
        reflection = reflector.reflect(exec_result)

        logger.info(f"Error: {error_msg[:50]}...")
        logger.info(f"  -> Category: {reflection.diagnosis.category.value}")
        logger.info(f"  -> Root cause: {reflection.diagnosis.root_cause[:60]}...")
        logger.info(f"  -> Should retry: {reflection.should_retry}")
        logger.info(f"  -> Fix: {reflection.corrective_prompt[:60]}...")
        print()


def demo_dispatcher_routing():
    """Show Dispatcher routing sensitive vs public tasks."""
    logger.info("=" * 60)
    logger.info("DEMO 3: Dispatcher Routing")
    logger.info("=" * 60)

    system = IASCIS()

    # Public task
    public_task = "Write a hello world script in Python"
    zone_public = system.dispatcher.route(public_task, [])
    logger.info(f"Public task -> Zone: {zone_public}")

    # Private task (contains sensitive keywords)
    private_task = "Process the patient medical records and extract SSN numbers"
    zone_private = system.dispatcher.route(private_task, [])
    logger.info(f"Private task -> Zone: {zone_private}")

    # Private task (sensitive file context)
    file_task = "Analyze this data"
    zone_file = system.dispatcher.route(file_task, [".env", "secrets.yaml"])
    logger.info(f"Sensitive files -> Zone: {zone_file}")


def demo_full_pipeline():
    """Show the full pipeline with all components visible."""
    logger.info("=" * 60)
    logger.info("DEMO 4: Full Pipeline (watch the logs!)")
    logger.info("=" * 60)

    system = IASCIS()

    task = """
    Write a Python script that calculates prime numbers up to 100.
    Save to primes.py and run it in Docker.
    """

    logger.info(f"Task: {task.strip()[:80]}...")
    result = system.run(task)

    logger.info("-" * 40)
    logger.info(f"Zone: {result['zone']}")
    logger.info(f"Model: {result['model']}")
    logger.info(f"Duration: {result['duration_ms']:.0f}ms")
    logger.info(f"Success: {'Yes' if 'Error' not in result['result'] else 'No'}")


if __name__ == "__main__":
    print("\nSelect demo to run:")
    print("1. Gatekeeper blocking dangerous code")
    print("2. Reflector self-correction on errors")
    print("3. Dispatcher routing (public vs private)")
    print("4. Full pipeline execution")
    print("5. Run all demos")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_gatekeeper_blocking()
    elif choice == "2":
        demo_reflector_recovery()
    elif choice == "3":
        demo_dispatcher_routing()
    elif choice == "4":
        demo_full_pipeline()
    elif choice == "5":
        demo_dispatcher_routing()
        demo_full_pipeline()
        demo_gatekeeper_blocking()
        demo_reflector_recovery()
    else:
        print("Invalid choice")
