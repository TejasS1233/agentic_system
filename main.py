"""
IASCIS - Independent Autonomous Self-Correcting Intelligent System

Main entry point integrating:
- Orchestrator: Domain-based decomposition + tool retrieval
- Executor: Plan execution with dependency ordering
- Toolsmith: Dynamic tool generation
- Gatekeeper: Safety validation
"""

import os
import time

from dotenv import load_dotenv

from architecture.dispatcher import Dispatcher
from architecture.gatekeeper import Gatekeeper, ValidationResult
from architecture.orchestrator import Orchestrator
from architecture.executor import ExecutorAgent
from architecture.toolsmith import Toolsmith
from architecture.reflector import ExecutionResult, Reflector
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

WORKSPACE_PATH = os.path.join(os.getcwd(), "workspace")
TOOLS_DIR = os.path.join(WORKSPACE_PATH, "tools")


class IASCIS:
    """Main system orchestrating all components."""

    def __init__(
        self,
        workspace_path: str = None,
        public_model: str = "groq/llama-3.3-70b-versatile",
        private_model: str = "groq/llama-3.3-70b-versatile",
        safe_mode: bool = True,
    ):
        self.workspace_path = workspace_path or WORKSPACE_PATH
        self.tools_dir = os.path.join(self.workspace_path, "tools")
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(self.tools_dir, exist_ok=True)

        self.public_model = public_model
        self.private_model = private_model

        # Initialize components
        self.gatekeeper = Gatekeeper(strict_mode=safe_mode, stateful=True)
        self.toolsmith = Toolsmith(safe_mode=safe_mode, gatekeeper=self.gatekeeper)
        self.dispatcher = Dispatcher()
        self.reflector = Reflector(max_retries=3)

        # Initialize Executor
        self.executor = ExecutorAgent(
            workspace_path=self.workspace_path,
            tools_dir=self.tools_dir
        )

        # Initialize Orchestrator with Executor
        self.orchestrator = Orchestrator(
            toolsmith=self.toolsmith,
            executor=self.executor,
            registry_path=os.path.join(self.tools_dir, "registry.json")
        )

        logger.info(f"IASCIS initialized (workspace={self.workspace_path})")

    def run(self, task: str, file_context: list[str] = None) -> dict:
        """Execute a task through the orchestrator-executor pipeline."""
        file_context = file_context or []
        start_time = time.perf_counter()

        logger.info(f"Task received: {task[:100]}...")

        # Route to appropriate zone
        zone = self.dispatcher.route(task, file_context)
        model = self.private_model if zone == "private" else self.public_model
        logger.info(f"Routed to {zone} zone, model: {model}")

        # Run through orchestrator (which calls executor)
        try:
            result = self.orchestrator.run(task)
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            result = f"Error: {e}"

        duration_ms = (time.perf_counter() - start_time) * 1000

        return {
            "task": task,
            "zone": zone,
            "model": model,
            "result": result,
            "duration_ms": duration_ms,
        }

    def run_with_reflection(self, task: str, max_attempts: int = 3) -> dict:
        """Execute with self-correction loop."""
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1}/{max_attempts}")

            result = self.run(task)

            if "Error" not in str(result.get("result", "")):
                return result

            # Analyze failure
            exec_result = ExecutionResult(
                success=False,
                error=str(result.get("result")),
                exit_code=1,
            )

            reflection = self.reflector.reflect(exec_result)
            logger.warning(f"Attempt {attempt + 1} failed: {reflection.diagnosis.category.value}")

            if not reflection.should_retry:
                logger.error("No retry recommended")
                break

            task = reflection.corrective_prompt

        return result

    def validate_code(self, code: str) -> ValidationResult:
        """Validate code through Gatekeeper."""
        return self.gatekeeper.validate(code)

    def create_tool(self, requirement: str) -> str:
        """Create a new tool via Toolsmith."""
        return self.toolsmith.create_tool(requirement)

    def reset(self):
        """Reset stateful components for new session."""
        self.gatekeeper.reset_state()
        self.reflector.reset()


def main():
    """Main entry point."""
    system = IASCIS()

    # Example task
    task = input("Enter your task (or press Enter for demo): ").strip()
    if not task:
        task = "Calculate the square root of 144, then multiply by 2, and save the result to a file called result.txt"

    logger.info("Starting IASCIS")
    result = system.run(task)

    logger.info(f"Completed in {result['duration_ms']:.2f}ms")
    logger.info(f"Zone: {result['zone']}")
    print(f"\n{'='*50}")
    print(f"Result:\n{result['result']}")


if __name__ == "__main__":
    main()
