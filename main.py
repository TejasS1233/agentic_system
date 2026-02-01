"""
IASCIS - Independent Autonomous Self-Correcting Intelligent System

Main entry point integrating all architecture components:
- Orchestrator: High-level planning
- Dispatcher: Privacy-aware routing (public/private zones)
- Gatekeeper: Three-layer static safety (AST, vulnerability, state)
- Reflector: Execution analysis and self-correction
- Toolsmith: Dynamic tool generation
"""

import os
import time

from dotenv import load_dotenv

from architecture.dispatcher import Dispatcher
from architecture.gatekeeper import Gatekeeper, ValidationResult
from architecture.orchestrator import Orchestrator
from architecture.reflector import ExecutionResult, Reflector
from architecture.toolsmith import Toolsmith
from execution.core import Agent
from execution.llm import LiteLLMClient
from execution.tools import ReadFileTool, RequestNewTool, RunCommandTool, WriteFileTool
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class IASCIS:
    """Main system orchestrating all components."""

    def __init__(
        self,
        workspace_path: str = None,
        public_model: str = "groq/llama-3.3-70b-versatile",
        private_model: str = "groq/llama-3.3-70b-versatile",
        safe_mode: bool = True,
    ):
        self.workspace_path = workspace_path or os.path.join(os.getcwd(), "workspace")
        os.makedirs(self.workspace_path, exist_ok=True)

        self.public_model = public_model
        self.private_model = private_model

        self.orchestrator = Orchestrator(model_name=public_model)
        self.dispatcher = Dispatcher()
        self.gatekeeper = Gatekeeper(strict_mode=safe_mode, stateful=True)
        self.reflector = Reflector(max_retries=3)
        self.toolsmith = Toolsmith(safe_mode=safe_mode, gatekeeper=self.gatekeeper)

        self.tools = [
            WriteFileTool(self.workspace_path),
            RunCommandTool(self.workspace_path),
            ReadFileTool(self.workspace_path),
            RequestNewTool(self.workspace_path),
        ]

        logger.info(f"IASCIS initialized (workspace={self.workspace_path})")

    def run(self, task: str, file_context: list[str] = None) -> dict:
        """Execute a task through the full pipeline."""
        file_context = file_context or []
        start_time = time.perf_counter()

        logger.info(f"Task received: {task[:100]}...")

        zone = self.dispatcher.route(task, file_context)
        model = self.private_model if zone == "private" else self.public_model
        logger.info(f"Routed to {zone} zone, model: {model}")

        plan = self.orchestrator.plan(task)
        logger.info(f"Plan generated: {len(plan)} chars")

        llm_client = LiteLLMClient(model_name=model, tools=self.tools)
        agent = Agent(
            workspace_path=self.workspace_path,
            tools=self.tools,
            llm_client=llm_client,
        )

        goal = f"Execute this plan:\n{plan}\n\nWrite the necessary code and run it. NOTE: Always use Docker to run the code AND This is a Windows Machine"
        result = self._execute_with_reflection(agent, goal)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return {
            "task": task,
            "zone": zone,
            "model": model,
            "plan": plan,
            "result": result,
            "duration_ms": duration_ms,
        }

    def _execute_with_reflection(
        self, agent: Agent, goal: str, max_attempts: int = 3
    ) -> str:
        """Execute with self-correction loop."""
        for attempt in range(max_attempts):
            logger.info(f"Execution attempt {attempt + 1}/{max_attempts}")

            try:
                result = agent.run(goal)
                return result

            except Exception as e:
                exec_result = ExecutionResult(
                    success=False,
                    error=str(e),
                    exit_code=1,
                )

                reflection = self.reflector.reflect(exec_result)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {reflection.diagnosis.category.value}"
                )

                if not reflection.should_retry:
                    logger.error("No retry recommended")
                    return f"Failed: {reflection.diagnosis.root_cause}"

                goal = reflection.corrective_prompt

        return "Failed after max attempts"

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
    system = IASCIS()

    task = "Write a Python function that calculates 20 armstrong numbers sequence up to n terms and save it to armstrong.py"

    logger.info("Starting IASCIS demo")
    result = system.run(task)

    logger.info(f"Completed in {result['duration_ms']:.2f}ms")
    logger.info(f"Zone: {result['zone']}")
    print(f"\nResult:\n{result['result']}")


if __name__ == "__main__":
    main()
