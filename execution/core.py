"""Agent core execution loop with retry logic."""

import time

from utils.logger import get_logger

from .llm import LLMClient
from .schemas import AgentState
from .tools import Tool

logger = get_logger(__name__)


class Agent:
    """Core agent managing execution loop and LLM interaction."""

    DEFAULT_MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 10

    RETRIABLE_ERRORS = {"429", "ResourceExhausted", "503", "rate_limit", "quota"}

    def __init__(
        self,
        workspace_path: str,
        tools: list[Tool],
        llm_client: LLMClient,
    ):
        self.state = AgentState(workspace_path=workspace_path)
        self.tools = {t.name: t for t in tools}
        self.llm_client = llm_client
        self.chat = None
        logger.info(
            f"Agent initialized (workspace={workspace_path}, tools={len(tools)})"
        )

    def run(self, goal: str) -> str:
        """Execute goal with automatic retry on transient failures."""
        self.chat = self.llm_client.start_chat()
        logger.info(f"Executing goal: {goal[:80]}...")
        return self._run_with_retry(goal)

    def _run_with_retry(self, goal: str, max_retries: int = None) -> str:
        """Execute with exponential backoff retry for transient errors."""
        max_retries = max_retries or self.DEFAULT_MAX_RETRIES
        delay = self.INITIAL_RETRY_DELAY

        for attempt in range(max_retries):
            try:
                result = self.llm_client.send_message(self.chat, goal)
                logger.info(f"Execution completed (attempt {attempt + 1})")
                return result

            except Exception as e:
                error_str = str(e)
                logger.error(f"Attempt {attempt + 1} failed: {error_str}")

                if self._is_retriable(error_str):
                    logger.warning(f"Retriable error, waiting {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    return f"Error: {error_str}"

        logger.error("Max retries exceeded")
        return "Failed after max retries."

    def _is_retriable(self, error_str: str) -> bool:
        """Check if error is transient and worth retrying."""
        return any(code in error_str for code in self.RETRIABLE_ERRORS)

    def reset(self):
        """Reset agent state for new task."""
        self.chat = None
        self.state = AgentState(workspace_path=self.state.workspace_path)
