"""Orchestrator for high-level task planning and decomposition."""

import os

from litellm import completion

from utils.logger import get_logger

logger = get_logger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_API_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are an expert technical planner. Given a user's goal:
1. Break it down into clear, actionable steps
2. Identify dependencies between steps
3. Specify expected outputs for each step
4. Consider error handling and edge cases

Be concise and practical."""


class Orchestrator:
    """Decomposes high-level goals into actionable execution plans."""

    def __init__(self, model_name: str = "gemini/gemini-3-flash-preview"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_base = OLLAMA_URL if "ollama" in model_name.lower() else None
        self.extra_headers = (
            {"ngrok-skip-browser-warning": "true"} if self.api_base else None
        )
        logger.info(f"Orchestrator initialized (model={model_name})")

    def plan(self, goal: str) -> str:
        """Generate an execution plan from a high-level goal."""
        logger.info(f"Planning: {goal[:80]}...")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]

        kwargs = {"model": self.model_name, "messages": messages}

        if "gemini" in self.model_name.lower():
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        try:
            response = completion(**kwargs)
            plan = response.choices[0].message.content
            logger.info(f"Plan generated: {len(plan)} chars")
            return plan
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return f"Planning failed: {e}"

    def run(self, goal: str) -> str:
        """Alias for plan() for compatibility."""
        return self.plan(goal)
