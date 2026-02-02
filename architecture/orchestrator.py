"""Orchestrator for high-level task planning and decomposition."""

import os

from architecture.llm_manager import get_llm_manager

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

    def __init__(self, model_name: str = "gemini/gemini-2.5-flash"):
        self.model_name = model_name
        
        # Determine provider from model name
        if "gemini" in model_name.lower():
            self.llm = get_llm_manager(provider="gemini")
        elif "groq" in model_name.lower() or "llama" in model_name.lower():
            self.llm = get_llm_manager(provider="groq")
        else:
            # For ollama or other providers, use direct litellm
            self.llm = None
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

        try:
            if self.llm:
                # Use centralized LLM manager
                response = self.llm.generate(messages=messages, temperature=0.3)
                if response.get("error"):
                    logger.error(f"Planning failed: {response['error']}")
                    return f"Planning failed: {response['error']}"
                plan = response["content"]
            else:
                # Fallback to direct litellm for ollama
                from litellm import completion
                kwargs = {"model": self.model_name, "messages": messages}
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                if self.extra_headers:
                    kwargs["extra_headers"] = self.extra_headers
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
