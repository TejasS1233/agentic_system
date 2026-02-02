"""LLM client abstraction for multi-provider support via LiteLLM."""

import json
import os
import platform
import warnings
from abc import ABC, abstractmethod

from architecture.llm_manager import get_llm_manager

from utils.logger import get_logger

from .schemas import Message
from .tools import Tool

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

logger = get_logger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_API_URL", "http://localhost:11434")

OS_INFO = f"OS: {platform.system()} ({platform.release()})"

SYSTEM_PROMPT = f"""You are an expert DevOps engineer and Python developer.
{OS_INFO}
Always create a docker container to run your code.
Your goal is to autonomously solve infrastructure and coding tasks.
You can write files and execute commands.
If a command fails, analyze the error output, fix the code or configuration, and try again.
Always verify your work by running the code you wrote.

IMPORTANT: On Windows, use %cd% instead of $(pwd) for Docker volume mounts.

CRITICAL: Once you have successfully completed the task and verified the output is correct, 
you MUST respond with a final summary message WITHOUT making any more tool calls. 
Do NOT repeat successful commands. If the output looks correct, STOP and report success."""


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def start_chat(self, history: list[Message] = None):
        pass

    @abstractmethod
    def send_message(self, chat, message: str) -> str:
        pass


class LiteLLMClient(LLMClient):
    """LiteLLM-based client supporting multiple LLM providers."""

    MAX_TURNS = 10

    def __init__(
        self,
        model_name: str,
        tools: list[Tool] = None,
        api_base: str = None,
        max_turns: int = MAX_TURNS,
    ):
        self.model_name = model_name
        self.tools = tools or []
        self.max_turns = max_turns
        self.history: list[dict] = []

        self.api_base = None
        self.extra_headers = None

        if "ollama" in model_name.lower():
            self.api_base = api_base or OLLAMA_URL
            self.extra_headers = {"ngrok-skip-browser-warning": "true"}
            logger.info(f"Using Ollama at: {self.api_base}")

        logger.info(
            f"LLMClient initialized (model={model_name}, tools={len(self.tools)})"
        )

    def _get_tools_schema(self) -> list[dict]:
        """Convert Tools to OpenAI function calling schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.model_json_schema(),
                },
            }
            for t in self.tools
        ]

    def start_chat(self, history: list[Message] = None):
        """Initialize or reset chat history."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            self.history.extend(
                [{"role": msg.role, "content": msg.content} for msg in history]
            )
        return self

    def send_message(self, chat, message: str) -> str:
        """Send message and execute tool calls in a loop until completion."""
        self.history.append({"role": "user", "content": message})

        tools_map = {t.name: t for t in self.tools}
        tools_schema = self._get_tools_schema()
        security_blocks = 0  # Track consecutive security blocks

        for turn in range(self.max_turns):
            logger.debug(f"Turn {turn + 1}/{self.max_turns}")

            response = self._call_llm(tools_schema)
            if isinstance(response, str):
                return response

            response_message = response.choices[0].message

            if not response_message.tool_calls:
                self.history.append(response_message)
                return response_message.content

            self.history.append(response_message)
            blocked = self._execute_tool_calls(response_message.tool_calls, tools_map)

            # Early stop if security is repeatedly blocking
            if blocked:
                security_blocks += 1
                if security_blocks >= 2:
                    logger.warning("Security blocked execution - stopping")
                    return "Task blocked: Security violations detected. The requested operation is not permitted."
            else:
                security_blocks = 0

        logger.warning("Max turns reached")
        return "Error: Max tool turns reached."

    def _call_llm(self, tools_schema: list[dict]):
        """Make LLM API call using the centralized LLM manager."""
        from litellm import completion
        
        kwargs = {
            "model": self.model_name,
            "messages": self.history,
            "tools": tools_schema if tools_schema else None,
            "tool_choice": "auto" if tools_schema else None,
            "temperature": 0.0,
        }

        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        try:
            # Use direct completion for tool calling (manager doesn't support tools yet)
            return completion(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"LiteLLM Error: {e}"

    def _execute_tool_calls(self, tool_calls, tools_map: dict) -> bool:
        """Execute tool calls and return True if any were blocked."""
        blocked = False
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            logger.info(f"Tool call: {func_name}")
            logger.debug(f"Args: {func_args}")

            if func_name in tools_map:
                try:
                    result = tools_map[func_name].run(**func_args)
                    # Detect security blocks
                    if isinstance(result, str) and result.startswith("BLOCKED:"):
                        blocked = True
                except Exception as e:
                    result = f"Error executing {func_name}: {e}"
                    logger.error(result)
            else:
                result = f"Error: Tool {func_name} not found"
                logger.warning(result)

            logger.debug(f"Result: {str(result)[:100]}...")

            self.history.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": str(result),
                }
            )

        return blocked
