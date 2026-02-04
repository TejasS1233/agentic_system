"""Agent core execution loop with retry logic and tool decay management."""

import time
from typing import Optional

from utils.logger import get_logger

from .llm import LLMClient
from .schemas import AgentState
from .tools import Tool
from .tool_decay import ToolDecayManager, create_decay_manager

logger = get_logger(__name__)


# Default semantic groups for agent tools
DEFAULT_SEMANTIC_GROUPS = {
    "file_ops": ["read_file", "write_file", "list_dir", "create_file", "delete_file"],
    "code_analysis": ["search_code", "find_references", "get_definition", "analyze"],
    "shell": ["run_command", "execute", "shell", "terminal"],
    "web": ["fetch_url", "http_request", "api_call", "web_search"],
}


class Agent:
    """Core agent managing execution loop and LLM interaction with tool decay."""

    DEFAULT_MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 10

    # Tool decay configuration
    TOOL_DECAY_PERIOD_MINUTES = 10.0  # Base TTL for tools
    TOOL_CLEANUP_INTERVAL = 60.0  # Cleanup check interval in seconds
    TOOL_MAX_CAPACITY = 100  # Maximum tools in cache

    RETRIABLE_ERRORS = {"429", "ResourceExhausted", "503", "rate_limit", "quota"}

    def __init__(
        self,
        workspace_path: str,
        tools: list[Tool],
        llm_client: LLMClient,
        enable_tool_decay: bool = True,
        protected_tools: Optional[list[str]] = None,
    ):
        self.state = AgentState(workspace_path=workspace_path)
        self.llm_client = llm_client
        self.chat = None
        self._enable_tool_decay = enable_tool_decay
        
        # Initialize tool decay manager
        if enable_tool_decay:
            self._decay_manager = create_decay_manager(
                decay_minutes=self.TOOL_DECAY_PERIOD_MINUTES,
                protected_tools=protected_tools or [],
                auto_cleanup=True,
                cleanup_interval=self.TOOL_CLEANUP_INTERVAL,
                max_capacity=self.TOOL_MAX_CAPACITY,
            )
            # Register all tools with the decay manager
            for tool in tools:
                self._register_tool(tool)
            logger.info(
                f"Tool decay enabled (capacity={self.TOOL_MAX_CAPACITY}, "
                f"decay_period={self.TOOL_DECAY_PERIOD_MINUTES}min)"
            )
        else:
            self._decay_manager = None
            self.tools = {t.name: t for t in tools}
            
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

    # ========== Tool Decay Integration ==========

    def _register_tool(self, tool: Tool, protected: bool = False) -> None:
        """Register a tool with the decay manager."""
        if self._decay_manager is None:
            self.tools[tool.name] = tool
            return
        
        # Determine semantic group based on tool name
        semantic_group = None
        for group, patterns in DEFAULT_SEMANTIC_GROUPS.items():
            if any(pattern in tool.name.lower() for pattern in patterns):
                semantic_group = group
                break
        
        self._decay_manager.register_tool(
            name=tool.name,
            tool=tool,
            protected=protected,
            semantic_group=semantic_group,
        )

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name. Uses decay-aware retrieval if enabled.
        
        This method will:
        - Return the tool from cache or archive
        - Restore archived tools automatically
        - Return None if tool doesn't exist
        """
        if self._decay_manager is None:
            return self.tools.get(name)
        
        # Use get_or_restore to automatically bring back archived tools
        return self._decay_manager.get_or_restore(name)

    def record_tool_usage(
        self, 
        name: str, 
        success: bool = True, 
        execution_time_ms: float = 0.0
    ) -> bool:
        """
        Record a tool usage event for decay scoring.
        
        Should be called after each tool execution to track:
        - Success/failure rates
        - Execution time
        - Usage frequency
        
        Args:
            name: The tool name
            success: Whether the execution succeeded
            execution_time_ms: Execution time in milliseconds
        
        Returns:
            True if the usage was recorded, False if tool not found
        """
        if self._decay_manager is None:
            return name in self.tools
        
        metrics = self._decay_manager.record_usage(name, success, execution_time_ms)
        return metrics is not None

    @property
    def tools(self) -> dict[str, Tool]:
        """Get all active tools as a dictionary."""
        if self._decay_manager is None:
            return self._tools_dict
        return self._decay_manager.get_all_tools()

    @tools.setter
    def tools(self, value: dict[str, Tool]):
        """Set tools dictionary (used when decay is disabled)."""
        self._tools_dict = value

    def get_tool_status_report(self) -> str:
        """Get a human-readable report of tool usage and decay status."""
        if self._decay_manager is None:
            return f"Tool decay disabled. {len(self.tools)} tools active."
        return self._decay_manager.get_status_report()

    def get_tool_statistics(self) -> dict:
        """Get detailed statistics about tool cache performance."""
        if self._decay_manager is None:
            return {"decay_enabled": False, "total_tools": len(self.tools)}
        stats = self._decay_manager.get_statistics()
        stats["decay_enabled"] = True
        return stats

    def protect_tool(self, name: str) -> bool:
        """Mark a tool as protected (never evicted)."""
        if self._decay_manager is None:
            return name in self.tools
        return self._decay_manager.protect_tool(name)

    def shutdown(self):
        """Clean shutdown of the agent, stopping background tasks."""
        if self._decay_manager:
            self._decay_manager.stop_background_cleanup()
            logger.info("Tool decay cleanup stopped")
