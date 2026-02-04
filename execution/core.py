"""Agent core execution loop with retry logic, tool decay management, and profiling."""

import time
from typing import Optional, Tuple, Any

from utils.logger import get_logger

from .llm import LLMClient
from .schemas import AgentState
from .tools import Tool
from .tool_decay import ToolDecayManager, create_decay_manager

# Import profiler (optional dependency)
try:
    from architecture.profiler import (
        Profiler,
        ProfilingMode,
        ProfileResult,
        profile_to_metrics_update,
    )
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    Profiler = None
    ProfilingMode = None
    ProfileResult = None

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
        enable_profiling: bool = True,
        profiling_mode: str = "lightweight",  # off, lightweight, standard, full
    ):
        self.state = AgentState(workspace_path=workspace_path)
        self.llm_client = llm_client
        self.chat = None
        self._enable_tool_decay = enable_tool_decay
        self._enable_profiling = enable_profiling and HAS_PROFILER
        
        # Initialize profiler
        if self._enable_profiling:
            mode_map = {
                "off": ProfilingMode.OFF,
                "lightweight": ProfilingMode.LIGHTWEIGHT,
                "standard": ProfilingMode.STANDARD,
                "full": ProfilingMode.FULL,
                "gpu": ProfilingMode.GPU,
            }
            self._profiler = Profiler(
                mode=mode_map.get(profiling_mode, ProfilingMode.LIGHTWEIGHT),
                history_size=500,  # Keep last 500 profiles
            )
            logger.info(f"Profiler enabled (mode={profiling_mode})")
        else:
            self._profiler = None
            if enable_profiling and not HAS_PROFILER:
                logger.warning("Profiling requested but profiler not available")
        
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
            f"Agent initialized (workspace={workspace_path}, tools={len(tools)}, "
            f"profiling={self._enable_profiling})"
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
        execution_time_ms: float = 0.0,
        profile: Optional[Any] = None,  # ProfileResult if available
    ) -> bool:
        """
        Record a tool usage event for decay scoring.
        
        Should be called after each tool execution to track:
        - Success/failure rates
        - Execution time
        - Usage frequency
        - Performance profile (if profiling enabled)
        
        Args:
            name: The tool name
            success: Whether the execution succeeded
            execution_time_ms: Execution time in milliseconds
            profile: Optional ProfileResult from profiler
        
        Returns:
            True if the usage was recorded, False if tool not found
        """
        # Use profile data if available
        if profile is not None and hasattr(profile, 'execution_time_ms'):
            execution_time_ms = profile.execution_time_ms
            success = profile.success
        
        if self._decay_manager is None:
            return name in self.tools
        
        metrics = self._decay_manager.record_usage(name, execution_time_ms)
        return metrics is not None

    def execute_tool(
        self,
        name: str,
        **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        """
        Execute a tool with automatic profiling and usage tracking.
        
        This is the recommended way to execute tools as it:
        1. Retrieves the tool (restoring from archive if needed)
        2. Profiles the execution (if profiling enabled)
        3. Records usage for decay scoring
        4. Returns both result and profile
        
        Args:
            name: The tool name
            **kwargs: Arguments to pass to the tool's run() method
        
        Returns:
            Tuple of (result, profile) where profile is ProfileResult or None
        """
        tool = self.get_tool(name)
        if tool is None:
            logger.warning(f"Tool not found: {name}")
            return None, None
        
        result = None
        profile = None
        
        if self._enable_profiling and self._profiler is not None:
            # Profile the tool execution
            try:
                result, profile = self._profiler.profile_tool(tool, kwargs)
            except Exception as e:
                # Even if an exception occurs, profile is recorded
                logger.error(f"Tool {name} failed: {e}")
                # Get the last profile from history (it was recorded by the profiler)
                if self._profiler.history:
                    profile = self._profiler.history[-1]
                raise
        else:
            # No profiling - execute directly
            start_time = time.perf_counter()
            try:
                result = tool.run(**kwargs)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
            except Exception as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self.record_tool_usage(name, success=False, execution_time_ms=execution_time_ms)
                raise
        
        # Record the usage
        self.record_tool_usage(name, profile=profile, execution_time_ms=execution_time_ms if profile is None else 0)
        
        return result, profile

    def get_profiler_statistics(self) -> dict:
        """Get aggregate statistics from the profiler."""
        if self._profiler is None:
            return {"profiling_enabled": False}
        
        stats = self._profiler.get_statistics()
        stats["profiling_enabled"] = True
        return stats

    def get_tool_profile_history(self, tool_name: str) -> list:
        """Get profiling history for a specific tool."""
        if self._profiler is None:
            return []
        
        return [
            p.to_dict() 
            for p in self._profiler.history 
            if p.tool_name == tool_name
        ]

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
