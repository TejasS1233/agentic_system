"""
Tool Decay Module - Backward Compatibility Layer

This module provides backward compatibility with the original ToolDecayManager
while using the new AdaptiveToolCache under the hood.

For new implementations, use AdaptiveToolCache directly:
    from execution.adaptive_tool_cache import AdaptiveToolCache

Decay Score Formula (Linear Frequency Scaling):
    half_life = 9400 × (1 + min(frequency × 0.01, 2.0))
    score = exp(-TSU / half_life)

Where:
    - 9400 seconds (~2.6 hours) = base half-life
    - Frequency bonus = 0.01 per call
    - Max bonus = 2.0 (total multiplier 3x) at 200+ calls

Examples (Survival Time):
    - 1 call:   6.0 hours (baseline)
    - 10 calls: 6.6 hours
    - 100 calls: 12.0 hours
    - 200+ calls: 18.0 hours
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import time
import json
import os

from .adaptive_tool_cache import (
    AdaptiveToolCache,
    ToolMetrics,
    ToolTier,
    create_adaptive_cache,
)


# ============================================================
#               BACKWARD COMPATIBILITY CLASS
# ============================================================

@dataclass
class ToolUsageStats:
    """
    Backward-compatible wrapper for ToolMetrics.
    
    Provides the original ToolUsageStats interface while internally
    using the new ToolMetrics structure.
    
    Original interface:
        - name, last_used, use_count, created_at
        - mark_used(), is_expired(), time_since_last_use()
    
    Note: Success/failure tracking is removed because tools always succeed
    (they are validated before registration).
    """
    name: str
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    created_at: float = field(default_factory=time.time)
    
    # Extended field for execution time tracking
    total_execution_time_ms: float = 0.0
    
    def mark_used(self, execution_time_ms: float = 0.0):
        """Mark the tool as used (resets decay timer)."""
        self.last_used = time.time()
        self.use_count += 1
        self.total_execution_time_ms += execution_time_ms
    
    def time_since_last_use(self) -> float:
        """Get seconds since last use (TSU)."""
        return time.time() - self.last_used
    
    def is_expired(self, decay_period: float) -> bool:
        """Check if tool has exceeded the decay period."""
        return self.time_since_last_use() > decay_period
    
    def calculate_decay_score(self, k: float = 60.0) -> float:
        """
        Calculate decay score using linear frequency scaling.
        
        Formula:
            half_life = 9400 × (1 + min(freq × 0.01, 2.0))
            score = exp(-TSU / half_life)
        
        Args:
            k: Unused, kept for API compatibility
        """
        import math
        
        frequency = self.use_count
        tsu = self.time_since_last_use()
        age = time.time() - self.created_at
        
        # Base half-life: 9400 seconds (~2.6 hours)
        BASE_HALF_LIFE = 9400.0
        
        # Handle never-used tools (grace period based on age)
        if frequency == 0:
            age_hours = age / 3600.0
            if age_hours < 1:
                return 1.0  # Brand new, keep for now
            elif age_hours < 6:
                return 0.5  # Give it some time
            else:
                return 0.05  # Old and never used, should decay
        
        # Frequency extends the half-life linearly but gently
        # Multiplier = 1 + (freq * 0.01)
        # Capped at 3.0x (bonus of 2.0)
        multiplier = 1.0 + min(frequency * 0.01, 2.0)
        
        effective_half_life = BASE_HALF_LIFE * multiplier
        
        # Exponential decay based on time since last use
        return math.exp(-tsu / effective_half_life)
    
    @classmethod
    def from_metrics(cls, metrics: ToolMetrics) -> "ToolUsageStats":
        """Create ToolUsageStats from ToolMetrics."""
        return cls(
            name=metrics.name,
            last_used=metrics.last_used,
            use_count=metrics.total_calls,
            created_at=metrics.created_at,
            total_execution_time_ms=metrics.total_execution_time_ms,
        )


class ToolDecayManager:
    """
    Backward-compatible wrapper around AdaptiveToolCache.
    
    Provides the same interface as the original ToolDecayManager
    but uses the advanced ARC-based cache internally.
    
    New Features Available:
        - record_usage(name, success, execution_time_ms) for detailed tracking
        - get_eviction_candidates(n) to see which tools will be evicted
        - boost_group(group, multiplier) to boost semantic groups
        - export_metrics_json() for benchmark integration
        - Registry sync: Automatically updates registry.json on eviction/restore
    """
    
    def __init__(
        self,
        decay_period: float = 300.0,
        on_tool_removed: Optional[Callable[[str], None]] = None,
        protected_tools: Optional[List[str]] = None,
        # New parameters (with defaults for backward compatibility)
        max_capacity: int = 100,
        decay_constant: float = 60.0,
        semantic_groups: Optional[Dict[str, List[str]]] = None,
        registry_path: Optional[str] = None  # Path to registry.json for status sync
    ):
        """
        Initialize the decay manager.
        
        Args:
            decay_period: Base TTL in seconds before unused tools decay
            on_tool_removed: Callback when a tool is removed (receives tool name)
            protected_tools: List of tool names that never decay
            max_capacity: Maximum number of tools in cache (default: 100)
            decay_constant: K constant in decay formula (default: 60.0)
            semantic_groups: Dict mapping group names to tool name patterns
            registry_path: Path to registry.json for status sync (optional)
        """
        self.decay_period = decay_period
        self.on_tool_removed = on_tool_removed
        self.protected_tools = set(protected_tools or [])
        self.decay_constant = decay_constant
        
        # Registry path for status sync
        self.registry_path = registry_path
        if registry_path is None:
            # Default to workspace/tools/registry.json
            workspace_root = os.path.join(os.getcwd(), "workspace")
            self.registry_path = os.path.join(workspace_root, "tools", "registry.json")
        
        # Create the underlying cache
        self._cache = AdaptiveToolCache(
            max_capacity=max_capacity,
            base_ttl=decay_period,
            decay_constant=decay_constant,
            on_tool_evicted=self._handle_eviction,
            protected_tools=list(self.protected_tools),
            semantic_groups=semantic_groups
        )
        
        # Set registry path for enhanced decay score calculation
        self._cache.set_registry_path(self.registry_path)
    
    def _update_registry_status(self, tool_name: str, status: str) -> bool:
        """
        Update the status of a tool in the registry.json file.
        
        Args:
            tool_name: Name of the tool to update
            status: New status - one of: active, inactive, deprecated, failed
            
        Returns:
            True if successful, False otherwise
        """
        if not self.registry_path or not os.path.exists(self.registry_path):
            return False
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            if tool_name in registry:
                registry[tool_name]["status"] = status
                registry[tool_name]["updated_at"] = time.time()
                
                with open(self.registry_path, "w") as f:
                    json.dump(registry, f, indent=2)
                
                print(f"[ToolDecayManager] Registry updated: {tool_name} -> {status}")
                return True
            return False
        except Exception as e:
            print(f"[ToolDecayManager] Failed to update registry: {e}")
            return False
    
    def _get_registry_status(self, tool_name: str) -> Optional[str]:
        """
        Get the current status of a tool from the registry.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Status string or None if not found
        """
        if not self.registry_path or not os.path.exists(self.registry_path):
            return None
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            if tool_name in registry:
                return registry[tool_name].get("status", "active")
            return None
        except Exception:
            return None
    
    def _update_registry_usage(self, tool_name: str) -> bool:
        """
        Update usage stats in the registry (use_count and last_used).
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if successful, False otherwise
        """
        if not self.registry_path or not os.path.exists(self.registry_path):
            return False
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            if tool_name in registry:
                # Increment use_count (default to 0 if not present)
                registry[tool_name]["use_count"] = registry[tool_name].get("use_count", 0) + 1
                registry[tool_name]["last_used"] = time.time()
                
                with open(self.registry_path, "w") as f:
                    json.dump(registry, f, indent=2)
                
                return True
            return False
        except Exception as e:
            print(f"[ToolDecayManager] Failed to update registry usage: {e}")
            return False
    
    def _handle_eviction(self, name: str, metrics: ToolMetrics):
        """Internal callback to handle eviction, invokes user callback and updates registry."""
        # Update registry status to 'inactive'
        self._update_registry_status(name, "inactive")
        
        if self.on_tool_removed:
            try:
                self.on_tool_removed(name)
            except Exception as e:
                print(f"[ToolDecayManager] Error in on_tool_removed callback: {e}")

    
    # ========== Core Operations ==========
    
    def _get_registry_usage(self, tool_name: str) -> tuple:
        """
        Get usage data from the registry (use_count and last_used).
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tuple of (use_count, last_used) or (0, None) if not found
        """
        if not self.registry_path or not os.path.exists(self.registry_path):
            return (0, None)
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            if tool_name in registry:
                use_count = registry[tool_name].get("use_count", 0)
                last_used = registry[tool_name].get("last_used")
                return (use_count, last_used)
            return (0, None)
        except Exception:
            return (0, None)

    def register_tool(
        self, 
        name: str, 
        tool: Any, 
        protected: bool = False,
        semantic_group: Optional[str] = None,
        created_at: Optional[float] = None,
        last_used: Optional[float] = None
    ):
        """
        Register a tool with the decay manager.
        
        Args:
            name: Unique tool name
            tool: The tool instance
            protected: If True, tool never gets evicted
            semantic_group: Optional semantic category for grouping
            created_at: Optional timestamp (Unix epoch) when tool was originally created.
                        If provided, decay will be calculated from this time.
            last_used: Optional timestamp (Unix epoch) when tool was last used.
                        If provided, time_since_use will be calculated from this.
        """
        # Check registry status - skip if deprecated or failed
        registry_status = self._get_registry_status(name)
        if registry_status in ["deprecated", "failed"]:
            print(f"[ToolDecayManager] Skipping {name}: registry status is '{registry_status}'")
            return
        
        # Load historical usage data from registry
        registry_use_count, registry_last_used = self._get_registry_usage(name)
        
        # Use registry last_used if not explicitly provided
        if last_used is None and registry_last_used is not None:
            last_used = registry_last_used
        
        self._cache.register_tool(
            name, 
            tool, 
            protected=protected, 
            semantic_group=semantic_group,
            created_at=created_at,
            last_used=last_used
        )
        
        # Set the initial use count from registry data
        if registry_use_count > 0:
            metrics = self._cache.get_metrics(name)
            if metrics:
                metrics.total_calls = registry_use_count
                print(f"[ToolDecayManager] Loaded {name} with use_count={registry_use_count} from registry")
        
        if protected:
            self.protected_tools.add(name)
        
        # Update registry status to 'active'
        self._update_registry_status(name, "active")
    
    def unregister_tool(self, name: str) -> Optional[Any]:
        """Remove a tool manually."""
        return self._cache.unregister_tool(name)
    
    def mark_used(self, name: str) -> bool:
        """
        Mark a tool as used (legacy method).
        
        For detailed tracking with execution time, use record_usage() instead.
        """
        metrics = self._cache.record_usage(name, execution_time_ms=0.0)
        return metrics is not None
    
    def record_usage(
        self, 
        name: str, 
        execution_time_ms: float = 0.0
    ) -> Optional[ToolMetrics]:
        """
        Record a tool usage with execution time.
        
        This is the preferred method for tracking tool usage as it
        enables accurate decay score calculation.
        
        Args:
            name: Tool name
            execution_time_ms: Execution time in milliseconds
        
        Returns:
            Updated ToolMetrics or None if tool not found
        
        Note: success parameter is removed because tools always succeed.
        """
        metrics = self._cache.record_usage(name, execution_time_ms)
        
        # Update registry with usage stats
        if metrics is not None:
            self._update_registry_usage(name)
        
        return metrics
    
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool and record a cache hit."""
        return self._cache.get_tool(name)
    
    # ========== Tool Queries ==========
    
    def get_all_tools(self) -> Dict[str, Any]:
        """Get all active tools as a dictionary."""
        return dict(self._cache._tools)
    
    def get_active_tools(self) -> List[str]:
        """Get names of all active tools."""
        return list(self._cache._tools.keys())
    
    def get_usage_stats(self, name: str) -> Optional[ToolUsageStats]:
        """
        Get usage stats for a specific tool.
        
        Returns a ToolUsageStats object for backward compatibility.
        """
        metrics = self._cache.get_metrics(name)
        if metrics:
            return ToolUsageStats.from_metrics(metrics)
        return None
    
    def get_metrics(self, name: str) -> Optional[ToolMetrics]:
        """Get detailed metrics for a specific tool (new API)."""
        return self._cache.get_metrics(name)
    
    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        """Get metrics for all tools."""
        return self._cache.get_all_metrics()
    
    # ========== Cleanup Operations ==========
    
    def cleanup_expired_tools(self) -> List[str]:
        """
        Remove expired tools (exceeding their TTL).
        
        Returns:
            List of evicted tool names
        """
        expired = self._cache.cleanup_expired()
        return [name for name, _ in expired]
    
    def cleanup_low_performers(self, min_score: float = 0.01) -> List[str]:
        """
        Remove tools with decay score below threshold.
        
        Args:
            min_score: Minimum decay score to keep
        
        Returns:
            List of evicted tool names
        """
        return self._cache.cleanup_low_performers(min_score)
    
    def start_background_cleanup(self, interval: float = 60.0):
        """Start background thread for automatic cleanup."""
        self._cache.start_background_cleanup(interval)
    
    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        self._cache.stop_background_cleanup()
    
    # ========== Protection ==========
    
    def protect_tool(self, name: str) -> bool:
        """
        Mark a tool as protected (never evicted).
        
        Returns:
            True if tool was found and protected, False otherwise
        """
        self.protected_tools.add(name)
        
        # Use cache's internal method properly
        metrics = self._cache.get_metrics(name)
        if metrics:
            # Update through the cache
            old_tier = metrics.tier
            metrics.tier = ToolTier.PROTECTED
            
            # Update tier tracking in cache
            with self._cache._lock:
                self._cache._tiers[old_tier].discard(name)
                self._cache._tiers[ToolTier.PROTECTED].add(name)
                self._cache.protected_tools.add(name)
            
            return True
        return False
    
    def unprotect_tool(self, name: str) -> bool:
        """
        Remove protection from a tool.
        
        Returns:
            True if tool was found and unprotected, False otherwise
        """
        self.protected_tools.discard(name)
        
        metrics = self._cache.get_metrics(name)
        if metrics and metrics.tier == ToolTier.PROTECTED:
            with self._cache._lock:
                self._cache._tiers[ToolTier.PROTECTED].discard(name)
                self._cache.protected_tools.discard(name)
                
                # Re-evaluate tier based on usage
                if metrics.total_calls >= self._cache.HOT_THRESHOLD:
                    metrics.tier = ToolTier.HOT
                    self._cache._tiers[ToolTier.HOT].add(name)
                elif metrics.total_calls >= self._cache.WARM_THRESHOLD:
                    metrics.tier = ToolTier.WARM
                    self._cache._tiers[ToolTier.WARM].add(name)
                else:
                    metrics.tier = ToolTier.COLD
                    self._cache._tiers[ToolTier.COLD].add(name)
            return True
        return False
    
    # ========== Advanced Features (New) ==========
    
    def get_eviction_candidates(self, n: int = 5) -> List[tuple]:
        """
        Get top N tools most likely to be evicted.
        
        Returns:
            List of (name, decay_score) tuples, sorted lowest first
        """
        return self._cache.get_eviction_candidates(n)
    
    def boost_group(self, group: str, ttl_multiplier: float = 1.5):
        """
        Boost TTL for all tools in a semantic group.
        
        Args:
            group: Semantic group name (e.g., "file_ops", "network")
            ttl_multiplier: Multiplier for current TTL
        """
        self._cache.boost_group(group, ttl_multiplier)
    
    def get_tier_breakdown(self) -> Dict[str, List[str]]:
        """Get tools organized by tier."""
        return self._cache.get_tier_breakdown()
    
    def get_group_tools(self, group: str) -> List[str]:
        """Get all tools in a semantic group."""
        return self._cache.get_group_tools(group)
    
    # ========== Archive Management ==========
    
    def restore_from_archive(self, name: str) -> Optional[Any]:
        """
        Restore a tool from archive back to active cache.
        
        Tools are archived (not deleted) when evicted. Use this to bring them back.
        
        Args:
            name: Name of the tool to restore
        
        Returns:
            The restored tool instance, or None if not found in archive
        """
        tool = self._cache.restore_from_archive(name)
        if tool is not None:
            # Update registry status to 'active'
            self._update_registry_status(name, "active")
        return tool
    
    def get_archived_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all archived tools with their metadata.
        
        Returns:
            Dict mapping tool name to metadata including:
            - archived_at: ISO timestamp
            - age_seconds: How long it's been archived
            - ttl_remaining_seconds: Time until permanent deletion
            - reason: Why it was archived
            - will_delete_at: When it will be permanently deleted
        """
        return self._cache.get_archived_tools()
    
    def is_archived(self, name: str) -> bool:
        """Check if a tool is in the archive."""
        return self._cache.is_archived(name)
    
    def cleanup_archive(self) -> List[str]:
        """
        Permanently delete archived tools that have exceeded archive_ttl.
        
        Returns:
            List of permanently deleted tool names
        """
        return self._cache.cleanup_archive()
    
    def get_or_restore(self, name: str) -> Optional[Any]:
        """
        Get a tool from cache, or restore from archive if available.
        
        This is the smart getter that automatically restores archived tools.
        Use this instead of get_tool() if you want automatic restoration.
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance or None
        """
        # Check if tool is archived before calling get_or_restore
        was_archived = self._cache.is_archived(name)
        
        tool = self._cache.get_or_restore(name)
        
        # If tool was restored from archive, update registry status
        if tool is not None and was_archived:
            self._update_registry_status(name, "active")
        
        return tool
    
    # ========== Reporting ==========
    
    def get_status_report(self) -> str:
        """Get human-readable status report."""
        return self._cache.get_status_report()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_statistics()
    
    def export_metrics_json(self) -> Dict[str, Any]:
        """
        Export all metrics in JSON format.
        
        Compatible with METRICS_DOCUMENTATION.md structure.
        """
        return self._cache.export_metrics_json()
    
    # ========== Dunder Methods ==========
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, name: str) -> bool:
        return name in self._cache
    
    def __repr__(self) -> str:
        archived_count = len(self._cache._archive)
        return f"ToolDecayManager(active={len(self)}, archived={archived_count}, decay_period={self.decay_period}s)"


# ============================================================
#                    CONVENIENCE FUNCTIONS
# ============================================================

def create_decay_manager(
    decay_minutes: float = 5.0,
    protected_tools: Optional[List[str]] = None,
    auto_cleanup: bool = True,
    cleanup_interval: float = 60.0,
    max_capacity: int = 100,
    decay_constant: float = 60.0,
    registry_path: Optional[str] = None
) -> ToolDecayManager:
    """
    Create a pre-configured ToolDecayManager.
    
    This is a convenience function for backward compatibility.
    Consider using create_adaptive_cache() for new code.
    
    Args:
        decay_minutes: Base TTL in minutes (default: 5)
        protected_tools: Tools that never get evicted
        auto_cleanup: Start background cleanup automatically
        cleanup_interval: Cleanup check interval in seconds
        max_capacity: Maximum tools in cache
        decay_constant: K constant in decay formula
        registry_path: Path to registry.json for status sync (optional)
    
    Returns:
        Configured ToolDecayManager
    """
    manager = ToolDecayManager(
        decay_period=decay_minutes * 60,
        protected_tools=protected_tools,
        max_capacity=max_capacity,
        decay_constant=decay_constant,
        registry_path=registry_path
    )
    
    if auto_cleanup:
        manager.start_background_cleanup(interval=cleanup_interval)
    
    return manager


# Re-export for convenience
__all__ = [
    "ToolDecayManager",
    "ToolUsageStats", 
    "ToolMetrics",
    "ToolTier",
    "AdaptiveToolCache",
    "create_decay_manager",
    "create_adaptive_cache",
]
