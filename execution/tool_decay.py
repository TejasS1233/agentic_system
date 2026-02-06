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
)


@dataclass
class ToolUsageStats:
    """
    Backward-compatible wrapper for ToolMetrics.
    """
    name: str
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    created_at: float = field(default_factory=time.time)
    total_execution_time_ms: float = 0.0
    
    def mark_used(self, execution_time_ms: float = 0.0):
        self.last_used = time.time()
        self.use_count += 1
        self.total_execution_time_ms += execution_time_ms
    
    def time_since_last_use(self) -> float:
        return time.time() - self.last_used
    
    def is_expired(self, decay_period: float) -> bool:
        return self.time_since_last_use() > decay_period
    
    def calculate_decay_score(self, k: float = 60.0) -> float:
        import math
        frequency = self.use_count
        tsu = self.time_since_last_use()
        age = time.time() - self.created_at
        BASE_HALF_LIFE = 9400.0
        
        if frequency == 0:
            age_hours = age / 3600.0
            if age_hours < 1: return 1.0
            elif age_hours < 6: return 0.5
            else: return 0.05
        
        multiplier = 1.0 + min(frequency * 0.01, 2.0)
        effective_half_life = BASE_HALF_LIFE * multiplier
        return math.exp(-tsu / effective_half_life)
    
    @classmethod
    def from_metrics(cls, metrics: ToolMetrics) -> "ToolUsageStats":
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
    """
    
    def __init__(
        self,
        decay_period: float = 300.0,
        on_tool_removed: Optional[Callable[[str], None]] = None,
        protected_tools: Optional[List[str]] = None,
        max_capacity: int = 50,  # Default changed to 50 as requested
        decay_constant: float = 60.0,
        semantic_groups: Optional[Dict[str, List[str]]] = None,
        registry_path: Optional[str] = None,
        inactive_ttl: float = 86400.0,
    ):
        self.decay_period = decay_period
        self.on_tool_removed = on_tool_removed
        self.protected_tools = set(protected_tools or [])
        self.decay_constant = decay_constant
        self.inactive_ttl = inactive_ttl
        
        # Registry path for status sync
        self.registry_path = registry_path
        if registry_path is None:
            workspace_root = os.path.join(os.getcwd(), "workspace")
            self.registry_path = os.path.join(workspace_root, "tools", "registry.json")
        
        self._cache = AdaptiveToolCache(
            max_capacity=max_capacity,
            base_ttl=decay_period,
            decay_constant=decay_constant,
            on_tool_evicted=self._handle_eviction,
            protected_tools=list(self.protected_tools),
            semantic_groups=semantic_groups,
            inactive_ttl=inactive_ttl,
        )
        
        self._cache.set_registry_path(self.registry_path)
    
    def _update_registry_status(self, tool_name: str, status: str, inactive_since: float = None) -> bool:
        """Update status in registry.json."""
        if not self.registry_path or not os.path.exists(self.registry_path):
            return False
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            if tool_name in registry:
                registry[tool_name]["status"] = status
                registry[tool_name]["updated_at"] = time.time()
                
                if status == "inactive":
                    registry[tool_name]["inactive_since"] = inactive_since or time.time()
                elif status == "active":
                    registry[tool_name].pop("inactive_since", None)

                with open(self.registry_path, "w") as f:
                    # preserve order if possible, but json.dump usually doesn't
                    json.dump(registry, f, indent=2)
                return True
            return False
        except Exception as e:
            print(f"[ToolDecayManager] Failed to update registry: {e}")
            return False
    
    def _get_registry_status(self, tool_name: str) -> Optional[str]:
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
        if not self.registry_path or not os.path.exists(self.registry_path):
            return False
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            if tool_name in registry:
                registry[tool_name]["use_count"] = registry[tool_name].get("use_count", 0) + 1
                registry[tool_name]["last_used"] = time.time()
                with open(self.registry_path, "w") as f:
                    json.dump(registry, f, indent=2)
                return True
            return False
        except Exception:
            return False
    
    def _handle_eviction(self, name: str, metrics: ToolMetrics):
        """Callback for tool eviction (moves to inactive state visually, or archived)."""
        # In the new logic, eviction means 'removed from active cache'. 
        # But wait, AdaptiveToolCache puts it in 'archive' dictionary.
        # The user's request: "instead of tool being evicted... only if limit reaches 50".
        # So 'eviction' = removal from cache due to capacity.
        self._update_registry_status(name, "inactive")
        if self.on_tool_removed:
            try:
                self.on_tool_removed(name)
            except Exception:
                pass
    
    def _get_registry_usage(self, tool_name: str) -> tuple:
        if not self.registry_path or not os.path.exists(self.registry_path):
            return (0, None)
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            if tool_name in registry:
                return (registry[tool_name].get("use_count", 0), registry[tool_name].get("last_used"))
            return (0, None)
        except Exception:
            return (0, None)

    def _get_registry_inactive_since(self, tool_name: str) -> Optional[float]:
        if not self.registry_path or not os.path.exists(self.registry_path):
            return None
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            if tool_name in registry:
                return registry[tool_name].get("inactive_since")
            return None
        except Exception:
            return None

    def register_tool(self, name: str, tool: Any, protected: bool = False, semantic_group: Optional[str] = None, created_at: Optional[float] = None, last_used: Optional[float] = None):
        registry_status = self._get_registry_status(name)
        if registry_status in ["deprecated", "failed"]:
            return
        
        registry_use_count, registry_last_used = self._get_registry_usage(name)
        registry_inactive_since = self._get_registry_inactive_since(name)

        if last_used is None and registry_last_used is not None:
            last_used = registry_last_used
        
        self._cache.register_tool(name, tool, protected=protected, semantic_group=semantic_group, created_at=created_at, last_used=last_used)
        
        if registry_use_count > 0:
            metrics = self._cache.get_metrics(name)
            if metrics:
                metrics.total_calls = registry_use_count
        
        if registry_status == "inactive" and registry_inactive_since:
            metrics = self._cache.get_metrics(name)
            if metrics:
                metrics.is_active = False
                metrics.inactive_since = registry_inactive_since
            if protected:
                self.protected_tools.add(name)
            return

        if protected:
            self.protected_tools.add(name)
        
        self._update_registry_status(name, "active")
    
    def unregister_tool(self, name: str) -> Optional[Any]:
        return self._cache.unregister_tool(name)
    
    def mark_used(self, name: str) -> bool:
        metrics = self._cache.record_usage(name, execution_time_ms=0.0)
        return metrics is not None
    
    def record_usage(self, name: str, execution_time_ms: float = 0.0) -> Optional[ToolMetrics]:
        was_inactive = False
        old_metrics = self._cache.get_metrics(name)
        if old_metrics and not old_metrics.is_active:
            was_inactive = True
        
        metrics = self._cache.record_usage(name, execution_time_ms)
        
        if metrics is not None:
            self._update_registry_usage(name)
            if was_inactive:
                self._update_registry_status(name, "active")
        return metrics
    
    def get_tool(self, name: str) -> Optional[Any]:
        return self._cache.get_tool(name)
    
    def get_all_tools(self) -> Dict[str, Any]:
        return dict(self._cache._tools)
    
    def get_active_tools(self) -> List[str]:
        return list(self._cache._tools.keys())
    
    def get_usage_stats(self, name: str) -> Optional[ToolUsageStats]:
        metrics = self._cache.get_metrics(name)
        if metrics:
            return ToolUsageStats.from_metrics(metrics)
        return None
    
    def get_metrics(self, name: str) -> Optional[ToolMetrics]:
        return self._cache.get_metrics(name)
    
    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        return self._cache.get_all_metrics()
    
    def cleanup_expired_tools(self) -> List[str]:
        """
        Cleanup tools that are expired.
        
        NOTE: With the new capacity-only eviction strategy, this NO LONGER deletes
        tools simply because they are 'old' or 'inactive'.
        It delegates to the cache, which only cleans up the archive.
        """
        # Delegated to cache - now mostly controls archive cleanup
        self._cache.cleanup_expired()
        
        # We also used to check for inactive tools > 1 day here and delete them.
        # That logic has been removed to satisfy the "only evict on capacity 50" rule.
        
        return []

    def start_background_cleanup(self, interval: float = 60.0):
        self._cache.start_background_cleanup(interval)
    
    def stop_background_cleanup(self):
        self._cache.stop_background_cleanup()
    
    def protect_tool(self, name: str) -> bool:
        self.protected_tools.add(name)
        metrics = self._cache.get_metrics(name)
        if metrics:
            with self._cache._lock:
                # Direct access for simpler implementation in wrapper
                metrics.tier = ToolTier.PROTECTED
                self._cache._tiers[ToolTier.PROTECTED].add(name)
                self._cache.protected_tools.add(name)
            return True
        return False
    
    def unprotect_tool(self, name: str) -> bool:
        self.protected_tools.discard(name)
        metrics = self._cache.get_metrics(name)
        if metrics and metrics.tier == ToolTier.PROTECTED:
            with self._cache._lock:
                self._cache._tiers[ToolTier.PROTECTED].discard(name)
                self._cache.protected_tools.discard(name)
                # Re-eval tier logic simplified
                metrics.tier = ToolTier.COLD 
                self._cache._tiers[ToolTier.COLD].add(name)
            return True
        return False

    def get_eviction_candidates(self, n: int = 5) -> List[tuple]:
        return self._cache.get_eviction_candidates(n)
        
    def boost_group(self, group: str, ttl_multiplier: float = 1.5):
        self._cache.boost_group(group, ttl_multiplier)
        
    def get_tier_breakdown(self) -> Dict[str, List[str]]:
        return self._cache.get_tier_breakdown()
        
    def get_group_tools(self, group: str) -> List[str]:
        return self._cache.get_group_tools(group)

    def get_inactive_tools(self) -> List[tuple]:
        return self._cache.get_inactive_tools()

    def get_active_tool_count(self) -> int:
        return self._cache.get_active_tool_count()

    def get_inactive_tool_count(self) -> int:
        return self._cache.get_inactive_tool_count()

    def get_lru_status(self) -> Dict[str, Any]:
        inactive_tools = self._cache.get_inactive_tools()
        return {
            "active_count": self._cache.get_active_tool_count(),
            "inactive_count": self._cache.get_inactive_tool_count(),
            "inactive_tools": [
                {"name": name, "ttl_remaining_seconds": ttl}
                for name, ttl in inactive_tools
            ],
            "next_to_evict": inactive_tools[0][0] if inactive_tools else None,
        }

    def update_inactive_status(self) -> List[str]:
        newly_inactive = self._cache._update_inactive_status()
        for name in newly_inactive:
            metrics = self._cache.get_metrics(name)
            inactive_since = metrics.inactive_since if metrics else None
            self._update_registry_status(name, "inactive", inactive_since)
        return newly_inactive
        
    def _delete_tool_file(self, filename: str) -> bool:
        if not self.registry_path: return False
        try:
            tools_dir = os.path.dirname(self.registry_path)
            file_path = os.path.join(tools_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False

    def _delete_from_registry(self, tool_name: str) -> bool:
        if not self.registry_path or not os.path.exists(self.registry_path): return False
        try:
            with open(self.registry_path, "r") as f: registry = json.load(f)
            if tool_name in registry:
                del registry[tool_name]
                with open(self.registry_path, "w") as f: json.dump(registry, f, indent=2)
                return True
            return False
        except Exception:
            return False
