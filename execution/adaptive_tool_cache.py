"""
Adaptive Tool Cache (ATC) for IASCIS

A sophisticated tool management system combining multiple caching strategies:
- ARC (Adaptive Replacement Cache): LFU + LRU hybrid
- Segmented LRU: Tier-based tool organization
- Dynamic TTL: Adaptive time-to-live based on frequency
- Semantic Grouping: Tools categorized by functionality

Decay Score Formula (Linear Frequency Scaling):
    half_life = 9400 × (1 + min(frequency × 0.01, 2.0))
    score = exp(-TSU / half_life)

Where:
    - 9400 seconds (~2.6 hours) = base half-life
    - Frequency bonus = 0.01 per call (gentle linear scaling)
    - Max bonus = 2.0 (total multiplier 3x) at 200+ calls

Examples (Survival Time):
    - 1 call:   6.0 hours (baseline)
    - 10 calls: 6.6 hours
    - 100 calls: 12.0 hours
    - 200+ calls: 18.0 hours (max capped)

Tools with score < 0.1 are candidates for eviction.
"""

import time
import math
import threading
from typing import Dict, Optional, List, Set, Callable, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import OrderedDict, defaultdict


class ToolTier(Enum):
    """Tool tier classification based on usage patterns."""
    PROTECTED = "protected"      # Never decays (core tools)
    HOT = "hot"                  # Frequently used, high priority
    WARM = "warm"                # Moderate usage
    COLD = "cold"                # Rarely used, decay candidates
    PROBATION = "probation"     # Recently added, being evaluated


@dataclass
class ToolMetrics:
    """
    Comprehensive metrics for a single tool.
    Aligns with METRICS_DOCUMENTATION.md tool metrics.
    
    Note: Success/failure tracking is removed because tools are validated
    before registration, so they always succeed when executed.
    """
    name: str
    
    # Usage metrics
    total_calls: int = 0
    
    # Timing metrics
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    total_execution_time_ms: float = 0.0
    
    # Tier management
    tier: ToolTier = field(default=ToolTier.PROBATION)
    tier_promoted_at: Optional[float] = None
    tier_demoted_at: Optional[float] = None
    
    # Semantic grouping
    semantic_group: Optional[str] = None  # e.g., "file_ops", "network", "data_processing"
    
    # Dynamic TTL
    base_ttl: float = 300.0  # Base TTL in seconds
    current_ttl: float = 300.0  # Adaptive TTL

    # LRU-style inactive tracking
    # When decay_score <= 0.1, tool becomes inactive.
    # It remains inactive until used (rescue) or evicted by capacity limit.
    inactive_since: Optional[float] = None  # Timestamp when tool became inactive
    is_active: bool = True  # Active status for LRU priority

    @property
    def avg_execution_time_ms(self) -> float:
        """Average execution time per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_calls
    
    @property
    def time_since_use(self) -> float:
        """Seconds since last use (TSU)."""
        return time.time() - self.last_used
    
    @property
    def age(self) -> float:
        """Seconds since tool was created."""
        return time.time() - self.created_at
    
    def record_call(self, execution_time_ms: float = 0.0):
        """
        Record a tool invocation.
        
        Note: success parameter is removed because tools always succeed.
        """
        self.total_calls += 1
        self.last_used = time.time()
        self.total_execution_time_ms += execution_time_ms
        
        # Adjust dynamic TTL based on frequency
        self._adjust_ttl()
    
    def _adjust_ttl(self):
        """
        Dynamically adjust TTL based on tool frequency.
        
        Higher frequency = longer TTL (tool is valuable, keep it around)
        Lower frequency = shorter TTL (tool may decay sooner)
        
        Note: Success rate is NOT considered because tools always succeed
        (they are validated before registration).
        """
        # Frequency boost (logarithmic scale)
        # More calls = higher boost (1.0x to ~1.6x for 100 calls)
        freq_boost = 1.0 + math.log10(max(1, self.total_calls)) * 0.5
        
        # Calculate new TTL (base * frequency factor)
        self.current_ttl = self.base_ttl * freq_boost
        
        # Clamp TTL between 60s and 3600s (1 min to 1 hour)
        self.current_ttl = max(60.0, min(3600.0, self.current_ttl))
    
    def calculate_decay_score(self, k: float = 60.0, registry_data: dict = None) -> float:
        """
        Calculate decay score using linear frequency scaling.
        
        Formula:
            half_life = 9400 × (1 + min(frequency × 0.01, 2.0))
            score = exp(-TSU / half_life)
        
        Key insight: More uses = SLOWER decay (frequency extends half-life)
        
        Examples (time to reach score=0.01 eviction threshold):
            - 1 call:     half-life = 2.6h,  survives ~12 hours
            - 10 calls:   half-life = 2.9h,  survives ~13 hours
            - 100 calls:  half-life = 5.2h,  survives ~24 hours
            - 200+ calls: half-life = 7.8h,  survives ~36 hours (max capped)
        
        Args:
            k: Unused, kept for API compatibility
            registry_data: Unused, kept for API compatibility
        
        Returns:
            Decay score (higher = keep, lower = evict candidate)
        """
        frequency = self.total_calls
        tsu = self.time_since_use
        
        # Base half-life: 9400 seconds (~2.6 hours)
        # At threshold 0.01: TSU_max = -H × ln(0.01) = H × 4.605 ≈ 12 hours
        BASE_HALF_LIFE = 9400.0
        
        # Handle never-used tools (fixed grace period based on age)
        if frequency == 0:
            age_hours = self.age / 3600.0
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
        score = math.exp(-tsu / effective_half_life)
        
        return score
    
    def is_expired(self) -> bool:
        """Check if tool has exceeded its dynamic TTL."""
        return self.time_since_use > self.current_ttl

    def mark_inactive(self):
        """Mark tool as inactive with timestamp for LRU tracking."""
        if self.is_active:
            self.is_active = False
            self.inactive_since = time.time()

    def mark_active(self):
        """Mark tool as active (resets inactive tracking)."""
        self.is_active = True
        self.inactive_since = None

    def get_inactive_ttl_remaining(self, inactive_ttl: float = 86400.0) -> float:
        """
        Get duration since becoming inactive.
        Using infinity for consistency with old API, but effectively ignored for auto-eviction.
        """
        return float('inf')  # No longer auto-expires

    def to_dict(self) -> dict:
        """Export metrics to dictionary (for JSON serialization)."""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "time_since_use_seconds": round(self.time_since_use, 2),
            "current_ttl_seconds": round(self.current_ttl, 2),
            "tier": self.tier.value,
            "semantic_group": self.semantic_group,
            "decay_score": round(self.calculate_decay_score(), 4),
            "age_seconds": round(self.age, 2),
            "is_active": self.is_active,
            "inactive_since": self.inactive_since,
            "inactive_ttl_remaining": None 
        }
    
    def __repr__(self):
        status = "active" if self.is_active else "inactive"
        return (
            f"ToolMetrics({self.name}, calls={self.total_calls}, "
            f"tier={self.tier.value}, status={status})"
        )


class AdaptiveToolCache:
    """
    Adaptive Replacement Cache (ARC) for tools with:
    - LRU (Least Recently Used) tracking
    - LFU (Least Frequently Used) tracking
    - Segmented tiers (Hot/Warm/Cold)
    - Dynamic TTL based on usage frequency
    - Semantic grouping for related tools
    
    Eviction Priority (Capacity based):
    1. Inactive Tools: Evict Longest Inactive First.
    2. Active Tools: Evict Lowest Decay Score First.
    """
    
    # Tier thresholds
    HOT_THRESHOLD = 10      # 10+ calls = hot
    WARM_THRESHOLD = 3      # 3+ calls = warm
    PROBATION_PERIOD = 60   # 60 seconds before evaluation
    
    # Default constants
    DEFAULT_BASE_TTL = 300.0  # 5 minutes base TTL
    DECAY_CONSTANT_K = 60.0   # Smoothing constant for score formula
    
    def __init__(
        self,
        max_capacity: int = 50,
        base_ttl: float = DEFAULT_BASE_TTL,
        decay_constant: float = DECAY_CONSTANT_K,
        on_tool_evicted: Optional[Callable[[str, ToolMetrics], None]] = None,
        on_tool_archived: Optional[Callable[[str, ToolMetrics], None]] = None,
        on_tool_deleted: Optional[Callable[[str], None]] = None,
        protected_tools: Optional[List[str]] = None,
        semantic_groups: Optional[Dict[str, List[str]]] = None,
        archive_ttl: float = 3600.0,  # 1 hour before permanent deletion
        max_archive_size: int = 100,  # Max tools in archive
        inactive_decay_threshold: float = 0.1,  # Decay score threshold for inactive
        inactive_ttl: float = 86400.0,  # Unused for eviction now, kept for compat
    ):
        """
        Initialize the Adaptive Tool Cache.
        
        Args:
            max_capacity: Maximum number of tools to cache
            base_ttl: Base time-to-live in seconds (will be adjusted dynamically)
            decay_constant: K constant in decay score formula
            on_tool_evicted: Callback when a tool is evicted
            protected_tools: List of tool names that never get evicted
            semantic_groups: Dict mapping group names to tool name patterns
        """
        self.max_capacity = max_capacity
        self.base_ttl = base_ttl
        self.decay_constant = decay_constant
        self.on_tool_evicted = on_tool_evicted
        self.on_tool_archived = on_tool_archived
        self.on_tool_deleted = on_tool_deleted
        self.protected_tools: Set[str] = set(protected_tools or [])
        self.archive_ttl = archive_ttl
        self.max_archive_size = max_archive_size
        
        # LRU-style inactive tool management
        self.inactive_decay_threshold = inactive_decay_threshold
        self.inactive_ttl = inactive_ttl

        # Semantic group mappings (tool_name_pattern -> group_name)
        self.semantic_groups = semantic_groups or {
            "file_ops": ["write_file", "read_file", "delete_file", "list_dir"],
            "execution": ["run_command", "execute_script", "shell"],
            "network": ["http_request", "fetch_url", "api_call"],
            "data": ["parse_json", "transform_data", "query_db"],
        }
        
        # Core data structures
        self._tools: Dict[str, Any] = {}  # name -> tool instance
        self._metrics: Dict[str, ToolMetrics] = {}  # name -> metrics
        
        # LRU tracking (OrderedDict maintains insertion order)
        self._lru_order: OrderedDict = OrderedDict()
        
        # Tier buckets
        self._tiers: Dict[ToolTier, Set[str]] = {
            ToolTier.PROTECTED: set(),
            ToolTier.HOT: set(),
            ToolTier.WARM: set(),
            ToolTier.COLD: set(),
            ToolTier.PROBATION: set(),
        }
        
        # Ghost entries (recently evicted - for ARC learning)
        self._ghost_lru: OrderedDict = OrderedDict()  # Evicted due to recency
        self._ghost_lfu: OrderedDict = OrderedDict()  # Evicted due to frequency
        self._ghost_max_size = max_capacity // 2
        
        # Adaptive parameter (p) for ARC
        # Higher p = favor LRU, Lower p = favor LFU
        self._arc_p: float = 0.5
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "tier_promotions": 0,
            "tier_demotions": 0,
            "archive_restorations": 0,
            "permanent_deletions": 0,
        }
        
        # Archive storage (evicted but not permanently deleted)
        # Structure: {name: {"tool": instance, "metrics": ToolMetrics, "archived_at": timestamp}}
        self._archive: Dict[str, Dict[str, Any]] = {}
        
        # Registry path for enriched data loading
        self.registry_path: Optional[str] = None
        self._registry_cache: Dict[str, dict] = {}  # Cached registry data
        self._registry_cache_time: float = 0.0  # Last cache update time
        self._registry_cache_ttl: float = 30.0  # Cache registry for 30 seconds
    
    def set_registry_path(self, path: str):
        """Set the path to registry.json for enriched decay score calculation."""
        self.registry_path = path
        self._registry_cache = {}  # Clear cache
        self._registry_cache_time = 0.0
    
    def _load_registry_data(self, tool_name: str) -> Optional[dict]:
        """
        Load registry data for a specific tool.
        
        Returns dict with: domain, tags, status, use_count, last_used, created_at
        """
        if not self.registry_path:
            return None
        
        import os
        import json
        
        # Check cache
        now = time.time()
        if now - self._registry_cache_time < self._registry_cache_ttl:
            return self._registry_cache.get(tool_name)
        
        # Reload from file
        if not os.path.exists(self.registry_path):
            return None
        
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            self._registry_cache = registry
            self._registry_cache_time = now
            
            return registry.get(tool_name)
        except Exception as e:
            print(f"[ATC] Failed to load registry: {e}")
            return None
    
    def _get_decay_score_with_registry(self, metrics: ToolMetrics) -> float:
        """Calculate decay score with registry data enhancement."""
        registry_data = self._load_registry_data(metrics.name)
        return metrics.calculate_decay_score(self.decay_constant, registry_data)
    
    # ========== Core Operations ==========
    
    def register_tool(
        self, 
        name: str, 
        tool: Any, 
        protected: bool = False,
        semantic_group: Optional[str] = None,
        created_at: Optional[float] = None,
        last_used: Optional[float] = None
    ) -> ToolMetrics:
        """
        Register a new tool with the cache.
        
        Args:
            name: Unique tool name
            tool: The tool instance
            protected: If True, tool never gets evicted
            semantic_group: Optional semantic category
            created_at: Optional timestamp (Unix epoch) when tool was originally created.
                        If provided, decay will be calculated from this time.
                        If None, uses current time.
            last_used: Optional timestamp (Unix epoch) when tool was last used.
                        If provided, time_since_use will be calculated from this.
                        If None, uses current time.
        
        Returns:
            ToolMetrics object for the registered tool
        """
        with self._lock:
            # Check capacity and evict if needed
            if name not in self._tools and len(self._tools) >= self.max_capacity:
                self._evict_one()
            
            # Use provided timestamps or default to current time
            now = time.time()
            actual_created_at = created_at if created_at is not None else now
            actual_last_used = last_used if last_used is not None else now
            
            # Create metrics with provided or default timestamps
            metrics = ToolMetrics(
                name=name,
                base_ttl=self.base_ttl,
                current_ttl=self.base_ttl,
                semantic_group=semantic_group or self._infer_semantic_group(name),
                created_at=actual_created_at,
                last_used=actual_last_used
            )
            
            # If using historical timestamps, adjust TTL to account for elapsed time
            if created_at is not None or last_used is not None:
                # Recalculate current_ttl based on historical data
                metrics._adjust_ttl()
            
            # Set tier
            if protected:
                metrics.tier = ToolTier.PROTECTED
                self.protected_tools.add(name)
                self._tiers[ToolTier.PROTECTED].add(name)
            else:
                metrics.tier = ToolTier.PROBATION
                self._tiers[ToolTier.PROBATION].add(name)
            
            # Store
            self._tools[name] = tool
            self._metrics[name] = metrics
            self._lru_order[name] = time.time()
            self._lru_order.move_to_end(name)
            
            # Check if it was in ghost (ARC adaptation)
            if name in self._ghost_lru:
                # Tool was evicted for recency but used again
                # Increase preference for LRU
                self._arc_p = min(1.0, self._arc_p + 0.1)
                del self._ghost_lru[name]
            elif name in self._ghost_lfu:
                # Tool was evicted for frequency but used again
                # Increase preference for LFU
                self._arc_p = max(0.0, self._arc_p - 0.1)
                del self._ghost_lfu[name]
            
            print(f"[ATC] Registered: '{name}' (tier={metrics.tier.value}, group={metrics.semantic_group})")
            return metrics
    
    def get_tool(self, name: str) -> Optional[Any]:
        """
        Get a tool by name. Records a cache hit/miss.
        
        This does NOT record a usage - call record_usage() when the tool is executed.
        """
        with self._lock:
            if name in self._tools:
                self._stats["hits"] += 1
                # Update LRU
                self._lru_order.move_to_end(name)
                return self._tools[name]
            else:
                self._stats["misses"] += 1
                return None
    
    def record_usage(
        self, 
        name: str, 
        execution_time_ms: float = 0.0
    ) -> Optional[ToolMetrics]:
        """
        Record a tool usage event.
        
        This should be called after the tool is executed.
        
        LRU BEHAVIOR: Using an inactive tool makes it active again!
        This "rescues" the tool from the inactive pool.

        Args:
            name: Tool name
            execution_time_ms: Time taken to execute
        
        Returns:
            Updated ToolMetrics or None if tool not found
        
        Note: success parameter is removed because tools always succeed.
        """
        with self._lock:
            if name not in self._metrics:
                return None
            
            metrics = self._metrics[name]
            
            # LRU: Using a tool makes it active again (rescue from inactive)
            if not metrics.is_active:
                metrics.mark_active()
                print(f"[ATC] LRU Rescue: '{name}' used, now active again")
            
            metrics.record_call(execution_time_ms)
            
            # Update LRU
            self._lru_order.move_to_end(name)
            
            # Re-evaluate tier
            self._update_tier(name)
            
            return metrics
    
    def unregister_tool(self, name: str) -> Optional[Any]:
        """Remove a tool manually."""
        with self._lock:
            return self._remove_tool(name, reason="manual")
    
    # ========== Tier Management ==========
    
    def _update_tier(self, name: str):
        """Update tool tier based on current metrics."""
        if name not in self._metrics:
            return
        
        metrics = self._metrics[name]
        old_tier = metrics.tier
        
        # Protected tools never change tier
        if old_tier == ToolTier.PROTECTED:
            return
        
        # Determine new tier based on usage
        new_tier = old_tier
        
        if metrics.total_calls >= self.HOT_THRESHOLD:
            new_tier = ToolTier.HOT
        elif metrics.total_calls >= self.WARM_THRESHOLD:
            new_tier = ToolTier.WARM
        elif metrics.age > self.PROBATION_PERIOD:
            # Past probation period with low usage = cold
            new_tier = ToolTier.COLD
        
        # Apply tier change
        if new_tier != old_tier:
            # Remove from old tier
            self._tiers[old_tier].discard(name)
            # Add to new tier
            self._tiers[new_tier].add(name)
            metrics.tier = new_tier
            
            if self._tier_rank(new_tier) > self._tier_rank(old_tier):
                metrics.tier_promoted_at = time.time()
                self._stats["tier_promotions"] += 1
                print(f"[ATC] Promoted: '{name}' ({old_tier.value} → {new_tier.value})")
            else:
                metrics.tier_demoted_at = time.time()
                self._stats["tier_demotions"] += 1
                print(f"[ATC] Demoted: '{name}' ({old_tier.value} → {new_tier.value})")
    
    def _tier_rank(self, tier: ToolTier) -> int:
        """Get numeric rank for tier comparison."""
        ranks = {
            ToolTier.PROTECTED: 100,
            ToolTier.HOT: 3,
            ToolTier.WARM: 2,
            ToolTier.COLD: 1,
            ToolTier.PROBATION: 0,
        }
        return ranks.get(tier, 0)
    
    # ========== Eviction Logic ==========
    
    def _evict_one(self):
        """
        Evict a single tool when capacity is full.

        Strategy (Priority Order):
        1. NEVER evict protected tools
        2. FIRST: Evict inactive tools
           - Among inactive: evict the one that has been inactive the LONGEST (smallest inactive_since)
        3. ONLY IF no inactive tools: Evict active tools
           - Evict based on lowest decay score
        4. Track in ghost lists for ARC learning
        """
        # Separate candidates into inactive and active
        inactive_candidates = []
        active_candidates = []

        for tier in [ToolTier.COLD, ToolTier.PROBATION, ToolTier.WARM, ToolTier.HOT]:
            for name in self._tiers[tier]:
                if name not in self.protected_tools:
                    metrics = self._metrics[name]
                    score = self._get_decay_score_with_registry(metrics)
                    
                    if not metrics.is_active:
                        # Inactive tool
                        inactive_candidates.append((name, metrics.inactive_since or 0.0, score, tier, metrics))
                    else:
                        # Active tool
                        active_candidates.append((name, score, tier, metrics))

        # Priority 1: Evict from inactive tools first (Longest inactive first)
        if inactive_candidates:
            # Sort by inactive_since (ascending -> oldest timestamp first)
            inactive_candidates.sort(key=lambda x: x[1])
            
            victim_name, inactive_since, score, tier, metrics = inactive_candidates[0]
            print(f"[ATC] LRU Eviction: '{victim_name}' (inactive since: {time.ctime(inactive_since)})")
            
            # Track in ghost list
            self._add_to_ghost(victim_name, self._ghost_lru)
            self._remove_tool(victim_name, reason="lru_inactive_eviction")
            return

        # Priority 2: Only evict active tools if NO inactive tools exist
        if active_candidates:
            # Sort by decay score (lowest first = evict first)
            active_candidates.sort(key=lambda x: x[1])
            
            victim_name, score, tier, metrics = active_candidates[0]
            print(f"[ATC] Active Eviction: '{victim_name}' (score={score:.4f}, no inactive tools)")

            # Decide which ghost list based on ARC parameter
            recency_factor = metrics.time_since_use
            frequency_factor = max(1, metrics.total_calls)

            if recency_factor / (frequency_factor + 1) > self._arc_p:
                self._add_to_ghost(victim_name, self._ghost_lru)
            else:
                self._add_to_ghost(victim_name, self._ghost_lfu)

            self._remove_tool(victim_name, reason="active_eviction")
            return

        print("[ATC] Warning: No eviction candidates (all protected)")

    def _add_to_ghost(self, name: str, ghost_list: OrderedDict):
        """Add to ghost list (for ARC learning)."""
        if len(ghost_list) >= self._ghost_max_size:
            ghost_list.popitem(last=False)  # Remove oldest
        ghost_list[name] = time.time()
    
    def _remove_tool(self, name: str, reason: str = "unknown") -> Optional[Any]:
        """
        Remove a tool from the active cache and move to archive.
        
        The tool is NOT permanently deleted - it goes to archive first.
        Use cleanup_archive() to permanently delete old archived tools.
        """
        if name not in self._tools:
            return None
        
        tool = self._tools.pop(name)
        metrics = self._metrics.pop(name, None)
        self._lru_order.pop(name, None)
        
        # Remove from tier
        if metrics:
            self._tiers[metrics.tier].discard(name)
        
        self._stats["evictions"] += 1
        
        # Archive the tool instead of deleting
        self._archive_tool(name, tool, metrics, reason)
        
        print(f"[ATC] Archived: '{name}' (reason={reason})")
        
        # Callback for eviction (moved to archive)
        if self.on_tool_evicted and metrics:
            try:
                self.on_tool_evicted(name, metrics)
            except Exception as e:
                print(f"[ATC] Eviction callback error: {e}")
        
        return tool
    
    def _archive_tool(self, name: str, tool: Any, metrics: Optional[ToolMetrics], reason: str):
        """Move a tool to the archive storage."""
        # If archive is full, permanently delete oldest
        if len(self._archive) >= self.max_archive_size:
            self._delete_oldest_archived()
        
        self._archive[name] = {
            "tool": tool,
            "metrics": metrics,
            "archived_at": time.time(),
            "reason": reason,
        }
        
        if self.on_tool_archived and metrics:
            try:
                self.on_tool_archived(name, metrics)
            except Exception as e:
                print(f"[ATC] Archive callback error: {e}")

    def _delete_oldest_archived(self):
        """Permanently delete the oldest tool in the archive."""
        if not self._archive:
            return
        
        # Find oldest entry
        oldest_name = min(self._archive.items(), key=lambda x: x[1]["archived_at"])[0]
        
        self._archive.pop(oldest_name)
        self._stats["permanent_deletions"] += 1
        
        print(f"[ATC] Permanent Deletion: '{oldest_name}' (archive full)")
        
        if self.on_tool_deleted:
            self.on_tool_deleted(oldest_name)
    
    def restore_from_archive(self, name: str) -> bool:
        """
        Restore a tool from archive to active cache.
        Returns True if restored, False if not found.
        """
        with self._lock:
            if name not in self._archive:
                return False
            
            entry = self._archive.pop(name)
            tool = entry["tool"]
            metrics = entry["metrics"]
            
            # Re-register
            self._tools[name] = tool
            self._metrics[name] = metrics
            self._lru_order[name] = time.time()
            self._lru_order.move_to_end(name)
            
            # Put back in tier
            self._tiers[metrics.tier].add(name)
            
            self._stats["archive_restorations"] += 1
            print(f"[ATC] Restored: '{name}' from archive")
            return True

    def cleanup_expired(self) -> List[Tuple[str, ToolMetrics]]:
        """
        Legacy method for cleaning up expired tools.
        NOW DISABLED for active tools based on TTL.
        Only manages archive cleanup.
        """
        with self._lock:
            # 1. Cleanup Archive (retained)
            self._cleanup_archive()
            
            return []  # No longer auto-evict active tools by time
            
    def _cleanup_archive(self):
        """Remove archived tools that have exceeded archive_ttl."""
        now = time.time()
        to_delete = []
        
        for name, entry in self._archive.items():
            if now - entry["archived_at"] > self.archive_ttl:
                to_delete.append(name)
        
        for name in to_delete:
            self._archive.pop(name)
            self._stats["permanent_deletions"] += 1
            print(f"[ATC] Permanent Deletion: '{name}' (archive TTL expired)")
            
            if self.on_tool_deleted:
                self.on_tool_deleted(name)

    def _cleanup_expired_inactive(self) -> List[str]:
        """
        Cleanup inactive tools that exceeded TTL.
        DISABLED: The user wants eviction ONLY on capacity limit.
        """
        return []

    # ========== Inactive/Active Management ==========

    def _update_inactive_status(self) -> List[str]:
        """
        Update is_active status based on decay score.
        Returns list of newly inactive tools.
        """
        newly_inactive = []
        
        with self._lock:
            for name, metrics in self._metrics.items():
                if name in self.protected_tools:
                    continue
                    
                score = self._get_decay_score_with_registry(metrics)
                
                # If score drops below threshold, mark inactive
                if score <= self.inactive_decay_threshold and metrics.is_active:
                    metrics.mark_inactive()
                    newly_inactive.append(name)
                    print(f"[ATC] Tool '{name}' became INACTIVE (score={score:.4f})")
                
                # Note: We don't auto-reactivate here. 
                # Rescue happens on usage (record_usage).
        
        return newly_inactive

    def get_inactive_tools(self) -> List[tuple]:
        """
        Get all inactive tools sorted by how long they've been inactive.
        Returns: List of (name, seconds_inactive)
        """
        inactive = []
        now = time.time()
        for name, metrics in self._metrics.items():
            if not metrics.is_active:
                time_inactive = now - (metrics.inactive_since or now)
                inactive.append((name, time_inactive))
        
        # Sort by time_inactive descending (longest inactive first)
        inactive.sort(key=lambda x: x[1], reverse=True)
        return inactive

    def get_active_tool_count(self) -> int:
        return sum(1 for m in self._metrics.values() if m.is_active)

    def get_inactive_tool_count(self) -> int:
        return sum(1 for m in self._metrics.values() if not m.is_active)

    # ========== Auxiliary Methods ==========
    
    def _infer_semantic_group(self, name: str) -> str:
        """Guess semantic group from tool name."""
        for group, patterns in self.semantic_groups.items():
            for pattern in patterns:
                if pattern in name:
                    return group
        return "misc"
    
    def get_metrics(self, name: str) -> Optional[ToolMetrics]:
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        return self._metrics.copy()

    def get_eviction_candidates(self, n: int = 5) -> List[tuple]:
        """
        Get top N tools most likely to be evicted.
        Priority: Inactive (longest first) -> Active (score lowest first)
        """
        candidates = []
        
        # 1. Inactive tools
        now = time.time()
        for name, metrics in self._metrics.items():
            if name in self.protected_tools:
                continue
            
            score = self._get_decay_score_with_registry(metrics)
            
            if not metrics.is_active:
                # Rank by time inactive (negative timestamp so oldest comes first)
                rank_score = metrics.inactive_since or now
                candidates.append({
                    "name": name, 
                    "score": score, 
                    "type": "inactive",
                    "sort_key": rank_score
                })
            else:
                candidates.append({
                    "name": name, 
                    "score": score, 
                    "type": "active",
                    "sort_key": score # Low score comes first
                })
        
        # Sort inactive first (by oldest), then active (by lowest score)
        # Custom sort is tricky, let's just split
        inactive = [c for c in candidates if c["type"] == "inactive"]
        active = [c for c in candidates if c["type"] == "active"]
        
        # Sort inactive: oldest inactive_since first (smallest timestamp)
        inactive.sort(key=lambda x: x["sort_key"])
        
        # Sort active: lowest decay score first
        active.sort(key=lambda x: x["sort_key"])
        
        # Combine
        final_list = []
        for c in inactive:
            final_list.append((c["name"], f"Inactive ({time.ctime(c['sort_key'])})"))
        for c in active:
            final_list.append((c["name"], f"Active (score={c['score']:.4f})"))
            
        return final_list[:n]
        
    def boost_group(self, group: str, ttl_multiplier: float = 1.5):
        """Boost TTL for a semantic group."""
        for name, metrics in self._metrics.items():
            if metrics.semantic_group == group:
                metrics.current_ttl *= ttl_multiplier

    def get_tier_breakdown(self) -> Dict[str, List[str]]:
        return {tier.value: list(tools) for tier, tools in self._tiers.items()}

    def get_group_tools(self, group: str) -> List[str]:
        return [
            name for name, metrics in self._metrics.items() 
            if metrics.semantic_group == group
        ]

    def cleanup_low_performers(self, min_score: float = 0.01) -> List[str]:
        """
        Manual cleanup of low performing tools.
        Respected even if auto-eviction is off.
        """
        with self._lock:
            evicted = []
            for name, metrics in list(self._metrics.items()):
                if name in self.protected_tools:
                    continue
                
                score = self._get_decay_score_with_registry(metrics)
                if score < min_score:
                    self._remove_tool(name, reason="low_score_cleanup")
                    evicted.append(name)
            return evicted

    def start_background_cleanup(self, interval: float = 60.0):
        """Start background thread."""
        if self._cleanup_thread is not None:
            return
            
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval,),
            daemon=True
        )
        self._cleanup_thread.start()
        print("[ATC] Background cleanup started")
    
    def stop_background_cleanup(self):
        """Stop background thread."""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join()
            self._cleanup_thread = None
            print("[ATC] Background cleanup stopped")
    
    def _cleanup_loop(self, interval: float):
        """
        Background loop.
        Only cleans up ARCHIVE now. No auto-eviction of active/inactive tools.
        """
        while not self._stop_cleanup.is_set():
            try:
                self.cleanup_expired()
                # Also update inactive status based on decay scores
                self._update_inactive_status()
            except Exception as e:
                print(f"[ATC] Cleanup error: {e}")
            
            # Sleep with interrupt check
            self._stop_cleanup.wait(interval)
