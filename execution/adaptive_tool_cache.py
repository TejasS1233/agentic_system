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
<<<<<<< Updated upstream
    
=======

    # LRU-style inactive tracking
    # When decay_score <= 0.1, tool becomes inactive and has 1 day to live
    inactive_since: Optional[float] = None  # Timestamp when tool became inactive
    is_active: bool = True  # Active status for LRU priority

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    
=======

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
        Get remaining TTL for inactive tool.
        
        Inactive tools have a fixed TTL (default 1 day = 86400 seconds).
        Returns remaining seconds, or infinity if tool is active.
        """
        if self.is_active or self.inactive_since is None:
            return float('inf')  # Active tools never expire via this mechanism
        
        time_inactive = time.time() - self.inactive_since
        return max(0, inactive_ttl - time_inactive)

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
            "age_seconds": round(self.age, 2)
=======
            "age_seconds": round(self.age, 2),
            "is_active": self.is_active,
            "inactive_since": self.inactive_since,
            "inactive_ttl_remaining": round(self.get_inactive_ttl_remaining(), 2) if not self.is_active else None,
>>>>>>> Stashed changes
        }
    
    def __repr__(self):
<<<<<<< Updated upstream
        return (f"ToolMetrics({self.name}, calls={self.total_calls}, "
                f"tier={self.tier.value})")
=======
        status = "active" if self.is_active else "inactive"
        return (
            f"ToolMetrics({self.name}, calls={self.total_calls}, "
            f"tier={self.tier.value}, status={status})"
        )
>>>>>>> Stashed changes


class AdaptiveToolCache:
    """
    Adaptive Replacement Cache (ARC) for tools with:
    - LRU (Least Recently Used) tracking
    - LFU (Least Frequently Used) tracking
    - Segmented tiers (Hot/Warm/Cold)
    - Dynamic TTL based on usage frequency
    - Semantic grouping for related tools
    
    Eviction Priority (lowest score first):
        score = frequency / (TSU + K)
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
<<<<<<< Updated upstream
        max_archive_size: int = 100   # Max tools in archive
=======
        max_archive_size: int = 100,  # Max tools in archive
        inactive_decay_threshold: float = 0.1,  # Decay score threshold for inactive
        inactive_ttl: float = 86400.0,  # 1 day TTL for inactive tools
>>>>>>> Stashed changes
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
        self.archive_ttl = archive_ttl  # Time before archived tools are permanently deleted
        self.max_archive_size = max_archive_size
        
<<<<<<< Updated upstream
=======
        # LRU-style inactive tool management
        self.inactive_decay_threshold = inactive_decay_threshold  # Score <= this = inactive
        self.inactive_ttl = inactive_ttl  # 1 day default TTL for inactive tools

>>>>>>> Stashed changes
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
        
<<<<<<< Updated upstream
=======
        LRU BEHAVIOR: Using an inactive tool makes it active again!
        This "rescues" the tool from the inactive pool.

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        Evict a single tool using adaptive ARC strategy.
        
        Strategy:
        1. Never evict protected tools
        2. Prefer evicting COLD tier
        3. Use decay score: lowest score = evict first
        4. Track in ghost lists for learning
        """
        # Find eviction candidates (not protected)
        candidates = []
        
=======
        Evict a single tool using LRU-style priority eviction.

        Strategy (Priority Order):
        1. NEVER evict protected tools
        2. FIRST: Evict inactive tools (decay_score <= 0.1)
           - Among inactive: evict the one CLOSEST to deletion (lowest TTL remaining)
           - This is the "LRU for inactive" behavior
        3. ONLY IF no inactive tools: Evict active tools
           - Use original tier-based + decay score strategy
        4. Track in ghost lists for ARC learning
        
        Example with 50 tools (28 active, 22 inactive):
        - New tool arrives, need to evict 1
        - Check inactive tools, find the one with least time remaining
        - Evict that one, NOT any active tool
        """
        # Separate candidates into inactive and active
        inactive_candidates = []
        active_candidates = []

>>>>>>> Stashed changes
        for tier in [ToolTier.COLD, ToolTier.PROBATION, ToolTier.WARM, ToolTier.HOT]:
            for name in self._tiers[tier]:
                if name not in self.protected_tools:
                    metrics = self._metrics[name]
                    score = self._get_decay_score_with_registry(metrics)
<<<<<<< Updated upstream
                    candidates.append((name, score, tier, metrics))
        
        if not candidates:
            print("[ATC] Warning: No eviction candidates (all protected)")
            return
        
        # Sort by score (lowest first = evict first)
        candidates.sort(key=lambda x: x[1])
        
        # Get the victim
        victim_name, score, tier, metrics = candidates[0]
        
        # Decide which ghost list based on ARC parameter
        # If score low due to recency → ghost_lru
        # If score low due to frequency → ghost_lfu
        recency_factor = metrics.time_since_use
        frequency_factor = max(1, metrics.total_calls)
        
        if recency_factor / (frequency_factor + 1) > self._arc_p:
            # Evicted mainly due to recency
            self._add_to_ghost(victim_name, self._ghost_lru)
        else:
            # Evicted mainly due to low frequency
            self._add_to_ghost(victim_name, self._ghost_lfu)
        
        self._remove_tool(victim_name, reason="eviction")
    
=======
                    
                    if not metrics.is_active:
                        # Inactive tool - calculate remaining TTL
                        ttl_remaining = metrics.get_inactive_ttl_remaining(self.inactive_ttl)
                        inactive_candidates.append((name, ttl_remaining, score, tier, metrics))
                    else:
                        # Active tool
                        active_candidates.append((name, score, tier, metrics))

        # Priority 1: Evict from inactive tools first (LRU-style)
        if inactive_candidates:
            # Sort by TTL remaining (lowest first = closest to deletion = evict first)
            inactive_candidates.sort(key=lambda x: x[1])
            
            victim_name, ttl_remaining, score, tier, metrics = inactive_candidates[0]
            print(f"[ATC] LRU Eviction: '{victim_name}' (inactive, TTL remaining: {ttl_remaining:.1f}s)")
            
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

>>>>>>> Stashed changes
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
        
        print(f"[ATC] Archived: '{name}' (reason={reason}, score={self._get_decay_score_with_registry(metrics):.4f})")
        
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
        
        # Callback for archiving
        if self.on_tool_archived and metrics:
            try:
                self.on_tool_archived(name, metrics)
            except Exception as e:
                print(f"[ATC] Archive callback error: {e}")
    
    def _delete_oldest_archived(self):
        """Permanently delete the oldest archived tool."""
        if not self._archive:
            return
        
        # Find oldest
        oldest_name = min(self._archive.keys(), key=lambda n: self._archive[n]["archived_at"])
        self._permanently_delete(oldest_name)
    
    def _permanently_delete(self, name: str):
        """Permanently delete a tool from archive."""
        if name not in self._archive:
            return
        
        del self._archive[name]
        self._stats["permanent_deletions"] += 1
        print(f"[ATC] Permanently deleted: '{name}'")
        
        # Callback for permanent deletion
        if self.on_tool_deleted:
            try:
                self.on_tool_deleted(name)
            except Exception as e:
                print(f"[ATC] Delete callback error: {e}")
    
    # ========== Archive Management ==========
    
    def restore_from_archive(self, name: str) -> Optional[Any]:
        """
        Restore a tool from archive back to active cache.
        
        Args:
            name: Name of the tool to restore
        
        Returns:
            The restored tool instance, or None if not found in archive
        """
        with self._lock:
            if name not in self._archive:
                return None
            
            entry = self._archive.pop(name)
            tool = entry["tool"]
            old_metrics = entry["metrics"]
            
            # Re-register with preserved metrics
            if old_metrics:
                # Check capacity
                if len(self._tools) >= self.max_capacity:
                    self._evict_one()
                
                # Restore tool and metrics
                self._tools[name] = tool
                old_metrics.last_used = time.time()  # Reset last used
                self._metrics[name] = old_metrics
                self._lru_order[name] = time.time()
                
                # Restore tier
                self._tiers[old_metrics.tier].add(name)
            else:
                # No metrics, register fresh
                self.register_tool(name, tool)
            
            self._stats["archive_restorations"] += 1
            print(f"[ATC] Restored from archive: '{name}'")
            
            return tool
    
    def get_archived_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all archived tools with their metadata."""
        with self._lock:
            result = {}
            for name, entry in self._archive.items():
                archived_at = entry["archived_at"]
                age = time.time() - archived_at
                ttl_remaining = self.archive_ttl - age
                
                result[name] = {
                    "archived_at": datetime.fromtimestamp(archived_at).isoformat(),
                    "age_seconds": age,
                    "ttl_remaining_seconds": max(0, ttl_remaining),
                    "reason": entry["reason"],
                    "will_delete_at": datetime.fromtimestamp(archived_at + self.archive_ttl).isoformat(),
                    "metrics": entry["metrics"].to_dict() if entry["metrics"] else None,
                }
            return result
    
    def is_archived(self, name: str) -> bool:
        """Check if a tool is in the archive."""
        return name in self._archive
    
    def cleanup_archive(self) -> List[str]:
        """
        Permanently delete archived tools that have exceeded archive_ttl.
        
        Returns:
            List of permanently deleted tool names
        """
        with self._lock:
            deleted = []
            now = time.time()
            
            for name in list(self._archive.keys()):
                entry = self._archive[name]
                age = now - entry["archived_at"]
                
                if age > self.archive_ttl:
                    self._permanently_delete(name)
                    deleted.append(name)
            
            return deleted
    
    def get_or_restore(self, name: str) -> Optional[Any]:
        """
        Get a tool from cache, or restore from archive if available.
        
        This is the smart getter that automatically restores archived tools.
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance or None
        """
        with self._lock:
            # First check active cache
            tool = self.get_tool(name)
            if tool:
                return tool
            
            # Check archive
            if name in self._archive:
                print(f"[ATC] Tool '{name}' found in archive, restoring...")
                return self.restore_from_archive(name)
            
            return None
<<<<<<< Updated upstream
    
=======

    # ========== LRU-Style Inactive Tool Management ==========

    def _update_inactive_status(self) -> List[str]:
        """
        Update the active/inactive status of all tools based on decay score.
        
        Tools with decay_score <= inactive_decay_threshold (default 0.1) 
        are marked as inactive. They now have 1 day to live.
        
        Returns:
            List of tool names that became inactive in this cycle
        """
        newly_inactive = []
        
        for name in list(self._tools.keys()):
            if name in self.protected_tools:
                continue
            
            metrics = self._metrics.get(name)
            if not metrics:
                continue
            
            score = self._get_decay_score_with_registry(metrics)
            
            # Check if tool should become inactive
            if score <= self.inactive_decay_threshold and metrics.is_active:
                metrics.mark_inactive()
                newly_inactive.append(name)
                print(f"[ATC] Tool '{name}' became inactive (score={score:.4f} <= {self.inactive_decay_threshold})")
        
        return newly_inactive

    def _cleanup_expired_inactive(self) -> List[str]:
        """
        Evict inactive tools that have exceeded their 1-day TTL.
        
        This is DIFFERENT from regular eviction:
        - Regular eviction happens when cache is full (LRU-style, picks closest to deletion)
        - This is automatic cleanup of inactive tools that have lived past their TTL
        
        Returns:
            List of evicted tool names
        """
        evicted = []
        
        for name in list(self._tools.keys()):
            if name in self.protected_tools:
                continue
            
            metrics = self._metrics.get(name)
            if not metrics:
                continue
            
            # Only check inactive tools
            if not metrics.is_active:
                ttl_remaining = metrics.get_inactive_ttl_remaining(self.inactive_ttl)
                
                if ttl_remaining <= 0:
                    # Inactive tool has exceeded its 1-day TTL - evict it
                    print(f"[ATC] Inactive tool '{name}' exceeded TTL, evicting")
                    self._remove_tool(name, reason="inactive_ttl_expired")
                    evicted.append(name)
        
        return evicted

    def get_inactive_tools(self) -> List[Tuple[str, float]]:
        """
        Get all inactive tools with their remaining TTL.
        
        Returns:
            List of (name, ttl_remaining_seconds) tuples, sorted by TTL (lowest first)
        """
        with self._lock:
            inactive = []
            for name, metrics in self._metrics.items():
                if not metrics.is_active:
                    ttl_remaining = metrics.get_inactive_ttl_remaining(self.inactive_ttl)
                    inactive.append((name, ttl_remaining))
            
            # Sort by TTL remaining (closest to deletion first)
            inactive.sort(key=lambda x: x[1])
            return inactive

    def get_active_tool_count(self) -> int:
        """Get count of active tools."""
        with self._lock:
            return sum(1 for m in self._metrics.values() if m.is_active)

    def get_inactive_tool_count(self) -> int:
        """Get count of inactive tools."""
        with self._lock:
            return sum(1 for m in self._metrics.values() if not m.is_active)

>>>>>>> Stashed changes
    # ========== Cleanup & Maintenance ==========
    
    def cleanup_expired(self) -> List[Tuple[str, ToolMetrics]]:
        """
        Remove all tools that have exceeded their dynamic TTL.
        
        Returns:
            List of (name, metrics) tuples for evicted tools
        """
        with self._lock:
            evicted = []
            
            # Check all non-protected tools
            for name in list(self._tools.keys()):
                if name in self.protected_tools:
                    continue
                
                metrics = self._metrics.get(name)
                if metrics and metrics.is_expired():
                    self._remove_tool(name, reason="ttl_expired")
                    evicted.append((name, metrics))
            
            return evicted
    
    def cleanup_low_performers(self, min_score: float = 0.01) -> List[str]:
        """
        Remove tools with decay score below threshold.
        
        Args:
            min_score: Minimum decay score to keep
        
        Returns:
            List of evicted tool names
        """
        with self._lock:
            evicted = []
            
            for name in list(self._tools.keys()):
                if name in self.protected_tools:
                    continue
                
                metrics = self._metrics.get(name)
                if metrics:
                    score = self._get_decay_score_with_registry(metrics)
                    if score < min_score:
                        self._remove_tool(name, reason="low_score")
                        evicted.append(name)
            
            return evicted
    
    def start_background_cleanup(self, interval: float = 60.0, verbose: bool = True):
        """Start background thread for automatic cleanup.
        
        Args:
            interval: Time between cleanup cycles in seconds
            verbose: If True, logs decay scores during each cycle
        """
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._stop_cleanup.clear()
        self._verbose_cleanup = verbose
        
        def cleanup_loop():
            cycle_count = 0
            while not self._stop_cleanup.wait(timeout=interval):
                cycle_count += 1
                with self._lock:
                    # Log current decay scores (verbose mode)
                    if self._verbose_cleanup:
                        self._log_decay_status(cycle_count)
<<<<<<< Updated upstream
                    
=======

                    # UPDATE INACTIVE STATUS based on decay score
                    # Tools with decay_score <= 0.1 become inactive
                    newly_inactive = self._update_inactive_status()
                    
                    # EVICT inactive tools that have exceeded their 1-day TTL
                    expired_inactive = self._cleanup_expired_inactive()

>>>>>>> Stashed changes
                    # Cleanup expired active tools (move to archive)
                    expired = self.cleanup_expired()
                    
                    # Cleanup old archived tools (permanent deletion)
                    deleted = self.cleanup_archive()
                    
                    # Re-tier all tools based on current metrics
                    for name in list(self._metrics.keys()):
                        self._update_tier(name)
<<<<<<< Updated upstream
                    
=======

                    if newly_inactive:
                        print(
                            f"[ATC] LRU Status: {len(newly_inactive)} tools became inactive"
                        )
                    if expired_inactive:
                        print(
                            f"[ATC] LRU Cleanup: {len(expired_inactive)} inactive tools evicted (TTL exceeded)"
                        )
>>>>>>> Stashed changes
                    if expired:
                        print(f"[ATC] Background cleanup: {len(expired)} tools archived due to TTL")
                    if deleted:
                        print(f"[ATC] Background cleanup: {len(deleted)} tools permanently deleted")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        print(f"[ATC] Started background cleanup (interval={interval}s, archive_ttl={self.archive_ttl}s, verbose={verbose})")
    
    def _log_decay_status(self, cycle: int):
        """Log current decay scores for all tools."""
        if not self._metrics:
            return
        
        # Get decay scores sorted by score (lowest first = most likely to be evicted)
        scores = []
        for name, metrics in self._metrics.items():
            if name not in self.protected_tools:
                score = self._get_decay_score_with_registry(metrics)
                time_since_use = metrics.time_since_use
<<<<<<< Updated upstream
                scores.append((name, score, time_since_use, metrics.current_ttl))
        
=======
                is_active = metrics.is_active
                inactive_ttl = metrics.get_inactive_ttl_remaining(self.inactive_ttl) if not is_active else None
                scores.append((name, score, time_since_use, metrics.current_ttl, is_active, inactive_ttl))

>>>>>>> Stashed changes
        if not scores:
            return
        
        scores.sort(key=lambda x: x[1])
        
        # Format time helper
        def fmt_time(secs):
            if secs >= 86400:
                return f"{secs/86400:.1f}d"
            elif secs >= 3600:
                return f"{secs/3600:.1f}h"
            elif secs >= 60:
                return f"{secs/60:.1f}m"
            return f"{secs:.0f}s"
<<<<<<< Updated upstream
        
        print(f"\n[ATC] === Decay Status (cycle #{cycle}) ===")
        for name, score, tsu, ttl in scores[:5]:  # Show top 5 lowest scores
            ttl_remaining = ttl - tsu
            status = "⚠️ EXPIRED" if ttl_remaining <= 0 else f"TTL: {fmt_time(ttl_remaining)}"
            print(f"  {name}: score={score:.4f}, idle={fmt_time(tsu)}, {status}")
        
=======

        # Count active vs inactive
        active_count = sum(1 for s in scores if s[4])
        inactive_count = len(scores) - active_count

        print(f"\n[ATC] === Decay Status (cycle #{cycle}) ===")
        print(f"[ATC] Tools: {active_count} active, {inactive_count} inactive")
        
        for name, score, tsu, ttl, is_active, inactive_ttl in scores[:5]:  # Show top 5 lowest scores
            if is_active:
                ttl_remaining = ttl - tsu
                status = (
                    "⚠️ EXPIRED" if ttl_remaining <= 0 else f"TTL: {fmt_time(ttl_remaining)}"
                )
                print(f"  {name}: score={score:.4f}, idle={fmt_time(tsu)}, {status} [ACTIVE]")
            else:
                status = f"inactive TTL: {fmt_time(inactive_ttl)}" if inactive_ttl > 0 else "⚠️ EXPIRED"
                print(f"  {name}: score={score:.4f}, idle={fmt_time(tsu)}, {status} [INACTIVE]")

>>>>>>> Stashed changes
        if len(scores) > 5:
            print(f"  ... and {len(scores) - 5} more tools")
    
    def stop_background_cleanup(self):
        """Stop the background cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            self._cleanup_thread = None
    
    # ========== Semantic Grouping ==========
    
    def _infer_semantic_group(self, name: str) -> Optional[str]:
        """Infer semantic group from tool name."""
        name_lower = name.lower()
        for group, patterns in self.semantic_groups.items():
            for pattern in patterns:
                if pattern.lower() in name_lower or name_lower in pattern.lower():
                    return group
        return None
    
    def get_group_tools(self, group: str) -> List[str]:
        """Get all tools in a semantic group."""
        with self._lock:
            return [
                name for name, metrics in self._metrics.items()
                if metrics.semantic_group == group
            ]
    
    def boost_group(self, group: str, ttl_multiplier: float = 1.5):
        """Boost TTL for all tools in a semantic group."""
        with self._lock:
            for name in self.get_group_tools(group):
                if name in self._metrics:
                    metrics = self._metrics[name]
                    metrics.current_ttl *= ttl_multiplier
                    print(f"[ATC] Boosted '{name}' TTL to {metrics.current_ttl:.1f}s")
    
    # ========== Queries & Reports ==========
    
    def get_metrics(self, name: str) -> Optional[ToolMetrics]:
        """Get metrics for a specific tool."""
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        """Get metrics for all tools."""
        with self._lock:
            return dict(self._metrics)
    
    def get_tier_breakdown(self) -> Dict[str, List[str]]:
        """Get tools organized by tier."""
        with self._lock:
            return {tier.value: list(names) for tier, names in self._tiers.items()}
    
    def get_eviction_candidates(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N tools most likely to be evicted (lowest scores)."""
        with self._lock:
            candidates = []
            for name, metrics in self._metrics.items():
                if name not in self.protected_tools:
                    score = self._get_decay_score_with_registry(metrics)
                    candidates.append((name, score))
            
            candidates.sort(key=lambda x: x[1])
            return candidates[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total_access = self._stats["hits"] + self._stats["misses"]
            if total_access > 0:
                hit_rate = self._stats["hits"] / total_access
            
            return {
                "total_tools": len(self._tools),
                "max_capacity": self.max_capacity,
                "utilization_percent": len(self._tools) / self.max_capacity * 100,
                "cache_hits": self._stats["hits"],
                "cache_misses": self._stats["misses"],
                "hit_rate_percent": round(hit_rate * 100, 2),
                "total_evictions": self._stats["evictions"],
                "tier_promotions": self._stats["tier_promotions"],
                "tier_demotions": self._stats["tier_demotions"],
                "arc_parameter": round(self._arc_p, 3),
                "ghost_lru_size": len(self._ghost_lru),
                "ghost_lfu_size": len(self._ghost_lfu),
                "protected_count": len(self.protected_tools),
            }
    
    def get_status_report(self) -> str:
        """Generate human-readable status report."""
        with self._lock:
            stats = self.get_statistics()
            
            lines = [
                "═" * 60,
                "         ADAPTIVE TOOL CACHE STATUS REPORT",
                "═" * 60,
                f"Capacity: {stats['total_tools']}/{stats['max_capacity']} ({stats['utilization_percent']:.1f}%)",
                f"Hit Rate: {stats['hit_rate_percent']:.1f}% ({stats['cache_hits']} hits / {stats['cache_misses']} misses)",
                f"Evictions: {stats['total_evictions']}",
                f"ARC Parameter (p): {stats['arc_parameter']:.3f} (0=LFU, 1=LRU)",
                "",
                "TIER BREAKDOWN:",
            ]
            
            for tier in ToolTier:
                tools = self._tiers[tier]
                lines.append(f"  {tier.value.upper():12} : {len(tools)} tools")
            
            lines.append("")
            lines.append("TOP 5 EVICTION CANDIDATES (lowest scores):")
            
            for name, score in self.get_eviction_candidates(5):
                metrics = self._metrics.get(name)
                if metrics:
                    lines.append(
                        f"  • {name}: score={score:.4f}, "
                        f"calls={metrics.total_calls}, "
                        f"tsu={metrics.time_since_use:.0f}s"
                    )
            
            lines.append("")
            lines.append("SEMANTIC GROUPS:")
            groups = defaultdict(list)
            for name, metrics in self._metrics.items():
                if metrics.semantic_group:
                    groups[metrics.semantic_group].append(name)
            
            for group, tools in groups.items():
                lines.append(f"  {group}: {len(tools)} tools")
            
            lines.append("═" * 60)
            
            return "\n".join(lines)
    
    def export_metrics_json(self) -> Dict[str, Any]:
        """
        Export all metrics in JSON format.
        Compatible with METRICS_DOCUMENTATION.md structure.
        """
        with self._lock:
            tool_metrics = {}
            for name, metrics in self._metrics.items():
                tool_metrics[name] = metrics.to_dict()
            
            return {
                "cache_statistics": self.get_statistics(),
                "tier_breakdown": self.get_tier_breakdown(),
                "tool_metrics": tool_metrics,
                "decay_formula": {
                    "formula": "score = (freq × success%) / (TSU + K)",
                    "k_constant": self.decay_constant,
                    "base_ttl_seconds": self.base_ttl,
                },
                "timestamp": datetime.now().isoformat(),
            }
    
    # ========== Dunder Methods ==========
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __repr__(self):
        return f"AdaptiveToolCache(size={len(self)}/{self.max_capacity}, arc_p={self._arc_p:.2f})"


# ============================================================
#                    CONVENIENCE FUNCTIONS
# ============================================================

def create_adaptive_cache(
    max_tools: int = 50,
    base_ttl_minutes: float = 5.0,
    protected_tools: Optional[List[str]] = None,
    auto_cleanup: bool = True,
    cleanup_interval: float = 60.0
) -> AdaptiveToolCache:
    """
    Create a pre-configured AdaptiveToolCache.
    
    Args:
        max_tools: Maximum number of tools to cache
        base_ttl_minutes: Base TTL in minutes
        protected_tools: Tools that never get evicted
        auto_cleanup: Start background cleanup automatically
        cleanup_interval: Cleanup check interval in seconds
    
    Returns:
        Configured AdaptiveToolCache
    """
    cache = AdaptiveToolCache(
        max_capacity=max_tools,
        base_ttl=base_ttl_minutes * 60,
        protected_tools=protected_tools
    )
    
    if auto_cleanup:
        cache.start_background_cleanup(interval=cleanup_interval)
    
    return cache
