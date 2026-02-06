# Tool Decay System - API Reference

This document provides a detailed reference for all classes and functions in the tool decay and eviction system.

## Files Covered
- `execution/adaptive_tool_cache.py`: Core caching logic, LRU/ARC eviction, and metrics.
- `execution/tool_decay.py`: High-level manager, backward compatibility, and registry synchronization.

---

# 1. execution/adaptive_tool_cache.py

## Class: `ToolMetrics`
Tracks usage statistics, timing, and lifecycle state for a single tool.

### Properties
- **`avg_execution_time_ms`** `-> float`: Returns the average execution time per call (total_time / calls).
- **`time_since_use`** `-> float`: Returns seconds elapsed since `last_used`.
- **`age`** `-> float`: Returns seconds elapsed since `created_at`.

### Methods
- **`record_call(execution_time_ms)`**: Increments usage count, updates `last_used` timestamp, adds execution time, and triggers dynamic TTL adjustment.
- **`_adjust_ttl()`**: Internal method. dynamically lengthens the tool's TTL based on usage frequency (logarithmic scale).
- **`calculate_decay_score(k, registry_data)`** `-> float`: Computes the "liveness" score of a tool.
    - **Formula**: `score = exp(-time_since_use / half_life)`
    - **Half-Life**: Scaling factor that increases with frequency (more use = slower decay).
- **`is_expired()`** `-> bool`: Checks if `time_since_use > current_ttl`. (Note: Currently not used for auto-deletion).
- **`mark_inactive()`**: Sets `is_active = False` and records `inactive_since` timestamp. Used for LRU tracking.
- **`mark_active()`**: Sets `is_active = True` and clears `inactive_since`. Used when a tool is rescued.
- **`to_dict()`** `-> dict`: Serializes metrics including name, calls, tier, decay_score, and active status for JSON export.

---

## Class: `AdaptiveToolCache`
The main in-memory cache implementing ARC (Adaptive Replacement Cache) and Tiered LRU.

### Initialization
- **`__init__(max_capacity=50, ...)`**: Sets up the cache storage (`_tools`), metrics (`_metrics`), eviction queues (`_lru_order`), and tiers (`_tiers`).

### Core Operations
- **`register_tool(name, tool, protected, ...)`** `-> ToolMetrics`: 
    - Adds a new tool to the cache.
    - **Triggers Eviction** if `current_size >= max_capacity`.
    - Initializes metrics (optionally from historical timestamps).
- **`get_tool(name)`** `-> Any`: Retrieves a tool instance without updating usage stats (records a cache hit/miss).
- **`record_usage(name, execution_time_ms)`** `-> ToolMetrics`: 
    - Marks a tool as used.
    - **Rescue Mechanism**: If tool was Inactive, it immediately becomes Active.
    - Updates LRU position and re-evaluates Tier.
- **`unregister_tool(name)`**: Manually removes a tool from the cache.

### Eviction Logic
- **`_evict_one()`**: 
    - The core eviction decision engine.
    - **Priority 1**: Inactive Tools (Sorted by `inactive_since` asc - Longest Inactive First).
    - **Priority 2**: Active Tools (Sorted by `decay_score` asc - Lowest Score First).
    - Moves evicted tool to `_archive` (does not permanently delete).
- **`_remove_tool(name, reason)`**: Removes from active cache, updates tiers, and calls `_archive_tool`.

### Status Updates
- **`_update_inactive_status()`** `-> List[str]`:
    - Scans all tools.
    - Marks tool as **Inactive** if `decay_score <= 0.1`.
    - Returns list of newly inactive tools.

### Query & Reporting
- **`get_eviction_candidates(n)`**: Returns top `n` tools likely to be evicted next.
- **`get_inactive_tools()`**: Returns list of inactive tools sorted by detailed duration.
- **`get_tier_breakdown()`**: Returns active tools grouped by tier (HOT, WARM, COLD, PROTECTED).

---

# 2. execution/tool_decay.py

## Class: `ToolUsageStats`
A legacy wrapper class to maintain backward compatibility with older parts of the system.

### Methods
- **`mark_used()`**: Legacy update method.
- **`calculate_decay_score()`**: Duplicate implementation of the decay formula for legacy callers.

---

## Class: `ToolDecayManager`
High-level controller that coordinates the `AdaptiveToolCache` with the file system (`registry.json`).

### Initialization
- **`__init__(...)`**: Initializes the underlying `AdaptiveToolCache` and determines the `registry.json` path.

### Registry Synchronization
- **`_update_registry_status(tool_name, status, inactive_since)`**: 
    - Writes the tool's state (`active`/`inactive`) to `registry.json`.
    - Persists `inactive_since` so Inactive status survives system restarts.
- **`_update_registry_usage(tool_name)`**: Persists `use_count` and `last_used` timestamps to `registry.json`.

### Lifecycle Management
- **`register_tool(...)`**: 
    - Checks registry for existing status/history.
    - **Loads History**: Restores `use_count` and `inactive_since` from file.
    - Registers tool in the cache.
- **`record_usage(...)`**:
    - Calls `cache.record_usage()`.
    - Syncs new usage data to registry.
    - If tool was Inactive, updates registry status to `active`.
- **`update_inactive_status()`**: 
    - Calls usage update in cache.
    - For any tool that becomes inactive, syncs status to `registry.json`.
- **`cleanup_expired_tools()`**:
    - **Changed Behavior**: No longer auto-deletes tools based on time.
    - Delegates to cache only for archive cleanup.

### Helper Methods
- **`get_lru_status()`**: Returns a summary dict with active/inactive counts and next eviction candidate.
- **`protect_tool(name)`** / **`unprotect_tool(name)`**: Manually sets/unsets `PROTECTED` tier.
