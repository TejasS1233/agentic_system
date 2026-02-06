# Tool Decay & LRU Eviction System

## Overview

The decay system manages the lifecycle of dynamically created tools, ensuring that:
- Frequently used tools stay available
- Unused tools are gradually removed
- System resources (memory, storage) are kept under control

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ToolDecayManager                           │
│  (High-level wrapper with registry sync)                        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  AdaptiveToolCache                      │   │
│   │  (In-memory cache with LRU eviction logic)              │   │
│   │                                                         │   │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │   │
│   │   │ ToolMetrics │   │ ToolMetrics │   │ ToolMetrics │   │   │
│   │   │  (tool_1)   │   │  (tool_2)   │   │  (tool_n)   │   │   │
│   │   └─────────────┘   └─────────────┘   └─────────────┘   │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ syncs with
                              ▼
                    ┌─────────────────────┐
                    │   registry.json     │
                    │   (persistent)      │
                    └─────────────────────┘
                              │
                              │ reads from
                              ▼
                    ┌─────────────────────┐
                    │   tool_graph.py     │
                    │   (knowledge graph) │
                    └─────────────────────┘
```

## Key Components

### 1. ToolMetrics
Tracks per-tool statistics:
- `total_calls` - Number of times the tool has been used
- `last_used` - Timestamp of last usage
- `is_active` - Whether the tool is active or inactive
- `inactive_since` - Timestamp when tool became inactive (for TTL tracking)
- `tier` - Classification: `hot`, `warm`, `cold`, `probation`, `protected`

### 2. AdaptiveToolCache
In-memory cache with:
- `max_capacity` - Maximum number of tools (default: 50)
- LRU-style eviction logic
- Tier-based promotion/demotion
- Background cleanup cycle

### 3. ToolDecayManager
High-level wrapper that:
- Syncs tool status with `registry.json`
- Handles complete tool deletion (registry + files)
- Provides backward-compatible API

---

## The Decay Score

### Formula
```python
# Base half-life: 9400 seconds (~2.6 hours)
BASE_HALF_LIFE = 9400.0

# Frequency extends half-life (capped at 3x)
multiplier = 1.0 + min(frequency * 0.01, 2.0)  # 1x to 3x

effective_half_life = BASE_HALF_LIFE * multiplier

score = exp(-time_since_use / effective_half_life)
```

Where:
- `time_since_use` = current_time - last_used (seconds)
- `frequency` = total_calls (number of times tool was used)

### Special Case: Never-Used Tools
```python
if frequency == 0:
    if age < 1 hour:  return 1.0   # Brand new, keep
    if age < 6 hours: return 0.5   # Give it time
    else:             return 0.05  # Old, never used, decay
```

### Examples

| Tool | Uses | Time Since Use | Effective Half-Life | Decay Score |
|------|------|----------------|---------------------|-------------|
| Calculator | 120 | 10 minutes | 5.7 hours | ~0.97 |
| WebSearch | 50 | 1 hour | 3.9 hours | ~0.84 |
| FileReader | 10 | 12 hours | 2.9 hours | ~0.02 |
| OldTool | 0 | 3 days | N/A | 0.05 (special case) |

---

## Tool States

### State Machine

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
    REGISTER        │           decay_score > 0.1             │
        │           │                                         │
        ▼           │                                         │
   ┌─────────┐      │      ┌──────────────┐                   │
   │         │      │      │              │                   │
   │ ACTIVE  │◄─────┼──────┤   INACTIVE   │                   │
   │         │   USED!     │              │                   │
   │ status: │  (rescue)   │ status:      │                   │
   │ "active"│             │ "inactive"   │                   │
   │         │─────────────►              │                   │
   └─────────┘             │ inactive_    │                   │
        │     score≤0.1    │ since: ts    │                   │
        │                  └──────┬───────┘                   │
        │                         │                           │
        │                         │ TTL expires (1 day)       │
        │                         ▼                           │
        │                  ┌──────────────┐                   │
        │                  │   DELETED    │                   │
        │                  │              │                   │
        │                  │ - registry   │                   │
        │                  │ - .py file   │                   │
        │                  │ - from graph │                   │
        │                  └──────────────┘                   │
        │                                                     │
        └─────────────────────────────────────────────────────┘
```

### Registry Fields

| State | `status` | `inactive_since` |
|-------|----------|------------------|
| Active | `"active"` | *(not present)* |
| Inactive | `"inactive"` | `1770332458.0` (Unix timestamp) |
| Deleted | *(entry removed)* | N/A |

---

## LRU Eviction Strategy

### When Eviction Happens
1. **Capacity limit reached** - Adding a new tool when at `max_capacity`
2. **TTL expired** - Inactive tool exceeds 1-day TTL
3. **Background cleanup** - Periodic check every 60 seconds (if enabled)

### Eviction Priority

```
┌────────────────────────────────────────────────────────────┐
│                    EVICTION ORDER                          │
├────────────────────────────────────────────────────────────┤
│  1. INACTIVE tools (sorted by TTL remaining - lowest first)│
│     └─ Tool closest to deletion gets evicted              │
├────────────────────────────────────────────────────────────┤
│  2. ACTIVE tools (only if no inactive tools exist)         │
│     └─ Sorted by decay_score (lowest first)               │
│     └─ Then by tier (cold → warm → hot)                   │
├────────────────────────────────────────────────────────────┤
│  3. PROTECTED tools                                        │
│     └─ NEVER evicted                                       │
└────────────────────────────────────────────────────────────┘
```

### Example Scenario

```
Cache: [Tool_A(active), Tool_B(inactive, 2h TTL), Tool_C(inactive, 23h TTL)]
Capacity: 3 (FULL)

New tool requested → Need to evict one

1. Check inactive tools:
   - Tool_B: 2h remaining ← LOWEST TTL
   - Tool_C: 23h remaining
   
2. Evict Tool_B (closest to deletion)

3. Add new tool
```

---

## Rescue Mechanism

When an **inactive** tool is used:

1. Tool is immediately marked as **active**
2. `inactive_since` is cleared
3. TTL countdown stops
4. Tool is protected from imminent eviction

```python
# Using an inactive tool
decay_manager.record_usage("FileReader", execution_time_ms=50)
# → FileReader is now ACTIVE again!
```

---

## Complete Deletion

When a tool's TTL expires, it is **permanently deleted**:

### What Gets Deleted
1. **Registry entry** - Removed from `registry.json`
2. **Tool file** - `.py` file deleted from `workspace/tools/`
3. **Knowledge graph** - Auto-rebuilds without the tool

### Code Flow
```python
# cleanup_expired_tools() does:
1. _cleanup_expired_inactive()  # Find expired tools
2. _get_tool_file_from_registry(name)  # Get file path
3. _delete_from_registry(name)  # Remove from JSON
4. _delete_tool_file(filename)  # Delete .py file
```

---

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_capacity` | 100 | Maximum tools in cache |
| `inactive_ttl` | 86400.0 | 1 day TTL for inactive tools before deletion |
| `registry_path` | `workspace/tools/registry.json` | Path to registry |

### Decay Score Constants (in ToolMetrics.calculate_decay_score)

| Constant | Value | Description |
|----------|-------|-------------|
| `BASE_HALF_LIFE` | 9400s (~2.6 hours) | Base half-life for decay calculation |
| Frequency multiplier | `1 + min(freq × 0.01, 2.0)` | 1x to 3x based on usage |
| `INACTIVE_THRESHOLD` | 0.1 | Score at which tool becomes inactive |

### Example Initialization
```python
from execution.tool_decay import ToolDecayManager

decay_manager = ToolDecayManager(
    max_capacity=50,
    inactive_ttl=86400,  # 1 day
    registry_path="workspace/tools/registry.json"
)
```

---

## API Reference

### Core Methods

```python
# Register a new tool
decay_manager.register_tool(
    name="MyTool",
    tool=tool_instance,
    protected=False,
    created_at=time.time(),
    last_used=None
)

# Record tool usage (resets decay, rescues from inactive)
decay_manager.record_usage("MyTool", execution_time_ms=150.0)

# Update inactive status based on decay scores
newly_inactive = decay_manager.update_inactive_status()

# Clean up expired tools (permanent deletion)
deleted = decay_manager.cleanup_expired_tools()

# Get LRU status summary
status = decay_manager.get_lru_status()
# Returns: {active_count, inactive_count, inactive_tools, next_to_evict}
```

### Query Methods

```python
# Get metrics for a specific tool
metrics = decay_manager.get_metrics("MyTool")

# Get all inactive tools (sorted by TTL)
inactive = decay_manager.get_inactive_tools()
# Returns: [(name, ttl_remaining_seconds), ...]

# Get eviction candidates
candidates = decay_manager.get_eviction_candidates(n=5)
```

---

## Integration Points

### With Registry
- Status synced on state changes (`active` ↔ `inactive`)
- `inactive_since` persisted for TTL calculation across restarts
- Complete deletion removes entry from registry

### With Knowledge Graph
- No direct integration needed
- Graph reads from `registry.json`
- Automatically excludes deleted tools on rebuild

### With Main Pipeline (TODO)
```python
# In IASCIS.__init__:
self.decay_manager = ToolDecayManager(...)

# In ExecutorAgent.execute():
result = tool.run(args)
self.decay_manager.record_usage(tool_name, execution_time)

# In background or on startup:
decay_manager.update_inactive_status()
decay_manager.cleanup_expired_tools()
```

---

## Files

| File | Purpose |
|------|---------|
| `execution/adaptive_tool_cache.py` | Core caching logic, ToolMetrics, eviction |
| `execution/tool_decay.py` | ToolDecayManager wrapper, registry sync |
| `workspace/tools/registry.json` | Persistent tool metadata |
| `architecture/tool_graph.py` | Knowledge graph (reads registry) |

---

## Summary

The decay system ensures tools that aren't being used don't consume resources forever:

1. **Decay Score** - Measures how "alive" a tool is (0.0 to 1.0)
2. **Active/Inactive** - Binary state triggered at score ≤ 0.1
3. **1-Day TTL** - Inactive tools have 24 hours to be used or get deleted
4. **LRU Eviction** - Inactive tools evicted first (oldest inactive = first out)
5. **Rescue** - Using inactive tool brings it back to active
6. **Complete Deletion** - Expired tools removed from registry, files deleted

The system is **self-cleaning** and **usage-aware** - frequently used tools thrive, unused tools fade away.
