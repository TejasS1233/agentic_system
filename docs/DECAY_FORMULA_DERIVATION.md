# Tool Decay Formula: Design & Derivation

> This document explains the mathematical foundation and design rationale behind the tool decay scoring system used in the Agentic System's adaptive tool cache.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Design Goals](#design-goals)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Formula Derivation](#formula-derivation)
5. [Parameter Selection](#parameter-selection)
6. [Behavior Analysis](#behavior-analysis)
7. [Alternative Approaches Considered](#alternative-approaches-considered)

---

## Problem Statement

### The Challenge

In an agentic system that can dynamically create tools, we face a **resource management problem**:

- Tools accumulate over time as the agent creates them for various tasks
- Memory and context window space are limited resources
- Not all tools remain useful — some become obsolete or are rarely needed
- We need an automated way to decide which tools to keep and which to evict

### Requirements

1. **Recently used tools should be preserved** — recency indicates relevance
2. **Frequently used tools should decay slower** — frequency indicates importance
3. **Unused tools should eventually be removed** — prevents unbounded growth
4. **The system should be predictable** — operators can understand and tune it
5. **No tool should be immortal** — even heavily-used tools must decay if abandoned

---

## Design Goals

| Goal | Rationale |
|------|-----------|
| **Smooth decay** | Avoid sudden evictions; give tools a "grace period" |
| **Frequency awareness** | Reward tools that prove their value through repeated use |
| **Bounded survival** | Prevent "immortal" tools that never decay |
| **Tunable parameters** | Allow operators to adjust behavior for their use case |
| **Computational efficiency** | Formula must be fast to evaluate (called frequently) |

---

## Mathematical Foundation

### Why Exponential Decay?

We chose **exponential decay** because it models real-world "forgetting" naturally:

```
score(t) = e^(-t/τ)
```

Where:
- `t` = time since last use (TSU)
- `τ` = time constant (related to half-life)

**Properties of exponential decay:**

1. **Smooth and continuous** — no sudden jumps
2. **Never reaches zero** — always some probability of relevance
3. **Half-life is intuitive** — "tool loses half its value every X hours"
4. **Memoryless** — only current state matters, not history of decay

### Half-Life Interpretation

The **half-life** (H) is the time it takes for the score to drop to 50%:

```
H = τ × ln(2) ≈ 0.693 × τ
```

Or equivalently:

```
score = e^(-TSU / H × 0.693) = 2^(-TSU / H)
```

For simplicity, we use the natural exponential form with the half-life directly:

```
score = e^(-TSU / H)
```

This means at `TSU = H`, the score is `e^(-1) ≈ 0.37` (not exactly 0.5, but close enough for our purposes and computationally simpler).

---

## Formula Derivation

### Step 1: Start with Basic Exponential Decay

```
score = e^(-TSU / H₀)
```

Where `H₀` is a base half-life constant.

**Problem:** All tools decay at the same rate, regardless of how useful they've proven to be.

### Step 2: Incorporate Frequency

We want frequently-used tools to decay **slower**. The intuition:

> "A tool used 100 times has proven its value and deserves more time before being evicted."

We extend the half-life based on frequency:

```
H = H₀ × multiplier(frequency)
```

### Step 3: Choose a Scaling Function

We considered several options for `multiplier(f)`:

| Option | Formula | Behavior |
|--------|---------|----------|
| Linear | `1 + k×f` | Simple, but unbounded growth |
| Logarithmic | `1 + k×log(f)` | Diminishing returns, good but complex |
| Square root | `1 + k×√f` | Similar to log, moderate growth |
| Capped linear | `1 + min(k×f, cap)` | Simple, bounded, predictable |

**We chose capped linear** for simplicity and predictability:

```
multiplier = 1 + min(f × 0.01, 2.0)
```

### Step 4: Final Formula

Combining everything:

```
H = H₀ × (1 + min(f × 0.01, 2.0))
score = e^(-TSU / H)
```

Expanded:

```
score = exp(-TSU / (9400 × (1 + min(frequency × 0.01, 2.0))))
```

---

## Parameter Selection

### Base Half-Life: 9400 seconds (~2.6 hours)

**Rationale:**

- Short enough that unused tools are cleaned up within a workday
- Long enough that a tool used in the morning survives until afternoon
- Aligns with typical human work sessions (2-3 hour focused sprints)

**Calculation for survival time at threshold 0.01:**

```
TSU_max = -H × ln(0.01) = H × 4.605
```

For base case (1 call): `9400 × 4.605 ≈ 43,300 seconds ≈ 12 hours`

### Frequency Scaling Factor: 0.01 per call

**Rationale:**

- At 100 calls: multiplier = 2.0 → half-life doubles
- Gradual increase that doesn't over-reward early usage
- Easy to reason about: "each call adds 1% to the multiplier"

### Maximum Bonus Cap: 2.0 (at 200 calls)

**Rationale:**

- Maximum half-life = 9400 × 3 = 28,200 seconds (~7.8 hours)
- Maximum survival = 28,200 × 4.605 ≈ 130,000 seconds ≈ **36 hours**
- Ensures even the most-used tools will decay within 1.5 days if abandoned
- Prevents "immortal" tools that would never be evicted

---

## Behavior Analysis

### Decay Curves by Frequency

```
Score
1.0 ┤━━━━╮
    │     ╲
    │      ╲  ← 200+ calls (slowest decay)
0.5 ┤       ╲____
    │            ╲ ← 100 calls
    │             ╲____
    │                  ╲ ← 50 calls
    │                   ╲____
    │                        ╲ ← 1 call (fastest decay)
0.0 ┼──────────────────────────────→ TSU
    0     6h     12h    18h    24h    30h    36h
```

### Survival Times Summary

| Usage Level | Calls | Half-Life | Score @ 12h | Max Survival |
|-------------|-------|-----------|-------------|--------------|
| Minimal     | 1     | 2.6h      | 0.01        | ~12 hours    |
| Light       | 10    | 2.9h      | 0.02        | ~13 hours    |
| Moderate    | 50    | 3.9h      | 0.05        | ~18 hours    |
| Heavy       | 100   | 5.2h      | 0.10        | ~24 hours    |
| Very Heavy  | 200+  | 7.8h      | 0.22        | ~36 hours    |

### Special Cases

#### Never-Used Tools (frequency = 0)

The exponential formula breaks down when `frequency = 0` (division issues and conceptual problems). We handle this with explicit age-based rules:

```python
if frequency == 0:
    age_hours = age / 3600.0
    if age_hours < 1:
        return 1.0   # Brand new, keep for now
    elif age_hours < 6:
        return 0.5   # Give it some time to prove value
    else:
        return 0.05  # Old and never used, should decay
```

**Rationale:** A newly created tool deserves a chance to be used. But if it sits unused for 6+ hours, it's probably not needed.

---

## Alternative Approaches Considered

### 1. Fixed TTL (Time-To-Live)

```
evict if (now - last_used) > TTL
```

**Rejected because:**
- All tools treated equally regardless of usage
- Binary decision (keep/evict) with no gradual degradation
- No way to compare tools for eviction priority

### 2. LRU (Least Recently Used)

```
evict the tool with oldest last_used timestamp
```

**Rejected because:**
- Ignores frequency entirely
- A tool used once yesterday beats a tool used 1000 times two days ago
- Prone to "thrashing" in cyclic usage patterns

### 3. LFU (Least Frequently Used)

```
evict the tool with lowest use_count
```

**Rejected because:**
- Ignores recency entirely
- Old heavily-used tools never decay even if obsolete
- New tools can never compete with established ones

### 4. Weighted Combination

```
score = α × recency_score + β × frequency_score
```

**Considered but rejected because:**
- Two arbitrary weights to tune (α, β)
- Unclear how to normalize different scales
- Not as intuitive as exponential decay

### 5. Logarithmic Frequency Scaling

```
multiplier = 1 + k × log(frequency + 1)
```

**Considered as alternative but chose linear because:**
- Harder to reason about ("what does log(100) mean?")
- Diminishing returns at high frequencies (maybe desirable, but we prefer the cap)
- Linear with cap is simpler and achieves similar bounds

---

## Conclusion

The final decay formula balances simplicity, predictability, and effectiveness:

```
score = exp(-TSU / (9400 × (1 + min(frequency × 0.01, 2.0))))
```

**Key properties:**

✅ Recently used tools are preserved (TSU drives decay)  
✅ Frequently used tools decay slower (frequency extends half-life)  
✅ All tools eventually decay (no immortality due to cap)  
✅ Predictable behavior (simple parameters, bounded outcomes)  
✅ Computationally efficient (single exp() call)  

---

## Appendix: Quick Reference

### The Formula

```
score = e^(-TSU / H)
H = 9400 × (1 + min(f × 0.01, 2.0))
```

### Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Base half-life | 9400s | Time scale for decay |
| Frequency factor | 0.01 | How much each call helps |
| Max bonus | 2.0 | Cap at 200 calls |
| Eviction threshold | 0.01 | When to remove tool |

### Key Numbers

| Metric | Value |
|--------|-------|
| Min survival (1 call) | ~12 hours |
| Max survival (200+ calls) | ~36 hours |
| Half-life range | 2.6h – 7.8h |
| Cap kicks in at | 200 calls |

---

*Document created: February 2026*  
*Last updated: February 2026*
