# Domain Matching System - Evolution & Improvements

This document details the progressive improvements made to the domain matching/classification system in the Agentic Toolsmith project, from the initial simple approach to the current sophisticated multi-tier classification system.

---

## Table of Contents
1. [Original Approach: LLM-Only Classification](#1-original-approach-llm-only-classification)
2. [Problem Statement](#2-problem-statement)
3. [Solution: Intent Classifier Architecture](#3-solution-intent-classifier-architecture)
4. [Key Components](#4-key-components)
5. [Improvement Timeline](#5-improvement-timeline)
6. [Performance Metrics](#6-performance-metrics)
7. [Technical Deep Dive](#7-technical-deep-dive)

---

## 1. Original Approach: LLM-Only Classification

### How It Worked
Initially, the `Toolsmith` relied entirely on the LLM (Gemini) to classify user queries into domains. Every request triggered an LLM call that would:
1. Parse the user's natural language requirement
2. Determine the appropriate domain (math, file, web, etc.)
3. Generate tool code for that domain

### Limitations
| Issue | Impact |
|-------|--------|
| **Latency** | Every classification required a full LLM round-trip (~500-2000ms) |
| **Cost** | Token usage for simple classification tasks |
| **Inconsistency** | LLM could return non-standard domain names |
| **Hallucination** | Novel domain names were sometimes invented |

---

## 2. Problem Statement

The core challenge was **ambiguous queries** where the same verb could mean different things depending on context:

```
"sum of numbers" → math (add numbers together)
"sum of files"   → file (count files or aggregate file sizes)
```

Other examples:
- `"check prime"` → math
- `"check file exists"` → file  
- `"check email format"` → validation
- `"download chart"` → web (not visualization!)

A naive keyword approach would fail on these cases, defaulting to the wrong domain and generating incorrect tools.

---

## 3. Solution: Intent Classifier Architecture

The solution was a **multi-tier classification pipeline** that avoids LLM calls for domain routing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Query Input                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1: Phrase Pattern Matching (Regex)                        │
│  - Matches multi-word patterns like "check if file exists"      │
│  - Highest confidence, returns immediately if matched           │
└─────────────────────────────────────────────────────────────────┘
                              │ (no match)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2: spaCy Verb-Object Parsing                              │
│  - Extracts (verb, direct_object) pairs                         │
│  - Looks up in 260+ explicit patterns table                     │
│  - Handles lemmatization for verb forms                         │
└─────────────────────────────────────────────────────────────────┘
                              │ (no match)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 3: Entity Indicator Matching                              │
│  - Detects domain-specific nouns (e.g., "factorial" → math)     │
│  - Weighted scoring system                                      │
└─────────────────────────────────────────────────────────────────┘
                              │ (low confidence)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 4: Weighted Keyword Matching                              │
│  - Each keyword has a weight (0.0-1.0) per domain               │
│  - Sums weights across all matches                              │
└─────────────────────────────────────────────────────────────────┘
                              │ (low confidence)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 5: TF-IDF + Logistic Regression (ML Fallback)             │
│  - Trained on tool registry + seed examples                     │
│  - sklearn-based classifier                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Classified Domain Output                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Components

### 4.1 Canonical Domain Registry
A single source of truth defining all valid domains:

```python
ALLOWED_DOMAINS = {
    "math": {...},
    "text": {...},
    "file": {...},
    "web": {...},
    "data": {...},
    "system": {...},
    "visualization": {...},
    "conversion": {...},
    "search": {...},
    "validation": {...}
}
```

Each domain includes:
- **Description**: Human-readable explanation
- **Aliases**: Alternative names that map to this domain
- **Examples**: Sample queries for documentation

### 4.2 Verb-Object Pattern Table
Over **260 explicit patterns** that resolve ambiguity:

```python
VERB_OBJECT_PATTERNS = {
    ("sum", "numbers"): "math",
    ("sum", "files"): "file",
    ("download", "chart"): "web",      # Not visualization!
    ("compress", "data"): "file",
    ("query", "endpoint"): "web",
    ("query", "database"): "search",
    # ... 250+ more patterns
}
```

### 4.3 spaCy NLP Integration
Uses `en_core_web_sm` model for:
- **Dependency parsing** to extract verb→object relationships
- **Lemmatization** to normalize verb forms (`downloading` → `download`)
- **POS tagging** to identify action verbs vs nouns

### 4.4 TF-IDF ML Classifier
A scikit-learn pipeline trained on:
1. **Seed data**: 350+ labeled examples per domain (built-in)
2. **Tool registry**: Real tools and their descriptions
3. **Vectorizer**: TF-IDF with 500 features, English stopwords removed
4. **Model**: Logistic Regression classifier

---

## 5. Improvement Timeline

### Phase 1: Basic Keyword Matching
**Accuracy: ~50-60%**
- Simple dictionary lookup
- Failed on ambiguous verbs

### Phase 2: Weighted Keywords
**Accuracy: ~65-70%**
- Added weights to keywords (e.g., `"factorial": 0.95` for math)
- Still failed on context-dependent queries

### Phase 3: Entity Indicators
**Accuracy: ~70-75%**
- Added noun-based domain detection
- Improved handling of domain-specific terminology

### Phase 4: spaCy Verb-Object Parsing
**Accuracy: ~76%** (baseline on stress test)
- Added NLP-based extraction
- Initial verb-object pattern table

### Phase 5: Massive Pattern Expansion
**Accuracy: ~85%**
- Expanded verb-object patterns from ~50 to 260+
- Fixed spaCy noun-as-verb parsing issue
- Added action verb priority scoring

### Phase 6: TF-IDF ML Fallback + Refinements
**Accuracy: ~91.4%**
- Trained ML classifier on registry + seed data
- Added phrase pattern matching (regex tier)
- Balanced training data across all domains
- Added typo correction

---

## 6. Performance Metrics

### Stress Test Results (500 Synthetic Cases)

| Metric | Value |
|--------|-------|
| **Total Cases** | 500 |
| **Passed** | 457 |
| **Failed** | 43 |
| **Accuracy** | 91.4% |

### Accuracy Progression

```
Phase 1 (Keywords only):        ████████░░░░░░░░░░░░  50-60%
Phase 2 (Weighted):             █████████░░░░░░░░░░░  65-70%
Phase 3 (Entity indicators):    ██████████░░░░░░░░░░  70-75%
Phase 4 (spaCy integration):    ███████████░░░░░░░░░  76%
Phase 5 (Pattern expansion):    █████████████░░░░░░░  85%
Phase 6 (TF-IDF + refinement):  ██████████████████░░  91.4%
```

### Hard Cases Test
Manual edge cases in `test_hard_cases.py` achieve **~100%** accuracy after targeted pattern additions.

---

## 7. Technical Deep Dive

### 7.1 spaCy Verb-Object Extraction

```python
def _extract_verb_object(self, query: str) -> Tuple[str, str]:
    """Extract main verb and object from query using spaCy."""
    nlp = get_nlp()
    doc = nlp(query.lower())
    
    verb = None
    obj = None
    
    for token in doc:
        # Find root verb
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            verb = token.lemma_
        # Find direct object
        if token.dep_ in ("dobj", "pobj", "attr"):
            obj = token.lemma_
    
    return (verb, obj)
```

### 7.2 Classification Cascade

```python
def classify(self, query: str) -> Tuple[str, str, float]:
    # Tier 1: Phrase patterns
    for pattern, domain in PHRASE_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return (domain, "phrase_pattern", 0.95)
    
    # Tier 2: Verb-Object patterns
    verb, obj = self._extract_verb_object(query)
    if (verb, obj) in VERB_OBJECT_PATTERNS:
        return (VERB_OBJECT_PATTERNS[(verb, obj)], "verb_object", 0.9)
    
    # Tier 3: Entity matching
    entity_domain, entity_conf = self._classify_by_entity(query)
    if entity_conf > 0.7:
        return (entity_domain, "entity", entity_conf)
    
    # Tier 4: Weighted keywords
    keyword_domain, keyword_conf = self._classify_by_keywords(query)
    if keyword_conf > 0.6:
        return (keyword_domain, "keyword", keyword_conf)
    
    # Tier 5: TF-IDF fallback
    tfidf_domain, tfidf_conf = self._classify_by_tfidf(query)
    return (tfidf_domain, "tfidf", tfidf_conf)
```

### 7.3 Key Bug Fixes

| Issue | Solution |
|-------|----------|
| spaCy parsed "query" as noun, not verb | Added noun-as-command detection for imperative queries |
| "database" keyword hijacked search queries | Added 100+ verb-object overrides to disambiguate |
| "check" verb was ambiguous | Mapped to domain based on object: `check disk`→system, `check grammar`→text |
| TF-IDF defaulted to "math" | Balanced seed data with equal examples per domain |

---

## Future Improvements

### Remaining Failure Modes (~8%)
The remaining errors are deeply ambiguous queries where multiple domains are valid:
- `"crawl the data"` → Web (crawling) or Data (processing)?
- `"export xml from file"` → File (I/O) or Data (serialization)?

### Recommended Next Steps
1. **Lightweight LLM fallback**: Use Gemini Flash for low-confidence (<0.6) cases only
2. **User feedback loop**: Learn from corrections to update patterns
3. **Context awareness**: Consider previous queries in session

---

## Files Reference

| File | Purpose |
|------|---------|
| [intent_classifier.py](../architecture/intent_classifier.py) | Main classifier implementation |
| [toolsmith.py](../architecture/toolsmith.py) | Integration with tool generation |
| [test_intent_classifier.py](../tests/test_intent_classifier.py) | Unit tests |
| [stress_test.py](../tests/stress_test.py) | 500-case synthetic stress test |
| [test_hard_cases.py](../tests/test_hard_cases.py) | Manual edge case tests |
