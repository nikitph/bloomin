# MCP Streaming & Pagination - Context Window Deep Dive

## Overview

This document explains how streaming and pagination interact with LLM context windows in MCP integrations, with specific examples for the APT Detection System.

---

## Table of Contents

1. [Streaming Results](#streaming-results)
2. [Paginated Results](#paginated-results)
3. [Context Accumulation](#context-accumulation)
4. [Design Patterns](#design-patterns)
5. [Implementation Examples](#implementation-examples)

---

## Streaming Results

### What is Streaming?

**Streaming** means MCP returns data incrementally in chunks rather than all at once.

```
Traditional (Non-Streaming):
  MCP: [Processes all data]
  MCP: [Returns everything at once]
  LLM: [Receives complete result]

Streaming:
  MCP: [Processes chunk 1] → Send chunk 1 → LLM receives
  MCP: [Processes chunk 2] → Send chunk 2 → LLM receives
  MCP: [Processes chunk 3] → Send chunk 3 → LLM receives
  ...
```

---

### Context Impact: Per-Chunk Limit

**Key Point**: Each chunk must fit in the REMAINING context window.

#### Example 1: Simple Streaming (Context Grows)

```python
# MCP Tool: Stream 1000 attack logs
def stream_attack_logs():
    """Stream logs in chunks of 10"""

    for i in range(100):  # 100 chunks × 10 logs
        chunk = get_logs(offset=i*10, limit=10)
        yield {
            "chunk_id": i,
            "logs": chunk  # 10 logs
        }

# What happens to LLM context:
```

```
Initial context: 0 tokens
────────────────────────────────────────────────────────

Chunk 1 arrives (10 logs × 500 tokens = 5K tokens)
  Context: 0 + 5K = 5K tokens ✅
  LLM: "Received 10 logs, analyzing..."

Chunk 2 arrives (10 logs × 500 tokens = 5K tokens)
  Context: 5K + 5K = 10K tokens ✅
  LLM: "Received 10 more logs, total 20..."

Chunk 3 arrives (10 logs × 500 tokens = 5K tokens)
  Context: 10K + 5K = 15K tokens ✅
  LLM: "Received 10 more logs, total 30..."

...

Chunk 40 arrives (10 logs × 500 tokens = 5K tokens)
  Context: 195K + 5K = 200K tokens ⚠️ LIMIT REACHED!
  LLM: [Context window full]

Chunk 41 arrives (10 logs × 500 tokens = 5K tokens)
  Context: Would be 205K tokens ❌ OVERFLOW!
  System: ERROR or oldest chunks evicted
```

**Problem**: Each chunk adds to context. Eventually hits limit.

---

### Streaming Pattern 1: Accumulation with Summarization

**Solution**: LLM summarizes old chunks to make room for new ones.

```python
# LLM's internal process:

# Chunk 1-10 received (50K tokens)
LLM Context: [All 100 logs in detail]

# Chunk 11 arrives, context getting full
LLM: "Let me summarize chunks 1-10 to save space"
LLM Context:
  [Summary: "100 logs, 12 HIGH severity, targeting servers 1-5"]  ← 100 tokens
  [Chunks 11-20 in detail]  ← Fresh data

# Chunk 21 arrives
LLM: "Summarize chunks 11-20"
LLM Context:
  [Summary of chunks 1-10]  ← 100 tokens
  [Summary of chunks 11-20]  ← 100 tokens
  [Chunks 21-30 in detail]  ← Fresh data

# Result: Can stream indefinitely!
```

**Characteristics**:
- ✅ Can process unlimited chunks
- ✅ Always has summary of old data
- ⚠️ Loses fine-grained detail of old chunks
- ⚠️ Requires LLM to actively summarize (uses tokens/time)

---

### Streaming Pattern 2: Stateless Chunks

**Solution**: Each chunk is independent, LLM doesn't need to remember previous chunks.

```python
# Use case: "Count attacks per day over 30 days"

def stream_daily_attack_counts():
    """Stream one day at a time"""

    for day in range(30):
        count = count_attacks_for_day(day)

        yield {
            "day": day,
            "count": count,
            "severity_breakdown": {...}
        }

# LLM receives:
# Chunk 1: {"day": 1, "count": 12, ...}  ← 50 tokens
# Chunk 2: {"day": 2, "count": 8, ...}   ← 50 tokens
# ...
# Chunk 30: {"day": 30, "count": 15, ...} ← 50 tokens
#
# Total: 30 × 50 = 1,500 tokens ✅ No problem!

# LLM doesn't need to hold all chunks in memory simultaneously
# because each chunk is independent.
```

**Characteristics**:
- ✅ Scales to any number of chunks
- ✅ No context accumulation problem
- ✅ Simple to implement
- ⚠️ Only works for independent data points
- ⚠️ Cannot do cross-chunk analysis

---

### Streaming Pattern 3: Rolling Window

**Solution**: Keep only last N chunks in context.

```python
# System maintains rolling window

Chunk 1 arrives → Context: [Chunk 1]
Chunk 2 arrives → Context: [Chunk 1, Chunk 2]
Chunk 3 arrives → Context: [Chunk 1, Chunk 2, Chunk 3]
...
Chunk 10 arrives → Context: [Chunks 1-10]  ← Window size = 10

Chunk 11 arrives → Context: [Chunks 2-11]  ← Chunk 1 evicted!
Chunk 12 arrives → Context: [Chunks 3-12]  ← Chunk 2 evicted!
```

**Characteristics**:
- ✅ Fixed memory usage
- ✅ Always has recent data
- ⚠️ Loses old data completely
- ⚠️ Cannot reference distant history

---

### Real Example: APT Detection System

```python
# Scenario: Stream all logs from a campaign

def stream_campaign_logs(campaign_id):
    """Stream logs from campaign in chronological order"""

    campaign = get_campaign(campaign_id)
    logs = campaign.logs  # Could be 10,000 logs!

    chunk_size = 10
    for i in range(0, len(logs), chunk_size):
        chunk = logs[i:i+chunk_size]

        yield {
            "chunk_id": i // chunk_size,
            "total_chunks": len(logs) // chunk_size,
            "logs": chunk,
            "progress": f"{i}/{len(logs)}"
        }

# User query: "Tell me about campaign camp_001"
# LLM: [Calls stream_campaign_logs("camp_001")]

# What happens:
# ──────────────────────────────────────────────────────

# Chunk 0 arrives (logs 0-9)
LLM Context:
  User: "Tell me about campaign camp_001"
  Tool call: stream_campaign_logs("camp_001")
  Chunk 0: [10 logs, 5K tokens]

  Total: 5K tokens ✅

# Chunk 1 arrives (logs 10-19)
LLM Context:
  User: "Tell me about campaign camp_001"
  Tool call: stream_campaign_logs("camp_001")
  Chunk 0: [10 logs, 5K tokens]
  Chunk 1: [10 logs, 5K tokens]

  Total: 10K tokens ✅

# ... chunks keep arriving ...

# Chunk 40 arrives (logs 400-409)
LLM Context:
  User: "Tell me about campaign camp_001"
  Tool call: stream_campaign_logs("camp_001")
  Chunk 0-39: [400 logs, 200K tokens]  ← FULL!
  Chunk 40: [10 logs, 5K tokens]       ← OVERFLOW!

  Total: 205K tokens ❌

# Options at this point:
# 1. Stop streaming (miss last 600 logs)
# 2. Summarize old chunks
# 3. Switch to pagination
# 4. Return error
```

---

## Paginated Results

### What is Pagination?

**Pagination** means dividing results into discrete pages, user requests specific pages.

```
Pagination Model:
  Results: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

  Page 1: [1, 2, 3]  ← User requests this
  Page 2: [4, 5, 6]  ← User requests this (separate request)
  Page 3: [7, 8, 9]  ← User requests this (separate request)
  Page 4: [10]       ← User requests this (separate request)
```

**Key Difference from Streaming**: Each page is a separate MCP call.

---

### Context Impact: Per-Page Limit

Each page is a separate tool call, so context includes:
1. Conversation history
2. Previous pages (if user hasn't moved on)
3. Current page

#### Example 1: Basic Pagination

```python
# MCP Tool
def get_attack_logs_page(page=1, page_size=20):
    """Get specific page of attack logs"""

    offset = (page - 1) * page_size
    logs = query_logs(offset=offset, limit=page_size)

    return {
        "page": page,
        "page_size": page_size,
        "total_pages": 500,  # 10,000 logs / 20 per page
        "total_logs": 10000,
        "logs": logs  # 20 logs
    }
```

**Conversation 1: Single Page (No Problem)**

```
User: "Show me attack logs, page 1"
  ↓
LLM: [Calls get_attack_logs_page(page=1)]
  ↓
MCP: Returns page 1 (20 logs, 10K tokens)
  ↓
LLM Context:
  User query: "Show me attack logs, page 1"
  Tool result: [20 logs, 10K tokens]
  LLM response: "Here are 20 attack logs..."

  Total: 10K tokens ✅ No problem!
```

---

**Conversation 2: Multiple Pages in Same Conversation (Accumulation!)**

```
User: "Show me attack logs, page 1"
  ↓
LLM: [Calls get_attack_logs_page(page=1)]
MCP: Returns page 1 (20 logs, 10K tokens)

LLM Context:
  [Page 1: 10K tokens]
  Total: 10K tokens ✅

───────────────────────────────────────────────

User: "Now show me page 2"
  ↓
LLM: [Calls get_attack_logs_page(page=2)]
MCP: Returns page 2 (20 logs, 10K tokens)

LLM Context:
  [Page 1: 10K tokens]  ← Still in conversation!
  [Page 2: 10K tokens]  ← New
  Total: 20K tokens ✅

───────────────────────────────────────────────

User: "Now show me page 3"
  ↓
LLM: [Calls get_attack_logs_page(page=3)]
MCP: Returns page 3 (20 logs, 10K tokens)

LLM Context:
  [Page 1: 10K tokens]  ← Still in conversation!
  [Page 2: 10K tokens]  ← Still in conversation!
  [Page 3: 10K tokens]  ← New
  Total: 30K tokens ✅

───────────────────────────────────────────────

... user continues requesting pages ...

───────────────────────────────────────────────

User: "Now show me page 20"

LLM Context:
  [Pages 1-19: 190K tokens]  ← All previous pages!
  [Page 20: 10K tokens]      ← New
  Total: 200K tokens ⚠️ LIMIT!

───────────────────────────────────────────────

User: "Now show me page 21"

LLM Context:
  [Pages 1-20: 200K tokens]  ← Hit limit!
  [Page 21: 10K tokens]      ← Would overflow!
  Total: 210K tokens ❌ OVERFLOW!
```

**Problem**: Each page stays in conversation context. Pages accumulate!

---

### Pagination Pattern 1: Stateless Pages

**Solution**: Design system so each page request is independent.

```python
# Implementation with stateless design

def get_attack_logs_page(page=1, page_size=20):
    """Each page is self-contained"""

    logs = query_logs(offset=(page-1)*page_size, limit=page_size)

    # Include context in EACH page so LLM doesn't need history
    return {
        "page": page,
        "page_size": page_size,
        "total_pages": 500,
        "logs": logs,
        "summary_stats": {  # ← Repeat on every page!
            "total_attacks": 10000,
            "severity_breakdown": {"HIGH": 1200, "MEDIUM": 8800},
            "date_range": "2024-01-01 to 2024-01-30"
        }
    }

# User conversation:

User: "Show me page 1"
Response: "Here are 20 logs from 10,000 total attacks (HIGH: 1200)..."

User: "Show me page 5"
Response: "Here are 20 logs from 10,000 total attacks (HIGH: 1200)..."
         ↑ LLM doesn't need page 1-4 to answer!

# Each page is self-explanatory, LLM can drop old pages from context.
```

**Characteristics**:
- ✅ Pages don't accumulate in context (LLM can discard old ones)
- ✅ User can jump to any page
- ✅ Simpler mental model
- ⚠️ Redundant data in each page (summary repeated)

---

### Pagination Pattern 2: Progressive Disclosure

**Solution**: Start with summary, let user drill down.

```python
# Level 1: High-level summary (always fits in context)
def get_attack_summary():
    return {
        "total_attacks": 10000,
        "by_severity": {"CRITICAL": 50, "HIGH": 1200, "MEDIUM": 8750},
        "by_cve": {"CVE-2024-6387": 247, "CVE-2021-44228": 1523, ...},
        "date_range": "2024-01-01 to 2024-01-30"
    }
    # Size: ~500 tokens ✅

# Level 2: Filtered subset (user narrows down)
def get_high_severity_attacks(page=1):
    return {
        "page": page,
        "total_pages": 60,  # 1200 HIGH / 20 per page
        "logs": [...]  # 20 HIGH severity logs
    }
    # Size: 10K tokens per page

# Level 3: Individual attack details
def get_attack_details(attack_id):
    return {
        "attack_id": attack_id,
        "full_details": {...}
    }
    # Size: 2K tokens

# User flow:
User: "Show me attacks"
→ Returns summary (500 tokens) ✅

User: "Show me HIGH severity attacks"
→ Returns page 1 of HIGH (10K tokens) ✅

User: "Tell me more about attack #42"
→ Returns detailed view (2K tokens) ✅

# Context never accumulates because user is drilling down, not browsing pages.
```

**Characteristics**:
- ✅ Minimal context usage
- ✅ User gets increasingly detailed view
- ✅ Natural exploration pattern
- ⚠️ Requires good filtering/search

---

### Pagination Pattern 3: Context-Aware Page Size

**Solution**: Adjust page size based on remaining context.

```python
def smart_pagination(page=1, max_context_tokens=None):
    """Dynamically adjust page size based on context availability"""

    # Check how much context is available
    if max_context_tokens is None:
        # Estimate based on conversation so far
        used_tokens = estimate_conversation_tokens()
        available_tokens = 200000 - used_tokens
    else:
        available_tokens = max_context_tokens

    # Adjust page size
    if available_tokens > 50000:
        page_size = 50  # Plenty of room, show 50 logs
    elif available_tokens > 20000:
        page_size = 20  # Medium room, show 20 logs
    elif available_tokens > 10000:
        page_size = 10  # Running low, show 10 logs
    else:
        page_size = 5   # Almost full, show 5 logs

    logs = query_logs(offset=(page-1)*page_size, limit=page_size)

    return {
        "page": page,
        "page_size": page_size,
        "logs": logs,
        "context_warning": available_tokens < 20000,
        "message": "Reduced page size due to context limits" if page_size < 20 else None
    }
```

**Characteristics**:
- ✅ Adapts to context availability
- ✅ Prevents overflow
- ✅ Transparent to user
- ⚠️ Page sizes vary (can be confusing)

---

## Context Accumulation

### The Core Problem

**Every tool call result stays in conversation context until explicitly removed.**

```
Conversation Timeline:

T=0: User: "Show me page 1"
     Context: [Query + Page 1 result] = 10K tokens

T=1: User: "Show me page 2"
     Context: [Query + Page 1 + Query + Page 2] = 20K tokens
                      ↑ STILL HERE!

T=2: User: "Show me page 3"
     Context: [Query + Page 1 + Query + Page 2 + Query + Page 3] = 30K tokens
                      ↑ AND HERE!       ↑ AND HERE!

... continues ...

T=19: User: "Show me page 20"
      Context: [All previous queries and pages] = 200K tokens ← FULL!
```

---

### Why Context Accumulation Happens

**LLM needs conversation context for coherence:**

```
Scenario WITHOUT context accumulation:

User: "Show me HIGH severity attacks on page 1"
Response: [20 HIGH attacks]

User: "Are any of these on server-1?"
         ↑ "these" refers to page 1

LLM: ❓ "these" = what?
     Without page 1 in context, LLM doesn't know!
```

**LLM needs previous results to answer follow-up questions!**

---

### Accumulation Example: Multi-Page Investigation

```python
# Real SOC analyst workflow:

User: "Show me all SSH attacks, page 1"
  → Page 1 in context (10K tokens)

User: "Which of these are CRITICAL?"
  → LLM needs Page 1 to answer ✅
  → Page 1 must stay in context

User: "Show me page 2"
  → Page 2 in context (10K tokens)
  → Page 1 ALSO still in context (user might ask about it)
  → Total: 20K tokens

User: "Compare page 1 and page 2, any patterns?"
  → LLM needs BOTH pages to compare ✅
  → Both pages must stay in context

User: "Show me page 3"
  → Page 3 in context (10K tokens)
  → Pages 1-2 still in context
  → Total: 30K tokens

User: "Show me page 4"
User: "Show me page 5"
...
User: "Show me page 20"
  → All pages in context
  → Total: 200K tokens ← LIMIT!
```

---

## Design Patterns

### Pattern 1: Cursor-Based Pagination

Instead of numeric pages, use cursors that encode position.

```python
def get_attacks_cursor(cursor=None, limit=20):
    """Cursor-based pagination (doesn't encourage browsing all pages)"""

    if cursor is None:
        # First page
        logs = query_logs(limit=limit)
    else:
        # Decode cursor to get position
        position = decode_cursor(cursor)
        logs = query_logs_after(position, limit=limit)

    next_cursor = None
    if len(logs) == limit:
        # More results available
        next_cursor = encode_cursor(logs[-1].id)

    return {
        "logs": logs,
        "next_cursor": next_cursor,
        "has_more": next_cursor is not None
    }

# User flow:
User: "Show me attacks"
Response: [20 logs + cursor "abc123"]

User: "Show me more"
Request: get_attacks_cursor(cursor="abc123")
Response: [20 more logs + cursor "def456"]

# Advantage: Discourages "show me page 50" requests
# User naturally flows forward, old pages can be dropped
```

---

### Pattern 2: Aggregation First, Details on Demand

```python
# Step 1: Show summary (tiny context)
def get_attack_overview():
    return {
        "total": 10000,
        "by_severity": {"CRITICAL": 50, "HIGH": 1200, ...},
        "by_entity": {"server-1": 234, "server-2": 456, ...}
    }
    # Size: 500 tokens ✅

# Step 2: User filters
User: "Show me CRITICAL attacks"

def get_critical_attacks():
    return {
        "total": 50,  # Only 50 CRITICAL
        "logs": [...]  # Can show all 50! (25K tokens)
    }
    # Size: 25K tokens ✅ (manageable!)

# Step 3: Drill into specific attack
User: "Tell me about attack #5"

def get_attack_full_details(attack_id):
    return {...}  # Full forensic details
    # Size: 5K tokens ✅

# Context usage:
# - Overview: 500 tokens
# - Critical list: 25K tokens
# - Detail: 5K tokens
# Total: ~31K tokens ✅ No problem!

# Compare to showing all 10,000 logs: 5M tokens ❌
```

---

### Pattern 3: Ephemeral Results

Explicitly mark results as "can be discarded after reading".

```python
def get_logs_ephemeral(page=1):
    """Logs that LLM can discard after summarizing"""

    logs = query_logs(page=page, page_size=20)

    return {
        "ephemeral": True,  # ← Signal to LLM
        "instruction": "Summarize these logs, then you can discard them",
        "logs": logs
    }

# LLM behavior:
# 1. Receives 20 logs (10K tokens)
# 2. Summarizes: "20 logs, 5 CRITICAL, targeting servers 1-3"
# 3. Discards original 20 logs from context
# 4. Keeps only summary (100 tokens)

# Result: Can process unlimited pages!
# Context: Summary of all pages (~100 tokens per page)
```

---

### Pattern 4: Sliding Window with Landmarks

Keep detailed view of recent pages, summaries of old pages.

```python
# Context management strategy:

Pages 1-5 viewed:
  Context: [Detailed Pages 1-5] = 50K tokens

Page 6 viewed:
  Context: [Summary of Pages 1-3] = 500 tokens
           [Detailed Pages 4-6] = 30K tokens
  Total: 30.5K tokens

Page 7 viewed:
  Context: [Summary of Pages 1-4] = 700 tokens
           [Detailed Pages 5-7] = 30K tokens
  Total: 30.7K tokens

# Window slides forward:
# - Recent 3 pages: Full detail
# - Old pages: Summarized
# - Ancient pages: Landmarks only ("Page 1 showed 12 CRITICAL attacks")
```

---

## Implementation Examples

### Example 1: Smart Streaming for APT System

```python
class StreamingCampaignAnalysis:
    """Stream campaign logs with automatic summarization"""

    def __init__(self, campaign_id, context_limit=50000):
        self.campaign_id = campaign_id
        self.context_limit = context_limit
        self.current_context_usage = 0
        self.summaries = []

    def stream_logs(self):
        """Stream logs, auto-summarize when needed"""

        campaign = get_campaign(self.campaign_id)
        chunk_size = 10

        for i in range(0, len(campaign.logs), chunk_size):
            chunk = campaign.logs[i:i+chunk_size]
            chunk_tokens = estimate_tokens(chunk)

            # Check if we need to summarize before sending
            if self.current_context_usage + chunk_tokens > self.context_limit:
                # Trigger summarization
                yield {
                    "type": "summarize_request",
                    "instruction": "Please summarize the logs you've seen so far, "
                                  "I'll continue with more logs",
                    "logs_so_far": i
                }

                # Wait for LLM to summarize (reduces context)
                self.current_context_usage = sum(
                    estimate_tokens(s) for s in self.summaries
                )

            # Send chunk
            yield {
                "type": "logs",
                "chunk": i // chunk_size,
                "total_chunks": len(campaign.logs) // chunk_size,
                "logs": chunk
            }

            self.current_context_usage += chunk_tokens

        # Final summary
        yield {
            "type": "final_summary_request",
            "instruction": "Please provide a complete analysis of the campaign"
        }
```

---

### Example 2: Context-Aware Pagination

```python
class SmartPaginator:
    """Pagination that adapts to context availability"""

    def __init__(self, query_func, total_results):
        self.query_func = query_func
        self.total_results = total_results
        self.conversation_tokens = 0

    def get_page(self, page=1, conversation_context_tokens=0):
        """Get page with size adapted to context availability"""

        self.conversation_tokens = conversation_context_tokens

        # Calculate available context
        max_context = 200000  # Claude's limit
        available = max_context - conversation_context_tokens

        # Determine page size
        if available > 100000:
            page_size = 100  # Plenty of room
            message = None
        elif available > 50000:
            page_size = 50
            message = "Context filling up, showing 50 results per page"
        elif available > 20000:
            page_size = 20
            message = "Context nearly full, showing 20 results per page"
        elif available > 10000:
            page_size = 10
            message = "⚠️ Context almost full, showing 10 results per page"
        else:
            # Critical: Suggest switching strategies
            return {
                "error": "Context window nearly exhausted",
                "suggestion": "Please either:\n"
                             "1. Summarize what you've seen so far\n"
                             "2. Start a new conversation\n"
                             "3. Use filtering to narrow results",
                "context_used": f"{conversation_context_tokens:,} / {max_context:,} tokens",
                "available": f"{available:,} tokens"
            }

        # Fetch data
        offset = (page - 1) * page_size
        results = self.query_func(offset=offset, limit=page_size)

        return {
            "page": page,
            "page_size": page_size,
            "total_pages": (self.total_results + page_size - 1) // page_size,
            "total_results": self.total_results,
            "results": results,
            "context_info": {
                "used": conversation_context_tokens,
                "available": available,
                "max": max_context,
                "warning": available < 50000
            },
            "message": message
        }

# Usage:
paginator = SmartPaginator(
    query_func=lambda offset, limit: query_attacks(offset, limit),
    total_results=10000
)

# First call - lots of context available
result = paginator.get_page(page=1, conversation_context_tokens=5000)
# Returns: 100 results

# After many pages - context filling up
result = paginator.get_page(page=15, conversation_context_tokens=180000)
# Returns: 10 results + warning

# Context nearly exhausted
result = paginator.get_page(page=20, conversation_context_tokens=195000)
# Returns: Error with suggestions
```

---

### Example 3: Progressive Disclosure for Threat Hunting

```python
class ProgressiveThreatHunter:
    """Multi-level threat hunting with progressive detail"""

    def level_1_overview(self, hunt_query):
        """High-level overview (always fits in context)"""

        results = semantic_search(hunt_query, limit=10000)

        return {
            "query": hunt_query,
            "total_matches": len(results),
            "severity_breakdown": count_by_severity(results),
            "entity_breakdown": count_by_entity(results),
            "time_distribution": group_by_day(results),
            "top_entities": get_top_n(results, key="entity", n=10),
            "recommendation": self._suggest_next_step(results)
        }
        # Size: ~1K tokens ✅

    def level_2_filtered(self, hunt_query, filters):
        """Filtered subset based on Level 1 insights"""

        results = semantic_search(
            hunt_query,
            severity=filters.get("severity"),
            entity=filters.get("entity"),
            time_range=filters.get("time_range")
        )

        # Now that we've filtered, we can show more detail
        return {
            "filters": filters,
            "total_matches": len(results),
            "logs": results[:50],  # Top 50 of filtered set
            "has_more": len(results) > 50,
            "statistics": calculate_stats(results)
        }
        # Size: ~25K tokens ✅

    def level_3_details(self, log_id):
        """Deep dive into specific log"""

        log = get_log_details(log_id)
        related = find_related_logs(log)
        timeline = build_timeline_around(log)

        return {
            "log": log,
            "related_logs": related[:20],
            "timeline": timeline,
            "iocs": extract_iocs(log),
            "recommendations": generate_recommendations(log)
        }
        # Size: ~10K tokens ✅

# User flow:
User: "Hunt for JNDI injection attempts"

Response: Level 1 Overview
  "Found 1,523 matches across 45 entities.
   Severity: 127 HIGH, 1,396 MEDIUM
   Top entities: server-7 (234), server-12 (189)
   Most active: Jan 15-17, 2024

   → Recommend: Filter to HIGH severity on server-7"

User: "Show me HIGH severity on server-7"

Response: Level 2 Filtered
  "Showing 50 of 234 HIGH severity logs on server-7
   [Detailed log entries 1-50]

   → Want to see specific log in detail? Ask about log ID"

User: "Tell me about log #42"

Response: Level 3 Details
  "Log #42 full analysis:
   [Complete forensic view]
   Related: 5 logs in same attack chain
   Timeline: Part of 3-day campaign
   IOCs: evil.com, /tmp/backdoor.sh"

# Total context usage:
# - Level 1: 1K tokens
# - Level 2: 26K tokens (1K + 25K)
# - Level 3: 36K tokens (1K + 25K + 10K)
#
# Compare to: "Show all 1,523 logs" = 750K tokens ❌
```

---

## Best Practices Summary

### For Streaming:

| Pattern | When to Use | Context Impact |
|---------|-------------|----------------|
| **Auto-summarization** | Long data streams | Grows slowly (summaries accumulate) |
| **Stateless chunks** | Independent data points | Constant (no accumulation) |
| **Rolling window** | Recent data important | Constant (fixed window size) |

### For Pagination:

| Pattern | When to Use | Context Impact |
|---------|-------------|----------------|
| **Stateless pages** | Independent pages | Constant (LLM can drop old pages) |
| **Progressive disclosure** | Hierarchical data | Minimal (user drills down, not across) |
| **Context-aware sizing** | Long browsing sessions | Adaptive (shrinks as context fills) |
| **Cursor-based** | Sequential access | Grows (but discourages jumping around) |

### General Guidelines:

1. **Return summaries, not raw data**
   - ✅ "247 attacks, 12 CRITICAL" (50 tokens)
   - ❌ All 247 attack logs (125K tokens)

2. **Filter before showing details**
   - ✅ "Show HIGH severity" → 50 results
   - ❌ "Show all" → 10,000 results

3. **Use hierarchical disclosure**
   - Level 1: Overview (1K tokens)
   - Level 2: Filtered list (25K tokens)
   - Level 3: Individual detail (10K tokens)

4. **Signal ephemeral data**
   - Tell LLM it can discard after summarizing
   - "Summarize these, then I'll send more"

5. **Monitor context usage**
   - Track how much context is used
   - Warn user when approaching limits
   - Suggest alternative strategies

6. **Design for natural workflows**
   - Analysts rarely need all pages
   - They filter, drill down, and pivot
   - Design tools for this pattern

---

## Final Answer

**Per-Chunk/Per-Page Context Limits:**

### Streaming (Per-Chunk):
- ✅ Each chunk CAN be any size
- ❌ BUT all chunks ACCUMULATE in context
- ⚠️ Must manage accumulation via summarization, rolling windows, or stateless design

### Pagination (Per-Page):
- ✅ Each page CAN be any size
- ❌ BUT all pages in conversation ACCUMULATE in context
- ⚠️ Must manage accumulation via stateless pages, progressive disclosure, or context-aware sizing

**Key Insight**: The constraint is not "each chunk must be small", it's "all chunks together must fit in the conversation context window (which includes everything said so far)".

**Best Strategy**: Design MCP tools that return **summaries and insights**, not raw data. Process millions of records in the tool, return compact results to LLM. 🎯
