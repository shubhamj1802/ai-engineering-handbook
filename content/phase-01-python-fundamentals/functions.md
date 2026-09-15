---
title: Functions
order: 8
difficulty: Beginner
duration: 16
badges: ["Hands-on"]
summary: Parameters, defaults, *args and **kwargs, return values, scope and docstrings — written so that testing them later is easy.
prereqs: ["Control Flow — if, for, while"]
keyConcepts: ["def", "parameters", "*args", "**kwargs", "scope", "pure function"]
---

:::note In one line
**A function is a name for a job.** Give it clear inputs, one clear output, and no surprises — and never use a list or dict as a default argument, for a reason this lesson shows you.
:::

## Why this matters

Functions are the unit of reuse, the unit of testing, and — once you reach Phase 22 — the
unit an LLM can call as a tool. A well-shaped function (clear inputs, one job, a real
return value, no hidden state) is trivially testable and trivially exposable to an agent. A
badly shaped one is neither.

## Mental Model

```text
        inputs                    output
  ┌──────────────────┐     ┌────────────────┐
  │ required args    │ ──▶ │  return value  │
  │ defaults         │     └────────────────┘
  │ *args  **kwargs  │            │
  └──────────────────┘            ▼
                            side effects (files, network, prints)
                            ← minimise these; they are what makes testing hard
```

A **pure function** returns the same output for the same input and changes nothing outside
itself. Push impurity (I/O, randomness, clocks) to the edges of your program and keep the
core pure — this single habit is why some codebases are easy to test and others are not.

## Core Concepts

### Defining and calling

```python
def score_to_label(score: float, threshold: float = 0.6) -> str:
    """Convert a similarity score into a human label.

    Args:
        score: similarity in [0, 1].
        threshold: minimum score considered a match.

    Returns:
        "match" or "no match".
    """
    return "match" if score >= threshold else "no match"


score_to_label(0.72)                       # positional
score_to_label(0.72, 0.8)                  # positional
score_to_label(score=0.72, threshold=0.8)  # keyword - clearer at call sites
score_to_label(0.72, threshold=0.8)        # mixed: positional first
```

A function with no `return` returns `None`.

### Default arguments

```python
def retrieve(query: str, k: int = 5, min_score: float = 0.35) -> list[dict]:
    ...
```

:::danger The mutable default argument bug
```python
def add(item, bucket=[]):        # the list is created ONCE, at definition time
    bucket.append(item)
    return bucket

add("a")      # ['a']
add("b")      # ['a', 'b']   ← surprise: same list, still holding 'a'

def add(item, bucket=None):      # correct
    bucket = [] if bucket is None else bucket
    bucket.append(item)
    return bucket
```
Never use `[]`, `{}` or `set()` as a default. This appears in real agent code that
accumulates message history and is genuinely hard to spot in review.
:::

### Keyword-only and positional-only parameters

```python
def search(query: str, *, k: int = 5, rerank: bool = False) -> list[dict]:
    """Everything after * must be passed by keyword."""
```

`search("rag", k=3)` works; `search("rag", 3)` raises `TypeError`. Use this for any
parameter whose meaning is not obvious at the call site — it prevents
`search("rag", 3, True)`, which nobody can read.

### `*args` and `**kwargs`

```python
def log_event(event: str, *tags: str, **fields: object) -> None:
    """*tags collects extra positional args into a tuple.
    **fields collects extra keyword args into a dict."""
    print(f"{event} tags={tags} fields={fields}")


log_event("retrieval", "rag", "prod", latency_ms=120, k=5)
# retrieval tags=('rag', 'prod') fields={'latency_ms': 120, 'k': 5}
```

Unpacking works in the other direction too:

```python
args = ["rag", "prod"]
fields = {"latency_ms": 120}
log_event("retrieval", *args, **fields)
```

The most common real use is a wrapper that must forward whatever it was given:

```python
def with_retries(fn, *args, attempts: int = 3, **kwargs):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except TransientError:
            if attempt == attempts:
                raise
```

### Returning multiple values

```python
def evaluate(predictions: list[int], labels: list[int]) -> tuple[float, float]:
    ...
    return precision, recall


precision, recall = evaluate(preds, labels)
```

Beyond two or three values, return a dataclass or a dict instead — positional tuples get
misread.

### Scope: LEGB

Python resolves names **L**ocal → **E**nclosing → **G**lobal → **B**uilt-in.

```python
THRESHOLD = 0.6                 # global

def check(score):
    limit = THRESHOLD           # reads the global - fine
    return score >= limit

def broken():
    count += 1                  # UnboundLocalError: assignment makes `count` local

def works():
    global count                # possible, but avoid: hidden state
    count += 1
```

Reading globals is fine (constants, configuration). Writing them is a design smell: pass
values in and return values out instead.

### Lambdas

```python
sorted(chunks, key=lambda c: c["score"], reverse=True)
```

A `lambda` is a one-expression anonymous function. Use it for tiny key functions; anything
longer gets a real `def` with a name and a docstring.

## Minimal Example

```python title="functions_demo.py"
def build_prompt(question: str, chunks: list[str], *, max_chunks: int = 3) -> str:
    """Assemble a grounded prompt from a question and retrieved chunks."""
    selected = chunks[:max_chunks]
    context = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(selected, start=1))
    return (
        "Answer using only the context. Cite chunk numbers.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


print(build_prompt("What is HNSW?", ["HNSW is a graph index.", "It is approximate."], max_chunks=2))
```

```text
Answer using only the context. Cite chunk numbers.

Context:
[1] HNSW is a graph index.

[2] It is approximate.

Question: What is HNSW?
Answer:
```

## Real-World Example

A module of small, pure, testable functions — the shape you want before any of these become
agent tools in Phase 22.

```python title="src/retrieval/scoring.py"
"""Scoring helpers.

Every function here is pure: same input, same output, no I/O. That makes the
whole module testable without a database, a network or a model.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors, in [-1, 1].

    Raises:
        ValueError: if the vectors have different lengths or either is all zeros.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cannot compute similarity for a zero vector")

    return dot / (norm_a * norm_b)


def normalise_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalise to [0, 1]. A constant input maps to all 1.0."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if math.isclose(lo, hi):
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def reciprocal_rank_fusion(*rankings: Sequence[str], k: int = 60) -> list[tuple[str, float]]:
    """Combine several ranked id lists into one ranking.

    Used for hybrid search: fuse vector results with keyword results without
    needing their scores to be on the same scale.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def weighted_score(
    similarity: float,
    *,
    recency_days: float = 0.0,
    half_life_days: float = 90.0,
    recency_weight: float = 0.2,
) -> float:
    """Blend semantic similarity with exponential recency decay.

    A three-year-old policy document and today's version look equally similar to
    an embedding model; recency is how you break the tie.
    """
    decay = 0.5 ** (recency_days / half_life_days)
    return (1 - recency_weight) * similarity + recency_weight * decay


if __name__ == "__main__":
    print(round(cosine_similarity([1, 0, 1], [1, 1, 1]), 4))
    print([round(s, 3) for s in normalise_scores([2.0, 5.0, 11.0])])
    print(reciprocal_rank_fusion(["c3", "c1", "c7"], ["c1", "c9", "c3"])[:3])
    print(round(weighted_score(0.8, recency_days=180), 4))
```

```text
0.8165
[0.0, 0.333, 1.0]
[('c1', 0.03252247488101534), ('c3', 0.032266458495966696), ('c7', 0.015873015873015872)]
0.69
```

Every one of these is a function you can test with three lines and no infrastructure. That
is the standard to aim for.

## Common Mistakes

:::mistake
```python
# 1. Mutable default (see above) - the classic

# 2. Mutating an argument the caller still owns
def trim(history):
    history.pop(0)          # caller's list changed - surprising
    return history
def trim(history):
    return history[1:]      # new list, caller untouched

# 3. Doing two jobs
def fetch_and_save_and_report(url): ...    # impossible to test one part

# 4. Returning different types
def get(key):
    if missing: return False    # sometimes bool, sometimes dict - callers break
    return {"value": 1}         # return None or raise instead, consistently

# 5. Silent failure
def parse(text):
    try:
        return json.loads(text)
    except Exception:
        return {}               # swallows the bug; caller cannot tell
```
:::

## Best Practices

1. One job per function; if the name needs "and", split it.
2. Type-hint the signature — it is documentation the editor can check (Phase 2).
3. Write a docstring saying what it returns and what it raises.
4. Prefer keyword-only parameters for options; positional for the one obvious subject.
5. Return data; let the caller decide about printing, logging and storing.
6. Validate inputs at the boundary and raise specific exceptions.

## Performance Considerations

- Function calls in Python are not free (~50–100 ns). Do not micro-optimise, but avoid
  calling a trivial function inside a million-iteration loop when a vectorised operation
  exists.
- Default arguments are evaluated once at definition time — which is why the mutable
  default bug exists, and also why `def f(now=time.time())` freezes the clock forever.
- `functools.lru_cache` memoises pure functions in one line:

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def embed_query(text: str) -> tuple[float, ...]:   # must return a hashable type
    ...
```

## Hands-on Exercise

:::exercise A token budget planner
Write `plan_context(system_tokens, history, chunks, *, window, output_reserve=1000)` where
`history` and `chunks` are lists of `(id, token_count)` tuples. It must return a dict:

```python
{
  "included_chunks": [...ids...],
  "included_history": [...ids...],
  "tokens_used": int,
  "tokens_available": int,
  "dropped": [...ids...],
}
```

Rules: the system prompt is always included; history is included newest-first; chunks are
included in the order given; nothing may push the total past `window - output_reserve`.
Raise `ValueError` if the system prompt alone does not fit.
:::

:::solution Solution
```python title="plan_context.py"
from __future__ import annotations


def plan_context(
    system_tokens: int,
    history: list[tuple[str, int]],
    chunks: list[tuple[str, int]],
    *,
    window: int,
    output_reserve: int = 1_000,
) -> dict[str, object]:
    """Decide what fits in the context window. Pure: no I/O, fully testable."""
    budget = window - output_reserve
    if system_tokens > budget:
        raise ValueError(
            f"system prompt ({system_tokens} tokens) exceeds the budget ({budget})"
        )

    used = system_tokens
    included_history: list[str] = []
    included_chunks: list[str] = []
    dropped: list[str] = []

    for item_id, cost in reversed(history):          # newest turns matter most
        if used + cost <= budget:
            included_history.insert(0, item_id)      # restore chronological order
            used += cost
        else:
            dropped.append(item_id)

    for item_id, cost in chunks:                     # already ranked by the retriever
        if used + cost <= budget:
            included_chunks.append(item_id)
            used += cost
        else:
            dropped.append(item_id)

    return {
        "included_chunks": included_chunks,
        "included_history": included_history,
        "tokens_used": used,
        "tokens_available": budget - used,
        "dropped": dropped,
    }


if __name__ == "__main__":
    plan = plan_context(
        system_tokens=800,
        history=[("t1", 600), ("t2", 900), ("t3", 400)],
        chunks=[("c1", 1200), ("c2", 1500), ("c3", 900)],
        window=6_000,
        output_reserve=1_000,
    )
    for key, value in plan.items():
        print(f"{key:>18}: {value}")
```

```text
   included_chunks: ['c1', 'c2']
  included_history: ['t1', 't2', 't3']
       tokens_used: 5000
  tokens_available: 0
           dropped: ['c3']
```
:::

## Challenge

:::challenge Make it a tool schema
Take `plan_context` and write, by hand, the JSON schema an LLM would need to call it:
parameter names, types, descriptions, which are required. Then write a `call_tool(name,
arguments: dict)` dispatcher that validates the arguments against that schema before
calling the function, returning a structured error instead of raising when validation
fails. You have just written the core of Phase 22 — this is genuinely all a "tool" is.
:::

## Interview Questions

:::interview
1. What happens when you use a list as a default argument value?
2. What do `*args` and `**kwargs` do, and when do you need them?
3. What is the LEGB rule?
4. Why prefer keyword-only arguments for options?
5. What makes a function easy to unit test?
:::

## Cheat Sheet

```python
def f(a, b=2, *args, c, d=4, **kwargs) -> R: ...
#     |  |      |      |  |      └ extra keyword args -> dict
#     |  |      |      |  └ keyword-only with default
#     |  |      |      └ keyword-only, REQUIRED (after *args or bare *)
#     |  |      └ extra positional args -> tuple
#     |  └ default (evaluated ONCE at def time)
#     └ required positional

def f(x, *, verbose=False)      # verbose must be passed by keyword
def f(x, /, y)                  # x must be positional (rare)

f(*list_of_args, **dict_of_kwargs)     # unpack at the call site
lambda x: x["score"]                    # tiny anonymous function

@lru_cache(maxsize=128)                 # memoise a pure function
global x / nonlocal x                   # avoid; pass values instead
```

```quiz
[
  {
    "question": "def add(item, bucket=[]): bucket.append(item); return bucket — what does the third call return?",
    "options": ["['c']", "['a', 'b', 'c']", "[]", "TypeError"],
    "answer": 1,
    "explanation": "The default list is created once when the function is defined and reused by every call that omits the argument. Use None as the default and create the list inside."
  },
  {
    "question": "Why declare options as keyword-only (after a bare *)?",
    "options": [
      "It is faster",
      "It forces readable call sites and lets you reorder parameters safely",
      "It is required for type hints",
      "It allows more arguments"
    ],
    "answer": 1,
    "explanation": "search('rag', 3, True) is unreadable and breaks if you reorder parameters. search('rag', k=3, rerank=True) is self-documenting."
  },
  {
    "question": "Which function is easiest to unit test?",
    "options": [
      "One that reads a file, calls an API and prints the result",
      "One that takes data as arguments and returns a value",
      "One that mutates a global cache",
      "One that uses a mutable default argument"
    ],
    "answer": 1,
    "explanation": "Pure functions need no mocks, no fixtures and no cleanup - so they actually get tested."
  }
]
```

## Summary

- Functions take inputs and return outputs; push I/O to the edges and keep the core pure.
- Never use a mutable default; use `None` and create the value inside.
- `*args`/`**kwargs` forward arbitrary arguments — essential for wrappers and decorators.
- Keyword-only parameters make call sites readable and refactors safe.

## Next Step

Errors and exceptions: how to fail loudly in development, gracefully in production, and
never silently.
