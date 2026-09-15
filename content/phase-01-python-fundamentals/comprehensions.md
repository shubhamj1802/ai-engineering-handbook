---
title: Comprehensions and Generator Expressions
order: 7
difficulty: Beginner
duration: 11
badges: ["Hands-on"]
summary: Transform and filter collections in one readable line — list, dict and set comprehensions, plus the lazy generator form that keeps memory flat.
prereqs: ["Control Flow — if, for, while", "Dictionaries and Sets"]
keyConcepts: ["list comprehension", "dict comprehension", "generator expression", "laziness"]
---

:::note In one line
**A comprehension is a `for` loop that builds a list, written on one line.** Use it when you are transforming or filtering. Go back to a normal loop the moment it stops being readable.
:::

## Why this matters

Comprehensions are everywhere in Python code — data cleaning, building prompts, reshaping
API responses. Reading them fluently is a prerequisite for reading any real codebase. And
the lazy generator form is what lets you process a 2 GB JSONL file of documents on a laptop
without running out of memory.

## Mental Model

A comprehension is **a `for` loop that builds a list, folded onto one line**.

Read it left to right in the same order you would say it out loud: *give me this, for each
of those, if it passes this test.*

<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Diagram showing a four-line for loop that builds a list being folded into a single-line comprehension, with the three parts labelled: what to keep, what to loop over, and the filter condition.">
  <defs>
    <marker id="cp-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>

  <text class="dg-sub" x="14" y="22">the long way</text>
  <rect x="14" y="32" width="300" height="94" rx="9" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-mono" x="28" y="54" style="font-size:11.5px">names = []</text>
  <text class="dg-mono" x="28" y="72" style="font-size:11.5px">for user in users:</text>
  <text class="dg-mono" x="28" y="90" style="font-size:11.5px">    if user.active:</text>
  <text class="dg-mono" x="28" y="108" style="font-size:11.5px">        names.append(user.name)</text>

  <path d="M322,79 L364,79" stroke="var(--accent)" stroke-width="2" fill="none" marker-end="url(#cp-a)"/>

  <text class="dg-sub" x="376" y="22">the short way</text>
  <rect x="376" y="32" width="270" height="94" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-mono" x="390" y="66" style="font-size:11.5px">names = [</text>
  <text class="dg-mono" x="390" y="86" style="font-size:11.5px">  user.name</text>
  <text class="dg-mono" x="390" y="104" style="font-size:11.5px">  for user in users</text>
  <text class="dg-mono" x="390" y="122" style="font-size:11.5px">  if user.active ]</text>

  <rect x="14" y="152" width="196" height="56" rx="8" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.5"/>
  <text class="dg-label" x="28" y="174" fill="var(--accent)">1. what to keep</text>
  <text class="dg-mono"  x="28" y="196" style="font-size:11.5px">user.name</text>

  <rect x="230" y="152" width="196" height="56" rx="8" fill="var(--panel)" stroke="var(--accent-3)" stroke-width="1.5"/>
  <text class="dg-label" x="244" y="174" fill="var(--accent-3)">2. what to loop over</text>
  <text class="dg-mono"  x="244" y="196" style="font-size:11.5px">for user in users</text>

  <rect x="446" y="152" width="200" height="56" rx="8" fill="var(--panel)" stroke="var(--accent-2)" stroke-width="1.5"/>
  <text class="dg-label" x="460" y="174" fill="var(--accent-2)">3. which ones (optional)</text>
  <text class="dg-mono"  x="460" y="196" style="font-size:11.5px">if user.active</text>

  <text class="dg-sub" x="330" y="232" text-anchor="middle">Always these three parts, always in this order.</text>
</svg>
<figcaption>
<strong>Same three parts, rearranged.</strong> The thing you want comes first, then the loop,
then the filter. Once you see the three slots, any comprehension becomes readable.
</figcaption>
</figure>

```python
# All three shapes work the same way
[user.name for user in users]                      # just transform
[user.name for user in users if user.active]       # transform + filter
{user.name for user in users}                      # a set: unique names
{user.id: user.name for user in users}             # a dict
```

:::warning Stop when it stops being readable
Comprehensions are good for **one** loop and **one** condition. Past that, a plain loop is
clearer and there is no prize for cramming it onto one line:

```python
# Nobody can read this
result = [f(x, y) for x in xs if x > 0 for y in ys if y != x and g(y)]

# This is better code, even though it is longer
result = []
for x in xs:
    if x <= 0:
        continue
    for y in ys:
        if y != x and g(y):
            result.append(f(x, y))
```
:::

## Core Concepts

### The four forms

```python
scores = [0.91, 0.34, 0.72, 0.15]

[s * 100 for s in scores]                      # list        [91.0, 34.0, 72.0, 15.0]
{round(s, 1) for s in scores}                  # set         {0.9, 0.3, 0.7, 0.2}
{f"c{i}": s for i, s in enumerate(scores)}     # dict        {'c0': 0.91, ...}
(s for s in scores if s > 0.5)                 # generator   lazy, nothing computed yet
```

### Filtering and conditional values

```python
# filter (if at the end)
kept = [c for c in chunks if c["score"] >= 0.5]

# conditional expression (if/else before the for)
labels = ["pass" if s >= 0.6 else "fail" for s in scores]

# both together
labels = ["high" if s > 0.8 else "low" for s in scores if s >= 0.5]
```

The position matters: `if` **after** the `for` filters items out; `if/else` **before** the
`for` chooses what to produce for every item.

### Nested loops and flattening

```python
documents = [["a", "b"], ["c"], ["d", "e"]]

flat = [chunk for doc in documents for chunk in doc]    # ['a', 'b', 'c', 'd', 'e']
```

The loop order matches the equivalent nested `for` statements, reading left to right. Two
levels are readable; three are not — write a loop instead.

### Generator expressions and laziness

```python
total_chars = sum(len(c["text"]) for c in chunks)      # no intermediate list at all
first_match = next((c for c in chunks if c["score"] > 0.9), None)
has_citation = any("[" in c["text"] for c in chunks)   # stops at the first hit
```

A list comprehension builds the whole list in memory immediately. A generator expression
produces items one at a time, on demand:

```python
import sys

sys.getsizeof([x * x for x in range(1_000_000)])   # ~8 MB
sys.getsizeof((x * x for x in range(1_000_000)))   # ~200 bytes
```

Use a generator when you will consume the values once — summing, feeding a loop, streaming
to a file. Use a list when you need to index, slice, or iterate more than once.

:::warning A generator is exhausted after one pass
```python
gen = (x for x in range(3))
list(gen)     # [0, 1, 2]
list(gen)     # []  ← already consumed
```
:::

## Minimal Example

```python title="comprehensions.py"
chunks = [
    {"id": "c1", "text": "Vector search finds similar embeddings.", "score": 0.91},
    {"id": "c2", "text": "", "score": 0.44},
    {"id": "c3", "text": "Rerankers reorder candidates.", "score": 0.67},
]

kept = [c for c in chunks if c["text"].strip() and c["score"] >= 0.5]
by_id = {c["id"]: c["score"] for c in kept}
context = "\n".join(f"[{c['id']}] {c['text']}" for c in kept)
total_chars = sum(len(c["text"]) for c in kept)

print(f"kept {len(kept)} of {len(chunks)}")
print(by_id)
print(context)
print("chars:", total_chars)
```

```text
kept 2 of 3
{'c1': 0.91, 'c3': 0.67}
[c1] Vector search finds similar embeddings.
[c3] Rerankers reorder candidates.
chars: 68
```

## Real-World Example

Normalising a batch of documents before embedding — a step every ingestion pipeline has.

```python title="src/ingest/prepare.py"
"""Turn raw records into embedding-ready payloads.

Comprehensions keep each transformation on one line and each line honest about
what it filters. The generator at the bottom means a 2 GB file never lands in RAM.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

MIN_CHARS = 80
ALLOWED_LANGS = {"en", "de", "fr"}


def prepare(records: list[dict]) -> list[dict]:
    """Filter, normalise and enrich records in one readable pass each."""
    # 1. drop anything unusable - one condition per line for readability
    usable = [
        r
        for r in records
        if r.get("text")
        and len(r["text"]) >= MIN_CHARS
        and r.get("lang", "en") in ALLOWED_LANGS
        and not r.get("is_draft", False)
    ]

    # 2. normalise
    normalised = [
        {
            **r,
            "text": " ".join(r["text"].split()),      # collapse whitespace
            "title": (r.get("title") or "untitled").strip(),
        }
        for r in usable
    ]

    # 3. enrich with derived fields
    return [
        {**r, "char_count": len(r["text"]), "approx_tokens": len(r["text"]) // 4}
        for r in normalised
    ]


def stream_jsonl(path: Path) -> Iterator[dict]:
    """Yield records one at a time - memory stays flat regardless of file size."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def corpus_stats(path: Path) -> dict[str, int | float]:
    """Two passes over a generator would be a bug, so build one list of sizes."""
    sizes = [len(r.get("text", "")) for r in stream_jsonl(path)]
    if not sizes:
        return {"documents": 0, "total_chars": 0, "mean_chars": 0.0}
    return {
        "documents": len(sizes),
        "total_chars": sum(sizes),
        "mean_chars": round(sum(sizes) / len(sizes), 1),
        "over_10k": sum(1 for s in sizes if s > 10_000),
    }


if __name__ == "__main__":
    records = [
        {"text": "a" * 200, "lang": "en", "title": "  Intro "},
        {"text": "too short", "lang": "en"},
        {"text": "b" * 300, "lang": "es"},
        {"text": "c" * 150, "lang": "de", "is_draft": True},
        {"text": "d" * 400, "lang": "fr", "title": None},
    ]
    for record in prepare(records):
        print(f'{record["title"]:<10} {record["lang"]} {record["char_count"]:>4} chars '
              f'~{record["approx_tokens"]} tokens')
```

```text
Intro      en  200 chars ~50 tokens
untitled   fr  400 chars ~100 tokens
```

## Common Mistakes

:::mistake
```python
# 1. Cramming too much in - if it needs a comment, write a loop
result = [transform(x) if check(x) else fallback(x) for y in outer for x in y if valid(x)]

# 2. Side effects inside a comprehension
[print(x) for x in items]        # builds a list of Nones to throw away - use a loop

# 3. Reusing an exhausted generator
gen = (x for x in items)
count = sum(1 for _ in gen)
first = next(gen, None)          # always None - gen is spent

# 4. Building a huge list when a generator would do
sum([x ** 2 for x in range(10_000_000)])    # allocates 10M items
sum(x ** 2 for x in range(10_000_000))      # constant memory

# 5. Forgetting that the filter runs before the transform
[1 / x for x in values if x != 0]    # correct: filter first
[1 / x if x != 0 else 0 for x in values]   # different meaning - produces 0s
```
:::

## Debugging

Comprehensions hide the intermediate values, so when one misbehaves, temporarily expand it:

```python
kept = []
for c in chunks:
    ok = c["score"] >= 0.5
    print(f"{c['id']}: score={c['score']} keep={ok}")
    if ok:
        kept.append(c)
```

Then collapse it back once you know why. Debugging a comprehension by staring at it is
usually slower than expanding it for thirty seconds.

## Performance Considerations

| Pattern | Relative speed | Memory |
| --- | --- | --- |
| `for` loop with `.append()` | 1.0× | list |
| list comprehension | ~1.3–1.6× faster | list |
| generator expression | similar speed, lazy | constant |
| `map`/`filter` with a lambda | usually slower than a comprehension | lazy |

Comprehensions are faster because the append happens in C rather than through a Python
method lookup each iteration. For numeric arrays, NumPy (Phase 3) is another 10–100× beyond
any of these.

## Hands-on Exercise

:::exercise Build a citation map
Given:

```python
answer = "RAG grounds answers [c1] and reduces hallucination [c3][c1]."
chunks = [
    {"id": "c1", "doc": "handbook.pdf", "page": 4},
    {"id": "c2", "doc": "faq.md", "page": 1},
    {"id": "c3", "doc": "handbook.pdf", "page": 9},
]
```

Using comprehensions (and `re.findall`), produce:

1. `cited_ids`: the unique ids referenced in the answer, in order of first appearance.
2. `citations`: a list of `{"id", "doc", "page"}` dicts for the cited chunks only.
3. `unused`: the ids of retrieved chunks that were never cited.
4. `invalid`: ids cited in the answer that do not exist in `chunks` — a hallucinated
   citation, which must be caught before the answer reaches a user.
:::

:::solution Solution
```python title="citations.py"
import re

answer = "RAG grounds answers [c1] and reduces hallucination [c3][c1]."
chunks = [
    {"id": "c1", "doc": "handbook.pdf", "page": 4},
    {"id": "c2", "doc": "faq.md", "page": 1},
    {"id": "c3", "doc": "handbook.pdf", "page": 9},
]

referenced = re.findall(r"\[(c\d+)\]", answer)
cited_ids = list(dict.fromkeys(referenced))            # unique, order preserved

by_id = {c["id"]: c for c in chunks}
citations = [by_id[cid] for cid in cited_ids if cid in by_id]
unused = [c["id"] for c in chunks if c["id"] not in set(cited_ids)]
invalid = [cid for cid in cited_ids if cid not in by_id]

print("cited  :", cited_ids)
print("sources:", [f'{c["doc"]} p{c["page"]}' for c in citations])
print("unused :", unused)
print("invalid:", invalid)
assert not invalid, f"answer cites non-existent chunks: {invalid}"
```

```text
cited  : ['c1', 'c3']
sources: ['handbook.pdf p4', 'handbook.pdf p9']
unused : ['c2']
invalid: []
```

That final `assert` is a real guardrail: in Phase 19 it becomes a validation function that
rejects the answer instead of crashing.
:::

## Challenge

:::challenge Chunk overlap matrix
Given a list of chunk texts, build a dict-of-dicts
`{chunk_i_id: {chunk_j_id: jaccard_similarity}}` for all pairs where the Jaccard similarity
of their word sets exceeds 0.6 — i.e. near-duplicate chunks that waste context window.

Use set comprehensions for the word sets and a dict comprehension for the matrix, skipping
self-comparisons. Then print the pairs you would drop. Near-duplicate elimination is a real
retrieval optimisation (Phase 13).
:::

## Interview Questions

:::interview
1. What is the difference between a list comprehension and a generator expression?
2. Where does the filtering `if` go, and how does that differ from a conditional value?
3. When is a comprehension the wrong choice?
4. Why does `sum(x for x in big)` use less memory than `sum([x for x in big])`?
5. What happens if you iterate a generator twice?
:::

## Cheat Sheet

```python
[f(x) for x in it]                   # list
[f(x) for x in it if cond(x)]        # filtered
[a if cond(x) else b for x in it]    # conditional value
{k(x): v(x) for x in it}             # dict
{f(x) for x in it}                   # set
(f(x) for x in it)                   # generator - lazy

[y for row in matrix for y in row]   # flatten (outer loop first)

sum(len(x) for x in it)
any(cond(x) for x in it)             # short-circuits
all(cond(x) for x in it)
next((x for x in it if cond(x)), None)   # first match or None
```

```quiz
[
  {
    "question": "What does [x for x in range(10) if x % 2 == 0] produce?",
    "options": ["[0, 2, 4, 6, 8]", "[1, 3, 5, 7, 9]", "[True, False, ...]", "A generator"],
    "answer": 0,
    "explanation": "The trailing `if` filters, keeping even numbers. Square brackets make it a list, evaluated immediately."
  },
  {
    "question": "You need to sum the character counts of 5 million documents streamed from disk. Which is best?",
    "options": [
      "sum([len(d.text) for d in docs])",
      "sum(len(d.text) for d in docs)",
      "total = 0; for d in docs: total += len(d.text)  (identical to B in memory)",
      "Both B and C are fine; A needlessly materialises 5 million ints"
    ],
    "answer": 3,
    "explanation": "The generator form and the explicit loop both use constant memory; the list comprehension allocates five million integers first for no reason."
  }
]
```

## Summary

- Comprehensions express filter-and-transform in one line: source, filter, then output.
- `if` after `for` filters; `if/else` before `for` chooses a value per item.
- Generator expressions are lazy and single-use — ideal for streaming and aggregation.
- If a comprehension needs a comment, it should be a loop.

## Next Step

Functions: parameters, return values, `*args`, `**kwargs`, scope — and how to write ones
that are easy to test.
