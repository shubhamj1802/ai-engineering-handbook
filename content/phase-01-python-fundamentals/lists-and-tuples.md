---
title: Lists and Tuples
order: 4
difficulty: Beginner
duration: 15
badges: ["Hands-on"]
summary: Ordered collections — building, slicing, sorting and copying them safely, and knowing when a tuple is the better choice.
prereqs: ["Variables and Data Types", "Strings — The Material Prompts Are Made Of"]
keyConcepts: ["list", "tuple", "slicing", "sort", "shallow copy", "unpacking"]
---

:::note In one line
**Lists can be changed, tuples cannot.** Reach for a list when you are collecting things, and a tuple when the shape is fixed and should stay that way.
:::

## Why this matters

A conversation is a list of messages. A retrieval result is a list of chunks. A batch is a
list of inputs. Lists are the workhorse container of every AI pipeline, and the two things
that go wrong with them — accidental aliasing and sorting by the wrong key — account for a
remarkable share of real bugs.

## Mental Model

Both hold a row of things in order. The difference is one word: **can it change?**

<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Diagram: a list has an open end where items can be appended or removed, while a tuple is sealed so its contents are fixed once created.">
  <defs>
    <marker id="lt-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <text class="dg-label" x="14" y="26" fill="var(--accent)">LIST — can change</text>
  <text class="dg-mono"  x="210" y="26" fill="var(--accent)">[ ]</text>
  <rect x="14" y="44" width="60" height="46" rx="7" class="dg-box"/>
  <text class="dg-mono" x="44" y="73" text-anchor="middle">1</text>
  <rect x="82" y="44" width="60" height="46" rx="7" class="dg-box"/>
  <text class="dg-mono" x="112" y="73" text-anchor="middle">2</text>
  <rect x="150" y="44" width="60" height="46" rx="7" class="dg-box"/>
  <text class="dg-mono" x="180" y="73" text-anchor="middle">3</text>
  <rect x="218" y="44" width="60" height="46" rx="7" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-dasharray="5 4"/>
  <text class="dg-sub" x="248" y="72" text-anchor="middle" fill="var(--accent)">room</text>
  <path d="M320,67 L288,67" stroke="var(--accent)" stroke-width="1.8" marker-end="url(#lt-a)"/>
  <text class="dg-mono" x="330" y="64">.append(4)</text>
  <text class="dg-sub"  x="330" y="84">also .remove(), .pop(), .sort()</text>
  <text class="dg-sub" x="14" y="118">Use it when you are collecting things as you go.</text>
  <line x1="14" y1="140" x2="646" y2="140" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-label" x="14" y="168">TUPLE — sealed</text>
  <text class="dg-mono"  x="180" y="168">( )</text>
  <rect x="14" y="186" width="264" height="46" rx="9" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="2"/>
  <text class="dg-mono" x="44"  y="215" text-anchor="middle">12.5</text>
  <line x1="88" y1="192" x2="88" y2="226" stroke="var(--border-strong)" stroke-width="1.2"/>
  <text class="dg-mono" x="140" y="215" text-anchor="middle">48.9</text>
  <line x1="196" y1="192" x2="196" y2="226" stroke="var(--border-strong)" stroke-width="1.2"/>
  <text class="dg-sub"  x="238" y="215" text-anchor="middle">no room</text>
  <text class="dg-sub" x="320" y="204">Nothing can be added or removed.</text>
  <text class="dg-sub" x="320" y="224">Use it when the shape is fixed: (latitude, longitude).</text>
</svg>
<figcaption>
<strong>A list has an open end; a tuple is sealed shut.</strong> That is nearly the whole
difference. Because a tuple cannot change, Python can also use it as a dictionary key — a
list can never be one.
</figcaption>
</figure>

How to choose, in one question: **will the number of items change while the program runs?**

```python
messages = []                       # list: you will append to it all day
messages.append({"role": "user"})

coordinate = (12.5, 48.9)           # tuple: a place. Two numbers, forever.
```

| | List | Tuple |
| --- | --- | --- |
| Written with | `[ ]` | `( )` |
| Can add or remove | yes | no |
| Can be a dict key | no | yes |
| Good for | a growing collection | a fixed record |

:::tip When a tuple starts getting long, stop
`(12.5, 48.9)` is fine. But if you find yourself writing
`("Ada", 36, "London", "admin", True)` you now have to remember that position 3 is the role.

That is the moment to use a `dataclass` instead (Phase 2), where the fields have names.
:::

## Core Concepts

### Creating and accessing

```python
chunks = ["intro", "methods", "results"]
empty: list[str] = []
from_range = list(range(5))          # [0, 1, 2, 3, 4]
repeated = [0] * 3                   # [0, 0, 0]

chunks[0]        # 'intro'
chunks[-1]       # 'results'
chunks[0:2]      # ['intro', 'methods']    slicing returns a NEW list
len(chunks)      # 3
"methods" in chunks   # True
chunks.index("methods")   # 1  (raises ValueError if absent)
```

### Mutating

```python
chunks.append("discussion")        # add one at the end
chunks.extend(["a", "b"])          # add many  (chunks += [...] is the same)
chunks.insert(0, "abstract")       # insert at index
chunks.remove("a")                 # remove first match (ValueError if absent)
last = chunks.pop()                # remove and return last
first = chunks.pop(0)              # remove and return by index (O(n))
chunks[1:3] = ["x"]                # slice assignment: replace a range
chunks.clear()
```

:::warning `append` vs `extend`
```python
a = [1, 2]
a.append([3, 4])    # [1, 2, [3, 4]]   ← nested
a = [1, 2]
a.extend([3, 4])    # [1, 2, 3, 4]     ← flattened
```
:::

### Sorting

```python
scores = [0.42, 0.91, 0.73]

sorted(scores)                    # NEW list, ascending: [0.42, 0.73, 0.91]
sorted(scores, reverse=True)      # [0.91, 0.73, 0.42]
scores.sort()                     # sorts IN PLACE, returns None

results = [
    {"id": "c1", "score": 0.42},
    {"id": "c2", "score": 0.91},
    {"id": "c3", "score": 0.73},
]
top = sorted(results, key=lambda r: r["score"], reverse=True)
top[:2]     # the two best chunks - exactly what a retriever does
```

:::mistake `x = mylist.sort()` sets x to None
`.sort()` mutates and returns `None`; `sorted()` returns a new list. Using the wrong one is
the most common list bug in Python.
:::

### Tuples, unpacking and immutability

```python
point = (12.5, 48.9)
lat, lon = point                   # unpacking
single = (42,)                     # note the comma - (42) is just an int

point[0] = 1                       # TypeError: tuples are immutable
```

Because they are immutable and hashable, tuples can be dictionary keys and set members:

```python
cache: dict[tuple[str, int], list[float]] = {}
cache[("hello world", 1536)] = [0.1, 0.2]     # (text, dimension) as a key
```

Functions returning several values return a tuple:

```python
def split_name(full: str) -> tuple[str, str]:
    first, _, last = full.partition(" ")
    return first, last

first, last = split_name("Ada Lovelace")
```

### Copying — the part that causes bugs

```python
a = [1, 2, 3]
b = a               # alias: same object
c = a.copy()        # shallow copy (also list(a) or a[:])

nested = [[1, 2], [3, 4]]
shallow = nested.copy()
shallow[0].append(99)
print(nested)       # [[1, 2, 99], [3, 4]]  ← the inner lists are shared!

import copy
deep = copy.deepcopy(nested)   # fully independent
```

## Minimal Example

```python title="messages.py"
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is RAG?"},
]

messages.append({"role": "assistant", "content": "Retrieval-augmented generation."})
messages.append({"role": "user", "content": "Give an example."})

print(f"{len(messages)} messages")
for i, m in enumerate(messages, start=1):
    print(f"  {i}. {m['role']:<9} {m['content'][:40]}")

recent = messages[-3:]          # keep only the last 3 turns
print("trimmed to", len(recent))
```

```text
4 messages
  1. system    You are concise.
  2. user      What is RAG?
  3. assistant Retrieval-augmented generation.
  4. user      Give an example.
trimmed to 3
```

## Real-World Example

A retrieval result set: score, sort, deduplicate, trim to a token budget.

```python title="src/retrieval/select.py"
"""Select which retrieved chunks actually go into the prompt.

Retrieval returns more candidates than fit in the context window, so this step
ranks, removes near-duplicates and stops at the token budget.
"""
from __future__ import annotations

Chunk = dict[str, object]     # {"id": str, "text": str, "score": float, "doc": str}


def select_chunks(
    candidates: list[Chunk],
    *,
    max_tokens: int = 3_000,
    min_score: float = 0.35,
    max_per_document: int = 3,
) -> list[Chunk]:
    """Return the chunks to include, best first, within budget.

    The input list is never mutated - callers often reuse it for logging.
    """
    ranked = sorted(
        (c for c in candidates if float(c["score"]) >= min_score),
        key=lambda c: float(c["score"]),
        reverse=True,
    )

    selected: list[Chunk] = []
    seen_ids: set[str] = set()
    per_doc: dict[str, int] = {}
    used_tokens = 0

    for chunk in ranked:
        chunk_id = str(chunk["id"])
        doc = str(chunk["doc"])

        if chunk_id in seen_ids:
            continue
        if per_doc.get(doc, 0) >= max_per_document:
            continue

        approx_tokens = len(str(chunk["text"])) // 4
        if used_tokens + approx_tokens > max_tokens:
            continue                      # skip, but keep trying smaller chunks

        selected.append(chunk)
        seen_ids.add(chunk_id)
        per_doc[doc] = per_doc.get(doc, 0) + 1
        used_tokens += approx_tokens

    return selected


if __name__ == "__main__":
    candidates: list[Chunk] = [
        {"id": "a1", "doc": "handbook", "score": 0.91, "text": "x" * 1200},
        {"id": "a2", "doc": "handbook", "score": 0.88, "text": "y" * 1200},
        {"id": "a3", "doc": "handbook", "score": 0.84, "text": "z" * 1200},
        {"id": "a4", "doc": "handbook", "score": 0.80, "text": "w" * 1200},
        {"id": "b1", "doc": "faq", "score": 0.62, "text": "q" * 800},
        {"id": "c1", "doc": "blog", "score": 0.21, "text": "low" * 100},
    ]
    chosen = select_chunks(candidates, max_tokens=1_200)
    for c in chosen:
        print(f'{c["id"]:>3} {c["doc"]:<9} score={c["score"]:.2f} chars={len(str(c["text"]))}')
    print("original list untouched:", len(candidates), "candidates")
```

```text
 a1 handbook  score=0.91 chars=1200
 b1 faq       score=0.62 chars=800
original list untouched: 6 candidates
```

Three list ideas carry the whole function: `sorted` with a `key`, a `set` for
deduplication, and a running total to enforce a budget.

## Common Mistakes

:::mistake
```python
# 1. Mutating a list while iterating over it
for c in chunks:
    if c["score"] < 0.3:
        chunks.remove(c)          # skips elements - the index shifts under you
chunks = [c for c in chunks if c["score"] >= 0.3]      # correct

# 2. Mutable default argument
def add(item, bucket=[]):         # ONE list shared across all calls, forever
    bucket.append(item)
    return bucket
def add(item, bucket=None):       # correct
    bucket = [] if bucket is None else bucket
    ...

# 3. Multiplying nested lists
grid = [[0] * 3] * 2              # both rows are THE SAME list
grid[0][0] = 1                    # -> [[1, 0, 0], [1, 0, 0]]
grid = [[0] * 3 for _ in range(2)]   # correct

# 4. `in` on a large list inside a loop
if item in big_list:              # O(n) every time
if item in big_set:               # O(1)
```
:::

## Debugging

```python
print(len(x), type(x))
print(x[:3])                      # peek at the head of a large list
print([type(i).__name__ for i in x[:5]])   # are the elements what you think?
assert all(isinstance(c, dict) for c in chunks), "non-dict in chunks"
```

## Performance Considerations

| Operation | Cost | Note |
| --- | --- | --- |
| `lst[i]` | O(1) | |
| `append` | O(1) amortised | |
| `insert(0, x)` / `pop(0)` | O(n) | use `collections.deque` for queues |
| `x in lst` | O(n) | use a `set` for membership |
| `sorted(lst)` | O(n log n) | stable: equal elements keep their order |
| `lst.copy()` | O(n) | shallow |

```python
from collections import deque

window = deque(maxlen=10)    # a sliding window of the last 10 turns
window.append(message)       # O(1) at both ends; oldest drops automatically
```

## Hands-on Exercise

:::exercise Conversation trimmer
Write `trim_history(messages, max_messages=10, keep_system=True)` that:

1. always keeps the first message if its role is `"system"` (when `keep_system` is true),
2. keeps the most recent messages up to `max_messages` total,
3. never returns a history that starts with an assistant message (drop it if so),
4. does not mutate the input list.

Test it with a 15-message history and print the roles of the result.
:::

:::solution Solution
```python title="trim.py"
def trim_history(
    messages: list[dict], *, max_messages: int = 10, keep_system: bool = True
) -> list[dict]:
    if not messages:
        return []

    system: list[dict] = []
    rest = list(messages)                       # copy: never mutate the caller's list

    if keep_system and rest[0].get("role") == "system":
        system = [rest.pop(0)]

    budget = max(max_messages - len(system), 0)
    recent = rest[-budget:] if budget else []

    # A history should not open with an assistant turn - it confuses the model.
    while recent and recent[0].get("role") == "assistant":
        recent.pop(0)

    return system + recent


if __name__ == "__main__":
    history = [{"role": "system", "content": "sys"}]
    for i in range(14):
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})

    trimmed = trim_history(history, max_messages=6)
    print([m["role"] for m in trimmed])
    print("original length still", len(history))
```

```text
['system', 'user', 'assistant', 'user', 'assistant', 'user']
original length still 15
```
:::

## Challenge

:::challenge Reciprocal rank fusion
Two retrievers return ranked lists of chunk ids (a vector search and a keyword search).
Merge them with reciprocal rank fusion: each document's score is
`sum(1 / (k + rank))` over the lists it appears in, with `k = 60` and `rank` starting at 1.
Return the ids sorted by fused score.

```python
vector = ["c3", "c1", "c7", "c2"]
keyword = ["c1", "c9", "c3"]
# expected: c1 and c3 rank highest because they appear in both
```

You will implement exactly this in Phase 13 for hybrid search — it is a handful of list and
dict operations, no libraries required.
:::

## Interview Questions

:::interview
1. What is the difference between `list.sort()` and `sorted(list)`?
2. When would you choose a tuple over a list?
3. Why is `[[0] * 3] * 2` dangerous?
4. What is the cost of `pop(0)` and what should you use instead?
5. Explain shallow versus deep copy with an example where it matters.
:::

## Cheat Sheet

```python
lst[i] lst[a:b] lst[::-1] len(lst) x in lst
.append(x) .extend(it) .insert(i, x) .remove(x) .pop() .pop(i) .clear()
.index(x) .count(x) .reverse() .sort(key=..., reverse=True)
sorted(it, key=lambda r: r["score"], reverse=True)
min(it, key=...)  max(it, key=...)  sum(it)  any(it)  all(it)
list(t) tuple(l)  a, b = pair   first, *rest = seq
copy: b = a.copy() / list(a) / a[:]     deep: copy.deepcopy(a)
deque(maxlen=n)   # O(1) at both ends
```

```quiz
[
  {
    "question": "What does `best = results.sort(key=lambda r: r['score'])` assign to `best`?",
    "options": ["The sorted list", "None", "A copy of results", "The highest-scoring item"],
    "answer": 1,
    "explanation": "`.sort()` sorts in place and returns None. Use `sorted(results, key=...)` when you want a new list back."
  },
  {
    "question": "Which snippet safely removes low-scoring chunks?",
    "options": [
      "for c in chunks: if c['score'] < 0.3: chunks.remove(c)",
      "chunks = [c for c in chunks if c['score'] >= 0.3]",
      "del chunks[c['score'] < 0.3]",
      "chunks.pop(low_scores)"
    ],
    "answer": 1,
    "explanation": "Removing while iterating skips elements because indices shift. Build a new list instead."
  },
  {
    "question": "Why is a tuple usable as a dict key but a list is not?",
    "options": [
      "Tuples are faster",
      "Tuples are immutable and therefore hashable",
      "Lists are too long",
      "Dict keys must be numbers"
    ],
    "answer": 1,
    "explanation": "Hash-based containers require keys whose hash never changes. Mutable objects would break the invariant, so lists are unhashable."
  }
]
```

## Summary

- Lists are mutable and ordered; tuples are immutable records, hashable, usable as keys.
- `sorted(..., key=...)` is the ranking primitive you will use in every retriever.
- Never mutate a list while iterating it; build a new one.
- Copies are shallow by default — nested structures stay shared.

## Next Step

Dictionaries and sets: the lookup structures behind every configuration, JSON payload and
deduplication step.
