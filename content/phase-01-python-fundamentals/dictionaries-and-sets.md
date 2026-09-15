---
title: Dictionaries and Sets
order: 5
difficulty: Beginner
duration: 15
badges: ["Hands-on"]
summary: Key–value lookup and uniqueness — the structures behind JSON, configuration, message objects, caches and deduplication.
prereqs: ["Lists and Tuples"]
keyConcepts: ["dict", "set", "get", "hashable", "O(1) lookup", "merge"]
---

:::note In one line
**A dict looks things up by name, instantly, no matter how big it gets.** A set remembers only the unique items. These two carry more real Python code than anything else.
:::

## Why this matters

Every LLM API request and response is a dictionary. Every configuration object is a
dictionary. Every JSON document you load is a dictionary. And every time you deduplicate
chunk ids, filter documents already seen, or check membership in a loop, a set is the
difference between an instant answer and a slow one.

## Mental Model

A **dict** stores things under a name, and finds them again instantly.

A **set** stores things and remembers only whether it has seen them before.

The "instantly" part is the whole reason these exist. Compare how each one finds something:

<figure class="lesson-figure">
<svg viewBox="0 0 660 260" role="img" aria-label="Diagram comparing a list and a dict. Searching a list means checking each item one after another. A dict computes the location from the key and jumps straight to it in a single step.">
  <defs>
    <marker id="dc-s" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--danger)"/>
    </marker>
    <marker id="dc-f" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <text class="dg-label" x="14" y="22" fill="var(--danger)">A list: look at every item</text>
  <text class="dg-sub"   x="14" y="40">slower and slower as it grows</text>
  <rect x="150" y="52" width="66" height="38" rx="7" class="dg-box"/>
  <text class="dg-sub" x="183" y="76" text-anchor="middle">nope</text>
  <rect x="230" y="52" width="66" height="38" rx="7" class="dg-box"/>
  <text class="dg-sub" x="263" y="76" text-anchor="middle">nope</text>
  <rect x="310" y="52" width="66" height="38" rx="7" class="dg-box"/>
  <text class="dg-sub" x="343" y="76" text-anchor="middle">nope</text>
  <rect x="390" y="52" width="66" height="38" rx="7" class="dg-box"/>
  <text class="dg-sub" x="423" y="76" text-anchor="middle">nope</text>
  <rect x="470" y="48" width="86" height="46" rx="8" fill="var(--panel-2)" stroke="var(--danger)" stroke-width="2"/>
  <text class="dg-sub" x="513" y="76" text-anchor="middle" fill="var(--danger)">found it</text>
  <path d="M120,71 L144,71" stroke="var(--danger)" stroke-width="1.6" marker-end="url(#dc-s)"/>
  <path d="M216,71 L224,71" stroke="var(--danger)" stroke-width="1.6" marker-end="url(#dc-s)"/>
  <path d="M296,71 L304,71" stroke="var(--danger)" stroke-width="1.6" marker-end="url(#dc-s)"/>
  <path d="M376,71 L384,71" stroke="var(--danger)" stroke-width="1.6" marker-end="url(#dc-s)"/>
  <path d="M456,71 L464,71" stroke="var(--danger)" stroke-width="1.6" marker-end="url(#dc-s)"/>
  <text class="dg-sub" x="580" y="76">5 steps</text>
  <line x1="14" y1="120" x2="646" y2="120" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-label" x="14" y="152" fill="var(--accent)">A dict: work out where it is</text>
  <text class="dg-sub"   x="14" y="170">same speed with 10 items or 10 million</text>
  <rect x="150" y="186" width="120" height="48" rx="8" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-mono" x="210" y="206" text-anchor="middle" fill="var(--accent)">"role"</text>
  <text class="dg-sub"  x="210" y="224" text-anchor="middle">the key</text>
  <rect x="330" y="182" width="150" height="56" rx="9" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.5"/>
  <text class="dg-sub"  x="405" y="204" text-anchor="middle">hash it to a slot</text>
  <text class="dg-sub"  x="405" y="222" text-anchor="middle">a bit of arithmetic</text>
  <rect x="530" y="182" width="116" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-mono" x="588" y="204" text-anchor="middle" fill="var(--accent)">"user"</text>
  <text class="dg-sub"  x="588" y="222" text-anchor="middle">1 step</text>
  <path d="M270,210 L324,210" stroke="var(--accent)" stroke-width="1.8" marker-end="url(#dc-f)"/>
  <path d="M480,210 L524,210" stroke="var(--accent)" stroke-width="1.8" marker-end="url(#dc-f)"/>
</svg>
<figcaption>
<strong>A dict does not search. It calculates.</strong> It turns the key into a number, and
that number says where to look. This is why looking something up in a dict of ten million
items is just as fast as in a dict of ten.
</figcaption>
</figure>

Which is why you reach for each one:

| You want to... | Use | Looks like |
| --- | --- | --- |
| store things under a name | `dict` | `{"role": "user", "content": "hi"}` |
| ask "have I seen this?" | `set` | `{"chunk_1", "chunk_7"}` |
| keep things in order | `list` | `[1, 2, 3]` |

:::tip Why some things cannot be dict keys
To calculate a slot, Python has to hash the key — and it can only hash things that never
change. So `str`, `int`, `float`, `bool` and tuples of those all work as keys.

Lists and dicts do **not** work as keys. If they could change after being filed away, the
stored location would be wrong and the value would be lost.
:::

## Core Concepts

### Creating and reading dictionaries

```python
message = {"role": "user", "content": "What is RAG?", "tokens": 6}

message["role"]                # 'user'      raises KeyError if missing
message.get("name")            # None        safe
message.get("name", "unknown") # 'unknown'   with default
"tokens" in message            # True
len(message)                   # 3
```

:::tip `.get()` is the difference between a crash and a default
Use `[]` when a missing key is a bug you want to hear about immediately, and `.get()` when
absence is legitimate. Deciding that consciously, per key, is a mark of careful code.
:::

### Writing and deleting

```python
message["tokens"] = 7                 # add or overwrite
message.setdefault("meta", {})        # insert only if absent, return the value
message.update({"model": "opus", "tokens": 8})
del message["meta"]                   # KeyError if absent
value = message.pop("model", None)    # remove and return, with a default
```

### Iterating

```python
config = {"model": "opus", "temperature": 0.2, "max_tokens": 512}

for key in config:                 # keys by default
    ...
for value in config.values():
    ...
for key, value in config.items():  # the one you will use most
    print(f"{key:>12}: {value}")
```

### Nested dictionaries — the JSON shape

```python
response = {
    "id": "msg_01",
    "usage": {"input_tokens": 1200, "output_tokens": 310},
    "content": [{"type": "text", "text": "Hello"}],
}

response["usage"]["input_tokens"]                 # 1200
response.get("usage", {}).get("cache_tokens", 0)  # 0 - safe chained access
response["content"][0]["text"]                    # 'Hello'
```

### Merging

```python
defaults = {"temperature": 0.2, "max_tokens": 512}
overrides = {"max_tokens": 1024}

merged = defaults | overrides        # {'temperature': 0.2, 'max_tokens': 1024}
merged = {**defaults, **overrides}   # same, works in older versions
defaults |= overrides                # in-place update
```

Right wins. This is exactly how layered configuration works: defaults → file → environment
→ per-request overrides.

### Sets

```python
seen = {"c1", "c7"}
seen.add("c3")
seen.discard("c9")          # no error if absent (remove() raises)
"c1" in seen                # O(1)

a = {1, 2, 3}
b = {3, 4}
a | b        # {1, 2, 3, 4}   union
a & b        # {3}            intersection
a - b        # {1, 2}         difference
a ^ b        # {1, 2, 4}      symmetric difference
a <= b       # subset test

unique = set([1, 1, 2])     # {1, 2}     deduplicate
ordered_unique = list(dict.fromkeys([3, 1, 3, 2]))   # [3, 1, 2] - dedupe, keep order
```

## Minimal Example

```python title="usage_counter.py"
calls = [
    {"model": "small", "tokens": 1200},
    {"model": "large", "tokens": 3400},
    {"model": "small", "tokens": 800},
    {"model": "medium", "tokens": 2000},
]

by_model: dict[str, int] = {}
for call in calls:
    model = call["model"]
    by_model[model] = by_model.get(model, 0) + call["tokens"]

for model, tokens in sorted(by_model.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{model:>7}: {tokens:>6,} tokens")

print("models used:", sorted(set(c["model"] for c in calls)))
```

```text
  large:  3,400 tokens
 medium:  2,000 tokens
  small:  2,000 tokens
models used: ['large', 'medium', 'small']
```

`collections.Counter` and `defaultdict` do this with less code:

```python
from collections import Counter, defaultdict

totals = defaultdict(int)
for call in calls:
    totals[call["model"]] += call["tokens"]

counts = Counter(c["model"] for c in calls)     # Counter({'small': 2, 'large': 1, ...})
counts.most_common(2)
```

## Real-World Example

An in-memory cache with TTL and statistics — the structure behind every "don't pay for the
same embedding twice" optimisation.

```python title="src/cache.py"
"""A tiny TTL cache.

Used in Phase 11 to avoid re-embedding identical text and in Phase 25 as the
first layer in front of an LLM call. Production would use Redis; the semantics
are the same.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


def cache_key(*parts: object) -> str:
    """Stable key from arbitrary parts. Hashing keeps keys short and avoids
    putting user text into log lines."""
    raw = "\0".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class TTLCache:
    ttl_seconds: float = 300.0
    max_entries: int = 10_000
    _store: dict[str, tuple[float, object]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def get(self, key: str) -> object | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None

        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            self.misses += 1
            self.evictions += 1
            return None

        self.hits += 1
        return value

    def set(self, key: str, value: object) -> None:
        if len(self._store) >= self.max_entries:
            # Cheap eviction: drop the oldest inserted key (dicts keep insertion order).
            oldest = next(iter(self._store))
            del self._store[oldest]
            self.evictions += 1
        self._store[key] = (time.monotonic() + self.ttl_seconds, value)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 3),
        }


if __name__ == "__main__":
    cache = TTLCache(ttl_seconds=60)
    key = cache_key("embed", "text-embedding-3-small", "what is rag?")

    if (vector := cache.get(key)) is None:
        vector = [0.01, 0.02, 0.03]          # pretend this cost an API call
        cache.set(key, vector)

    cache.get(key)                           # hit
    cache.get(cache_key("embed", "other"))   # miss

    print(cache.stats())
```

```text
{'entries': 1, 'hits': 1, 'misses': 2, 'evictions': 0, 'hit_rate': 0.333}
```

## Common Mistakes

:::mistake
```python
# 1. KeyError from assuming a key exists
tokens = response["usage"]["cache_tokens"]              # blows up when absent
tokens = response.get("usage", {}).get("cache_tokens", 0)   # safe

# 2. Mutable default shared across calls
def add_tag(tags={}):             # one dict for the lifetime of the process
    ...

# 3. Using a list as a key
index[[1, 2]] = "x"               # TypeError: unhashable type
index[(1, 2)] = "x"               # tuples are fine

# 4. Modifying a dict while iterating it
for k in config:
    if not config[k]:
        del config[k]             # RuntimeError: dictionary changed size
for k in list(config):            # iterate over a snapshot of the keys
    if not config[k]:
        del config[k]

# 5. Assuming sets preserve order
list({"b", "a", "c"})             # order is arbitrary - use dict.fromkeys to keep it
```
:::

## Debugging

```python
import json
print(json.dumps(payload, indent=2, default=str)[:2000])   # readable nested structures
print(sorted(payload.keys()))
print({k: type(v).__name__ for k, v in payload.items()})    # shape at a glance
```

Printing a nested API response with `json.dumps(..., indent=2)` is the fastest way to
understand an unfamiliar payload.

## Performance Considerations

| Operation | dict / set | list |
| --- | --- | --- |
| membership (`in`) | O(1) | O(n) |
| insert | O(1) | O(1) append |
| delete by key/value | O(1) | O(n) |
| ordered iteration | insertion order | index order |
| memory | higher | lower |

Deduplicating 100,000 chunk ids: a set does it in milliseconds, a list in minutes. This is
the single most common "why is my ingest pipeline slow?" answer.

## Security Considerations

:::security Dictionaries are where PII hides
Request payloads, traces and caches are all dicts, and they are the objects you are most
likely to log. Before logging, redact:

```python
SENSITIVE = {"api_key", "authorization", "email", "phone", "ssn", "password"}

def redact(payload: dict) -> dict:
    return {
        k: ("***" if k.lower() in SENSITIVE else redact(v) if isinstance(v, dict) else v)
        for k, v in payload.items()
    }
```
Build this once, use it everywhere you log. Phase 24 makes it part of the tracing layer.
:::

## Hands-on Exercise

:::exercise Merge configuration layers
Write `load_config(defaults, file_config, env_config, overrides)` that merges four
dictionaries with increasing priority, but:

1. nested dictionaries merge recursively (so `{"llm": {"temp": 0.2}}` and
   `{"llm": {"max_tokens": 100}}` produce both keys),
2. a value of `None` in a higher-priority layer does **not** override a lower one,
3. the result is a new dictionary; no input is mutated.

Print the merged config for a realistic example with nested `llm` and `retrieval` sections.
:::

:::solution Solution
```python title="config_merge.py"
from copy import deepcopy
from typing import Any


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: overlay wins, except for None values, nested dicts merge."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(*layers: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, layer)
    return merged


if __name__ == "__main__":
    defaults = {
        "llm": {"model": "small", "temperature": 0.2, "max_tokens": 512},
        "retrieval": {"k": 5, "min_score": 0.35},
    }
    file_config = {"llm": {"model": "medium"}, "retrieval": {"k": 8}}
    env_config = {"llm": {"temperature": None}}          # unset -> ignored
    overrides = {"llm": {"max_tokens": 1024}}

    import json
    print(json.dumps(load_config(defaults, file_config, env_config, overrides), indent=2))
    print("defaults untouched:", defaults["llm"]["model"] == "small")
```

```text
{
  "llm": {
    "model": "medium",
    "temperature": 0.2,
    "max_tokens": 1024
  },
  "retrieval": {
    "k": 8,
    "min_score": 0.35
  }
}
defaults untouched: True
```
:::

## Challenge

:::challenge Inverted index
Build `build_index(documents: dict[str, str]) -> dict[str, set[str]]` mapping each lowercase
word to the set of document ids containing it. Then write `search(index, query)` returning
the ids containing **all** query words, using set intersection.

This is a keyword search engine in fifteen lines — the "sparse" half of the hybrid retrieval
you will build in Phase 13, and a good reminder that not every search problem needs
embeddings.
:::

## Interview Questions

:::interview
1. What makes an object usable as a dictionary key?
2. When do you use `.get()` instead of `[]`?
3. How do you merge two dictionaries with the right one winning?
4. Why is `x in some_set` faster than `x in some_list`?
5. How would you deduplicate a list while preserving order?
:::

## Cheat Sheet

```python
d[k] d.get(k, default) d.setdefault(k, v) d.pop(k, None) del d[k]
d.keys() d.values() d.items()   k in d   len(d)
{**a, **b}   a | b   a |= b            # merge, right wins
dict.fromkeys(seq)                     # dedupe preserving order
sorted(d.items(), key=lambda kv: kv[1], reverse=True)

s.add(x) s.discard(x) s.remove(x)  x in s
a | b  a & b  a - b  a ^ b  a <= b
set(seq)                               # dedupe

from collections import Counter, defaultdict, deque
Counter(seq).most_common(3)
defaultdict(int) / defaultdict(list)
```

```quiz
[
  {
    "question": "Which expression safely reads a possibly-missing nested key?",
    "options": [
      "payload['usage']['cached']",
      "payload.get('usage').get('cached')",
      "payload.get('usage', {}).get('cached', 0)",
      "payload['usage'].get('cached', 0)"
    ],
    "answer": 2,
    "explanation": "Only option C survives both a missing 'usage' key and a missing 'cached' key. Option B raises AttributeError when 'usage' is absent, because .get() returned None."
  },
  {
    "question": "You need to check 50,000 chunk ids against ids you have already ingested. What do you store the seen ids in?",
    "options": ["A list", "A set", "A tuple", "A string"],
    "answer": 1,
    "explanation": "Set membership is O(1) versus O(n) for a list - the difference between milliseconds and minutes at this size."
  },
  {
    "question": "What does {'a': 1} | {'a': 2, 'b': 3} produce?",
    "options": ["{'a': 1, 'b': 3}", "{'a': 2, 'b': 3}", "{'a': 3, 'b': 3}", "TypeError"],
    "answer": 1,
    "explanation": "The right-hand operand wins for duplicate keys - which is exactly why configuration layers are merged left (defaults) to right (overrides)."
  }
]
```

## Summary

- Dicts map hashable keys to values with O(1) lookup; `.get()` handles legitimate absence.
- `|` merges dicts with the right side winning — the basis of layered configuration.
- Sets give O(1) membership and set algebra; use them for dedup and "have I seen this".
- Never mutate a dict or set while iterating it; iterate a snapshot instead.

## Next Step

Control flow: `if`, `for`, `while`, and the iteration helpers (`range`, `enumerate`, `zip`)
that make Python loops readable.
