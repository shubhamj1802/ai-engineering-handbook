---
title: Type Hints, Generics and Protocols
order: 6
difficulty: Intermediate
duration: 14
badges: ["Hands-on"]
summary: "Annotations that a checker can verify — unions, optionals, generics, TypedDict, Protocol and type aliases — plus what type hints do and do not guarantee at runtime."
prereqs: ["Functions", "Classes and Objects"]
keyConcepts: ["type hints", "Optional", "Generic", "Protocol", "TypedDict", "mypy"]
---

:::note In one line
**Type hints are notes for humans and tools, not rules Python enforces.** They pay for themselves the moment a codebase outgrows one person.
:::

## Why this matters

Type hints are how a large Python codebase stays navigable. Your editor autocompletes,
`mypy` or `pyright` catches the `None` you forgot to handle before it reaches production,
and — crucially for AI work — Pydantic and FastAPI *derive runtime behaviour* from your
annotations: request validation, JSON schemas, and the tool schemas an LLM reads.

## Mental Model

Type hints are **notes for humans and tools**. Python itself ignores them completely at
runtime.

<figure class="lesson-figure">
<svg viewBox="0 0 660 220" role="img" aria-label="Diagram: type hints are read by your editor and by a type checker before the code runs, but Python discards them at runtime, so validation of external data still needs a runtime check.">
  <defs>
    <marker id="ty-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="14" y="76" width="140" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-mono" x="84" y="100" text-anchor="middle" fill="var(--accent)" style="font-size:11.5px">k: int = 5</text>
  <text class="dg-sub"  x="84" y="119" text-anchor="middle">a type hint</text>
  <rect x="222" y="16" width="190" height="48" rx="8" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-sub" x="317" y="38" text-anchor="middle" fill="var(--ok)">your editor</text>
  <text class="dg-sub" x="317" y="55" text-anchor="middle">autocomplete, red squiggles</text>
  <rect x="222" y="80" width="190" height="48" rx="8" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-sub" x="317" y="102" text-anchor="middle" fill="var(--ok)">mypy / pyright</text>
  <text class="dg-sub" x="317" y="119" text-anchor="middle">catches it before you run</text>
  <rect x="222" y="144" width="190" height="52" rx="8" fill="var(--panel)" stroke="var(--danger)" stroke-width="1.7"/>
  <text class="dg-sub" x="317" y="166" text-anchor="middle" fill="var(--danger)">Python at runtime</text>
  <text class="dg-sub" x="317" y="184" text-anchor="middle">ignores them entirely</text>
  <path class="dg-arrow" d="M154,94 Q190,94 216,44" marker-end="url(#ty-a)"/>
  <path class="dg-arrow" d="M154,104 L216,104" marker-end="url(#ty-a)"/>
  <path class="dg-arrow" d="M154,114 Q190,114 216,164" marker-end="url(#ty-a)"/>
  <rect x="452" y="84" width="194" height="80" rx="9" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.8"/>
  <text class="dg-label" x="549" y="108" text-anchor="middle" fill="var(--accent-2)">So for outside data</text>
  <text class="dg-sub"   x="549" y="130" text-anchor="middle">use Pydantic to CHECK</text>
  <text class="dg-sub"   x="549" y="148" text-anchor="middle">at runtime, not just hint</text>
  <path class="dg-arrow" d="M412,170 Q440,170 446,140" marker-end="url(#ty-a)"/>
</svg>
<figcaption>
<strong>Hints are not validation.</strong> <code>def f(k: int)</code> will happily accept the
string <code>"5"</code> and fail later somewhere confusing. For anything arriving from an
API, a file or a model, you need a real runtime check.
</figcaption>
</figure>

```python
def search(query: str, k: int = 5) -> list[str]:
    ...

search("hello", k="3")      # Python runs this. mypy would have stopped you.
```

That is why the handbook uses Pydantic for anything crossing a boundary:

```python
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=50)

SearchRequest(query="hi", k="3")     # k becomes 3 - coerced and checked
SearchRequest(query="", k=999)       # ValidationError, raised immediately
```

:::tip Where to spend your typing effort
Type the **edges** of your code — function signatures, dataclasses, anything public. That is
where hints pay for themselves in autocomplete and caught mistakes.

Do not bother annotating every local variable. `count = 0` is obviously an int, and
`count: int = 0` just adds noise.
:::

## Core Concepts

### The basics

```python
name: str = "rag"
k: int = 5
score: float = 0.91
enabled: bool = True
tags: list[str] = []
meta: dict[str, int] = {}
pair: tuple[str, float] = ("c1", 0.9)
fixed: tuple[int, ...] = (1, 2, 3)          # variable length, all ints


def search(query: str, k: int = 5) -> list[dict[str, float]]:
    ...
```

Modern syntax uses built-ins (`list[str]`), not `typing.List`. Nothing from `typing` is
needed for basic containers on Python 3.9+.

### Optional and unions

```python
def find(chunk_id: str) -> dict | None:        # may return None
    ...

def parse(value: str | int) -> float:          # accepts either
    ...

result = find("c1")
result["text"]              # mypy error: result may be None
if result is not None:
    result["text"]          # narrowed to dict - fine
```

That narrowing is the everyday payoff: the checker forces you to handle the `None` branch
you would otherwise discover in production.

### Type aliases

```python
type Vector = list[float]                      # Python 3.12+ syntax
type ChunkId = str
type SearchResults = list[tuple[ChunkId, float]]

def rank(query_vector: Vector, k: int = 5) -> SearchResults: ...
```

Aliases turn `list[tuple[str, float]]` into something that says what it means.

### Generics

```python
from collections.abc import Callable, Iterable, Sequence


def first[T](items: Sequence[T], default: T | None = None) -> T | None:
    """Python 3.12 generic syntax: [T] declares the type variable."""
    return items[0] if items else default


class Cache[K, V]:
    def __init__(self) -> None:
        self._data: dict[K, V] = {}

    def get(self, key: K) -> V | None:
        return self._data.get(key)

    def set(self, key: K, value: V) -> None:
        self._data[key] = value


cache: Cache[str, list[float]] = Cache()
cache.set("query", [0.1, 0.2])
vector = cache.get("query")        # inferred as list[float] | None
```

### Protocols — structural typing

A `Protocol` says "anything with these methods", with no inheritance required. This is the
cleanest way to define an interface in modern Python.

```python
from typing import Protocol, runtime_checkable


class Embedder(Protocol):
    """Anything that can turn texts into vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...


def build_rag(embedder: Embedder, index: VectorIndex) -> "RagService":
    ...                     # any object with those methods satisfies this, no base class
```

:::tip Protocol vs ABC
Use a **Protocol** when you do not control the implementations (third-party clients, test
doubles) or want zero coupling. Use an **ABC** when you own the implementations and want
shared concrete helpers plus construction-time enforcement. Many codebases use both: a
Protocol at module boundaries, an ABC where a family of classes shares code.
:::

### TypedDict — typing the JSON you receive

```python
from typing import TypedDict, NotRequired


class Message(TypedDict):
    role: str
    content: str


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int
    cached_tokens: NotRequired[int]          # optional key


class LLMResponse(TypedDict):
    id: str
    content: list[Message]
    usage: Usage


def total_tokens(response: LLMResponse) -> int:
    usage = response["usage"]
    return usage["input_tokens"] + usage["output_tokens"] + usage.get("cached_tokens", 0)
```

`TypedDict` documents API payload shapes without converting them into objects. For data you
must *validate* (anything crossing a trust boundary), use Pydantic instead (Phase 19).

### Literal, Final and Annotated

```python
from typing import Annotated, Final, Literal

Strategy = Literal["vector", "keyword", "hybrid"]

def retrieve(query: str, strategy: Strategy = "hybrid") -> list[dict]: ...

retrieve("x", "fuzzy")          # mypy error: not a valid literal

MAX_RETRIES: Final = 3          # reassignment is an error

Temperature = Annotated[float, "range 0..2"]   # extra metadata; Pydantic reads these
```

### Callables and callbacks

```python
from collections.abc import Callable

Handler = Callable[[str, int], bool]         # (str, int) -> bool

def register(event: str, handler: Handler) -> None: ...
```

### Typing generators and async

```python
from collections.abc import AsyncIterator, Iterator

def stream(text: str) -> Iterator[str]: ...
async def astream(text: str) -> AsyncIterator[str]: ...
```

## Minimal Example

```python title="typed_retrieval.py"
from collections.abc import Sequence
from typing import Protocol, TypedDict

type Vector = list[float]


class Chunk(TypedDict):
    id: str
    text: str
    score: float


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[Vector]: ...


def best_chunk(chunks: Sequence[Chunk], *, min_score: float = 0.5) -> Chunk | None:
    """Return the highest-scoring chunk above the threshold, or None."""
    eligible = [c for c in chunks if c["score"] >= min_score]
    return max(eligible, key=lambda c: c["score"]) if eligible else None


class FakeEmbedder:
    """No inheritance needed - it structurally satisfies Embedder."""

    def embed(self, texts: list[str]) -> list[Vector]:
        return [[float(len(t))] for t in texts]


def use(embedder: Embedder) -> list[Vector]:
    return embedder.embed(["a", "bb"])


print(use(FakeEmbedder()))
print(best_chunk([{"id": "c1", "text": "x", "score": 0.4}], min_score=0.5))
print(best_chunk([{"id": "c1", "text": "x", "score": 0.8}]))
```

```text
[[1.0], [2.0]]
None
{'id': 'c1', 'text': 'x', 'score': 0.8}
```

## Real-World Example

A fully typed retrieval module — and the mypy configuration that keeps it honest.

```python title="src/service/retrieval/interfaces.py"
"""Typed interfaces for the retrieval layer.

Protocols here mean the RAG service depends on *shapes*, not on classes, so a
test double, an in-memory index and Qdrant are all equally valid collaborators.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable

type Vector = Sequence[float]
type ChunkId = str

Strategy = Literal["vector", "keyword", "hybrid"]


class ChunkPayload(TypedDict):
    """The JSON shape stored alongside each vector."""

    text: str
    document: str
    page: int
    ingested_at: str


@dataclass(frozen=True, slots=True)
class Hit:
    id: ChunkId
    score: float
    payload: ChunkPayload


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Implemented by providers and by fakes."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorIndex(Protocol):
    def upsert(self, ids: Sequence[ChunkId], vectors: Sequence[Vector],
               payloads: Sequence[ChunkPayload]) -> int: ...

    def search(self, vector: Vector, *, k: int = 5,
               where: dict[str, object] | None = None) -> list[Hit]: ...

    def count(self) -> int: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit], *, top_n: int = 5) -> list[Hit]: ...


class Retriever:
    """Composes an embedder, an index and an optional reranker."""

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        *,
        reranker: Reranker | None = None,
        strategy: Strategy = "vector",
        k: int = 10,
        top_n: int = 5,
        min_score: float = 0.3,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.strategy = strategy
        self.k = k
        self.top_n = top_n
        self.min_score = min_score

    def retrieve(self, query: str, *, where: dict[str, object] | None = None) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        hits = self.index.search(vector, k=self.k, where=where)
        hits = [h for h in hits if h.score >= self.min_score]

        if self.reranker is not None and hits:
            hits = self.reranker.rerank(query, hits, top_n=self.top_n)

        return hits[: self.top_n]

    def stats(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "vectors": self.index.count(),
            "dimension": self.embedder.dimension,
            "reranked": self.reranker is not None,
        }


class HashEmbedder:
    """Deterministic fake embedder for tests - no API, no model, no network."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in text.lower().split():
            vector[hash(token) % self._dimension] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]


if __name__ == "__main__":
    embedder = HashEmbedder()
    print("satisfies Embedder:", isinstance(embedder, Embedder))   # runtime_checkable
    print("dimension:", embedder.dimension)
    print("vector:", [round(v, 2) for v in embedder.embed_query("vector search")])
```

```text
satisfies Embedder: True
dimension: 8
vector: [0.0, 0.71, 0.0, 0.0, 0.71, 0.0, 0.0, 0.0]
```

```toml title="pyproject.toml (type checking config)"
[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
warn_return_any = true
disallow_untyped_defs = true
ignore_missing_imports = true          # third-party packages without stubs

[[tool.mypy.overrides]]
module = ["tests.*"]
disallow_untyped_defs = false          # pragmatic: tests may stay lighter
```

```bash
uv add --dev mypy
uv run mypy src
```

:::note `runtime_checkable` checks method *names*, not signatures
`isinstance(x, Embedder)` returns `True` if `x` has `embed` and `embed_query` attributes,
regardless of their parameters. It is a smoke test, not a guarantee — the real check is
static.
:::

## Common Mistakes

:::mistake
```python
# 1. Believing annotations are enforced
def f(x: int) -> int: return x
f("3")                              # runs; returns "3". Only mypy complains.

# 2. Mutable default with an Optional type
def g(items: list[str] = []) -> None: ...       # the mutable-default bug, now typed

# 3. Any as a habit
def h(data: Any) -> Any: ...        # switches the checker off exactly where you need it

# 4. Forgetting | None on things that can be missing
def find(id: str) -> dict: return INDEX.get(id)   # actually dict | None

# 5. Old-style imports
from typing import List, Dict, Optional          # use list, dict, X | None

# 6. Typing a mutable argument as a concrete container when you only read it
def f(items: list[str]): ...        # Sequence[str] accepts tuples and is clearer about intent
```
:::

## Debugging

```bash
uv run mypy src --show-error-codes --pretty
uv run mypy src --disallow-any-explicit          # find where Any crept in
```

```python
from typing import reveal_type

reveal_type(hits)      # mypy prints the inferred type; remove before committing
```

Adopting types in an existing codebase: start with `mypy --ignore-missing-imports` and no
strictness, fix what it finds, then enable `disallow_untyped_defs` module by module. A
big-bang `strict = true` on a large untyped codebase produces thousands of errors and gets
abandoned.

## Security Considerations

:::security Type hints are not validation
An annotation does nothing to data arriving from an HTTP request, a file or a model.
Anything crossing a trust boundary must be validated at runtime:

```python
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    k: int = Field(default=5, ge=1, le=20)
```
Phase 19 makes this the standard for every input and every model output.
:::

## Hands-on Exercise

:::exercise Type an existing module
Take the `select_chunks` function from Phase 1 and give it a fully typed home:

1. `type Score = float`, `ChunkId = str`, and a `TypedDict` for the chunk shape.
2. A `Protocol` `TokenCounter` with `count(text: str) -> int`, so the token estimate is
   injected rather than hard-coded as `len(text) // 4`.
3. A generic `def top_n[T](items: Sequence[T], key: Callable[[T], float], n: int) -> list[T]`.
4. `select_chunks` typed end-to-end, returning `list[Chunk]`.
5. `uv run mypy --strict` passing with zero errors and no `Any`.
:::

:::solution Solution
```python title="typed_select.py"
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict

type Score = float
type ChunkId = str


class Chunk(TypedDict):
    id: ChunkId
    text: str
    score: Score
    document: str


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """Cheap heuristic; swap for a real tokenizer in Phase 10."""

    def __init__(self, chars_per_token: int = 4) -> None:
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)


def top_n[T](items: Sequence[T], key: Callable[[T], float], n: int) -> list[T]:
    return sorted(items, key=key, reverse=True)[:n]


def select_chunks(
    candidates: Sequence[Chunk],
    counter: TokenCounter,
    *,
    max_tokens: int = 3_000,
    min_score: Score = 0.35,
    max_per_document: int = 3,
) -> list[Chunk]:
    eligible = [c for c in candidates if c["score"] >= min_score]
    ranked = top_n(eligible, key=lambda c: c["score"], n=len(eligible))

    selected: list[Chunk] = []
    seen: set[ChunkId] = set()
    per_document: dict[str, int] = {}
    used = 0

    for chunk in ranked:
        if chunk["id"] in seen:
            continue
        if per_document.get(chunk["document"], 0) >= max_per_document:
            continue
        cost = counter.count(chunk["text"])
        if used + cost > max_tokens:
            continue
        selected.append(chunk)
        seen.add(chunk["id"])
        per_document[chunk["document"]] = per_document.get(chunk["document"], 0) + 1
        used += cost

    return selected


if __name__ == "__main__":
    chunks: list[Chunk] = [
        {"id": "c1", "text": "x" * 1_200, "score": 0.91, "document": "handbook"},
        {"id": "c2", "text": "y" * 1_200, "score": 0.62, "document": "handbook"},
        {"id": "c3", "text": "z" * 400, "score": 0.20, "document": "faq"},
    ]
    chosen = select_chunks(chunks, ApproxTokenCounter(), max_tokens=400)
    print([c["id"] for c in chosen])
```

```text
['c1']
```

```bash
uv run mypy --strict typed_select.py
```

```text
Success: no issues found in 1 source file
```
:::

## Challenge

:::challenge Type the tool registry
Return to the `@tool` decorator from the previous lesson and make it fully typed with
`ParamSpec` and `TypeVar`, so that decorating a function preserves its exact signature for
callers and the checker:

```python
def tool[**P, R](fn: Callable[P, R]) -> Callable[P, R]: ...
```

Then verify that calling a decorated tool with wrong argument types is an error under mypy.
Signature-preserving decorators are what make framework-heavy code navigable in an editor.
:::

## Interview Questions

:::interview
1. Are type hints enforced at runtime?
2. What is the difference between a Protocol and an ABC?
3. When would you use `TypedDict` instead of a dataclass or a Pydantic model?
4. What does `x: str | None` force a caller to do?
5. How would you introduce typing into a large untyped codebase?
:::

## Cheat Sheet

```python
x: int  y: str  z: float  flag: bool
items: list[str]  meta: dict[str, int]  pair: tuple[str, float]  many: tuple[int, ...]
maybe: str | None            either: int | str          const: Final = 3
mode: Literal["a", "b"]      type Vector = list[float]

from collections.abc import Callable, Sequence, Iterable, Iterator, AsyncIterator, Mapping
handler: Callable[[str, int], bool]

from typing import Protocol, TypedDict, NotRequired, Annotated, runtime_checkable
class Iface(Protocol):
    def method(self, x: int) -> str: ...

def first[T](xs: Sequence[T]) -> T | None: ...     # 3.12 generics
class Cache[K, V]: ...

uv run mypy src --strict
```

```quiz
[
  {
    "question": "What happens at runtime when you call f('hello') on def f(x: int) -> int?",
    "options": [
      "TypeError",
      "It runs normally; only a static checker would complain",
      "The string is converted to an int",
      "A warning is printed"
    ],
    "answer": 1,
    "explanation": "Python does not enforce annotations. Static checking (mypy/pyright) and runtime validation (Pydantic) are two different, complementary tools."
  },
  {
    "question": "You want any object with .embed() to be acceptable, including third-party clients you cannot modify. What do you use?",
    "options": ["An abstract base class", "A Protocol", "A TypedDict", "isinstance checks"],
    "answer": 1,
    "explanation": "Protocols are structural: conformance is by shape, so classes you do not own - and simple test doubles - satisfy them without inheriting anything."
  },
  {
    "question": "Which annotation best expresses 'this function may not find anything'?",
    "options": ["-> dict", "-> dict | None", "-> Any", "-> object"],
    "answer": 1,
    "explanation": "`dict | None` forces every caller to handle the missing case, which is exactly the class of bug a type checker should catch for you."
  }
]
```

## Summary

- Type hints are checked by tools, not by Python — but Pydantic, FastAPI and agent
  frameworks read them at runtime to build validation and schemas.
- Prefer built-in generics (`list[str]`), `X | None`, `Literal` and type aliases.
- Protocols give interfaces without inheritance; ABCs enforce and share implementation.
- Types document intent and catch `None` bugs; they never replace runtime validation of
  untrusted input.

## Next Step

Async Python — concurrency for I/O-bound work, which is what almost all AI code is.
