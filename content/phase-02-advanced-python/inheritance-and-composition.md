---
title: Inheritance, Composition and Polymorphism
order: 2
difficulty: Intermediate
duration: 14
badges: ["Hands-on"]
summary: "Abstract base classes, method overriding and super() — and the reason experienced engineers reach for composition first."
prereqs: ["Classes and Objects"]
keyConcepts: ["inheritance", "composition", "polymorphism", "ABC", "super()"]
---

:::note In one line
**Prefer composition over inheritance.** Inheritance says *is a*; composition says *has a* - and *has a* stays flexible far longer.
:::

## Why this matters

Every provider abstraction you will build — three vector stores behind one interface, four
LLM providers behind one `complete()` method — is this lesson. Get it right and swapping
Chroma for Qdrant is a one-line change. Get it wrong (deep inheritance trees, base classes
that know about their children) and the codebase calcifies.

## Mental Model

Two ways to reuse code, and one of them ages much better.

- **Inheritance** says *is a*. A `CachedRetriever` **is a** `Retriever`.
- **Composition** says *has a*. A `Service` **has a** retriever.

<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Diagram comparing inheritance, where subclasses form a rigid tree and a change at the top affects everything below, with composition, where a service holds interchangeable parts that can be swapped freely.">
  <text class="dg-label" x="14" y="22" fill="var(--warn)">Inheritance — a rigid tree</text>
  <rect x="96" y="32" width="120" height="34" rx="7" fill="var(--panel-2)" stroke="var(--warn)" stroke-width="1.8"/>
  <text class="dg-sub" x="156" y="54" text-anchor="middle">Retriever</text>
  <rect x="20" y="86" width="120" height="34" rx="7" class="dg-box"/>
  <text class="dg-sub" x="80" y="108" text-anchor="middle">CachedRetriever</text>
  <rect x="164" y="86" width="126" height="34" rx="7" class="dg-box"/>
  <text class="dg-sub" x="227" y="108" text-anchor="middle">HybridRetriever</text>
  <path class="dg-arrow" d="M140,66 L92,82"/>
  <path class="dg-arrow" d="M180,66 L218,82"/>
  <text class="dg-sub" x="14" y="142" fill="var(--warn)">Change the parent and everything below it changes too.</text>
  <text class="dg-sub" x="14" y="160" fill="var(--warn)">Need caching AND hybrid? Now you are stuck.</text>
  <line x1="320" y1="14" x2="320" y2="236" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-label" x="344" y="22" fill="var(--ok)">Composition — parts you can swap</text>
  <rect x="344" y="32" width="150" height="40" rx="8" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="2"/>
  <text class="dg-sub" x="419" y="57" text-anchor="middle" fill="var(--ok)">Service</text>
  <rect x="344" y="92" width="136" height="32" rx="6" class="dg-box"/>
  <text class="dg-sub" x="412" y="112" text-anchor="middle">a retriever</text>
  <rect x="344" y="132" width="136" height="32" rx="6" class="dg-box"/>
  <text class="dg-sub" x="412" y="152" text-anchor="middle">a cache</text>
  <rect x="344" y="172" width="136" height="32" rx="6" class="dg-box"/>
  <text class="dg-sub" x="412" y="192" text-anchor="middle">a reranker</text>
  <path class="dg-arrow" d="M400,72 L400,88"/>
  <path class="dg-arrow" d="M400,124 L400,128"/>
  <path class="dg-arrow" d="M400,164 L400,168"/>
  <rect x="502" y="92" width="144" height="112" rx="9" fill="var(--panel)" stroke="var(--ok)" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text class="dg-sub" x="574" y="122" text-anchor="middle" fill="var(--ok)">swap any part</text>
  <text class="dg-sub" x="574" y="142" text-anchor="middle">for a fake one</text>
  <text class="dg-sub" x="574" y="162" text-anchor="middle">in a test, with</text>
  <text class="dg-sub" x="574" y="182" text-anchor="middle">no subclassing</text>
  <text class="dg-sub" x="344" y="228">Mix and match freely. Nothing is locked to anything.</text>
</svg>
<figcaption>
<strong>Prefer composition.</strong> The tree looks tidier on day one, but the moment you
need two behaviours at once it forces you into awkward multiple inheritance. Holding parts
keeps every combination available.
</figcaption>
</figure>

```python
# Inheritance: locked into one chain
class CachedRetriever(Retriever):
    def search(self, query):
        ...

# Composition: parts you pass in, and can replace
class SearchService:
    def __init__(self, retriever, cache, reranker=None):
        self.retriever = retriever          # any object with .search()
        self.cache = cache
        self.reranker = reranker

    def search(self, query):
        if hit := self.cache.get(query):
            return hit
        results = self.retriever.search(query)
        if self.reranker:
            results = self.reranker.rerank(query, results)
        self.cache.set(query, results)
        return results
```

The second version can be tested with a fake retriever and a dict as the cache — no
subclassing, no mocking framework.

:::tip The question that decides it
Ask: *would I ever want this behaviour without the parent?*

If yes, it is a **part** (composition). If it genuinely only makes sense as a specialised
version of the parent, inheritance is fine. In practice the answer is usually "yes" and
inheritance is usually the wrong call.
:::

## Core Concepts

### Inheritance basics

```python
class Tool:
    name = "tool"

    def __init__(self, timeout_s: float = 10.0) -> None:
        self.timeout_s = timeout_s

    def run(self, **kwargs) -> str:
        raise NotImplementedError


class CalculatorTool(Tool):
    name = "calculator"

    def __init__(self, *, precision: int = 4, timeout_s: float = 2.0) -> None:
        super().__init__(timeout_s=timeout_s)     # always call super().__init__
        self.precision = precision

    def run(self, expression: str = "", **kwargs) -> str:
        return str(round(eval_safely(expression), self.precision))
```

`super()` calls the next class in the method resolution order — not necessarily the literal
parent, which matters with multiple inheritance. Forgetting `super().__init__()` leaves the
base class's attributes unset, producing `AttributeError` far from the cause.

### Abstract base classes

An ABC declares an interface and refuses to be instantiated if it is incomplete.

```python
from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Every vector store implementation must provide these three operations."""

    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        """Insert or replace vectors."""

    @abstractmethod
    def search(self, vector: list[float], k: int = 5, where: dict | None = None) -> list[dict]:
        """Return the k nearest neighbours, optionally filtered by metadata."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored vectors."""

    # concrete helper shared by all implementations
    def is_empty(self) -> bool:
        return self.count() == 0


VectorStore()            # TypeError: Can't instantiate abstract class
```

### Polymorphism

Code written against the abstraction works with every implementation:

```python
def index_corpus(store: VectorStore, chunks: list[dict], embedder) -> int:
    vectors = embedder.embed([c["text"] for c in chunks])
    store.upsert([c["id"] for c in chunks], vectors, [c["meta"] for c in chunks])
    return store.count()

index_corpus(InMemoryStore(), chunks, embedder)   # in tests
index_corpus(QdrantStore(url), chunks, embedder)  # in production
```

Python also allows **duck typing**: any object with the right methods works, no base class
required. Combined with `Protocol` (two lessons on), this gives you interfaces without
inheritance at all — often the cleanest option.

### Composition

```python
class RagService:
    """Composed from collaborators, each independently testable and replaceable."""

    def __init__(self, store: VectorStore, llm: LLMClient, *, k: int = 5) -> None:
        self.store = store
        self.llm = llm
        self.k = k

    def answer(self, question: str) -> dict:
        chunks = self.store.search(self.llm.embed(question), k=self.k)
        prompt = build_prompt(question, chunks)
        text = self.llm.complete([{"role": "user", "content": prompt}])
        return {"answer": text, "citations": [c["id"] for c in chunks]}
```

`RagService` is not a `VectorStore` and not an `LLMClient`; it *uses* both. Swapping either
is a constructor argument.

:::tip The test: say it out loud
"A Qdrant store **is a** vector store" — inheritance. "A RAG service **is a** vector store"
— obviously false, so composition. If you find yourself inheriting to reuse one useful
method, you wanted composition.
:::

### Multiple inheritance and mixins

```python
class TimingMixin:
    """Adds timing to any class with a `run` method. Mixins add behaviour, not identity."""

    def run_timed(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return self.run(*args, **kwargs)
        finally:
            self.last_duration_s = time.monotonic() - start


class SearchTool(TimingMixin, Tool):
    ...
```

Python resolves this with the MRO (`SearchTool.__mro__`). Keep mixins shallow and
single-purpose; three levels of multiple inheritance is a debugging nightmare.

## Minimal Example

```python title="stores.py"
from abc import ABC, abstractmethod


class Store(ABC):
    @abstractmethod
    def put(self, key: str, value: str) -> None: ...

    @abstractmethod
    def get(self, key: str) -> str | None: ...

    def get_or_default(self, key: str, default: str) -> str:
        found = self.get(key)
        return default if found is None else found


class MemoryStore(Store):
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        self._data[key] = value

    def get(self, key: str) -> str | None:
        return self._data.get(key)


class LoggingStore(Store):
    """Decorator-by-composition: wraps another Store instead of subclassing it."""

    def __init__(self, inner: Store) -> None:
        self.inner = inner
        self.operations: list[str] = []

    def put(self, key: str, value: str) -> None:
        self.operations.append(f"put {key}")
        self.inner.put(key, value)

    def get(self, key: str) -> str | None:
        self.operations.append(f"get {key}")
        return self.inner.get(key)


store = LoggingStore(MemoryStore())
store.put("a", "1")
print(store.get("a"), store.get_or_default("missing", "fallback"))
print(store.operations)
```

```text
1 fallback
['put a', 'get a', 'get missing']
```

`LoggingStore` wraps any `Store` — including another wrapper. That is composition doing
what deep inheritance cannot.

## Real-World Example

One interface, three implementations, chosen by configuration.

```python title="src/service/adapters/vectorstore.py"
"""Vector store abstraction.

The only file in the project that knows which vector database is in use.
Everything else depends on the abstract VectorStore.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


class VectorStore(ABC):
    """Minimal interface every backend must satisfy."""

    @abstractmethod
    def upsert(self, records: list[dict]) -> int:
        """records: [{'id', 'vector', 'text', 'metadata'}]. Returns the count written."""

    @abstractmethod
    def search(self, vector: list[float], *, k: int = 5, where: dict | None = None) -> list[SearchHit]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    def health(self) -> dict[str, object]:
        """Shared concrete behaviour - subclasses rarely need to override this."""
        try:
            return {"ok": True, "backend": type(self).__name__, "vectors": self.count()}
        except Exception as exc:                      # surface, do not hide, backend failures
            return {"ok": False, "backend": type(self).__name__, "error": str(exc)}


class InMemoryStore(VectorStore):
    """Exact brute-force search. Perfect for tests and for corpora under ~50k chunks."""

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}

    def upsert(self, records: list[dict]) -> int:
        for record in records:
            self._records[record["id"]] = record
        return len(records)

    def search(self, vector: list[float], *, k: int = 5, where: dict | None = None) -> list[SearchHit]:
        def matches(meta: dict) -> bool:
            return all(meta.get(key) == value for key, value in (where or {}).items())

        scored = [
            SearchHit(
                id=r["id"],
                score=_cosine(vector, r["vector"]),
                text=r["text"],
                metadata=r.get("metadata", {}),
            )
            for r in self._records.values()
            if matches(r.get("metadata", {}))
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._records)


class ChromaStore(VectorStore):
    """Persistent local store. Imports the SDK lazily so tests never need it installed."""

    def __init__(self, path: str = ".data/chroma", collection: str = "chunks") -> None:
        import chromadb                                  # lazy: keeps import cost local

        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection)

    def upsert(self, records: list[dict]) -> int:
        self._collection.upsert(
            ids=[r["id"] for r in records],
            embeddings=[r["vector"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[r.get("metadata", {}) for r in records],
        )
        return len(records)

    def search(self, vector: list[float], *, k: int = 5, where: dict | None = None) -> list[SearchHit]:
        result = self._collection.query(
            query_embeddings=[vector], n_results=k, where=where or None
        )
        hits: list[SearchHit] = []
        for id_, doc, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0],
            strict=True,
        ):
            hits.append(SearchHit(id=id_, score=1.0 - distance, text=doc, metadata=meta or {}))
        return hits

    def count(self) -> int:
        return self._collection.count()


def build_store(kind: str, **kwargs) -> VectorStore:
    """Factory: configuration decides the backend, callers never know which."""
    backends: dict[str, type[VectorStore]] = {"memory": InMemoryStore, "chroma": ChromaStore}
    try:
        return backends[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown store kind {kind!r}; known: {sorted(backends)}") from exc


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


if __name__ == "__main__":
    store = build_store("memory")
    store.upsert([
        {"id": "c1", "vector": [1.0, 0.0], "text": "vector search", "metadata": {"doc": "a"}},
        {"id": "c2", "vector": [0.0, 1.0], "text": "graph index", "metadata": {"doc": "b"}},
    ])
    for hit in store.search([0.9, 0.1], k=2):
        print(f"{hit.id} {hit.score:.3f} {hit.text}")
    print(store.health())
```

```text
c1 0.994 vector search
c2 0.110 graph index
{'ok': True, 'backend': 'InMemoryStore', 'vectors': 2}
```

## Common Mistakes

:::mistake
```python
# 1. Inheriting to reuse a method
class ReportGenerator(DatabaseClient):     # a report generator is NOT a database client
    ...                                     # compose: self.db = DatabaseClient()

# 2. Forgetting super().__init__()
class Child(Parent):
    def __init__(self, x):
        self.x = x                          # Parent's attributes never initialised

# 3. Base classes that know their subclasses
class Tool:
    def run(self):
        if isinstance(self, SearchTool): ...  # backwards: the base must not know children

# 4. Deep hierarchies
class A(B(C(D)))                            # three levels is already too many

# 5. Overriding with a different signature
class Base:  def search(self, q, k=5): ...
class Impl:  def search(self, query): ...   # breaks every caller (LSP violation)
```
:::

## Best Practices

1. Default to composition; inherit only to implement a declared interface.
2. Keep hierarchies one level deep: ABC → implementations.
3. Subclasses must be substitutable — same signature, same contract, no surprises.
4. Import heavy vendor SDKs lazily inside the implementation that needs them.
5. Provide an in-memory implementation of every interface; it makes tests fast and offline.
6. Use a factory function to turn configuration into an implementation.

## Hands-on Exercise

:::exercise Pluggable chunkers
Define an abstract `Chunker` with `chunk(text: str) -> list[str]` and a concrete shared
helper `stats(text)` returning `{chunks, mean_chars, max_chars}`. Implement three:

1. `FixedSizeChunker(size, overlap)` — character windows.
2. `ParagraphChunker(max_chars)` — split on blank lines, merging short paragraphs.
3. `SentenceChunker(max_chars)` — split on sentence boundaries, packing to the limit.

Then write `compare(text, chunkers)` printing a table of stats for each, and a factory
`build_chunker(name, **kwargs)`. Every strategy must be usable without changing any code
that consumes `Chunker`.
:::

:::solution Solution sketch with the key parts
```python title="chunkers.py"
from __future__ import annotations

import re
from abc import ABC, abstractmethod


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...

    def stats(self, text: str) -> dict[str, float | int]:
        pieces = self.chunk(text)
        if not pieces:
            return {"chunks": 0, "mean_chars": 0, "max_chars": 0}
        sizes = [len(p) for p in pieces]
        return {
            "chunks": len(pieces),
            "mean_chars": round(sum(sizes) / len(sizes), 1),
            "max_chars": max(sizes),
        }


class FixedSizeChunker(Chunker):
    def __init__(self, size: int = 400, overlap: int = 80) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size, self.overlap = size, overlap

    def chunk(self, text: str) -> list[str]:
        step = self.size - self.overlap
        out = []
        for start in range(0, max(len(text), 1), step):
            piece = text[start : start + self.size].strip()
            if piece:
                out.append(piece)
            if start + self.size >= len(text):
                break
        return out


class ParagraphChunker(Chunker):
    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[str]:
        out, buffer = [], ""
        for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
            if buffer and len(buffer) + len(para) + 2 > self.max_chars:
                out.append(buffer)
                buffer = para
            else:
                buffer = f"{buffer}\n\n{para}" if buffer else para
        if buffer:
            out.append(buffer)
        return out


class SentenceChunker(Chunker):
    SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chars: int = 600) -> None:
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[str]:
        out, buffer = [], ""
        for sentence in (s.strip() for s in self.SPLIT.split(text) if s.strip()):
            if buffer and len(buffer) + len(sentence) + 1 > self.max_chars:
                out.append(buffer)
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}" if buffer else sentence
        if buffer:
            out.append(buffer)
        return out


def build_chunker(name: str, **kwargs) -> Chunker:
    options: dict[str, type[Chunker]] = {
        "fixed": FixedSizeChunker, "paragraph": ParagraphChunker, "sentence": SentenceChunker,
    }
    if name not in options:
        raise ValueError(f"unknown chunker {name!r}; known: {sorted(options)}")
    return options[name](**kwargs)


def compare(text: str, chunkers: dict[str, Chunker]) -> None:
    print(f"{'strategy':<12}{'chunks':>8}{'mean':>9}{'max':>7}")
    for label, chunker in chunkers.items():
        s = chunker.stats(text)
        print(f"{label:<12}{s['chunks']:>8}{s['mean_chars']:>9}{s['max_chars']:>7}")


if __name__ == "__main__":
    doc = ("RAG grounds answers in retrieved context. It reduces hallucination. "
           "Chunking decides what can be retrieved.\n\n"
           "Rerankers reorder candidates. They cost latency but raise precision. ") * 4
    compare(doc, {
        "fixed": build_chunker("fixed", size=300, overlap=60),
        "paragraph": build_chunker("paragraph", max_chars=400),
        "sentence": build_chunker("sentence", max_chars=300),
    })
```

```text
strategy      chunks     mean    max
fixed              4    281.5    300
paragraph          3    297.3    381
sentence           4    276.2    297
```

You will compare real chunking strategies with real retrieval metrics in Phase 11 — the
interface stays exactly this.
:::

## Challenge

:::challenge Add a caching decorator store
Write `CachedStore(inner: VectorStore, cache: TTLCache)` that caches `search` results keyed
by `(vector_hash, k, where)`. It must satisfy `VectorStore` completely, so any code taking a
`VectorStore` accepts it. Then stack it: `CachedStore(LoggingStore(ChromaStore()))`.

If that composes cleanly, you have understood why composition beats inheritance — you cannot
stack subclasses.
:::

## Interview Questions

:::interview
1. When do you choose composition over inheritance?
2. What does an abstract base class give you that duck typing does not?
3. What is the Liskov substitution principle, in practical terms?
4. What does `super()` actually resolve to with multiple inheritance?
5. How would you support three vector databases without touching business logic?
:::

## Cheat Sheet

```python
from abc import ABC, abstractmethod

class Interface(ABC):
    @abstractmethod
    def required(self) -> None: ...
    def shared(self) -> None: ...          # concrete helper, inherited by all

class Impl(Interface):
    def __init__(self, dep):
        super().__init__()
        self.dep = dep                     # composition
    def required(self) -> None: ...

isinstance(obj, Interface)      type(obj).__mro__      Impl.__bases__

# wrapper by composition (stackable)
class Wrapper(Interface):
    def __init__(self, inner: Interface): self.inner = inner
    def required(self): return self.inner.required()
```

```quiz
[
  {
    "question": "You need logging, caching and retries around a vector store. What is the cleanest design?",
    "options": [
      "A LoggingCachingRetryingStore subclass",
      "Three wrapper classes that each implement VectorStore and hold an inner store",
      "Add flags to the base class",
      "Copy the store class three times"
    ],
    "answer": 1,
    "explanation": "Wrappers compose in any order and combination; a subclass per combination explodes combinatorially."
  },
  {
    "question": "What happens if a subclass does not implement every @abstractmethod?",
    "options": [
      "It works until the method is called",
      "Instantiating it raises TypeError",
      "Python inserts a default",
      "It becomes abstract silently"
    ],
    "answer": 1,
    "explanation": "ABCs fail at construction, which is far earlier and clearer than an AttributeError deep in production."
  }
]
```

## Summary

- Inheritance expresses "is-a" and is best used for one flat interface with several
  implementations.
- Composition expresses "has-a", stacks freely, and is the default choice.
- ABCs enforce the interface at construction time; an in-memory implementation makes tests
  fast.
- Subclasses must be substitutable: same signature, same contract.

## Next Step

Dataclasses and properties — much less boilerplate for the data-shaped classes that make up
most of a real codebase.
