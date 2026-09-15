---
title: Generators, Iterators and Context Managers
order: 5
difficulty: Intermediate
duration: 14
badges: ["Hands-on"]
summary: "Lazy sequences with yield, the iterator protocol, and with-blocks that guarantee cleanup — the machinery behind streaming LLM responses and safe resource handling."
prereqs: ["Comprehensions and Generator Expressions", "Classes and Objects"]
keyConcepts: ["yield", "iterator protocol", "lazy evaluation", "with", "contextmanager"]
---

:::note In one line
**A generator produces values one at a time instead of building a whole list.** That is how you process a file larger than your memory.
:::

## Why this matters

Streaming an LLM response token by token is a generator. Reading a 10 GB corpus without
filling RAM is a generator. Guaranteeing that a database connection, a file handle or a
trace span is closed even when the code raises is a context manager. Both appear in every
production AI system, and both are simpler than they look.

## Mental Model

```text
ITERABLE   something you can loop over            list, dict, file, generator
ITERATOR   something with __next__(), stateful    iter(list) -> list_iterator
GENERATOR  a function with `yield`; an iterator whose state is its paused execution

with block:
    __enter__()  → set up (open file, start span, acquire lock)
    ... body ...
    __exit__()   → tear down, ALWAYS, even on exception or return
```

```mermaid
flowchart LR
  CALL["gen = stream()"] --> PAUSE["function body paused<br/>nothing has run yet"]
  PAUSE -->|next()| Y1["runs until first yield<br/>produces a value, pauses again"]
  Y1 -->|next()| Y2["resumes after the yield"]
  Y2 -->|no more yields| STOP["StopIteration → loop ends"]
```

## Core Concepts

### Generators with `yield`

```python
def count_tokens_streaming(chunks: list[str]):
    """Yields a running total as each chunk is processed."""
    total = 0
    for chunk in chunks:
        total += len(chunk) // 4
        yield total                     # pause here, hand the value to the caller


for running_total in count_tokens_streaming(["a" * 400, "b" * 800]):
    print(running_total)
```

```text
100
300
```

Calling a generator function runs **none** of the body — it returns a generator object. The
body advances only when you iterate it.

```python
gen = count_tokens_streaming(["x" * 100])
next(gen)            # 25
next(gen)            # StopIteration
```

### `yield from`

Delegates to another iterable, forwarding its values:

```python
def all_chunks(documents: list[list[str]]):
    for document in documents:
        yield from document          # instead of: for c in document: yield c
```

### The iterator protocol

Any class implementing `__iter__` and `__next__` works with `for`:

```python
class Paginated:
    """Iterate an API's pages as if they were one sequence."""

    def __init__(self, fetch_page, page_size: int = 100):
        self.fetch_page = fetch_page
        self.page_size = page_size

    def __iter__(self):
        offset = 0
        while True:
            page = self.fetch_page(offset=offset, limit=self.page_size)
            if not page:
                return                      # ends the generator → StopIteration
            yield from page
            offset += len(page)
            if len(page) < self.page_size:
                return
```

Implementing `__iter__` as a generator (as above) is almost always easier than writing
`__next__` by hand.

### Generators that receive values

`yield` is an expression: `sent = yield value`. This underpins coroutines, though in modern
code `async`/`await` (next lesson) covers most of those cases.

```python
def accumulator():
    total = 0
    while True:
        value = yield total          # receives what the caller sends
        total += value or 0


acc = accumulator()
next(acc)                 # prime it: advance to the first yield
acc.send(10)              # 10
acc.send(5)               # 15
acc.close()
```

### Context managers

```python
with open("file.txt", encoding="utf-8") as handle:
    data = handle.read()
# handle.close() has already happened, even if read() raised
```

Write your own with a class:

```python
class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self                          # bound to `as t`

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        print(f"{self.label}: {self.elapsed_ms:.1f}ms")
        return False                         # False → exceptions propagate (usually right)
```

Or, far more often, with a decorator:

```python
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield                                # everything before yield = __enter__
    finally:                                 # everything after = __exit__
        print(f"{label}: {(time.perf_counter() - start) * 1000:.1f}ms")


with timer("retrieval"):
    chunks = retrieve(query)
```

:::warning Put the teardown in `finally`
Without it, an exception in the body skips your cleanup entirely — the exact situation a
context manager exists to prevent.
:::

### Multiple context managers and `ExitStack`

```python
with open("in.jsonl") as src, open("out.jsonl", "w") as dst:
    ...

from contextlib import ExitStack

with ExitStack() as stack:
    files = [stack.enter_context(open(p)) for p in paths]   # dynamic number, all closed
```

`contextlib.suppress` replaces the try/except/pass idiom honestly:

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    Path("cache.json").unlink()
```

## Minimal Example

```python title="streaming.py"
import time
from contextlib import contextmanager


def stream_answer(text: str, *, chunk_size: int = 8):
    """Simulate a streaming LLM response."""
    for i in range(0, len(text), chunk_size):
        time.sleep(0.01)
        yield text[i : i + chunk_size]


@contextmanager
def stream_metrics(label: str):
    stats = {"chunks": 0, "chars": 0}
    start = time.perf_counter()
    try:
        yield stats
    finally:
        elapsed = time.perf_counter() - start
        rate = stats["chars"] / elapsed if elapsed else 0
        print(f"\n[{label}] {stats['chunks']} chunks, {stats['chars']} chars, {rate:.0f} chars/s")


with stream_metrics("answer") as stats:
    for piece in stream_answer("Retrieval augmented generation grounds answers."):
        print(piece, end="", flush=True)
        stats["chunks"] += 1
        stats["chars"] += len(piece)
```

```text
Retrieval augmented generation grounds answers.
[answer] 6 chunks, 47 chars, 741 chars/s
```

Note that the metrics are printed even if the loop raises — that is the whole point.

## Real-World Example

A streaming ingest pipeline: read → clean → chunk → batch, all lazily, with resources
guaranteed closed.

```python title="src/ingest/pipeline.py"
"""Streaming ingestion.

Every stage is a generator, so a 50 GB corpus flows through in constant memory.
Nothing is materialised except the current batch.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    text: str
    index: int


def read_documents(paths: Iterable[Path]) -> Iterator[Document]:
    """Yield documents from .jsonl and .txt files without loading them all."""
    for path in paths:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("%s:%d invalid JSON, skipped", path, line_number)
                        continue
                    yield Document(
                        id=raw.get("id", f"{path.stem}-{line_number}"),
                        text=raw.get("text", ""),
                        source=str(path),
                    )
        else:
            yield Document(id=path.stem, text=path.read_text(encoding="utf-8"), source=str(path))


def clean(documents: Iterator[Document], *, min_chars: int = 50) -> Iterator[Document]:
    """Drop empties and normalise whitespace. Still lazy."""
    for document in documents:
        text = " ".join(document.text.split())
        if len(text) >= min_chars:
            yield Document(document.id, text, document.source)


def chunk(documents: Iterator[Document], *, size: int = 800, overlap: int = 150) -> Iterator[Chunk]:
    step = size - overlap
    for document in documents:
        for index, start in enumerate(range(0, len(document.text), step)):
            piece = document.text[start : start + size].strip()
            if piece:
                yield Chunk(f"{document.id}::{index}", document.id, piece, index)
            if start + size >= len(document.text):
                break


def batched(items: Iterator[Chunk], size: int = 64) -> Iterator[list[Chunk]]:
    """Group a stream into fixed-size batches - embedding APIs want batches."""
    batch: list[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


@contextmanager
def pipeline_run(label: str):
    """Log start/finish and duration whatever happens."""
    counters = {"documents": 0, "chunks": 0, "batches": 0}
    start = time.perf_counter()
    logger.info("%s: starting", label)
    try:
        yield counters
    except Exception:
        logger.exception("%s: FAILED after %.1fs", label, time.perf_counter() - start)
        raise
    else:
        logger.info(
            "%s: done in %.1fs (%s)", label, time.perf_counter() - start, counters
        )


def ingest(paths: list[Path], *, batch_size: int = 64) -> int:
    with pipeline_run("ingest") as counters:
        documents = read_documents(paths)
        cleaned = clean(documents)
        chunks = chunk(cleaned)

        for batch in batched(chunks, batch_size):
            counters["batches"] += 1
            counters["chunks"] += len(batch)
            # embed_and_upsert(batch)   ← the only step that costs money
        return counters["chunks"]


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

    demo = Path("data/demo")
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"d{i}", "text": "Retrieval grounds answers. " * 40})
            for i in range(20)
        ),
        encoding="utf-8",
    )

    total = ingest(list(demo.glob("*.jsonl")), batch_size=16)
    print("chunks produced:", total)
```

```text
INFO ingest: starting
INFO ingest: done in 0.0s ({'documents': 0, 'chunks': 40, 'batches': 3})
chunks produced: 40
```

Because every stage is a generator, memory use is independent of corpus size — swap the
20-document demo for a million and nothing changes.

## Common Mistakes

:::mistake
```python
# 1. Consuming a generator twice
chunks = chunk(docs)
print(sum(1 for _ in chunks))     # 400
print(list(chunks)[:3])           # [] - exhausted

# 2. len() on a generator
len(chunks)                       # TypeError - no length without consuming it

# 3. Late binding in a generator over a loop variable
gens = [(x for _ in range(3)) for x in range(3)]   # fine
funcs = [lambda: x for x in range(3)]              # all return 2 - classic closure trap

# 4. Returning instead of yielding
def gen():
    return [1, 2, 3]              # a list, not a generator

# 5. Cleanup outside finally
@contextmanager
def bad():
    resource = acquire()
    yield resource
    resource.close()              # skipped if the body raises

# 6. Swallowing exceptions in __exit__
def __exit__(self, *exc):
    return True                   # True SUPPRESSES the exception - almost never what you want
```
:::

## Debugging

```python
import itertools

peek = list(itertools.islice(gen, 5))     # look at the first 5 without draining everything
gen = itertools.chain(peek, gen)          # put them back

from itertools import tee
a, b = tee(gen, 2)                        # two independent iterators (buffers in memory)
```

`inspect.getgeneratorstate(gen)` reports `GEN_CREATED`, `GEN_SUSPENDED`, `GEN_CLOSED` — handy
when a pipeline mysteriously produces nothing.

## Performance Considerations

| Aspect | List | Generator |
| --- | --- | --- |
| Memory | proportional to size | constant |
| First result available | after everything is built | immediately |
| Re-iterable | yes | no |
| Random access | yes | no |
| Best for | small collections, repeated use | streams, pipelines, large files |

Generators also improve *perceived* latency: streaming the first token of an LLM answer in
300 ms feels far faster than a complete answer in 3 s, even though total time is the same.

## Hands-on Exercise

:::exercise A streaming JSON accumulator
LLM APIs stream partial JSON. Write:

1. `stream_deltas(text, size=5)` — a generator yielding slices of a JSON string.
2. `accumulate_json(deltas)` — a generator that yields `(partial_text, parsed_or_None)` for
   each delta, attempting `json.loads` each time and yielding the parsed object once it
   succeeds.
3. A `@contextmanager` `stream_guard(timeout_s, max_chars)` that raises if the stream
   exceeds either limit, and always reports total chars and duration.

Demonstrate it with `{"answer": "yes", "confidence": 0.91}` and show the first delta at
which the JSON becomes valid.
:::

:::solution Solution
```python title="stream_json.py"
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager


class StreamLimitExceeded(RuntimeError):
    pass


def stream_deltas(text: str, size: int = 5) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


def accumulate_json(deltas: Iterator[str]) -> Iterator[tuple[str, dict | None]]:
    buffer = ""
    for delta in deltas:
        buffer += delta
        try:
            parsed = json.loads(buffer)
        except json.JSONDecodeError:
            parsed = None
        yield buffer, parsed


@contextmanager
def stream_guard(*, timeout_s: float = 5.0, max_chars: int = 10_000):
    state = {"chars": 0}
    start = time.perf_counter()

    def check(delta_len: int) -> None:
        state["chars"] += delta_len
        if state["chars"] > max_chars:
            raise StreamLimitExceeded(f"stream exceeded {max_chars} chars")
        if time.perf_counter() - start > timeout_s:
            raise StreamLimitExceeded(f"stream exceeded {timeout_s}s")

    try:
        yield check
    finally:
        print(f"[stream] {state['chars']} chars in {(time.perf_counter() - start) * 1000:.1f}ms")


if __name__ == "__main__":
    payload = json.dumps({"answer": "yes", "confidence": 0.91})

    with stream_guard(timeout_s=2.0, max_chars=1_000) as check:
        for step, (partial, parsed) in enumerate(accumulate_json(stream_deltas(payload, 5)), 1):
            check(len(partial))
            status = "VALID" if parsed else "partial"
            print(f"{step:>2} {status:<8} {partial}")
            if parsed:
                print("   parsed:", parsed)
                break
```

```text
 1 partial  {"ans
 2 partial  {"answer"
 3 partial  {"answer": "y
 4 partial  {"answer": "yes",
 5 partial  {"answer": "yes", "con
 6 partial  {"answer": "yes", "confiden
 7 partial  {"answer": "yes", "confidence":
 8 partial  {"answer": "yes", "confidence": 0.9
 9 VALID    {"answer": "yes", "confidence": 0.91}
   parsed: {'answer': 'yes', 'confidence': 0.91}
[stream] 198 chars in 0.3ms
```

Phase 10 uses exactly this to render structured output progressively while it is still
arriving.
:::

## Challenge

:::challenge A resource pool context manager
Write `@contextmanager def borrow(pool)` that takes a connection from a pool, yields it, and
always returns it — even if the body raises — plus:

- a maximum wait with a clear timeout error,
- a check that a broken connection is discarded rather than returned,
- statistics (`borrowed`, `returned`, `discarded`, `peak_in_use`).

Then use `ExitStack` to borrow three connections at once and confirm all are returned when
one of them fails. This is how database and HTTP connection pools work under the hood, and
why forgetting `finally` leaks connections until an outage.
:::

## Interview Questions

:::interview
1. What is the difference between an iterable, an iterator and a generator?
2. What happens when you call a generator function?
3. Why can a generator only be consumed once?
4. What does returning `True` from `__exit__` do?
5. When would you use `contextlib.ExitStack`?
:::

## Cheat Sheet

```python
def gen():
    yield value                 # pause and produce
    yield from other_iterable   # delegate
    return                      # stop (raises StopIteration)

next(gen, default)  iter(obj)  itertools.islice(gen, n)  itertools.tee(gen, 2)
itertools.chain(a, b)  itertools.groupby(sorted_items, key=...)

from contextlib import contextmanager, ExitStack, suppress

@contextmanager
def cm():
    setup()
    try:
        yield resource
    finally:
        teardown()

class CM:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False   # False = propagate
```

```quiz
[
  {
    "question": "What does calling a generator function do?",
    "options": [
      "Runs the body and returns a list",
      "Returns a generator object without running any of the body",
      "Raises StopIteration",
      "Runs the body up to the first yield"
    ],
    "answer": 1,
    "explanation": "Nothing executes until you iterate it. This laziness is what makes generator pipelines memory-flat."
  },
  {
    "question": "Where must cleanup go in a @contextmanager?",
    "options": [
      "Before the yield",
      "Immediately after the yield",
      "In a finally block around the yield",
      "In __del__"
    ],
    "answer": 2,
    "explanation": "Code after a bare yield is skipped when the body raises. Only finally guarantees the teardown runs."
  },
  {
    "question": "You must process a 40 GB JSONL corpus on a 16 GB laptop. What is the key technique?",
    "options": [
      "Read it with json.load()",
      "Chain generators so each stage yields one record at a time",
      "Increase swap space",
      "Split the file manually"
    ],
    "answer": 1,
    "explanation": "A generator pipeline keeps only the current record (and current batch) in memory, so file size stops mattering."
  }
]
```

## Summary

- `yield` turns a function into a lazy, resumable iterator; calling it runs nothing.
- Generator pipelines process unbounded data in constant memory and deliver first results
  immediately.
- Generators are single-use; materialise a list when you need to iterate twice.
- Context managers guarantee cleanup; put teardown in `finally` and let exceptions propagate.

## Next Step

Type hints and protocols — making all of this checkable by a tool instead of by hope.
