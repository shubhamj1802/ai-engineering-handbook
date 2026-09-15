---
title: Dataclasses, Properties and Enums
order: 3
difficulty: Intermediate
duration: 13
badges: ["Hands-on"]
summary: "Data-shaped classes without boilerplate — dataclasses with slots and frozen, computed properties, enums for closed sets of values."
prereqs: ["Classes and Objects"]
keyConcepts: ["dataclass", "frozen", "slots", "property", "Enum", "field"]
---

:::note In one line
**`@dataclass` writes the boring parts of a class for you.** You declare the fields; Python generates the constructor, the comparison and the printout.
:::

## Why this matters

Most classes in an AI codebase are *data with a little behaviour*: a chunk, a search hit, a
tool call, a usage record, a config block. Writing `__init__`, `__repr__` and `__eq__` by
hand for each is wasted effort and a source of bugs. Dataclasses remove the boilerplate and
make the shape of your data obvious at a glance.

## Mental Model

`@dataclass` writes the boring parts of a class for you. You declare the fields; Python
generates the constructor, the printout and the comparison.

<figure class="lesson-figure">
<svg viewBox="0 0 660 230" role="img" aria-label="Diagram: a hand-written class needs a long init method plus repr and eq, while a dataclass declares three fields and Python generates all of that automatically.">
  <defs>
    <marker id="dz-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <text class="dg-sub" x="14" y="20">by hand</text>
  <rect x="14" y="30" width="264" height="136" rx="9" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-mono" x="28" y="52" style="font-size:11px">class Chunk:</text>
  <text class="dg-mono" x="28" y="70" style="font-size:11px">  def __init__(self, text, page, score):</text>
  <text class="dg-mono" x="28" y="86" style="font-size:11px">    self.text = text</text>
  <text class="dg-mono" x="28" y="102" style="font-size:11px">    self.page = page</text>
  <text class="dg-mono" x="28" y="118" style="font-size:11px">    self.score = score</text>
  <text class="dg-mono" x="28" y="136" style="font-size:11px">  def __repr__(self): ...</text>
  <text class="dg-mono" x="28" y="154" style="font-size:11px">  def __eq__(self, other): ...</text>
  <path d="M286,98 L332,98" stroke="var(--accent)" stroke-width="2" fill="none" marker-end="url(#dz-a)"/>
  <text class="dg-sub" x="352" y="20">with @dataclass</text>
  <rect x="352" y="30" width="294" height="82" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-mono" x="366" y="52" style="font-size:11px">@dataclass(frozen=True, slots=True)</text>
  <text class="dg-mono" x="366" y="70" style="font-size:11px">class Chunk:</text>
  <text class="dg-mono" x="366" y="86" style="font-size:11px">  text: str</text>
  <text class="dg-mono" x="366" y="102" style="font-size:11px">  page: int</text>
  <rect x="352" y="124" width="294" height="42" rx="8" fill="var(--panel)" stroke="var(--ok)" stroke-width="1.6"/>
  <text class="dg-sub" x="499" y="142" text-anchor="middle" fill="var(--ok)">__init__, __repr__, __eq__</text>
  <text class="dg-sub" x="499" y="159" text-anchor="middle">all generated for you</text>
  <text class="dg-sub" x="330" y="200" text-anchor="middle">frozen=True makes it unchangeable. slots=True makes it smaller and faster.</text>
  <text class="dg-sub" x="330" y="218" text-anchor="middle">Both are good defaults for data that travels through a system.</text>
</svg>
<figcaption>
<strong>Same object, a quarter of the code.</strong> And because the fields carry types, your
editor can autocomplete them and a type checker can catch mistakes before you run anything.
</figcaption>
</figure>

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    page: int
    score: float = 0.0          # a default, like any function argument

chunk = Chunk("hello", page=3)
print(chunk)                    # Chunk(text='hello', page=3, score=0.0)
chunk == Chunk("hello", 3)      # True - compares by value, not identity
```

:::warning Why frozen=True is worth the small inconvenience
An unfrozen dataclass can be changed by anything that holds a reference to it — the same
"two labels, one object" problem from Phase 1, but now across your whole program.

```python
chunk.score = 0.9        # FrozenInstanceError - and that is a good thing
new = replace(chunk, score=0.9)   # make a changed copy instead
```
Frozen objects are also hashable, so they work as dict keys and set members.
:::

## Core Concepts

### The basic dataclass

```python
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)   # NOT metadata: dict = {}


chunk = Chunk("c1", "vector search", score=0.91)
chunk                        # Chunk(id='c1', text='vector search', score=0.91, metadata={})
chunk == Chunk("c1", "vector search", 0.91)                  # True - field-by-field equality
```

You get `__init__`, `__repr__` and `__eq__` free. Annotations are required — a bare
`id = ""` without a type is treated as a class attribute, not a field.

### frozen and slots

```python
@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    text: str


hit = SearchHit("c1", 0.91, "…")
hit.score = 0.5              # FrozenInstanceError - immutable
{hit}                        # works: frozen dataclasses are hashable
```

Use `frozen=True` for anything that crosses a boundary (results returned from a function,
values put in a cache or a set). It makes accidental mutation impossible, which removes an
entire class of bug in concurrent and pipeline code.

`slots=True` cuts memory roughly in half and speeds attribute access — worth it for objects
you create by the million (chunks, events, vectors).

### Post-init validation and derived fields

```python
@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    k: int = 5
    min_score: float = 0.35
    max_tokens: int = 3_000
    strategy: str = "hybrid"

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {self.min_score}")
        if self.strategy not in {"vector", "keyword", "hybrid"}:
            raise ValueError(f"unknown strategy: {self.strategy!r}")
```

An object that cannot exist in an invalid state removes the need for defensive checks
everywhere downstream.

### field() options

```python
@dataclass
class Record:
    id: str
    tags: list[str] = field(default_factory=list)      # fresh list per instance
    secret: str = field(repr=False, default="")        # kept out of __repr__ and logs
    created: float = field(default_factory=time.time)
    internal: dict = field(default_factory=dict, compare=False)   # ignored by ==
```

`repr=False` on secrets is a genuine security control: dataclass reprs end up in logs and
tracebacks.

### Properties

A property is a method that is accessed like an attribute — use it for values derived from
other fields.

```python
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def output_ratio(self) -> float:
        return self.output_tokens / self.total if self.total else 0.0


usage = Usage(1200, 300)
usage.total            # 1500  - no parentheses
usage.total = 9        # AttributeError: no setter defined
```

With a setter, when you need validation on assignment:

```python
class Model:
    def __init__(self, temperature: float = 0.2):
        self._temperature = 0.0
        self.temperature = temperature          # goes through the setter

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {value}")
        self._temperature = value
```

:::tip Property or method?
If it is cheap and has no side effects, a property reads better (`usage.total`). If it is
expensive, does I/O, or could fail, make it a method (`client.fetch_usage()`) so the cost is
visible at the call site.
:::

### Enums

```python
from enum import Enum, StrEnum, auto


class Role(StrEnum):           # StrEnum members ARE strings - JSON-friendly
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(Enum):
    COMPLETED = auto()
    MAX_ITERATIONS = auto()
    BUDGET_EXHAUSTED = auto()
    HUMAN_REJECTED = auto()


Role.USER               # <Role.USER: 'user'>
Role.USER == "user"     # True for StrEnum
Role("user")            # lookup by value; ValueError if unknown
list(Role)              # every member - great for validation and docs
```

Enums replace magic strings. `if reason == "max_iter"` compiles fine with a typo;
`if reason is StopReason.MAX_ITERATIONS` does not.

## Minimal Example

```python title="models.py"
from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    tokens: int = 0

    @property
    def preview(self) -> str:
        return self.content[:40] + ("…" if len(self.content) > 40 else "")

    def to_payload(self) -> dict[str, str]:
        return {"role": str(self.role), "content": self.content}


@dataclass
class Thread:
    thread_id: str
    messages: list[Message] = field(default_factory=list)

    def add(self, role: Role, content: str, tokens: int = 0) -> None:
        self.messages.append(Message(role, content, tokens))

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages)


thread = Thread("t_01")
thread.add(Role.SYSTEM, "You are concise.", 12)
thread.add(Role.USER, "Explain reciprocal rank fusion in one paragraph.", 34)

for message in thread.messages:
    print(f"{message.role:<10} {message.preview}")
print("total tokens:", thread.total_tokens)
```

```text
system     You are concise.
user       Explain reciprocal rank fusion in one p…
total tokens: 46
```

## Real-World Example

The domain model of a RAG service — the objects every other module passes around.

```python title="src/service/models.py"
"""Domain models.

Frozen where values are results (safe to cache, hash and share) and mutable only
where something genuinely accumulates. Nothing here does I/O, so every module can
import it without cost or cycles.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class Decision(StrEnum):
    ANSWER = "answer"
    ANSWER_WITH_CAVEAT = "answer_with_caveat"
    ESCALATE = "escalate"
    REFUSED = "refused"


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXHAUSTED = "budget_exhausted"
    GUARDRAIL_BLOCKED = "guardrail_blocked"


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    document: str
    page: int | None = None

    def label(self) -> str:
        return f"{self.document}" + (f" p.{self.page}" if self.page is not None else "")


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: str
    text: str
    score: float
    document: str
    page: int | None = None
    metadata: dict[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score out of range for {self.id}: {self.score}")

    @property
    def approx_tokens(self) -> int:
        return len(self.text) // 4

    def to_citation(self) -> Citation:
        return Citation(self.chunk_id_or_id(), self.document, self.page)

    def chunk_id_or_id(self) -> str:
        return self.metadata.get("source_chunk_id", self.id)


@dataclass(slots=True)
class AnswerBuilder:
    """Mutable while the pipeline runs; produces an immutable Answer at the end."""

    question: str
    started_at: float = field(default_factory=time.monotonic)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    text: str = ""
    decision: Decision = Decision.ESCALATE
    stop_reason: StopReason = StopReason.COMPLETED
    input_tokens: int = 0
    output_tokens: int = 0

    def add_chunks(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks.extend(chunks)

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)

    @property
    def context_tokens(self) -> int:
        return sum(c.approx_tokens for c in self.chunks)

    def build(self) -> "Answer":
        return Answer(
            question=self.question,
            text=self.text,
            decision=self.decision,
            stop_reason=self.stop_reason,
            citations=tuple(c.to_citation() for c in self.chunks),
            latency_ms=self.elapsed_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    text: str
    decision: Decision
    stop_reason: StopReason
    citations: tuple[Citation, ...]
    latency_ms: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_grounded(self) -> bool:
        return bool(self.citations) and self.decision is not Decision.ESCALATE

    def to_api(self) -> dict[str, object]:
        return {
            "answer": self.text,
            "decision": str(self.decision),
            "citations": [c.label() for c in self.citations],
            "usage": {"input": self.input_tokens, "output": self.output_tokens},
            "latency_ms": self.latency_ms,
        }


if __name__ == "__main__":
    builder = AnswerBuilder("What is HNSW?")
    builder.add_chunks([
        RetrievedChunk("c1", "HNSW is a graph-based ANN index." * 6, 0.91, "handbook.pdf", 12),
        RetrievedChunk("c2", "It trades recall for speed." * 6, 0.77, "handbook.pdf", 13),
    ])
    builder.text = "HNSW is a graph-based approximate nearest neighbour index [c1]."
    builder.decision = Decision.ANSWER
    builder.input_tokens, builder.output_tokens = 1420, 96

    answer = builder.build()
    print(answer.to_api())
    print("grounded:", answer.is_grounded, "| context tokens:", builder.context_tokens)
```

```text
{'answer': 'HNSW is a graph-based approximate nearest neighbour index [c1].', 'decision': 'answer', 'citations': ['handbook.pdf p.12', 'handbook.pdf p.13'], 'usage': {'input': 1420, 'output': 96}, 'latency_ms': 0}
grounded: True | context tokens: 45
```

The **builder → frozen result** pattern is worth internalising: mutate freely while working,
then freeze what you hand out.

## Common Mistakes

:::mistake
```python
# 1. Mutable default without default_factory
@dataclass
class A:
    tags: list = []                   # ValueError at class creation (dataclasses catch it)
    tags: list = field(default_factory=list)   # correct

# 2. Missing annotation
@dataclass
class B:
    name = "x"                        # NOT a field - just a class attribute

# 3. Fields without defaults after fields with defaults
@dataclass
class C:
    a: int = 1
    b: str                            # TypeError: non-default follows default

# 4. Expecting frozen to be deep
@dataclass(frozen=True)
class D:
    items: list[str] = field(default_factory=list)
d.items.append("x")                   # allowed! frozen blocks rebinding, not mutation
# use tuple[str, ...] for genuinely immutable collections

# 5. Expensive property
@property
def embedding(self):
    return call_embedding_api(self.text)    # a hidden network call behind attribute syntax
```
:::

## Debugging

```python
from dataclasses import asdict, astuple, fields, replace

asdict(chunk)                  # nested dict - handy for JSON and logging
fields(Chunk)                  # introspect names, types, defaults
replace(config, k=10)          # copy with changes (the way to "modify" a frozen object)
```

`replace()` is how you work with frozen objects: it returns a new instance with the given
fields changed, leaving the original untouched.

## Performance Considerations

| Variant | Memory per instance (approx.) | Attribute access |
| --- | --- | --- |
| plain class | baseline + `__dict__` | dict lookup |
| `@dataclass` | same as plain class | dict lookup |
| `@dataclass(slots=True)` | ~40–50% less | faster (direct slot) |
| `NamedTuple` | lowest | fastest, but immutable and tuple-like |

For 1M chunk objects, `slots=True` is the difference between comfortably fitting in memory
and swapping. Pydantic models (Phase 19) add runtime validation and cost more — use
dataclasses internally and Pydantic at the boundaries.

## Hands-on Exercise

:::exercise Model a tool call
Build the domain model for tool calling, ready for Phase 22:

1. `ToolStatus` enum: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `DENIED`, `TIMED_OUT`.
2. Frozen `ToolSpec`: `name`, `description`, `parameters: dict`, `requires_approval: bool`,
   `timeout_s: float` with validation (name must be a valid identifier, timeout positive).
3. Mutable `ToolCall`: `id`, `spec`, `arguments`, `status`, `started_at`, `finished_at`,
   `result`, `error`; a `duration_ms` property; `mark_running/succeeded/failed` methods that
   enforce legal transitions (e.g. you cannot succeed a call that was denied).
4. A `to_audit_record()` returning a dict safe to log — arguments redacted if the spec is
   marked sensitive.
:::

:::solution Solution
```python title="tool_models.py"
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class ToolStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


TERMINAL = {ToolStatus.SUCCEEDED, ToolStatus.FAILED, ToolStatus.DENIED, ToolStatus.TIMED_OUT}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, object] = field(default_factory=dict)
    requires_approval: bool = False
    sensitive: bool = False
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"tool name must be a valid identifier: {self.name!r}")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self.description.strip():
            raise ValueError("every tool needs a description - the model reads it")


@dataclass(slots=True)
class ToolCall:
    id: str
    spec: ToolSpec
    arguments: dict[str, object] = field(default_factory=dict)
    status: ToolStatus = ToolStatus.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    result: object | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return int((self.finished_at - self.started_at) * 1000)

    def _require(self, *allowed: ToolStatus) -> None:
        if self.status not in allowed:
            raise ValueError(f"illegal transition from {self.status} for call {self.id}")

    def mark_running(self) -> None:
        self._require(ToolStatus.PENDING)
        self.status = ToolStatus.RUNNING
        self.started_at = time.monotonic()

    def mark_succeeded(self, result: object) -> None:
        self._require(ToolStatus.RUNNING)
        self.status, self.result = ToolStatus.SUCCEEDED, result
        self.finished_at = time.monotonic()

    def mark_failed(self, error: str, *, timed_out: bool = False) -> None:
        self._require(ToolStatus.RUNNING)
        self.status = ToolStatus.TIMED_OUT if timed_out else ToolStatus.FAILED
        self.error, self.finished_at = error, time.monotonic()

    def deny(self, reason: str) -> None:
        self._require(ToolStatus.PENDING)
        self.status, self.error = ToolStatus.DENIED, reason
        self.finished_at = time.monotonic()

    def to_audit_record(self) -> dict[str, object]:
        return {
            "call_id": self.id,
            "tool": self.spec.name,
            "status": str(self.status),
            "duration_ms": self.duration_ms,
            "arguments": "[redacted]" if self.spec.sensitive else self.arguments,
            "error": self.error,
        }


if __name__ == "__main__":
    spec = ToolSpec(
        name="issue_refund",
        description="Refund an order. Requires human approval.",
        parameters={"order_id": "string", "amount": "number"},
        requires_approval=True,
        sensitive=True,
    )
    call = ToolCall("call_1", spec, {"order_id": "A-993", "amount": 129.99})
    call.mark_running()
    call.mark_succeeded({"refunded": True})
    print(call.to_audit_record())

    denied = ToolCall("call_2", spec, {"order_id": "B-100", "amount": 5000})
    denied.deny("above approval limit")
    print(denied.to_audit_record())
```

```text
{'call_id': 'call_1', 'tool': 'issue_refund', 'status': 'succeeded', 'duration_ms': 0, 'arguments': '[redacted]', 'error': None}
{'call_id': 'call_2', 'tool': 'issue_refund', 'status': 'denied', 'duration_ms': 0, 'arguments': '[redacted]', 'error': 'above approval limit'}
```
:::

## Challenge

:::challenge Config from environment, validated
Write `@dataclass(frozen=True) class Settings` with nested `LLMSettings` and
`RetrievalSettings`, plus a `from_env()` classmethod that reads every value from environment
variables with sensible defaults, coerces types, validates ranges in `__post_init__`, and
raises one aggregated `ConfigError` listing *all* problems at once rather than the first.
Aggregated validation errors turn a frustrating trial-and-error deployment into a single
fix.
:::

## Interview Questions

:::interview
1. What does `@dataclass` generate for you?
2. What is the difference between `frozen=True` and true immutability?
3. When do you use `field(default_factory=...)`?
4. Property versus method — how do you choose?
5. Why prefer an Enum to string constants?
:::

## Cheat Sheet

```python
@dataclass(frozen=True, slots=True, order=False, kw_only=False)
class X:
    required: str
    optional: int = 0
    items: list[str] = field(default_factory=list)
    secret: str = field(default="", repr=False)
    ignored: dict = field(default_factory=dict, compare=False)

    def __post_init__(self): ...        # validation

asdict(x) astuple(x) fields(X) replace(x, optional=5)

@property
def derived(self) -> int: ...
@derived.setter
def derived(self, value): ...

class Role(StrEnum): USER = "user"      # members are strings
class Reason(Enum):  DONE = auto()      # opaque, comparison by identity
```

```quiz
[
  {
    "question": "@dataclass class C: tags: list = [] — what happens?",
    "options": [
      "All instances share one list",
      "ValueError at class definition time",
      "It works correctly",
      "tags becomes None"
    ],
    "answer": 1,
    "explanation": "Dataclasses detect mutable defaults and refuse the class outright - a rare case of Python protecting you. Use field(default_factory=list)."
  },
  {
    "question": "You return a search result that will be cached and shared across threads. Which declaration is safest?",
    "options": [
      "@dataclass class Hit",
      "@dataclass(frozen=True, slots=True) class Hit",
      "A plain dict",
      "A class with setters"
    ],
    "answer": 1,
    "explanation": "Frozen prevents accidental mutation of a shared object, makes it hashable, and slots cuts memory for large result sets."
  },
  {
    "question": "How do you 'change' a field on a frozen dataclass?",
    "options": [
      "obj.field = value",
      "dataclasses.replace(obj, field=value)",
      "obj.__dict__['field'] = value",
      "You cannot, ever"
    ],
    "answer": 1,
    "explanation": "replace() returns a new instance with the change applied, leaving the original untouched - the functional-update pattern."
  }
]
```

## Summary

- `@dataclass` generates `__init__`, `__repr__` and `__eq__`; annotations define the fields.
- `frozen=True` for values you hand out, `slots=True` for objects you create in bulk.
- `field(default_factory=...)` for mutable defaults; `repr=False` for secrets.
- `__post_init__` validates so invalid objects cannot exist.
- Enums replace magic strings and make invalid states unrepresentable.

## Next Step

Decorators — how `@dataclass`, `@property`, `@app.get` and `@tool` actually work, and how to
write your own for retries, timing and caching.
