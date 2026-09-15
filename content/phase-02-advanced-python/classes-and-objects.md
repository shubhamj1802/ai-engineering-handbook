---
title: Classes and Objects
order: 1
difficulty: Intermediate
duration: 15
badges: ["Hands-on"]
summary: "Objects that bundle state with behaviour — __init__, methods, attributes, dunder methods — and the honest answer to when a class beats a function."
prereqs: ["Functions", "Modules, Packages and Imports"]
keyConcepts: ["class", "__init__", "self", "instance state", "dunder methods"]
---

:::note In one line
**A class is a template; an object is one thing built from it.** Use a class when data and the behaviour that belongs to it should travel together.
:::

## Why this matters

Every framework you will use — LangChain runnables, LangGraph graphs, Pydantic models,
PyTorch modules — is built from classes. You need to read them fluently. You also need
judgement about when to write one: a class that holds no state is just a namespace with
extra ceremony, and beginner codebases are full of them.

## Mental Model

```text
class  = the blueprint        class Retriever: ...
object = one built thing      retriever = Retriever(index, k=5)
self   = "this particular object", passed automatically as the first argument

state    → attributes   (retriever.k, retriever.index)
behaviour → methods     (retriever.search(query))
```

A class earns its place when **state and behaviour belong together and the state outlives a
single call** — a client holding a connection, a cache holding entries, a graph holding
nodes. If you would pass the same three arguments to five functions, those arguments want
to be a class.

## Core Concepts

### Defining a class

```python
class Retriever:
    """Finds relevant chunks for a query."""

    DEFAULT_K = 5                       # class attribute: shared by all instances

    def __init__(self, index: dict[str, str], *, k: int = DEFAULT_K) -> None:
        self.index = index              # instance attributes: one per object
        self.k = k
        self._calls = 0                 # leading underscore: internal, not public API

    def search(self, query: str) -> list[str]:
        """Return up to k chunk ids matching the query."""
        self._calls += 1
        terms = set(query.lower().split())
        hits = [
            chunk_id
            for chunk_id, text in self.index.items()
            if terms & set(text.lower().split())
        ]
        return hits[: self.k]

    @property
    def call_count(self) -> int:
        """Read-only view of internal state."""
        return self._calls


retriever = Retriever({"c1": "vector search", "c2": "graph index"}, k=2)
retriever.search("vector")          # ['c1']
retriever.call_count                # 1
```

`__init__` is not a constructor in the C++/Java sense; it is an *initialiser* that receives
the already-created object as `self`.

### Instance vs class attributes

```python
class Counter:
    total = 0                 # shared by every instance

    def __init__(self):
        self.count = 0        # per instance

a, b = Counter(), Counter()
a.count += 1                  # only a
Counter.total += 1            # everyone sees it
```

:::danger Never use a mutable class attribute as per-instance state
```python
class Session:
    messages = []             # ONE list shared by every Session ever created

    def add(self, m): self.messages.append(m)

s1, s2 = Session(), Session()
s1.add("hi")
s2.messages                   # ['hi']  ← s2 sees s1's message
```
Initialise mutable state in `__init__`: `self.messages = []`.
:::

### Methods, class methods, static methods

```python
class Chunk:
    def __init__(self, chunk_id: str, text: str):
        self.id = chunk_id
        self.text = text

    def preview(self, n: int = 40) -> str:          # instance method: needs self
        return self.text[:n]

    @classmethod
    def from_dict(cls, raw: dict) -> "Chunk":       # alternative constructor
        return cls(raw["id"], raw["text"])

    @staticmethod
    def estimate_tokens(text: str) -> int:          # related, but needs no state
        return len(text) // 4
```

`@classmethod` receives the class, so subclasses build the right type — this is why
`from_dict` is the idiomatic alternative constructor.

### Dunder methods

Special methods let your objects work with Python's built-in syntax.

```python
class Vector:
    def __init__(self, values: list[float]):
        self.values = values

    def __repr__(self) -> str:              # for developers - shows in tracebacks/debugger
        return f"Vector(dim={len(self.values)})"

    def __len__(self) -> int:               # len(v)
        return len(self.values)

    def __getitem__(self, i: int) -> float: # v[0], and makes it iterable
        return self.values[i]

    def __eq__(self, other: object) -> bool:        # v == w
        return isinstance(other, Vector) and self.values == other.values

    def __hash__(self) -> int:                       # needed if __eq__ is defined and you
        return hash(tuple(self.values))              # want the object in a set/dict

    def __add__(self, other: "Vector") -> "Vector":  # v + w
        return Vector([a + b for a, b in zip(self.values, other.values, strict=True)])

    def __bool__(self) -> bool:                      # if v:
        return any(self.values)
```

Always define `__repr__`. A log line reading `<Chunk object at 0x7f3c>` has wasted more
engineering hours than almost anything else.

## Minimal Example

```python title="conversation.py"
class Conversation:
    """Holds message history with a bounded window."""

    def __init__(self, system_prompt: str, *, max_turns: int = 10) -> None:
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {role!r}")
        self._messages.append({"role": role, "content": content})
        excess = len(self._messages) - self.max_turns
        if excess > 0:
            del self._messages[:excess]

    def to_payload(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, *self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"Conversation(turns={len(self._messages)}, max={self.max_turns})"


chat = Conversation("You are concise.", max_turns=3)
for i in range(5):
    chat.add("user", f"question {i}")

print(chat)
print([m["content"] for m in chat.to_payload()[1:]])
```

```text
Conversation(turns=3, max=3)
['question 2', 'question 3', 'question 4']
```

## Real-World Example

An LLM client wrapper — the shape of every adapter you will write in this handbook.

```python title="src/service/adapters/llm.py"
"""LLM provider adapter.

State that justifies a class: the HTTP session, the configuration, the running
token/cost tally and the rate limiter. All of it outlives a single call, and all
of it is needed by every method.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """The provider failed in a way the caller should know about."""


@dataclass(slots=True)
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "errors": self.errors,
        }


class LLMClient:
    """A thin, testable wrapper around a chat completion API."""

    PRICES: dict[str, tuple[float, float]] = {       # USD per 1M tokens
        "small": (0.25, 1.25),
        "medium": (3.00, 15.00),
    }

    def __init__(
        self,
        transport,                                    # any object with .post(payload) -> dict
        *,
        model: str = "small",
        max_requests: int = 100,
        temperature: float = 0.2,
    ) -> None:
        if model not in self.PRICES:
            raise ValueError(f"unknown model tier: {model!r}. Known: {sorted(self.PRICES)}")
        self._transport = transport
        self.model = model
        self.temperature = temperature
        self.max_requests = max_requests
        self.usage = UsageTotals()
        self._started = time.monotonic()

    # --- public API ------------------------------------------------------
    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        """Send messages, return the assistant text.

        Raises:
            ModelError: on provider failure or when the request budget is spent.
        """
        if self.usage.requests >= self.max_requests:
            raise ModelError(f"request budget exhausted ({self.max_requests} calls)")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self._transport.post(payload)
        except Exception as exc:
            self.usage.errors += 1
            raise ModelError(f"provider call failed: {exc}") from exc

        self._record(response)
        return self._extract_text(response)

    def stats(self) -> dict[str, float | int | str]:
        elapsed = time.monotonic() - self._started
        return {
            **self.usage.as_dict(),
            "model": self.model,
            "elapsed_s": round(elapsed, 2),
            "requests_per_min": round(self.usage.requests / max(elapsed / 60, 1e-9), 1),
        }

    # --- internals -------------------------------------------------------
    def _record(self, response: dict) -> None:
        usage = response.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        price_in, price_out = self.PRICES[self.model]

        self.usage.requests += 1
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.cost_usd += input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out

    @staticmethod
    def _extract_text(response: dict) -> str:
        try:
            return response["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"unexpected response shape: {str(response)[:200]}") from exc

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, requests={self.usage.requests})"


class FakeTransport:
    """Test double - because `transport` is injected, no network is needed."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.calls: list[dict] = []

    def post(self, payload: dict) -> dict:
        self.calls.append(payload)
        return {
            "content": [{"type": "text", "text": self.reply}],
            "usage": {"input_tokens": 120, "output_tokens": 30},
        }


if __name__ == "__main__":
    client = LLMClient(FakeTransport("Hello!"), model="medium")
    print(client.complete([{"role": "user", "content": "hi"}]))
    print(client)
    print(client.stats())
```

```text
Hello!
LLMClient(model='medium', requests=1)
{'requests': 1, 'input_tokens': 120, 'output_tokens': 30, 'cost_usd': 0.00081, 'errors': 0, 'model': 'medium', 'elapsed_s': 0.0, 'requests_per_min': 1.0}
```

The key design choice is **dependency injection**: `transport` is passed in, not created
inside. That is what makes this class testable without a network, and it is the pattern
every adapter in this handbook follows.

## Common Mistakes

:::mistake
```python
# 1. Mutable class attribute as instance state (see above)

# 2. A class with no state
class MathUtils:
    @staticmethod
    def add(a, b): return a + b        # this is a module with extra steps

# 3. Forgetting self
class A:
    def method(): ...                  # TypeError when called on an instance

# 4. Doing I/O in __init__
def __init__(self):
    self.data = requests.get(url)      # now you cannot construct one in a test

# 5. No __repr__
print(obj)                             # <service.LLMClient object at 0x7f9c1a>

# 6. Defining __eq__ without __hash__
# the object becomes unhashable and silently unusable in sets and dict keys
```
:::

## Best Practices

1. If the class has no state, make it a module of functions.
2. Initialise all mutable state in `__init__`.
3. Inject dependencies; never construct clients or open connections inside `__init__`.
4. Prefix internals with `_`; expose read-only state via `@property`.
5. Always write `__repr__`.
6. Validate arguments in `__init__` and raise immediately — an object should never exist in
   an invalid state.

## Performance Considerations

- `__slots__` removes the per-instance `__dict__`, cutting memory substantially for
  millions of small objects (chunks, vectors, events):

```python
class Chunk:
    __slots__ = ("id", "text", "score")
```

- Attribute access is a dict lookup; in a hot loop, bind to a local first
  (`search = self.search`) — but only after profiling proves it matters.
- Creating objects is cheap; creating them inside a tight loop that runs a million times is
  not. Dataclasses with `slots=True` (next lesson) give you both brevity and speed.

## Hands-on Exercise

:::exercise A token bucket rate limiter
Implement `RateLimiter(rate_per_minute, burst)` with:

- `allow(tokens=1) -> bool` — consume capacity if available, refilling continuously based
  on elapsed time,
- `wait_time(tokens=1) -> float` — seconds until that many tokens would be available,
- `__repr__` showing current capacity,
- state that is per-instance (two limiters must not interfere).

Test it by creating a limiter with 60/minute and burst 5, calling `allow()` seven times in a
loop, and printing which calls succeeded.
:::

:::solution Solution
```python title="rate_limiter.py"
from __future__ import annotations

import time


class RateLimiter:
    """Token bucket: `burst` tokens capacity, refilled at rate_per_minute."""

    def __init__(self, rate_per_minute: float, burst: int | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate_per_second = rate_per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_minute)))
        self._tokens = self.capacity
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)

    def allow(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: float = 1.0) -> float:
        self._refill()
        if self._tokens >= tokens:
            return 0.0
        return (tokens - self._tokens) / self.rate_per_second

    def __repr__(self) -> str:
        return f"RateLimiter(tokens={self._tokens:.2f}/{self.capacity:.0f})"


if __name__ == "__main__":
    limiter = RateLimiter(rate_per_minute=60, burst=5)
    for i in range(7):
        ok = limiter.allow()
        print(f"call {i}: {'allowed' if ok else f'denied (wait {limiter.wait_time():.2f}s)'}")
    print(limiter)

    other = RateLimiter(rate_per_minute=60, burst=5)
    print("independent instance:", other.allow())
```

```text
call 0: allowed
call 1: allowed
call 2: allowed
call 3: allowed
call 4: allowed
call 5: denied (wait 1.00s)
call 6: denied (wait 1.00s)
RateLimiter(tokens=0.00/5)
independent instance: True
```

You will reuse this class almost verbatim in Phase 19 to cap per-user agent spend.
:::

## Challenge

:::challenge Make the client observable
Extend `LLMClient` so that every call records a structured event
(`{request_id, model, tokens, latency_ms, error}`) into an injected `recorder` object, and
so a `with client.batch():` context temporarily raises `max_requests`. You will need
`__enter__`/`__exit__` — covered two lessons from now — so attempt it, then come back and
compare with the context-manager lesson.
:::

## Interview Questions

:::interview
1. What is `self`, and why is it explicit in Python?
2. Difference between a class attribute and an instance attribute?
3. When would you use `@classmethod` versus `@staticmethod`?
4. Why must you define `__hash__` when you define `__eq__`?
5. What is dependency injection and why does it make a class testable?
:::

## Cheat Sheet

```python
class Thing:
    CONSTANT = 1                       # class attribute
    __slots__ = ("a", "b")             # optional: memory + speed

    def __init__(self, a): self.a = a  # instance state
    def method(self): ...              # takes self
    @classmethod
    def create(cls, raw): return cls(...)
    @staticmethod
    def helper(x): ...
    @property
    def derived(self): return ...      # computed, read-only

    def __repr__(self): ...            # ALWAYS
    __len__ __getitem__ __iter__ __contains__
    __eq__ __hash__ __bool__ __call__
    __enter__ __exit__                 # context manager
```

```quiz
[
  {
    "question": "class Session: messages = []  — what goes wrong?",
    "options": [
      "Nothing",
      "Every Session instance shares the same list",
      "It raises a TypeError",
      "messages becomes read-only"
    ],
    "answer": 1,
    "explanation": "Mutable class attributes are shared across all instances. Per-instance state must be created inside __init__."
  },
  {
    "question": "When is a class the wrong choice?",
    "options": [
      "When it manages a connection",
      "When it has no state and only static methods",
      "When it needs a __repr__",
      "When it is subclassed"
    ],
    "answer": 1,
    "explanation": "A stateless class is a module with extra ceremony. Plain functions are easier to import, test and compose."
  },
  {
    "question": "Why inject the transport rather than create it in __init__?",
    "options": [
      "It is faster",
      "It lets tests pass a fake, so the class is testable without a network",
      "Python requires it",
      "It avoids circular imports"
    ],
    "answer": 1,
    "explanation": "Dependency injection is the single most important habit for testable code, and it is what lets you swap providers later without touching business logic."
  }
]
```

## Summary

- A class bundles state with the behaviour that operates on it; without state, use functions.
- Mutable state belongs in `__init__`, never as a class attribute.
- Dunder methods integrate your object with Python syntax; `__repr__` is mandatory.
- Inject dependencies so objects can be constructed in tests without side effects.

## Next Step

Inheritance, composition and polymorphism — including why composition is usually the right
answer.
