---
title: Testing with pytest
order: 8
difficulty: Intermediate
duration: 15
badges: ["Hands-on"]
summary: "Fixtures, parametrisation, mocking, async tests and coverage — including how to test code that calls an LLM without ever calling one in CI."
prereqs: ["Functions", "Modules, Packages and Imports", "Async Python — asyncio, Concurrency and Parallelism"]
keyConcepts: ["pytest", "fixture", "parametrize", "mock", "test double", "coverage"]
---

:::note In one line
**A test is a sentence about what your code should do, written so a machine can check it.** Write the failing one first.
:::

## Why this matters

In an AI system the model is non-deterministic, but **everything around it is not** —
chunking, retrieval selection, guardrails, schema parsing, routing, cost accounting. That
deterministic 90% is where your bugs actually live, and it is entirely testable. Phase 24
adds evaluations for the probabilistic part; this lesson covers the part where ordinary
tests apply.

## Mental Model

A test is **one sentence about what your code should do**, written so a machine can check it.

Every test has the same three beats, and keeping them visibly separate makes tests readable
years later:

<figure class="lesson-figure">
<svg viewBox="0 0 660 220" role="img" aria-label="Diagram of the arrange, act, assert pattern: first set up the inputs, then call the one thing under test, then check exactly one outcome.">
  <defs>
    <marker id="ts-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="14" y="34" width="190" height="86" rx="10" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.8"/>
  <text class="dg-label" x="32" y="58" fill="var(--accent-3)">1. Arrange</text>
  <text class="dg-sub"   x="32" y="80">build the inputs</text>
  <text class="dg-sub"   x="32" y="98">set up the fake parts</text>
  <rect x="234" y="34" width="190" height="86" rx="10" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-label" x="252" y="58" fill="var(--accent)">2. Act</text>
  <text class="dg-sub"   x="252" y="80">call the ONE thing</text>
  <text class="dg-sub"   x="252" y="98">you are testing</text>
  <rect x="454" y="34" width="192" height="86" rx="10" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.8"/>
  <text class="dg-label" x="472" y="58" fill="var(--ok)">3. Assert</text>
  <text class="dg-sub"   x="472" y="80">check one outcome</text>
  <text class="dg-sub"   x="472" y="98">say what went wrong</text>
  <path class="dg-arrow" d="M204,77 L228,77" marker-end="url(#ts-a)"/>
  <path class="dg-arrow" d="M424,77 L448,77" marker-end="url(#ts-a)"/>
  <rect x="14" y="146" width="632" height="52" rx="9" fill="var(--panel)" stroke="var(--danger)" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text class="dg-sub" x="330" y="168" text-anchor="middle" fill="var(--danger)">If a test needs several Act steps, it is testing several things.</text>
  <text class="dg-sub" x="330" y="187" text-anchor="middle">Split it, so a failure names exactly what broke.</text>
</svg>
<figcaption>
<strong>One test, one claim.</strong> When a test with one assertion fails, the name tells
you what is wrong. When a test with nine assertions fails, you start debugging the test.
</figcaption>
</figure>

```python
def test_total_sales_ignores_refunds():
    # Arrange
    rows = [{"amount": 10}, {"amount": -4, "refund": True}]

    # Act
    result = total_sales(rows)

    # Assert
    assert result == 10
```

Notice the test name. `test_total_sales_ignores_refunds` tells you what broke without reading
a single line of the body — that is the point of a long test name.

:::tip Write the failing test first
It sounds like extra work. It is actually the cheapest way to be sure your test works at all.

A test written after the code often passes for the wrong reason — and a test that can never
fail is worse than no test, because it buys false confidence.
:::

## Core Concepts

### Setup

```bash
uv add --dev pytest pytest-cov pytest-asyncio
```

```toml title="pyproject.toml"
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q --strict-markers"
asyncio_mode = "auto"                 # plain `async def test_...` works
markers = [
    "slow: hits the network or a real database",
    "eval: LLM evaluation, costs money",
]
```

### Writing tests

```python title="tests/test_chunking.py"
import pytest

from ragkit.text import chunk_text


def test_produces_overlapping_windows() -> None:
    chunks = chunk_text("abcdefghij", size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_rejects_overlap_greater_than_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", size=2, overlap=5)


def test_empty_input_returns_no_chunks() -> None:
    assert chunk_text("   ") == []
```

Test names are documentation: `test_rejects_overlap_greater_than_size` tells you what broke
from the failure line alone.

### Parametrisation

```python
@pytest.mark.parametrize(
    ("similarity", "cited", "expected"),
    [
        (0.95, True, "answer"),
        (0.82, True, "answer"),              # exactly on the boundary
        (0.81, True, "answer_with_caveat"),
        (0.60, True, "answer_with_caveat"),
        (0.59, True, "escalate"),
        (0.99, False, "escalate"),           # no citation always escalates
    ],
)
def test_routing_boundaries(similarity: float, cited: bool, expected: str) -> None:
    assert route(similarity, cited) == expected
```

One function, six tests, each reported separately. Boundary values are where routing bugs
live, so name them explicitly.

### Fixtures

```python title="tests/conftest.py"
import pytest

from ragkit.stores import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    """A fresh store per test - no shared state between tests, ever."""
    return InMemoryStore()


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {"id": "c1", "text": "Vector search finds similar embeddings.", "score": 0.91},
        {"id": "c2", "text": "Rerankers reorder candidates.", "score": 0.64},
    ]


@pytest.fixture
def populated_store(store, sample_chunks):        # fixtures compose
    store.upsert(sample_chunks)
    return store


@pytest.fixture(scope="session")
def embedder():
    """Expensive to build, safe to share: created once for the whole run."""
    return HashEmbedder(dimension=16)


@pytest.fixture
def temp_corpus(tmp_path):                        # tmp_path is built in
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": "d1", "text": "hello"}\n', encoding="utf-8")
    return path
```

Built-in fixtures worth knowing: `tmp_path`, `monkeypatch`, `capsys`, `caplog`, `recwarn`.

### Fakes over mocks

```python title="tests/fakes.py"
class FakeLLM:
    """A fake, not a mock: it behaves like the real thing, deterministically."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or ["default answer"])
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **kwargs) -> str:
        self.calls.append(messages)
        return self.responses.pop(0) if self.responses else "default answer"


def test_answer_includes_citations(populated_store):
    llm = FakeLLM(["HNSW is a graph index [c1]."])
    service = RagService(store=populated_store, llm=llm)

    answer = service.answer("What is HNSW?")

    assert "[c1]" in answer.text
    assert answer.citations == ("c1",)
    assert len(llm.calls) == 1
    assert "Vector search" in llm.calls[0][-1]["content"]     # context really was passed
```

Fakes make tests readable and stable. Reserve `unittest.mock` for asserting *interactions*
with something you cannot reimplement.

### monkeypatch and mock

```python
def test_reads_config_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RETRIEVAL_K", "8")
    settings = Settings.from_env()
    assert settings.retrieval_k == 8


def test_retries_on_server_error(monkeypatch):
    from unittest.mock import MagicMock

    calls = MagicMock(side_effect=[TransientError("503"), TransientError("503"), "ok"])
    monkeypatch.setattr("service.resilience.time.sleep", lambda s: None)   # no real waiting
    assert retry(calls, attempts=3) == "ok"
    assert calls.call_count == 3
```

### Async tests

```python
async def test_embeds_concurrently():                  # asyncio_mode = "auto"
    embedder = AsyncEmbedder(FakeTransport(), concurrency=4)
    vectors = await embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3


async def test_times_out_slow_tools():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await asyncio.sleep(1)
```

### Testing that something raises, warns or logs

```python
def test_raises_with_a_useful_message():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_logs_a_warning_on_retry(caplog):
    with caplog.at_level("WARNING"):
        retry(flaky, attempts=2)
    assert "retrying" in caplog.text


def test_floats_compare_approximately():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
```

### Marks, skipping and selection

```python
@pytest.mark.slow
def test_against_real_qdrant(): ...

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs an API key")
def test_live_embedding(): ...
```

```bash
uv run pytest                      # everything
uv run pytest -m "not slow"        # CI default
uv run pytest tests/test_rag.py::test_answer_includes_citations
uv run pytest -k "citation"        # by name substring
uv run pytest -x --lf              # stop at first failure; rerun last failures
uv run pytest --cov=src --cov-report=term-missing
```

## Real-World Example

A complete test module for a RAG service, with no network access anywhere.

```python title="tests/test_rag_service.py"
"""Tests for the RAG service.

Nothing here touches a network, a database or a model. Every collaborator is a
fake, so the suite runs in milliseconds and never flakes.
"""
from __future__ import annotations

import pytest

from service.errors import GuardrailBlocked
from service.models import Decision
from service.rag import RagService


# --- fakes ------------------------------------------------------------------
class FakeEmbedder:
    dimension = 4

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            vector[hash(token) % self.dimension] += 1.0
        return vector

    def embed(self, texts):
        return [self.embed_query(t) for t in texts]


class FakeIndex:
    def __init__(self, hits: list[dict]):
        self._hits = hits
        self.queries: list[dict] = []

    def search(self, vector, *, k=5, where=None):
        self.queries.append({"k": k, "where": where})
        return self._hits[:k]

    def count(self) -> int:
        return len(self._hits)


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, messages, **kwargs) -> str:
        self.prompts.append(messages[-1]["content"])
        return self.responses.pop(0)


# --- fixtures ---------------------------------------------------------------
@pytest.fixture
def hits() -> list[dict]:
    return [
        {"id": "c1", "score": 0.92, "text": "HNSW is a graph-based index.", "document": "handbook"},
        {"id": "c2", "score": 0.71, "text": "IVF partitions the space.", "document": "handbook"},
        {"id": "c3", "score": 0.22, "text": "Unrelated content.", "document": "blog"},
    ]


@pytest.fixture
def service(hits):
    return RagService(
        embedder=FakeEmbedder(),
        index=FakeIndex(hits),
        llm=FakeLLM(["HNSW is a graph-based ANN index [c1]."]),
        k=3,
        min_score=0.3,
    )


# --- tests ------------------------------------------------------------------
def test_answers_with_citations(service):
    answer = service.answer("What is HNSW?")

    assert answer.decision is Decision.ANSWER
    assert answer.citations == ("c1",)
    assert "graph-based" in answer.text


def test_low_scoring_chunks_are_filtered(service):
    service.answer("What is HNSW?")
    prompt = service.llm.prompts[0]

    assert "HNSW is a graph-based index." in prompt
    assert "Unrelated content." not in prompt          # score 0.22 < min_score 0.3


def test_escalates_when_nothing_is_retrieved():
    service = RagService(
        embedder=FakeEmbedder(), index=FakeIndex([]), llm=FakeLLM(["should not be called"])
    )
    answer = service.answer("anything")

    assert answer.decision is Decision.ESCALATE
    assert answer.citations == ()
    assert service.llm.prompts == []                   # no model call: no wasted money


def test_rejects_hallucinated_citations(hits):
    service = RagService(
        embedder=FakeEmbedder(),
        index=FakeIndex(hits),
        llm=FakeLLM(["According to [c9], yes."]),      # c9 was never retrieved
    )
    with pytest.raises(GuardrailBlocked, match="c9"):
        service.answer("What is HNSW?")


@pytest.mark.parametrize(
    ("question", "expected_filter"),
    [
        ("What is HNSW?", None),
        ("What is HNSW in the handbook?", {"document": "handbook"}),
    ],
)
def test_metadata_filter_is_applied(hits, question, expected_filter):
    index = FakeIndex(hits)
    service = RagService(embedder=FakeEmbedder(), index=index, llm=FakeLLM(["ok [c1]"]))
    service.answer(question)
    assert index.queries[0]["where"] == expected_filter


async def test_async_answer_matches_sync(service):
    answer = await service.aanswer("What is HNSW?")
    assert answer.citations == ("c1",)
```

```bash
uv run pytest tests/test_rag_service.py -v
```

```text
tests/test_rag_service.py::test_answers_with_citations PASSED
tests/test_rag_service.py::test_low_scoring_chunks_are_filtered PASSED
tests/test_rag_service.py::test_escalates_when_nothing_is_retrieved PASSED
tests/test_rag_service.py::test_rejects_hallucinated_citations PASSED
tests/test_rag_service.py::test_metadata_filter_is_applied[What is HNSW?-None] PASSED
tests/test_rag_service.py::test_metadata_filter_is_applied[What is HNSW in the handbook?-expected_filter1] PASSED
tests/test_rag_service.py::test_async_answer_matches_sync PASSED

7 passed in 0.09s
```

Note `test_escalates_when_nothing_is_retrieved`: it asserts the LLM was **not** called. Not
spending money on a request that cannot succeed is a real production requirement, and it is
testable.

## Common Mistakes

:::mistake
```python
# 1. Tests that call real APIs
def test_answer():
    answer = service.answer("...")     # slow, costly, flaky, fails offline

# 2. Shared state between tests
STORE = InMemoryStore()                # module-level: test order now matters
@pytest.fixture
def store(): return InMemoryStore()    # fresh per test

# 3. Asserting on implementation instead of behaviour
assert service._internal_cache_size == 3    # breaks on every refactor

# 4. One test asserting ten things
def test_everything(): ...             # a failure tells you nothing specific

# 5. Testing the framework
def test_pydantic_validates_ints(): ...     # not your code

# 6. No boundary cases
# 0.82 works but 0.8199 does not? Only a boundary test finds that.

# 7. Sleeping in tests
time.sleep(2)                          # monkeypatch the clock or the sleep instead
```
:::

## Best Practices

1. Arrange–Act–Assert, in that order, with a blank line between each.
2. One behaviour per test; the name states the behaviour.
3. Fresh fixtures per test; no module-level mutable state.
4. Fakes over mocks; mock only what you cannot reimplement.
5. Test boundaries, empties and errors — not just the happy path.
6. Mark anything that touches the network `@pytest.mark.slow` and exclude it from the default
   run.
7. Aim for high coverage of *logic*, not of lines: a 100% covered module with no assertions
   on behaviour is worthless.

## Debugging

```bash
uv run pytest -x -vv --tb=short       # first failure, verbose, compact traceback
uv run pytest --pdb                   # drop into the debugger on failure
uv run pytest --lf                    # only the tests that failed last time
uv run pytest -p no:randomly          # if a plugin randomises order and you suspect coupling
```

`assert` failures in pytest are introspected, so `assert answer.citations == ("c1",)` shows
both sides of the comparison without you writing a message.

## Hands-on Exercise

:::exercise Test the token budget planner
Take `plan_context` from Phase 1 and write a full test module:

1. A fixture producing a standard history and chunk set.
2. Parametrised tests over four window sizes verifying that the total never exceeds the
   budget.
3. A test that the system prompt is always included.
4. A test that history is included newest-first.
5. A test that `ValueError` is raised when the system prompt alone exceeds the budget, with
   a message naming both numbers.
6. A property-style test: for 100 random inputs, `tokens_used + tokens_available` always
   equals the budget.

Target: the suite runs in under 0.1 s and fails clearly when you introduce an off-by-one.
:::

:::solution Solution
```python title="tests/test_plan_context.py"
from __future__ import annotations

import random

import pytest

from ragkit.context import plan_context


@pytest.fixture
def history() -> list[tuple[str, int]]:
    return [("t1", 600), ("t2", 900), ("t3", 400)]


@pytest.fixture
def chunks() -> list[tuple[str, int]]:
    return [("c1", 1200), ("c2", 1500), ("c3", 900)]


@pytest.mark.parametrize("window", [3_000, 6_000, 10_000, 20_000])
def test_never_exceeds_budget(history, chunks, window):
    plan = plan_context(800, history, chunks, window=window, output_reserve=1_000)
    assert plan["tokens_used"] <= window - 1_000


def test_system_prompt_is_always_included(history, chunks):
    plan = plan_context(800, history, chunks, window=2_000, output_reserve=1_000)
    assert plan["tokens_used"] >= 800


def test_history_is_kept_newest_first(history, chunks):
    plan = plan_context(100, history, chunks, window=1_700, output_reserve=1_000)
    # only 600 tokens of history fit: the newest turn (t3, 400) must win over t1
    assert "t3" in plan["included_history"]
    assert "t1" not in plan["included_history"]


def test_history_order_is_chronological(history, chunks):
    plan = plan_context(100, history, chunks, window=20_000, output_reserve=1_000)
    assert plan["included_history"] == ["t1", "t2", "t3"]


def test_rejects_impossible_system_prompt(history, chunks):
    with pytest.raises(ValueError, match=r"5000.*4000|4000.*5000"):
        plan_context(5_000, history, chunks, window=5_000, output_reserve=1_000)


def test_dropped_items_are_reported(history, chunks):
    plan = plan_context(800, history, chunks, window=6_000, output_reserve=1_000)
    included = set(plan["included_chunks"]) | set(plan["included_history"])
    dropped = set(plan["dropped"])
    assert not included & dropped                       # disjoint
    assert included | dropped == {"t1", "t2", "t3", "c1", "c2", "c3"}


@pytest.mark.parametrize("seed", range(20))
def test_invariant_used_plus_available_equals_budget(seed):
    rng = random.Random(seed)
    window = rng.randint(4_000, 30_000)
    reserve = rng.randint(200, 2_000)
    system = rng.randint(50, 1_000)
    history = [(f"t{i}", rng.randint(50, 900)) for i in range(rng.randint(0, 8))]
    chunks = [(f"c{i}", rng.randint(100, 2_000)) for i in range(rng.randint(0, 8))]

    plan = plan_context(system, history, chunks, window=window, output_reserve=reserve)
    assert plan["tokens_used"] + plan["tokens_available"] == window - reserve
```

```text
28 passed in 0.05s
```

The last test is the most valuable: an invariant checked against random inputs finds
off-by-one errors that hand-picked examples miss. `hypothesis` automates this style
properly if you want to go further.
:::

## Challenge

:::challenge Add a CI gate
Write a GitHub Actions workflow that runs `ruff check`, `mypy --strict src` and
`pytest -m "not slow" --cov=src --cov-fail-under=85` on every pull request, caching the uv
environment. Then deliberately break each of the three and confirm the build fails with a
clear message. A green pipeline you trust is what lets you refactor an AI system without
fear.
:::

## Interview Questions

:::interview
1. What is the difference between a stub, a fake and a mock?
2. How do you test code that calls an LLM?
3. What is a fixture, and what does its scope control?
4. Why parametrise instead of writing a loop inside a test?
5. What does high coverage guarantee, and what does it not?
:::

## Cheat Sheet

```python
def test_name(): assert actual == expected
with pytest.raises(ValueError, match="regex"): ...
assert value == pytest.approx(0.1 + 0.2)

@pytest.fixture(scope="function"|"module"|"session")
def thing(): yield resource     # code after yield = teardown

@pytest.mark.parametrize(("a", "b"), [(1, 2), (3, 4)])
@pytest.mark.slow / @pytest.mark.skipif(cond, reason="...")

monkeypatch.setenv / setattr / chdir
caplog.at_level("WARNING") ; caplog.text
capsys.readouterr().out
tmp_path  # pathlib.Path to a fresh temp dir

uv run pytest -q -x --lf -k "name" -m "not slow" --cov=src --cov-report=term-missing
```

```quiz
[
  {
    "question": "Why should unit tests never call a real LLM?",
    "options": [
      "It is against the terms of service",
      "They become slow, expensive and flaky because the output is non-deterministic",
      "LLMs cannot be called from tests",
      "The API key would leak"
    ],
    "answer": 1,
    "explanation": "Inject a fake for unit tests; put real-model checks in a separate, scheduled evaluation suite (Phase 24)."
  },
  {
    "question": "A test passes alone but fails when the suite runs. What is the most likely cause?",
    "options": [
      "A pytest bug",
      "Shared mutable state between tests (module-level objects or a session fixture)",
      "Insufficient coverage",
      "Missing type hints"
    ],
    "answer": 1,
    "explanation": "Order-dependent failures almost always mean state leaking between tests. Fresh function-scoped fixtures fix it."
  },
  {
    "question": "Which assertion tests behaviour rather than implementation?",
    "options": [
      "assert service._cache_size == 3",
      "assert answer.citations == ('c1',)",
      "assert service._retriever.__class__.__name__ == 'HybridRetriever'",
      "assert len(service.__dict__) == 5"
    ],
    "answer": 1,
    "explanation": "Behavioural assertions survive refactoring; assertions about internals turn every refactor into a test rewrite."
  }
]
```

## Summary

- Test the deterministic majority of an AI system with ordinary unit tests; evaluate the
  model separately.
- Fixtures give fresh state per test; parametrisation covers boundaries cheaply.
- Prefer fakes to mocks; assert behaviour, not internals.
- Keep network tests marked and excluded from the default run, and gate merges on the fast
  suite.

## Next Step

Packaging, logging, configuration and CLIs — turning a folder of modules into a tool other
people can install and operate.
