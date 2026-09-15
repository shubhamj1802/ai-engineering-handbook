---
title: "LangChain in Production — Caching, Cost, Tracing and Tests"
order: 5
difficulty: Production
duration: 20
badges: ["Production", "Hands-on"]
summary: The wrapper you put around every chain before it faces real traffic — caching, token budgets, rate limits, tracing with LangSmith or OpenTelemetry, and how to test code whose output changes every run.
prereqs: ["LangChain — Retrieval, Agents and Middleware", "Tracing, Metrics and Cost Observability"]
keyConcepts: ["caching", "token budget", "rate limiting", "callbacks", "tracing", "testing LLM code"]
---

:::note In one line
**A chain that works on your laptop is about a third of a production feature.** The rest is
caching, budgets, tracing and tests — and none of it is optional once real users arrive.
:::

:::warning Versions used on this page
`langchain` **1.4.0** · `langchain-core` **1.6.3**
:::

## Why this matters

The first time a LangChain feature meets real traffic, these are the things that break, in
roughly this order:

1. **Cost** — nobody measured per-request spend, and the bill arrives at month end
2. **Rate limits** — a burst of traffic turns into a wall of 429s
3. **Latency** — the p95 is four times the p50 and nobody knows which step is to blame
4. **Silent quality drift** — a prompt tweak helped one case and broke five others

Each has a standard answer. This lesson is those four answers.

## Caching: the cheapest win available

Identical requests are more common than you would guess — the same question from different
users, retries, a user reloading a page.

```python title="src/lc/caching.py"
"""One line at startup, applied to every model call in the process."""
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".cache/langchain.db"))
```

```python
# first call: 1,240ms, 380 tokens, costs money
# second identical call: 3ms, 0 tokens, free
```

The cache key is the **exact** prompt plus the model settings. Change one character and it
is a miss.

| Cache | Use when |
| --- | --- |
| `InMemoryCache` | tests, single process, fine to lose |
| `SQLiteCache` | one machine, survives restarts |
| Redis cache | several machines sharing one cache |

:::danger Never cache a personalised or permission-scoped answer
```python
# DANGEROUS if the prompt embeds one user's data
prompt = f"Given this account: {account_details}\n\nAnswer: {question}"
```
If two users ask the same question and the prompt includes account data, the cache key
differs so you are safe. But if you cache a *response* keyed only on the question while the
context differed, **user A can be served user B's answer.**

Rule: cache only calls whose full input is in the key, and never cache across tenants
without the tenant id in that key.
:::

## Token budgets and a hard cost ceiling

Track spend per request, and refuse to continue past a limit.

```python title="src/lc/budget.py"
"""A callback that adds up tokens and stops the run if it gets expensive."""
from dataclasses import dataclass, field
from langchain_core.callbacks import BaseCallbackHandler

# Prices per million tokens. Keep these in config, not in code.
PRICES = {
    "claude-opus-5":   {"in": 15.00, "out": 75.00},
    "claude-sonnet-5": {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
}

class BudgetExceeded(RuntimeError):
    pass

@dataclass
class CostTracker(BaseCallbackHandler):
    limit_usd: float = 0.50
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(generation.message, "usage_metadata", None) or {}
                model = generation.message.response_metadata.get("model", "claude-sonnet-5")
                price = PRICES.get(model, PRICES["claude-sonnet-5"])
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.cost_usd += (
                    usage.get("input_tokens", 0) / 1_000_000 * price["in"]
                    + usage.get("output_tokens", 0) / 1_000_000 * price["out"]
                )
        if self.cost_usd > self.limit_usd:
            # Raising here stops the run rather than letting it spiral.
            raise BudgetExceeded(
                f"spent ${self.cost_usd:.4f} over {self.calls} calls, limit ${self.limit_usd}"
            )
```

```python title="using it"
tracker = CostTracker(limit_usd=0.25)

try:
    result = chain.invoke(payload, config={"callbacks": [tracker]})
except BudgetExceeded as error:
    logger.error("budget stop: %s", error)
    result = fallback_answer()

logger.info(
    "request done tokens_in=%d tokens_out=%d cost=$%.4f calls=%d",
    tracker.input_tokens, tracker.output_tokens, tracker.cost_usd, tracker.calls,
)
```

:::production One log line per request, with the cost in it
That single line is what lets you answer "what does this feature cost per user?" without a
migration or a spreadsheet. Add it before you need it — retrofitting means you have no
history to compare against.
:::

## Rate limiting

LangChain has a built-in limiter, which is better than discovering the provider's one.

```python title="src/lc/rate_limit.py"
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model

limiter = InMemoryRateLimiter(
    requests_per_second=8,       # stay under your real limit, not at it
    check_every_n_seconds=0.1,
    max_bucket_size=16,          # allow a small burst
)

model = init_chat_model("anthropic:claude-sonnet-5", rate_limiter=limiter)
```

:::warning InMemoryRateLimiter is per process
Run four workers and you have four limiters, so your real rate is 4×8. For multiple
processes you need a shared limiter (Redis) or a per-worker budget that multiplies out to
below your provider limit. Getting this wrong is the usual cause of "we set a rate limit and
still got throttled".
:::

## Tracing: seeing inside a chain

Without a trace, a slow chain is a mystery. With one, it is obvious.

```bash title=".env - never commit"
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=support-assistant
```

That is all — every chain in the process is traced. To make traces useful, name your runs
and attach identifiers:

```python
result = chain.invoke(
    payload,
    config={
        "run_name": "answer_support_question",
        "tags": ["support", "rag", "v4"],
        "metadata": {"request_id": rid, "tenant": tenant, "user_tier": tier},
    },
)
```

If you would rather not send data to a third party, LangChain emits OpenTelemetry spans, so
traces can go to whatever you already run:

```bash
LANGSMITH_OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

<figure class="lesson-figure">
<svg viewBox="0 0 660 230" role="img" aria-label="Diagram: a production chain wrapped in four layers - a cache in front, a rate limiter, a cost tracker callback, and tracing around everything - with the bare chain in the middle.">
  <rect x="14" y="18" width="632" height="150" rx="12" fill="none" stroke="var(--accent-2)" stroke-width="1.7" stroke-dasharray="6 4"/>
  <text class="dg-label" x="30" y="40" fill="var(--accent-2)">tracing — wraps everything, adds request id</text>
  <rect x="40" y="52" width="580" height="104" rx="10" fill="none" stroke="var(--accent-3)" stroke-width="1.6"/>
  <text class="dg-label" x="56" y="74" fill="var(--accent-3)">cache — identical input returns instantly, free</text>
  <rect x="70" y="86" width="520" height="60" rx="9" fill="none" stroke="var(--warn)" stroke-width="1.6"/>
  <text class="dg-label" x="86" y="106" fill="var(--warn)">rate limiter + cost tracker — refuse before it spirals</text>
  <rect x="210" y="112" width="240" height="26" rx="6" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-mono" x="330" y="130" text-anchor="middle" fill="var(--accent)" style="font-size:11px">prompt | model | parser</text>
  <text class="dg-sub" x="14" y="192">The chain in the middle is the part you wrote in the last lesson. The four rings are this lesson.</text>
  <text class="dg-sub" x="14" y="214" fill="var(--danger)">Ship without the rings and you find out about each one from an incident instead of a dashboard.</text>
</svg>
<figcaption>
<strong>Four rings around every production chain.</strong> Cache for cost and speed, limiter
for survival under burst, tracker for the budget, tracing so you can explain any single
request afterwards.
</figcaption>
</figure>

## Testing code whose output changes

You cannot assert an exact string. You can still test almost everything.

### Layer 1 — fake the model, test your plumbing

Most of your bugs are in the wiring, not the model. Test that with no network at all.

```python title="tests/test_chain_wiring.py"
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain.messages import AIMessage

def test_prompt_includes_retrieved_context():
    """The retriever output must actually reach the prompt."""
    fake = FakeMessagesListChatModel(responses=[AIMessage("stub answer")])
    chain = build_rag_chain(model=fake, retriever=StubRetriever(["REFUND WINDOW IS 30 DAYS"]))

    result = chain.invoke("what is the refund window?")

    sent = fake.messages_sent[-1]                     # what the model actually received
    assert "REFUND WINDOW IS 30 DAYS" in str(sent)
    assert result.text == "stub answer"
```

This is fast, free, deterministic and catches the single most common RAG bug: context that
never made it into the prompt.

### Layer 2 — structure and properties, against the real model

```python title="tests/test_classification.py"
import pytest

CASES = [
    ("I was charged twice", "billing", {"high", "critical"}),
    ("How do I export my data?", "account", {"low", "normal"}),
    ("EVERYTHING IS DOWN", "technical", {"critical"}),
]

@pytest.mark.parametrize("text,category,allowed_urgency", CASES)
def test_classification_properties(text, category, allowed_urgency):
    result = classify_chain.invoke({"ticket_text": text})

    assert result.category == category               # deterministic enough to assert
    assert result.urgency in allowed_urgency         # a SET, not one value
    assert len(result.summary) <= 120                # the schema contract holds
```

Assert **sets and ranges**, never one exact string. A test that demands exact wording will
fail on a good day and teach your team to ignore it.

### Layer 3 — a dataset with a threshold

```python title="tests/test_eval_gate.py"
def test_accuracy_does_not_regress():
    """Runs 60 labelled cases. Gate the merge, not each case."""
    results = [classify_chain.invoke({"ticket_text": c.text}) for c in DATASET]
    correct = sum(r.category == c.expected for r, c in zip(results, DATASET, strict=True))
    accuracy = correct / len(DATASET)

    assert accuracy >= 0.90, f"accuracy dropped to {accuracy:.2%}"
```

| Layer | Speed | Cost | Runs |
| --- | --- | --- | --- |
| fake model, plumbing | milliseconds | free | every commit |
| properties, real model | seconds | pennies | every PR |
| dataset with threshold | minutes | dollars | before merge, nightly |

:::mistake Only having layer 3
Teams often jump straight to an eval suite, then find it takes eight minutes and $3 per run,
so nobody runs it. Layer 1 is where the day-to-day value is: it catches the wiring bugs
instantly and for nothing.
:::

## Streaming to a real client

For a web app you want tokens out of the door as they arrive.

```python title="src/lc/stream_api.py"
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/ask")
async def ask(question: str):
    async def generate():
        async for chunk in chain.astream({"question": question}):
            text = getattr(chunk, "text", None)
            if text:
                yield f"data: {text}\n\n"        # server-sent events
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

:::warning Streaming hides failures from your error handling
Once you have sent the first byte with a 200 status, you cannot change your mind. If the
model fails at token 300, the client has a half-finished answer and an HTTP success.

Send an explicit error event in the stream, and have the client treat a stream that ends
without `[DONE]` as a failure.
:::

## Production checklist

| Check | Why |
| --- | --- |
| Cost logged per request, with a request id | the only way to answer "what does this cost?" |
| A hard budget ceiling per request | one runaway loop cannot empty the account |
| Timeouts on every model call | a hung provider must not hold your worker |
| Retry plus fallback, and log which model answered | survive bad days without silent drift |
| Rate limiter sized for your worker count | not per process, per fleet |
| Cache with tenant in the key | speed without cross-tenant leaks |
| Tracing on, with request id in metadata | debug one specific complaint |
| Layer 1 tests in CI | catch wiring bugs for free |
| Eval threshold gating merges | prompt changes are code changes |
| No secrets in code or prompts | they end up in logs and traces |

## Hands-on Exercise

:::exercise Wrap a chain for production
Take any chain from the previous lesson and add all four rings:

1. `SQLiteCache`, with the tenant id included in what makes the key unique
2. A `CostTracker` callback with a $0.10 ceiling that raises and is handled
3. An `InMemoryRateLimiter` sized as **(your provider limit ÷ worker count)**
4. Tracing with `run_name`, tags and a `request_id` in metadata

Then write the three test layers: one fake-model plumbing test, three property tests, and
one dataset test with a threshold.

Finally, run the same 20 requests twice and report: wall time, tokens and cost for run 1 vs
run 2. Explain the difference.
:::

:::solution What the two runs show
```text
20 requests, 6 of them duplicates

run 1 (cold cache)    18.9s   14,880 tokens   $0.061
run 2 (warm cache)     9.2s    9,640 tokens   $0.039

the 6 duplicates cost nothing the second time: 3ms each instead of ~1.1s
```

```text
test layers
  layer 1  4 tests   0.08s    $0.000   ran 40 times today
  layer 2  3 tests   6.10s    $0.004   ran on each PR
  layer 3  1 test   84.00s    $0.180   ran before merge

The layer-1 test caught the real bug during the exercise: after refactoring, the
retriever output was being formatted into the prompt as "[Document(...)]" - the repr
of the list, not the page content. The model had been answering from nothing.
```
That is the characteristic layer-1 catch: a silent quality collapse from a wiring mistake
that no amount of prompt tuning would have fixed. It costs nothing to run and it found the
thing an eval suite would have blamed on the model.
:::

## Challenge

:::challenge Find your real p95, then fix it
Instrument a chain so each step reports its own duration, and run 200 requests with
realistic inputs — including a few pathological ones (very long input, an empty retrieval
result, a document with strange encoding).

Produce a table of p50, p95 and p99 for each step and for the whole request. Then find the
p95 driver and fix it. Candidates, in the order they usually turn out to matter: reranking,
an unnecessary second model call, a retriever with no timeout, and cold-start on first call.

Re-measure and report the before and after. The instructive part is usually that the p95 is
dominated by something that is not the model at all — which is exactly the finding the
tracing lesson in Phase 24 predicts.
:::

## Interview Questions

:::interview
1. What makes a model response cache dangerous in a multi-tenant app?
2. Why is `InMemoryRateLimiter` insufficient across four workers?
3. How do you test code whose output is different every run?
4. Why does a silent fallback to a cheaper model corrupt your metrics?
5. What breaks about error handling once you start streaming?
6. Which single log line would you add first to a new LangChain feature?
:::

## Cheat Sheet

```python
# caching
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".cache/langchain.db"))

# rate limiting
from langchain_core.rate_limiters import InMemoryRateLimiter
model = init_chat_model("anthropic:claude-sonnet-5",
                        rate_limiter=InMemoryRateLimiter(requests_per_second=8))

# cost + budget
class CostTracker(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs): ...
chain.invoke(x, config={"callbacks": [tracker]})

# resilience
model.with_retry(stop_after_attempt=3).with_fallbacks([cheaper])

# tracing
LANGSMITH_TRACING=true            # or LANGSMITH_OTEL_ENABLED=true for OTLP
chain.invoke(x, config={"run_name": "...", "metadata": {"request_id": rid}})

# testing
FakeMessagesListChatModel(responses=[AIMessage("stub")])   # layer 1
assert result.urgency in {"high", "critical"}              # layer 2
assert accuracy >= 0.90                                    # layer 3
```

## Summary

- Cache aggressively, but never across tenants or personalised context.
- Track cost per request with a callback, and enforce a hard ceiling.
- Size your rate limiter for the whole fleet, not one process.
- Turn tracing on and attach a request id; it pays for itself the first time someone
  complains about one specific answer.
- Test in three layers. The fake-model layer is the cheapest and catches the most.
- Streaming trades away your ability to fail cleanly — plan the error event.

## Next Step

Next: a full project — a document assistant that puts everything in this phase together,
with citations, budgets and tests.
