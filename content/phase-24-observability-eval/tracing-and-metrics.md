---
title: Tracing, Metrics and Cost Observability
order: 1
difficulty: Production
duration: 18
badges: ["Production", "Hands-on"]
summary: "Spans that capture an entire AI request, the metrics that matter (tokens, cost, latency, tool calls, trajectories), and wiring Langfuse or OpenTelemetry without leaking customer data."
prereqs: ["Logging, Configuration, HTTP Clients and CLIs", "Build an Agent from Scratch"]
keyConcepts: ["trace", "span", "token accounting", "p95 latency", "cost attribution"]
---

## Why this matters

A traditional service fails loudly: a 500, a stack trace, an alert. An AI system fails
quietly — a plausible wrong answer, a retrieval that returned nothing useful, an agent that
took eleven steps where it should have taken three. None of that shows up in HTTP metrics.
Tracing is how you see it at all.

## Mental Model

```text
TRACE    one user request, end to end
 └ SPAN  one operation inside it, with timing, inputs, outputs and attributes

  trace: req_8f21c  (2,340 ms · $0.0184 · outcome=answered)
   ├─ span guardrails.input        3 ms
   ├─ span retrieval              142 ms   k=12 → 5 · top_score=0.81
   │   ├─ span embed_query         18 ms
   │   └─ span vector_search      121 ms
   ├─ span generate             1,893 ms   in=2,140 out=181 · $0.0173 · cached=1,200
   ├─ span guardrails.output        2 ms   citations_valid=true
   └─ span persist                 34 ms
```

Four questions a trace must answer:

1. **What happened** — which steps ran, in what order.
2. **How long** — per step, so you know where the time went.
3. **What it cost** — tokens and dollars, per step.
4. **Why the outcome** — retrieved ids, scores, stop reason, guardrail verdicts.

## Core Concepts

### The metrics that matter

| Metric | Why | Alert when |
| --- | --- | --- |
| **p95 latency** | what users feel | above SLO |
| Time to first token | perceived speed | above 1.5 s |
| **Cost per request** | the bill | 2× the 7-day median |
| Tokens in / out | the cost driver | input creeping up |
| Cache hit rate | biggest cost lever | drops sharply |
| **Answered rate** | is it useful | falls |
| **Citation validity** | is it grounded | ever below 100% |
| Escalation rate | is it coping | spikes |
| Tool error rate | dependency health | above 5% per tool |
| Agent steps per task | efficiency | mean rises |
| Retrieval top score | index health | distribution shifts |

Means hide everything. Track **p50, p95 and p99** for latency and cost, always.

### What must never enter a trace

:::danger Traces are the easiest place to create a privacy incident
By default, record **metadata, not content**:

```text
record         question_length · chunk_ids · scores · token counts · latency · cost
                stop_reason · guardrail verdicts · model version · prompt fingerprint
do NOT record  full prompts · full answers · retrieved chunk text · user emails ·
                API keys · anything matching a PII pattern
```

If you need content for debugging, make it opt-in per environment, redact it, sample it (1%),
and give it a short retention. A trace store with full prompts is a customer-data store with
no access controls and unlimited retention.
:::

### OpenTelemetry or a purpose-built tool

| | OpenTelemetry | Langfuse / LangSmith |
| --- | --- | --- |
| Standard | vendor-neutral | product-specific |
| LLM concepts | you model them | built in (generations, scores, datasets) |
| Prompt/version tracking | manual | built in |
| Evaluation integration | none | datasets and scores in the same UI |
| Existing infra | fits your stack | another service |

A practical answer: **OpenTelemetry spans for the service, plus an LLM-specific layer for
generations and evaluations.** The instrumentation below is deliberately tool-agnostic so
either can consume it.

## Real-World Example

A tracing layer that works standalone and exports to Langfuse or OTel.

```python title="src/observability/tracing.py"
"""Request tracing for AI systems.

Deliberately dependency-free at the core: spans are plain dataclasses. Exporters
adapt them to Langfuse, OpenTelemetry or JSONL. Redaction happens before export,
not at the sink, so a misconfigured sink cannot leak content.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)
_current_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "current_span", default=None
)

PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36})\b"), "[KEY]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
]


class SpanKind(StrEnum):
    REQUEST = "request"
    GUARDRAIL = "guardrail"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    TOOL = "tool"
    AGENT_STEP = "agent_step"
    OTHER = "other"


def redact(text: str, *, max_chars: int = 500) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


@dataclass
class Span:
    name: str
    kind: SpanKind
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str = ""
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float = 0.0
    attributes: dict = field(default_factory=dict)
    error: str = ""

    # LLM-specific, populated on generation spans
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def duration_ms(self) -> int:
        end = self.ended_at or time.perf_counter()
        return int((end - self.started_at) * 1000)

    def set(self, **attributes) -> "Span":
        self.attributes.update(attributes)
        return self

    def record_usage(self, *, model: str, input_tokens: int, output_tokens: int,
                     cached_tokens: int = 0, prices: dict[str, tuple[float, float]]) -> "Span":
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens

        price_in, price_out = prices.get(model, (0.0, 0.0))
        self.cost_usd = (
            input_tokens / 1e6 * price_in
            + cached_tokens / 1e6 * price_in * 0.1
            + output_tokens / 1e6 * price_out
        )
        return self


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    name: str = "request"
    user_id_hash: str = ""            # hashed, never the raw id
    session_id: str = ""
    spans: list[Span] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def duration_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(s.input_tokens for s in self.spans),
            "output": sum(s.output_tokens for s in self.spans),
            "cached": sum(s.cached_tokens for s in self.spans),
        }

    def summary(self) -> dict:
        by_kind: dict[str, int] = {}
        for span in self.spans:
            by_kind[str(span.kind)] = by_kind.get(str(span.kind), 0) + span.duration_ms

        tokens = self.total_tokens
        cache_rate = tokens["cached"] / max(tokens["input"] + tokens["cached"], 1)

        return {
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "cost_usd": round(self.total_cost, 6),
            "tokens": tokens,
            "cache_hit_rate": round(cache_rate, 3),
            "spans": len(self.spans),
            "ms_by_kind": by_kind,
            "errors": [s.name for s in self.spans if s.error],
            **self.attributes,
        }

    def render(self) -> str:
        lines = [f"trace {self.trace_id}  {self.duration_ms}ms  ${self.total_cost:.5f}"]
        depth = {"": 0}
        for span in self.spans:
            level = depth.get(span.parent_id, 0)
            depth[span.span_id] = level + 1
            marker = "✗" if span.error else "·"
            detail = ""
            if span.kind is SpanKind.GENERATION:
                detail = f" in={span.input_tokens} out={span.output_tokens} ${span.cost_usd:.5f}"
            elif span.attributes:
                detail = " " + " ".join(f"{k}={v}" for k, v in list(span.attributes.items())[:3])
            lines.append(f"{'  ' * level}{marker} {span.name:<24} {span.duration_ms:>6}ms{detail}")
        return "\n".join(lines)


# --- the API you actually use ----------------------------------------------------
@contextmanager
def trace(name: str, *, user_id: str = "", session_id: str = "", **attributes):
    import hashlib

    new_trace = Trace(
        name=name,
        user_id_hash=hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else "",
        session_id=session_id,
        attributes=attributes,
    )
    token = _current_trace.set(new_trace)
    try:
        yield new_trace
    except Exception as exc:
        new_trace.attributes["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _current_trace.reset(token)
        for exporter in EXPORTERS:
            try:
                exporter.export(new_trace)
            except Exception:
                logger.exception("trace export failed")


@contextmanager
def span(name: str, kind: SpanKind = SpanKind.OTHER, **attributes):
    current_trace = _current_trace.get()
    if current_trace is None:                     # tracing off: near-zero overhead
        yield Span(name=name, kind=kind, trace_id="")
        return

    parent = _current_span.get()
    new_span = Span(name=name, kind=kind, trace_id=current_trace.trace_id,
                    parent_id=parent.span_id if parent else "", attributes=dict(attributes))
    current_trace.spans.append(new_span)
    token = _current_span.set(new_span)

    try:
        yield new_span
    except Exception as exc:
        new_span.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        new_span.ended_at = time.perf_counter()
        _current_span.reset(token)


# --- exporters --------------------------------------------------------------------
class JsonlExporter:
    """Always-on local export. Feeds the metrics job and offline analysis."""

    def __init__(self, path: Path = Path(".data/traces.jsonl")) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, trace: Trace) -> None:
        record = {
            "summary": trace.summary(),
            "spans": [
                {k: v for k, v in asdict(span).items() if k not in {"started_at", "ended_at"}}
                | {"duration_ms": span.duration_ms}
                for span in trace.spans
            ],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


class LangfuseExporter:
    """Optional: LLM-native tracing with datasets and scores in one place."""

    def __init__(self) -> None:
        from langfuse import Langfuse           # reads LANGFUSE_* from the environment

        self.client = Langfuse()

    def export(self, trace: Trace) -> None:
        remote = self.client.trace(
            id=trace.trace_id, name=trace.name,
            user_id=trace.user_id_hash or None, session_id=trace.session_id or None,
            metadata=trace.summary(),
        )
        for span in trace.spans:
            if span.kind is SpanKind.GENERATION:
                remote.generation(
                    name=span.name, model=span.model,
                    usage={"input": span.input_tokens, "output": span.output_tokens},
                    metadata={**span.attributes, "cost_usd": span.cost_usd},
                    level="ERROR" if span.error else "DEFAULT",
                )
            else:
                remote.span(name=span.name, metadata=span.attributes,
                            level="ERROR" if span.error else "DEFAULT")


EXPORTERS: list[object] = [JsonlExporter()]
```

### Instrumenting a real request

```python title="src/rag/traced_pipeline.py"
"""The Phase 12 pipeline, instrumented. Note what is recorded - and what is not."""
from __future__ import annotations

from ..observability.tracing import SpanKind, span, trace

PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def answer(question: str, *, user_id: str, session_id: str) -> dict:
    with trace("rag_answer", user_id=user_id, session_id=session_id,
               question_chars=len(question)) as current:          # LENGTH, not content

        with span("guardrails.input", SpanKind.GUARDRAIL) as guard_span:
            cleaned, verdicts = input_guardrails.run(question, {"user_id": user_id})
            guard_span.set(blocked=any(v.blocked for v in verdicts),
                           fired=[v.guardrail for v in verdicts if v.action != "allow"])
            if any(v.blocked for v in verdicts):
                current.attributes["outcome"] = "blocked_input"
                return {"status": "blocked"}

        with span("retrieval", SpanKind.RETRIEVAL) as retrieval_span:
            with span("embed_query", SpanKind.OTHER):
                vector = embedder.embed_query(cleaned)
            with span("vector_search", SpanKind.OTHER) as search_span:
                hits = store.search(vector, k=12, where={"tenant_id": user_id})
                search_span.set(candidates=len(hits))

            selected = select_chunks(hits, top_n=5)
            retrieval_span.set(
                chunk_ids=[c.id for c in selected],                # IDS, not text
                top_score=round(selected[0].score, 3) if selected else 0.0,
                mean_score=round(sum(c.score for c in selected) / len(selected), 3)
                           if selected else 0.0,
            )

        if not selected:
            current.attributes["outcome"] = "no_context"
            return {"status": "no_context"}

        with span("generate", SpanKind.GENERATION) as generation_span:
            text, meta = llm.complete(build_prompt(cleaned, selected), system=SYSTEM)
            generation_span.record_usage(
                model=meta["model"], input_tokens=meta["input_tokens"],
                output_tokens=meta["output_tokens"],
                cached_tokens=meta.get("cache_read_tokens", 0), prices=PRICES,
            ).set(stop_reason=meta["stop_reason"], answer_chars=len(text))

        with span("guardrails.output", SpanKind.GUARDRAIL) as output_span:
            checked, verdicts = output_guardrails.run(
                text, {"retrieved_ids": [c.id for c in selected]}
            )
            output_span.set(citations_valid=not any(v.blocked for v in verdicts),
                            fired=[v.guardrail for v in verdicts if v.action != "allow"])

        current.attributes.update(outcome="answered", citations=len(extract_citations(checked)))
        return {"status": "ok", "answer": checked, "trace_id": current.trace_id}
```

```text
trace req_8f21c94b0e6a  2340ms  $0.01841
· guardrails.input             3ms blocked=False fired=[]
· retrieval                  142ms chunk_ids=['pricing::2', 'pricing::3'] top_score=0.811
  · embed_query                18ms
  · vector_search             121ms candidates=12
· generate                  1893ms in=2140 out=181 $0.01730
· guardrails.output            2ms citations_valid=True fired=[]
```

### Metrics from traces

```python title="src/observability/metrics.py"
"""Aggregate traces into the dashboard numbers and the alerts."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load_traces(path: Path = Path(".data/traces.jsonl"), *, hours: int = 24) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line)["summary"]
            for line in path.read_text(encoding="utf-8").splitlines() if line]


def dashboard(traces: list[dict]) -> dict:
    if not traces:
        return {"requests": 0}

    latencies = sorted(t["duration_ms"] for t in traces)
    costs = sorted(t["cost_usd"] for t in traces)
    outcomes = Counter(t.get("outcome", "unknown") for t in traces)

    def percentile(values: list, p: float):
        return values[min(int(len(values) * p), len(values) - 1)]

    return {
        "requests": len(traces),
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95),
                       "p99": percentile(latencies, 0.99)},
        "cost_usd": {"total": round(sum(costs), 4),
                     "p50": round(percentile(costs, 0.5), 5),
                     "p95": round(percentile(costs, 0.95), 5),
                     "per_1k_requests": round(statistics.mean(costs) * 1_000, 2)},
        "tokens": {"input": sum(t["tokens"]["input"] for t in traces),
                   "output": sum(t["tokens"]["output"] for t in traces)},
        "cache_hit_rate": round(statistics.mean(t["cache_hit_rate"] for t in traces), 3),
        "outcomes": dict(outcomes),
        "answered_rate": round(outcomes.get("answered", 0) / len(traces), 3),
        "error_rate": round(sum(bool(t.get("errors")) for t in traces) / len(traces), 3),
        "time_by_stage": _mean_by_stage(traces),
    }


def _mean_by_stage(traces: list[dict]) -> dict[str, int]:
    totals: dict[str, list[int]] = {}
    for trace in traces:
        for kind, ms in trace.get("ms_by_kind", {}).items():
            totals.setdefault(kind, []).append(ms)
    return {kind: round(statistics.mean(values)) for kind, values in sorted(totals.items())}


ALERTS = {
    "p95_latency_ms": (3_000, "above"),
    "answered_rate": (0.80, "below"),
    "error_rate": (0.02, "above"),
    "cache_hit_rate": (0.30, "below"),
    "cost_per_1k_requests": (25.0, "above"),
}


def check_alerts(current: dict, baseline: dict) -> list[str]:
    problems: list[str] = []
    values = {
        "p95_latency_ms": current["latency_ms"]["p95"],
        "answered_rate": current["answered_rate"],
        "error_rate": current["error_rate"],
        "cache_hit_rate": current["cache_hit_rate"],
        "cost_per_1k_requests": current["cost_usd"]["per_1k_requests"],
    }

    for metric, (threshold, direction) in ALERTS.items():
        value = values[metric]
        if (direction == "above" and value > threshold) or \
           (direction == "below" and value < threshold):
            problems.append(f"{metric} = {value} ({direction} the {threshold} threshold)")

    # relative regression against the previous window
    if baseline:
        previous = baseline.get("cost_usd", {}).get("per_1k_requests", 0)
        if previous and values["cost_per_1k_requests"] > previous * 1.5:
            problems.append(f"cost per 1k rose from ${previous} to "
                            f"${values['cost_per_1k_requests']} (+50%)")
    return problems


if __name__ == "__main__":
    current = dashboard(load_traces())
    print(json.dumps(current, indent=2))
    for problem in check_alerts(current, baseline={}):
        print(f"ALERT: {problem}")
```

```text
{
  "requests": 4218,
  "latency_ms": {"p50": 1840, "p95": 3120, "p99": 6410},
  "cost_usd": {"total": 61.4212, "p50": 0.01204, "p95": 0.03891, "per_1k_requests": 14.56},
  "tokens": {"input": 9184203, "output": 812044},
  "cache_hit_rate": 0.612,
  "outcomes": {"answered": 3684, "no_context": 402, "blocked_input": 88, "escalated": 44},
  "answered_rate": 0.873,
  "error_rate": 0.004,
  "time_by_stage": {"generation": 1712, "retrieval": 118, "guardrail": 4, "other": 22}
}
ALERT: p95_latency_ms = 3120 (above the 3000 threshold)
```

`time_by_stage` settles every "why is it slow?" conversation: generation is 93% of the time,
so the fix is shorter answers, a faster model or streaming — not retrieval optimisation.

## Common Mistakes

:::mistake
```text
1. Logging full prompts and responses by default
   Your trace store becomes your largest uncontrolled customer-data store.

2. Reporting mean latency
   The mean hides the tail that users actually experience. p95 and p99.

3. No cost attribution per request
   You cannot tell which feature, tenant or model is driving the bill.

4. Tracing only the model call
   Retrieval quality and guardrail verdicts explain most bad answers.

5. No outcome attribute
   Without answered/no_context/escalated you cannot measure usefulness at all.

6. Traces without a request id in the user-facing response
   A support ticket becomes unresolvable.

7. Alerting on absolutes only
   A 50% cost rise inside the threshold is still a regression worth knowing about.

8. Retaining traces forever
   Same retention discipline as any other data store.
```
:::

## Hands-on Exercise

:::exercise Instrument your system end to end
1. Add tracing to every stage: guardrails, retrieval, generation, tools, persistence.
2. Record metadata only — ids, counts, scores, tokens, latency — and verify by grepping your
   trace file for an email address and an API key. Both must return nothing.
3. Return the `trace_id` in every API response and log it alongside the request id.
4. Build the dashboard aggregation and run it over a day of traffic.
5. Add the alert checks and deliberately trigger two of them.
6. Answer these from the traces alone: which stage dominates p95? what does the most
   expensive 1% of requests have in common? what fraction of requests are answered?

That last question is the one that tells you whether the system is useful, and almost nobody
instruments it.
:::

:::solution What the answers usually look like
```text
1. Which stage dominates p95?
   generation: 1,712ms of 1,876ms mean (91%). Retrieval is noise by comparison.

2. What do the most expensive 1% have in common?
   mean cost $0.18 vs $0.012 overall. All are agent-path requests with 6+ tool calls,
   and 80% of them ended in escalation - we pay the most for the requests we fail.
   Action: cap agent iterations lower and escalate earlier.

3. Answered rate: 87.3%. The 9.5% 'no_context' bucket is the retrieval gap, and those
   questions are the next evaluation dataset.

grep -c '@' .data/traces.jsonl   →  0
grep -c 'sk-'  .data/traces.jsonl →  0
```

"We pay the most for the requests we fail" is the single most common finding when teams first
instrument cost per request — and it is immediately actionable.
:::

## Challenge

:::challenge Cost attribution dashboard
Extend tracing to attribute cost along four dimensions: tenant, feature, model and outcome.
Then answer with data:

1. Which tenant costs the most per request, and why?
2. Which feature has the worst cost-to-value ratio?
3. What would routing the cheapest 60% of requests to a smaller model save, and what would it
   cost in answered rate?
4. What is the monthly saving from raising the cache hit rate from 0.61 to 0.85?

Present it as a one-page report with a recommendation. Cost attribution is usually the fastest
route to a 30–50% reduction in spend, and it is entirely a data exercise.
:::

## Interview Questions

:::interview
1. What belongs in a span for an AI request?
2. Why is mean latency the wrong metric?
3. What must never be recorded in a trace, and how do you enforce it?
4. How do you attribute cost to a feature or tenant?
5. What would you alert on for a RAG system?
:::

## Cheat Sheet

```text
TRACE   one request · SPAN one operation (nested, timed, attributed)
RECORD  ids · counts · scores · tokens · cost · latency · stop_reason · verdicts
NEVER   prompts · answers · chunk text · emails · keys (redact before export)

METRICS p50/p95/p99 latency · cost per request and per 1k · tokens in/out ·
        cache hit rate · answered rate · citation validity · escalation rate ·
        tool error rate · steps per task

ALERT   p95 > SLO · answered_rate drops · citation_validity < 1.0 ·
        cost/1k up >50% week over week · cache hit rate falls

TOOLS   OpenTelemetry (standard) + Langfuse/LangSmith (LLM-native datasets and scores)
RETURN  the trace_id in every response so a ticket maps to a trace
```

```quiz
[
  {
    "question": "Your dashboard shows mean latency 1.2s but users complain it is slow. What is likely happening?",
    "options": [
      "Users are exaggerating",
      "The tail is bad - p95 or p99 may be several seconds, and the mean hides it",
      "The dashboard is broken",
      "Network latency"
    ],
    "answer": 1,
    "explanation": "A few very slow requests barely move the mean but define the experience of 5% of users - and in agent systems, tail latency compounds across steps."
  },
  {
    "question": "What should a retrieval span record?",
    "options": [
      "The full text of every retrieved chunk",
      "Chunk ids, scores, candidate count and filter used - metadata that explains the outcome without storing content",
      "Only the duration",
      "The user's question verbatim"
    ],
    "answer": 1,
    "explanation": "Ids and scores are enough to diagnose a retrieval failure, and they turn your trace store into diagnostics rather than an uncontrolled copy of customer data."
  },
  {
    "question": "Cost per 1,000 requests rose from $14 to $22 with no code change. What do you check first?",
    "options": [
      "The provider's pricing page",
      "Cache hit rate and mean input tokens - a broken prompt prefix or growing context is the usual cause",
      "The number of users",
      "The vector database"
    ],
    "answer": 1,
    "explanation": "A cache hit rate collapse (a timestamp entering the prefix) or history growth are the two most common causes, and both are visible directly in the trace summaries."
  }
]
```

## Summary

- A trace answers what happened, how long, what it cost, and why the outcome occurred.
- Record metadata, never content; redact before export and verify by grepping your own store.
- Track percentiles, not means, and always include cost per request and answered rate.
- `time_by_stage` and cost attribution turn vague complaints into specific fixes.
- Return the trace id to the caller so support tickets map to traces.

## Next Step

Evaluation in CI: turning traces and datasets into a gate that blocks regressions before they
ship.
