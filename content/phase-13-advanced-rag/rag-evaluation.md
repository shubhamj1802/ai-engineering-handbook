---
title: RAG Evaluation and Hallucination Detection
order: 2
difficulty: Expert
duration: 18
badges: ["Hands-on", "Production"]
summary: "Build the evaluation that makes every RAG change measurable — retrieval metrics, faithfulness, answer relevance, LLM-as-judge, and a CI gate that blocks regressions."
prereqs: ["Query Transformation, Hybrid Search and Reranking"]
keyConcepts: ["faithfulness", "answer relevance", "context precision", "LLM-as-judge", "regression gate"]
---

## Why this matters

Without evaluation, every RAG change is a coin flip. With it, you can answer the two
questions that actually matter — *did this change help?* and *is the system getting worse?* —
and you can answer them in CI, before a release, rather than from support tickets.

## Mental Model

```text
Evaluate the two stages SEPARATELY, then the whole:

RETRIEVAL      did we find the right context?
  recall@k · precision@k · MRR · nDCG        ← deterministic, cheap, run on every commit

GENERATION     did we use it correctly?
  faithfulness      is every claim supported by the retrieved context?
  answer relevance  does the answer address the question?
  citation validity do cited ids exist and support the claim?   ← deterministic!
  correct refusal   does it decline when the context is insufficient?

END TO END     did the user get what they needed?
  correctness against a reference answer · human review on a sample
```

The discipline that makes this tractable: **check deterministically whatever you can, and
use a model judge only for what you genuinely cannot.**

## Core Concepts

### The dataset comes first

Fifty well-chosen cases beat five thousand scraped ones.

```python
{
  "id": "q_017",
  "question": "How long are logs retained on the Pro plan?",
  "relevant_chunk_ids": ["pricing::2"],          # for retrieval metrics
  "reference_answer": "30 days on Pro; Enterprise can configure up to 400 days.",
  "must_contain": ["30 day"],                     # deterministic check
  "must_not_contain": ["90 day"],
  "answerable": true,
  "category": "factual_lookup",
  "difficulty": "easy"
}
```

Compose the set deliberately:

| Category | Share | Why |
| --- | --- | --- |
| Factual lookup | 40% | the common case |
| Multi-hop (needs 2+ chunks) | 20% | tests retrieval breadth |
| Unanswerable | 15% | **tests refusal — the most important category** |
| Ambiguous | 10% | tests clarification behaviour |
| Adversarial / injection | 10% | tests guardrails |
| Exact identifier | 5% | tests hybrid search |

Source real questions from support tickets, search logs and Slack. Invented questions test
the system you imagined, not the one people use.

### Deterministic checks — do these first

They are free, instant, unambiguous, and catch most real failures:

```python
citations_exist        every [id] appears in the retrieved set
citations_nonempty     an answered question carries at least one citation
required_terms         must_contain phrases are present
forbidden_terms        must_not_contain phrases are absent
refusal_correct        refused exactly when the case is unanswerable
length_within_limit    respects the word limit
schema_valid           structured output parses
latency_within_budget  p95 under the SLO
```

If your evaluation has no deterministic layer, you are paying a model to check things
`in` and a regex could check for free.

### The four RAG metrics

| Metric | Question | Needs |
| --- | --- | --- |
| **Context precision** | were the retrieved chunks relevant? | judge or labels |
| **Context recall** | did retrieval find everything needed? | labelled chunk ids |
| **Faithfulness** | is every claim supported by the context? | judge |
| **Answer relevance** | does the answer address the question? | judge |

The diagnostic power comes from reading them together:

```text
low context recall  + high faithfulness  → retrieval problem: fix chunking/hybrid/rerank
high context recall + low faithfulness   → generation problem: fix the prompt
low faithfulness    + low relevance      → the model is improvising: check the context assembly
high everything     + unhappy users      → your dataset does not reflect real questions
```

### LLM-as-judge, done carefully

A model scoring model output is useful and biased. Rules that make it trustworthy:

1. **Score one narrow property at a time**, never "rate this 1–10 overall".
2. **Give a rubric with concrete anchors** for each score.
3. **Require structured output** so parsing cannot fail.
4. **Decompose first**: split the answer into atomic claims, then check each claim against
   the context. Far more reliable than judging a paragraph.
5. **Calibrate against humans**: have a person label 30 cases and measure agreement. If
   agreement is below ~80%, fix the rubric before trusting the judge.
6. Use a **different or stronger model** than the one generating, where budget allows —
   models favour their own output.

### Faithfulness by claim decomposition

```text
answer: "Logs are kept 30 days on Pro [c1] and Enterprise can extend to 400 days [c2].
         Pricing starts at $49."

claims:
  1. "Logs are kept 30 days on Pro"          → supported by c1   ✓
  2. "Enterprise can extend to 400 days"     → supported by c2   ✓
  3. "Pricing starts at $49"                 → NOT in context    ✗  ← unsupported, uncited

faithfulness = 2/3 = 0.667
```

That third claim is the hallucination: fluent, plausible, uncited and absent from the
context. Claim-level checking finds it; paragraph-level scoring usually does not.

## Real-World Example

A complete evaluation harness with deterministic checks, a claim-level judge, cost tracking
and a CI gate.

```python title="src/evals/harness.py"
"""RAG evaluation harness.

Design rules:
  - deterministic checks run first and are free
  - the judge runs only on what cannot be checked deterministically
  - every run is written to disk so runs are comparable over time
  - a regression gate compares against a stored baseline and fails the build
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CITATION = re.compile(r"\[([A-Za-z0-9_.#:/-]+)\]")
REFUSAL_MARKERS = ("could not find", "not in the provided", "insufficient context",
                   "no information", "cannot answer")


# --- dataset ----------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    answerable: bool = True
    relevant_chunk_ids: tuple[str, ...] = ()
    reference_answer: str = ""
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    category: str = "factual_lookup"
    difficulty: str = "medium"
    history: tuple[dict, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict) -> "EvalCase":
        return cls(
            id=raw["id"], question=raw["question"],
            answerable=raw.get("answerable", True),
            relevant_chunk_ids=tuple(raw.get("relevant_chunk_ids", [])),
            reference_answer=raw.get("reference_answer", ""),
            must_contain=tuple(raw.get("must_contain", [])),
            must_not_contain=tuple(raw.get("must_not_contain", [])),
            category=raw.get("category", "factual_lookup"),
            difficulty=raw.get("difficulty", "medium"),
            history=tuple(raw.get("history", [])),
        )


def load_dataset(path: Path) -> list[EvalCase]:
    return [
        EvalCase.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- deterministic checks ----------------------------------------------------
def deterministic_checks(case: EvalCase, answer, *, max_words: int = 220) -> dict[str, bool]:
    text = answer.text
    lowered = text.lower()
    refused = any(marker in lowered for marker in REFUSAL_MARKERS)

    cited = set(CITATION.findall(text))
    retrieved_ids = {c.id for c in answer.retrieved}
    supplied_ids = {c.id for c in answer.citations} | retrieved_ids

    return {
        "refusal_correct": refused if not case.answerable else not refused,
        "citations_valid": not (cited - supplied_ids),
        "has_citation": bool(cited) if case.answerable and not refused else True,
        "required_terms": all(t.lower() in lowered for t in case.must_contain) if not refused else True,
        "forbidden_terms": not any(t.lower() in lowered for t in case.must_not_contain),
        "within_length": len(text.split()) <= max_words,
    }


def retrieval_metrics(case: EvalCase, answer, *, k: int = 5) -> dict[str, float]:
    if not case.relevant_chunk_ids:
        return {}
    returned = [c.id for c in answer.retrieved][:k]
    relevant = set(case.relevant_chunk_ids)

    hit_rank = next((i for i, cid in enumerate(returned, 1) if cid in relevant), None)
    return {
        f"recall@{k}": len(relevant & set(returned)) / len(relevant),
        f"precision@{k}": len(relevant & set(returned)) / max(len(returned), 1),
        "mrr": 1.0 / hit_rank if hit_rank else 0.0,
    }


# --- LLM judge ---------------------------------------------------------------
class ClaimList(BaseModel):
    claims: list[str] = Field(description="atomic factual claims, one per statement")


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence_chunk_id: str | None = None
    explanation: str = Field(max_length=200)


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


class RelevanceVerdict(BaseModel):
    addresses_question: bool
    score: int = Field(ge=1, le=5, description="1 = ignores the question, 5 = fully answers it")
    missing: str = Field(default="", max_length=200)


DECOMPOSE = """\
Split the answer into atomic factual claims. One verifiable statement per claim.
Ignore hedging, meta-commentary and citation markers. If there are no factual claims,
return an empty list."""

VERIFY = """\
For each claim, decide whether the context supports it.

supported = true  only if a chunk states the claim or directly implies it
supported = false if the claim is absent, contradicted, or merely plausible

Cite the chunk id that supports each supported claim. Judge only support by the
context - not whether the claim is true in the world."""

RELEVANCE = """\
Judge whether the answer addresses the question asked.

5 fully answers every part
4 answers the main part, minor omission
3 partially answers
2 related but does not answer
1 does not address the question

A correct refusal for a genuinely unanswerable question scores 5."""


@dataclass
class Judge:
    llm: object                       # .structured(messages, schema, system=...) -> (obj, meta)
    cost_usd: float = 0.0
    calls: int = 0

    def faithfulness(self, answer_text: str, context: str) -> dict:
        claims, meta = self.llm.structured(
            [{"role": "user", "content": answer_text}], ClaimList, system=DECOMPOSE
        )
        self._record(meta)

        if not claims.claims:
            return {"faithfulness": 1.0, "claims": 0, "unsupported": []}

        payload = json.dumps({"context": context, "claims": claims.claims})
        verdicts, meta = self.llm.structured(
            [{"role": "user", "content": payload}], ClaimVerdicts, system=VERIFY
        )
        self._record(meta)

        supported = sum(v.supported for v in verdicts.verdicts)
        total = max(len(verdicts.verdicts), 1)
        return {
            "faithfulness": supported / total,
            "claims": total,
            "unsupported": [v.claim for v in verdicts.verdicts if not v.supported],
        }

    def relevance(self, question: str, answer_text: str) -> dict:
        verdict, meta = self.llm.structured(
            [{"role": "user", "content": json.dumps({"question": question, "answer": answer_text})}],
            RelevanceVerdict, system=RELEVANCE,
        )
        self._record(meta)
        return {"relevance": verdict.score / 5.0, "missing": verdict.missing}

    def _record(self, meta: dict) -> None:
        self.calls += 1
        self.cost_usd += float(meta.get("cost_usd", 0.0))


# --- the run -----------------------------------------------------------------
@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    retrieval: dict[str, float] = field(default_factory=dict)
    judged: dict[str, float] = field(default_factory=dict)
    latency_ms: int = 0
    answer: str = ""
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class EvalRun:
    results: list[CaseResult] = field(default_factory=list)
    judge_cost_usd: float = 0.0
    seconds: float = 0.0

    def aggregate(self) -> dict[str, float]:
        if not self.results:
            return {}

        def mean_of(key: str, source: str) -> float | None:
            values = [getattr(r, source).get(key) for r in self.results
                      if getattr(r, source).get(key) is not None]
            return round(statistics.mean(values), 4) if values else None

        summary: dict[str, float] = {
            "cases": len(self.results),
            "pass_rate": round(sum(r.passed for r in self.results) / len(self.results), 4),
            "p50_latency_ms": round(statistics.median(r.latency_ms for r in self.results)),
            "p95_latency_ms": round(
                sorted(r.latency_ms for r in self.results)[int(len(self.results) * 0.95) - 1]
            ),
            "judge_cost_usd": round(self.judge_cost_usd, 4),
        }
        for key in ("refusal_correct", "citations_valid", "has_citation",
                    "required_terms", "forbidden_terms"):
            summary[key] = round(
                sum(r.checks.get(key, True) for r in self.results) / len(self.results), 4
            )
        for key in ("recall@5", "precision@5", "mrr"):
            value = mean_of(key, "retrieval")
            if value is not None:
                summary[key] = value
        for key in ("faithfulness", "relevance"):
            value = mean_of(key, "judged")
            if value is not None:
                summary[key] = value
        return summary

    def by_category(self) -> dict[str, dict[str, float]]:
        categories: dict[str, list[CaseResult]] = {}
        for result in self.results:
            categories.setdefault(result.category, []).append(result)
        return {
            name: {
                "cases": len(items),
                "pass_rate": round(sum(i.passed for i in items) / len(items), 3),
                "faithfulness": round(
                    statistics.mean([i.judged.get("faithfulness", 1.0) for i in items]), 3
                ),
            }
            for name, items in sorted(categories.items())
        }

    def failures(self, limit: int = 10) -> list[dict]:
        return [
            {"case": r.case_id, "category": r.category,
             "failed_checks": [k for k, v in r.checks.items() if not v],
             "unsupported_claims": r.unsupported_claims[:3],
             "answer": r.answer[:200]}
            for r in self.results if not r.passed
        ][:limit]


def run_evaluation(
    pipeline, cases: list[EvalCase], *, judge: Judge | None = None,
    where: dict | None = None,
) -> EvalRun:
    run = EvalRun()
    started_all = time.perf_counter()

    for case in cases:
        started = time.perf_counter()
        answer = pipeline.answer(case.question, where=where)
        latency_ms = int((time.perf_counter() - started) * 1000)

        checks = deterministic_checks(case, answer)
        retrieval = retrieval_metrics(case, answer)
        judged: dict[str, float] = {}
        unsupported: list[str] = []

        # the judge only runs when there is something it alone can assess
        should_judge = judge is not None and case.answerable and all(
            checks[key] for key in ("refusal_correct", "citations_valid")
        )
        if should_judge:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in answer.citations)
            faithfulness = judge.faithfulness(answer.text, context)
            relevance = judge.relevance(case.question, answer.text)
            judged = {"faithfulness": faithfulness["faithfulness"],
                      "relevance": relevance["relevance"]}
            unsupported = faithfulness["unsupported"]

        passed = all(checks.values()) and judged.get("faithfulness", 1.0) >= 0.8

        run.results.append(CaseResult(
            case_id=case.id, category=case.category, passed=passed,
            checks=checks, retrieval=retrieval, judged=judged,
            latency_ms=latency_ms, answer=answer.text, unsupported_claims=unsupported,
        ))

    run.seconds = time.perf_counter() - started_all
    run.judge_cost_usd = judge.cost_usd if judge else 0.0
    return run


# --- regression gate ----------------------------------------------------------
GATES = {
    "pass_rate": 0.03,            # may drop by at most 3 points
    "faithfulness": 0.03,
    "recall@5": 0.05,
    "citations_valid": 0.00,      # must never regress at all
    "refusal_correct": 0.02,
}


def compare_to_baseline(current: dict, baseline_path: Path) -> tuple[bool, list[str]]:
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return True, ["no baseline found - current run stored as the baseline"]

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    for metric, allowed_drop in GATES.items():
        before, after = baseline.get(metric), current.get(metric)
        if before is None or after is None:
            continue
        if after < before - allowed_drop:
            problems.append(
                f"{metric}: {before:.3f} → {after:.3f} "
                f"(drop {before - after:.3f} exceeds the {allowed_drop:.2f} budget)"
            )

    latency_before = baseline.get("p95_latency_ms", 0)
    latency_after = current.get("p95_latency_ms", 0)
    if latency_before and latency_after > latency_before * 1.5:
        problems.append(f"p95 latency: {latency_before}ms → {latency_after}ms (+50% budget exceeded)")

    return not problems, problems


if __name__ == "__main__":
    import sys

    from ..llm.client import LLMClient
    from ..rag.app import build_pipeline

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

    cases = load_dataset(Path("evals/dataset.jsonl"))
    pipeline = build_pipeline()
    judge = Judge(LLMClient(model="claude-opus-5", max_calls=500))

    run = run_evaluation(pipeline, cases, judge=judge)
    summary = run.aggregate()

    print(f"\n=== {summary['cases']} cases in {run.seconds:.1f}s "
          f"(judge cost ${run.judge_cost_usd:.4f}) ===")
    for key, value in summary.items():
        print(f"  {key:<20} {value}")

    print("\nby category:")
    for name, metrics in run.by_category().items():
        print(f"  {name:<22} pass {metrics['pass_rate']:.2f}  "
              f"faithfulness {metrics['faithfulness']:.2f}  (n={metrics['cases']})")

    if run.failures():
        print("\nfailures:")
        for failure in run.failures(5):
            print(f"  {failure['case']} [{failure['category']}]: {failure['failed_checks']}")
            for claim in failure["unsupported_claims"]:
                print(f"      unsupported: {claim}")

    ok, problems = compare_to_baseline(summary, Path("evals/baseline.json"))
    if not ok:
        print("\nREGRESSION DETECTED:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("\nno regression against the baseline")
```

```bash
uv run python -m evals.harness
```

```text
=== 50 cases in 94.2s (judge cost $0.2140) ===
  cases                50
  pass_rate            0.88
  p50_latency_ms       1412
  p95_latency_ms       2681
  judge_cost_usd       0.214
  refusal_correct      0.96
  citations_valid      1.0
  has_citation         0.98
  required_terms       0.9
  forbidden_terms      1.0
  recall@5             0.92
  precision@5          0.41
  mrr                  0.864
  faithfulness         0.941
  relevance            0.912

by category:
  adversarial            pass 1.00  faithfulness 1.00  (n=5)
  ambiguous              pass 0.80  faithfulness 0.92  (n=5)
  exact_identifier       pass 1.00  faithfulness 1.00  (n=3)
  factual_lookup         pass 0.95  faithfulness 0.97  (n=20)
  multi_hop              pass 0.70  faithfulness 0.86  (n=10)
  unanswerable           pass 0.86  faithfulness 1.00  (n=7)

failures:
  q_023 [multi_hop]: ['required_terms']
      unsupported: Enterprise SLA is 99.95%
  q_041 [unanswerable]: ['refusal_correct']

no regression against the baseline
```

Read the category breakdown: multi-hop questions pass at 0.70 while everything else is above
0.85. That is a specific, actionable finding — retrieval is not bringing back both required
chunks — and it points straight at increasing `k`, reranking, or parent expansion. An
aggregate pass rate of 0.88 would have hidden it.

### Running it in CI

```yaml title=".github/workflows/eval.yml"
name: RAG evaluation

on:
  pull_request:
    paths: ["src/rag/**", "src/retrieval/**", "src/prompts/**", "evals/**"]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: { enable-cache: true }

      - run: uv sync --frozen

      - name: Fast deterministic evaluation
        run: uv run python -m evals.harness --no-judge      # free, ~15s, every PR
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - name: Full evaluation with judge
        if: contains(github.event.pull_request.labels.*.name, 'full-eval')
        run: uv run python -m evals.harness
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: eval-results, path: evals/runs/ }
```

Deterministic checks on every pull request (seconds, free); the judged run on demand and
nightly (minutes, cents). That split is what makes evaluation something a team actually
keeps running.

## Common Mistakes

:::mistake
```text
1. No unanswerable cases
   You never learn whether the system refuses, which is its most important behaviour.

2. Judging with the same model that generated
   Models favour their own output. Use a different or stronger judge.

3. "Rate this answer 1-10"
   Unreliable and uninterpretable. Score one narrow property with a rubric.

4. Never calibrating the judge against humans
   Label 30 cases by hand; if agreement is under 80%, the rubric is broken.

5. Aggregate metrics only
   0.88 overall hides multi-hop at 0.70. Always break down by category.

6. Dataset written by the person who built the system
   You test the questions you designed for. Use real user questions.

7. Evaluating generation with broken retrieval
   Fix recall first; generation metrics are meaningless without the right context.

8. A dataset that never changes
   Add every production failure as a case. The set should grow monotonically.
```
:::

## Existing tools

| Tool | What it gives you | Trade-off |
| --- | --- | --- |
| **RAGAS** | faithfulness, answer relevance, context precision/recall out of the box | opinionated metric definitions; version churn |
| **DeepEval** | pytest-style assertions, many metrics, CI-friendly | heavier dependency, learning curve |
| **Langfuse** | tracing plus datasets and scored runs in a UI | hosted or self-hosted service to operate |
| Hand-rolled | exactly your metrics, no dependencies, full control | you maintain it |

Start hand-rolled — the fifty lines of deterministic checks in this lesson catch most real
issues and cost nothing. Adopt RAGAS or DeepEval when you want standard metric definitions
and a comparison point; both are covered in Phase 24 alongside tracing.

## Hands-on Exercise

:::exercise Build a 50-case evaluation set
For your corpus:

1. Write 50 cases with the category distribution from this lesson, sourcing real questions
   wherever possible.
2. Label `relevant_chunk_ids` for at least the 20 factual cases.
3. Implement the deterministic checks and run them — free, a few seconds.
4. Add the claim-level faithfulness judge and run the full evaluation; record the cost.
5. Produce the aggregate table and the per-category breakdown.
6. Store the run as `baseline.json` and wire the regression gate.
7. Now make a change (chunk size, k, rerank on/off, a prompt rule) and re-run. Report what
   moved and by how much.

Deliverable: a table showing your change's effect on pass rate, faithfulness, recall@5, p95
latency and cost. That table is what you bring to a review instead of an opinion.
:::

:::solution Reference before/after
```text
metric             baseline   + rerank   delta
pass_rate             0.820      0.880   +0.060
faithfulness          0.897      0.941   +0.044
recall@5              0.840      0.920   +0.080
precision@5           0.310      0.410   +0.100
mrr                   0.701      0.864   +0.163
refusal_correct       0.860      0.960   +0.100
p95_latency_ms        2,410      2,681     +271
cost_per_query       $0.0031    $0.0031   +$0

by category (pass rate):
  multi_hop            0.50 → 0.70
  factual_lookup       0.90 → 0.95
  unanswerable         0.71 → 0.86

Decision: ship reranking. +6 points of pass rate and +0.16 MRR for 271 ms of p95 and no
extra model cost (the reranker runs locally). The unanswerable improvement is a bonus:
better ranking pushed irrelevant chunks out of the prompt, so the model stopped trying
to answer from them.
```

That last observation is common and worth internalising: **improving retrieval precision
improves refusal behaviour**, because the model stops being handed weak material that
invites a guess.
:::

## Challenge

:::challenge Calibrate your judge
Take 40 answers from a real run. Have two people independently label each one for
faithfulness (supported / not supported), resolving disagreements by discussion to produce a
gold label.

Then measure your LLM judge against the gold labels: accuracy, precision, recall and Cohen's
kappa. If kappa is below 0.6, iterate on the rubric — more concrete anchors, claim
decomposition, a stronger judge model — and re-measure.

Report the final agreement. A judge you have not calibrated is a number generator; a
calibrated one is a measurement instrument, and knowing which you have is the difference
between an evaluation and a comfort blanket.
:::

## Interview Questions

:::interview
1. What is faithfulness, and how does it differ from correctness?
2. How do you distinguish a retrieval failure from a generation failure using metrics?
3. Why decompose an answer into claims before judging it?
4. How do you validate that an LLM judge is trustworthy?
5. What would you gate a deployment on, and at what thresholds?
:::

## Cheat Sheet

```text
DATASET   50+ cases: 40% factual · 20% multi-hop · 15% unanswerable · 10% ambiguous
          10% adversarial · 5% identifiers. Source from real user questions.

DETERMINISTIC (free, first)
  citations exist · at least one citation · required/forbidden terms
  refusal correctness · length · schema validity · latency

JUDGED (only what code cannot check)
  faithfulness = supported claims / total claims   ← decompose first
  answer relevance with a 1-5 rubric
  context precision

DIAGNOSE  low recall + high faithfulness  → retrieval
          high recall + low faithfulness  → prompt/generation
          both low                        → context assembly

GATE      pass_rate -3pt · faithfulness -3pt · recall@5 -5pt
          citations_valid 0 tolerance · p95 latency +50%
          deterministic on every PR, judged nightly
```

```quiz
[
  {
    "question": "recall@5 is 0.95 but faithfulness is 0.62. What is the problem?",
    "options": [
      "Retrieval is failing",
      "Generation: the right context is being retrieved but the model adds unsupported claims",
      "The embedding model is wrong",
      "The dataset is too small"
    ],
    "answer": 1,
    "explanation": "High recall means the context was there; low faithfulness means the answer went beyond it. Fix the prompt (stricter grounding, permitted refusal), not the retriever."
  },
  {
    "question": "Which check does NOT need an LLM judge?",
    "options": [
      "Whether the answer addresses the question",
      "Whether every cited chunk id was actually retrieved",
      "Whether a claim is supported by the context",
      "Whether the answer is helpful"
    ],
    "answer": 1,
    "explanation": "Citation validity is a set operation on ids - free, instant and unambiguous. Run every deterministic check before paying a judge for anything."
  },
  {
    "question": "Your evaluation has 50 cases and none are unanswerable. What is the risk?",
    "options": [
      "The dataset is too small",
      "You have no measurement of refusal behaviour - the system could be inventing answers whenever context is missing, scoring perfectly",
      "Metrics will be too low",
      "Judging will cost more"
    ],
    "answer": 1,
    "explanation": "Refusal is the behaviour that separates a grounded system from a confident fabricator, and it is invisible unless your dataset contains questions the corpus genuinely cannot answer."
  }
]
```

## Summary

- Evaluate retrieval and generation separately; their metrics point at different fixes.
- Run deterministic checks first — citations, refusal, required terms — they are free and
  catch most failures.
- Judge faithfulness by decomposing the answer into claims and checking each against the
  context.
- Break results down by category; aggregates hide the failing slice.
- Gate deployments on a stored baseline, with a tight budget on citation validity.

## Next Step

Phase 14: agents — what happens when the model, not your code, decides the next step.
