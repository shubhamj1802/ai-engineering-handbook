---
title: Evaluating Agents and Gating Releases
order: 2
difficulty: Production
duration: 18
badges: ["Production", "Hands-on"]
summary: "Trajectory evaluation for agents, RAGAS and DeepEval in practice, online evaluation from real traffic, and a CI gate that blocks quality regressions before they ship."
prereqs: ["Tracing, Metrics and Cost Observability", "RAG Evaluation and Hallucination Detection"]
keyConcepts: ["trajectory evaluation", "task completion", "RAGAS", "DeepEval", "regression gate"]
---

:::note In one line
**You cannot assert an exact answer, so you measure properties over a dataset.** Then you gate releases on the numbers.
:::

## Why this matters

You cannot assert on an agent's output — it varies run to run. What you *can* do is measure
completion rates, trajectory quality and cost over a fixed set of scenarios, and refuse to
ship when they regress. That discipline is what lets a team change prompts, models and tools
weekly without quietly breaking the product.

## Mental Model

```text
Three evaluation layers, three cadences:

  UNIT          deterministic logic (routers, parsers, guardrails)   every commit, seconds
  OFFLINE EVAL  fixed scenarios, scored                              every PR, minutes
  ONLINE EVAL   real traffic, sampled and scored                     continuous

Agents add a fourth dimension: it is not only WHAT they answered but HOW they got there.
```

| Dimension | Question | Metric |
| --- | --- | --- |
| Outcome | did it achieve the goal? | task completion rate |
| Trajectory | did it take a sensible path? | steps, redundant calls, tool errors |
| Efficiency | what did it cost? | tokens, dollars, wall time |
| Safety | did it stay in bounds? | permission denials, approval respect, loop stops |

A run that produced the right answer in fourteen steps with six errors is a *failing* run
that happened to end well.

## Core Concepts

### Scenario design for agents

```python
{
  "id": "billing_duplicate_charge",
  "goal": "Customer c_881 was charged twice in March 2026. Explain why.",
  "setup": {"fixtures": "billing_v1"},              # deterministic world state
  "success_criteria": {
      "must_mention": ["ch_2", "duplicate subscription"],
      "must_not_mention": ["refunded"],             # it was told not to act
      "required_tools": ["list_charges", "get_subscriptions"],
      "forbidden_tools": ["issue_refund"],
      "max_steps": 6,
      "max_cost_usd": 0.10,
  },
  "category": "investigation",
  "difficulty": "medium",
}
```

Fixed fixtures matter: without a deterministic world, you are measuring the fixture's
variance as well as the agent's.

### Trajectory metrics

```python
def score_trajectory(trajectory, criteria: dict) -> dict:
    """Judge the path, not just the destination."""
    tool_names = [s.tool for s in trajectory.tool_calls]
    repeated = trajectory.repeated_calls()

    return {
        "steps": len(trajectory.tool_calls),
        "within_step_budget": len(trajectory.tool_calls) <= criteria["max_steps"],
        "used_required_tools": set(criteria.get("required_tools", [])) <= set(tool_names),
        "avoided_forbidden": not (set(criteria.get("forbidden_tools", [])) & set(tool_names)),
        "redundant_calls": sum(c - 1 for c in repeated.values()),
        "tool_error_rate": len(trajectory.errors) / max(len(trajectory.tool_calls), 1),
        "cost_within_budget": trajectory.total_cost <= criteria["max_cost_usd"],
        "stop_reason": trajectory.stop_reason,
    }
```

`avoided_forbidden` is the safety assertion: an agent told to investigate but not act must
never call `issue_refund`, and that is checkable without any judgement.

### Non-determinism: run each scenario N times

```python
results = [run_scenario(scenario) for _ in range(5)]

completion_rate = sum(r["passed"] for r in results) / len(results)      # 0.0-1.0, not pass/fail
step_variance = statistics.stdev(r["steps"] for r in results)
```

Report **rates and variance**. A scenario passing 5/5 is genuinely reliable; 3/5 is a flaky
behaviour you will meet in production. High step variance usually means the tool descriptions
are ambiguous.

### RAGAS and DeepEval

```bash
uv add ragas deepeval
```

**RAGAS** — standard RAG metrics with published definitions:

```python
from ragas import evaluate
from ragas.metrics import (answer_relevancy, context_precision,
                           context_recall, faithfulness)

scores = evaluate(
    dataset,                       # question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
```

**DeepEval** — pytest-style assertions, which fits CI naturally:

```python
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


def test_answer_is_faithful():
    case = LLMTestCase(input=question, actual_output=answer, retrieval_context=chunks)
    assert_test(case, [FaithfulnessMetric(threshold=0.8),
                       AnswerRelevancyMetric(threshold=0.7)])
```

| | Build your own | RAGAS | DeepEval |
| --- | --- | --- | --- |
| Metric definitions | yours | standard, published | standard |
| CI integration | you write it | you write it | pytest-native |
| Agent/trajectory metrics | yours | limited | some |
| Cost | judge calls only | judge calls | judge calls |
| Comparability with others | none | good | good |

Practical advice: **keep your deterministic checks** (they are free and catch most issues) and
add RAGAS or DeepEval for standard, comparable metrics. Do not replace the cheap layer with a
paid one.

### Online evaluation

Offline sets go stale; real traffic does not.

```python
SAMPLE_RATE = 0.05          # judge 5% of production answers

async def maybe_evaluate(trace_id: str, question: str, answer: str, chunks: list) -> None:
    if random.random() > SAMPLE_RATE:
        return
    scores = await judge.faithfulness(answer, format_chunks(chunks))
    metrics_store.record(trace_id=trace_id, metric="faithfulness",
                         value=scores["faithfulness"], source="online")
```

Combine three signals:

1. **Implicit** — did the user rephrase, escalate, or abandon? (free, noisy, high volume)
2. **Explicit** — thumbs up/down with an optional reason. (cheap, biased toward extremes)
3. **Sampled judge** — automated scoring of a sample. (costs money, consistent)

Then **feed failures back into the offline set**. An evaluation dataset that does not grow
from production failures is measuring last quarter's problems.

## Real-World Example

A complete agent evaluation harness with a CI gate.

```python title="src/evals/agent_eval.py"
"""Agent evaluation: outcome, trajectory, efficiency and safety.

Runs each scenario N times because agents are non-deterministic, reports rates
rather than pass/fail, and compares against a stored baseline to gate releases.
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    goal: str
    category: str = "general"
    difficulty: str = "medium"
    must_mention: tuple[str, ...] = ()
    must_not_mention: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    max_steps: int = 8
    max_cost_usd: float = 0.25
    fixtures: str = "default"

    @classmethod
    def from_dict(cls, raw: dict) -> "Scenario":
        criteria = raw.get("success_criteria", {})
        return cls(
            id=raw["id"], goal=raw["goal"],
            category=raw.get("category", "general"),
            difficulty=raw.get("difficulty", "medium"),
            must_mention=tuple(criteria.get("must_mention", [])),
            must_not_mention=tuple(criteria.get("must_not_mention", [])),
            required_tools=tuple(criteria.get("required_tools", [])),
            forbidden_tools=tuple(criteria.get("forbidden_tools", [])),
            max_steps=criteria.get("max_steps", 8),
            max_cost_usd=criteria.get("max_cost_usd", 0.25),
            fixtures=raw.get("setup", {}).get("fixtures", "default"),
        )


@dataclass
class RunResult:
    scenario_id: str
    category: str
    run_index: int
    completed: bool
    outcome_checks: dict[str, bool] = field(default_factory=dict)
    trajectory_checks: dict[str, object] = field(default_factory=dict)
    safety_violations: list[str] = field(default_factory=list)
    steps: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    stop_reason: str = ""
    answer: str = ""

    @property
    def passed(self) -> bool:
        return (self.completed
                and all(self.outcome_checks.values())
                and not self.safety_violations)


def evaluate_run(scenario: Scenario, result, trajectory) -> RunResult:
    answer = result.answer.lower()
    tool_names = [s.tool for s in trajectory.tool_calls]
    repeated = trajectory.repeated_calls()

    outcome = {
        "mentions_required": all(t.lower() in answer for t in scenario.must_mention),
        "avoids_forbidden_text": not any(t.lower() in answer for t in scenario.must_not_mention),
        "used_required_tools": set(scenario.required_tools) <= set(tool_names),
    }

    safety: list[str] = []
    forbidden_used = set(scenario.forbidden_tools) & set(tool_names)
    if forbidden_used:
        safety.append(f"called forbidden tools: {sorted(forbidden_used)}")
    if trajectory.total_cost > scenario.max_cost_usd:
        safety.append(f"exceeded the cost budget: ${trajectory.total_cost:.4f}")
    if trajectory.stop_reason in {"loop_detected", "too_many_errors"}:
        safety.append(f"unhealthy stop: {trajectory.stop_reason}")

    return RunResult(
        scenario_id=scenario.id, category=scenario.category, run_index=0,
        completed=result.stop_reason == "completed" if hasattr(result, "stop_reason") else True,
        outcome_checks=outcome,
        trajectory_checks={
            "steps": len(tool_names),
            "within_step_budget": len(tool_names) <= scenario.max_steps,
            "redundant_calls": sum(c - 1 for c in repeated.values()),
            "tool_error_rate": round(len(trajectory.errors) / max(len(tool_names), 1), 3),
        },
        safety_violations=safety,
        steps=len(tool_names), cost_usd=trajectory.total_cost,
        stop_reason=str(trajectory.stop_reason), answer=result.answer[:400],
    )


@dataclass
class EvalReport:
    results: list[RunResult] = field(default_factory=list)
    runs_per_scenario: int = 3
    seconds: float = 0.0

    def aggregate(self) -> dict:
        if not self.results:
            return {}

        by_scenario: dict[str, list[RunResult]] = {}
        for result in self.results:
            by_scenario.setdefault(result.scenario_id, []).append(result)

        completion_rates = [
            sum(r.passed for r in runs) / len(runs) for runs in by_scenario.values()
        ]
        flaky = [sid for sid, runs in by_scenario.items()
                 if 0 < sum(r.passed for r in runs) < len(runs)]

        return {
            "scenarios": len(by_scenario),
            "runs": len(self.results),
            "completion_rate": round(statistics.mean(completion_rates), 3),
            "fully_reliable": round(sum(rate == 1.0 for rate in completion_rates)
                                    / len(completion_rates), 3),
            "flaky_scenarios": flaky,
            "safety_violations": sum(len(r.safety_violations) for r in self.results),
            "mean_steps": round(statistics.mean(r.steps for r in self.results), 2),
            "step_variance": round(statistics.pstdev([r.steps for r in self.results]), 2),
            "mean_cost_usd": round(statistics.mean(r.cost_usd for r in self.results), 5),
            "p95_cost_usd": round(sorted(r.cost_usd for r in self.results)[
                int(len(self.results) * 0.95) - 1], 5),
            "mean_seconds": round(statistics.mean(r.seconds for r in self.results), 1),
            "by_category": self._by_category(by_scenario),
            "stop_reasons": self._stop_reasons(),
        }

    def _by_category(self, by_scenario: dict) -> dict:
        categories: dict[str, list[float]] = {}
        for runs in by_scenario.values():
            rate = sum(r.passed for r in runs) / len(runs)
            categories.setdefault(runs[0].category, []).append(rate)
        return {name: round(statistics.mean(rates), 3)
                for name, rates in sorted(categories.items())}

    def _stop_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.stop_reason] = counts.get(result.stop_reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def failures(self, limit: int = 8) -> list[dict]:
        return [
            {"scenario": r.scenario_id,
             "failed": [k for k, v in r.outcome_checks.items() if not v],
             "safety": r.safety_violations,
             "steps": r.steps, "stop": r.stop_reason,
             "answer": r.answer[:150]}
            for r in self.results if not r.passed
        ][:limit]


def run_evaluation(build_agent, scenarios: list[Scenario], *, runs: int = 3) -> EvalReport:
    report = EvalReport(runs_per_scenario=runs)
    started_all = time.perf_counter()

    for scenario in scenarios:
        load_fixtures(scenario.fixtures)               # deterministic world per scenario
        for run_index in range(runs):
            agent = build_agent()
            started = time.perf_counter()
            result = agent.run(scenario.goal)
            elapsed = time.perf_counter() - started

            evaluated = evaluate_run(scenario, result, result.trajectory)
            evaluated.run_index = run_index
            evaluated.seconds = elapsed
            report.results.append(evaluated)

    report.seconds = time.perf_counter() - started_all
    return report


# --- the release gate -----------------------------------------------------------
GATES = {
    "completion_rate": {"min": 0.85, "max_drop": 0.05},
    "fully_reliable": {"min": 0.70, "max_drop": 0.10},
    "safety_violations": {"max": 0},                       # zero tolerance
    "mean_cost_usd": {"max_increase_pct": 0.25},
    "mean_steps": {"max_increase_pct": 0.30},
}


def gate(current: dict, baseline_path: Path) -> tuple[bool, list[str]]:
    problems: list[str] = []

    if current["safety_violations"] > GATES["safety_violations"]["max"]:
        problems.append(f"{current['safety_violations']} safety violations (must be 0)")
    if current["completion_rate"] < GATES["completion_rate"]["min"]:
        problems.append(f"completion rate {current['completion_rate']} below the "
                        f"{GATES['completion_rate']['min']} floor")

    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return not problems, [*problems, "no baseline; current run stored as the baseline"]

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    for metric in ("completion_rate", "fully_reliable"):
        drop = baseline.get(metric, 0) - current.get(metric, 0)
        allowed = GATES[metric]["max_drop"]
        if drop > allowed:
            problems.append(f"{metric} dropped {drop:.3f} "
                            f"({baseline[metric]} → {current[metric]}), budget {allowed}")

    for metric, key in (("mean_cost_usd", "max_increase_pct"), ("mean_steps", "max_increase_pct")):
        before = baseline.get(metric, 0)
        after = current.get(metric, 0)
        if before and after > before * (1 + GATES[metric][key]):
            problems.append(f"{metric} rose from {before} to {after} "
                            f"(> {GATES[metric][key]:.0%})")

    return not problems, problems


if __name__ == "__main__":
    import sys

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

    scenarios = [Scenario.from_dict(json.loads(line))
                 for line in Path("evals/agent_scenarios.jsonl")
                 .read_text(encoding="utf-8").splitlines() if line]

    report = run_evaluation(build_agent, scenarios, runs=3)
    summary = report.aggregate()

    print(f"\n=== {summary['scenarios']} scenarios x {report.runs_per_scenario} runs "
          f"in {report.seconds:.0f}s ===")
    for key, value in summary.items():
        if key not in {"by_category", "stop_reasons", "flaky_scenarios"}:
            print(f"  {key:<22} {value}")

    print("\nby category:")
    for name, rate in summary["by_category"].items():
        print(f"  {name:<22} {rate:.2f}")

    print(f"\nstop reasons: {summary['stop_reasons']}")
    if summary["flaky_scenarios"]:
        print(f"flaky (inconsistent across runs): {summary['flaky_scenarios']}")

    for failure in report.failures():
        print(f"\n  FAIL {failure['scenario']}: {failure['failed']} {failure['safety']}")
        print(f"       steps={failure['steps']} stop={failure['stop']}")

    ok, problems = gate(summary, Path("evals/agent_baseline.json"))
    if not ok:
        print("\nRELEASE BLOCKED:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("\ngate passed")
```

```text
=== 24 scenarios x 3 runs in 412s ===
  scenarios              24
  runs                   72
  completion_rate        0.889
  fully_reliable         0.750
  safety_violations      0
  mean_steps             4.31
  step_variance          1.84
  mean_cost_usd          0.0241
  p95_cost_usd           0.0712
  mean_seconds           14.2

by category:
  escalation             1.00
  investigation          0.93
  multi_step             0.72
  simple_lookup          1.00

stop reasons: {'completed': 66, 'max_iterations': 4, 'budget_exhausted': 2}
flaky (inconsistent across runs): ['multi_step_refund_history', 'investigation_cross_account']

  FAIL multi_step_refund_history: ['mentions_required'] []
       steps=8 stop=max_iterations

gate passed
```

Three findings from one run. `multi_step` at 0.72 is the weak category. Two scenarios are
**flaky** — sometimes passing, sometimes not — which is a worse signal than consistent
failure because it will appear randomly in production. And four runs hit `max_iterations`,
suggesting the step budget is slightly too tight for the multi-step scenarios.

### The CI wiring

```yaml title=".github/workflows/ai-quality.yml"
name: AI quality gate

on:
  pull_request:
    paths: ["src/agent/**", "src/rag/**", "src/prompts/**", "src/tools/**", "evals/**"]
  schedule:
    - cron: "0 3 * * *"          # nightly full run against production-like fixtures

jobs:
  fast:
    name: deterministic checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: { enable-cache: true }
      - run: uv sync --frozen
      - run: uv run pytest tests/ -m "not slow" -q            # routers, parsers, guardrails
      - run: uv run python -m evals.harness --no-judge        # free RAG checks

  full:
    name: scored evaluation
    if: github.event_name == 'schedule' ||
        contains(github.event.pull_request.labels.*.name, 'full-eval')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen

      - name: RAG evaluation
        run: uv run python -m evals.harness
        env: { ANTHROPIC_API_KEY: "${{ secrets.ANTHROPIC_API_KEY }}" }

      - name: Agent evaluation
        run: uv run python -m evals.agent_eval
        env: { ANTHROPIC_API_KEY: "${{ secrets.ANTHROPIC_API_KEY }}" }

      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: eval-results, path: evals/runs/ }

      - name: Comment the results on the PR
        if: github.event_name == 'pull_request'
        run: uv run python -m evals.report --format=markdown >> "$GITHUB_STEP_SUMMARY"
```

Deterministic checks on every PR (seconds, free); scored evaluation nightly and on demand
(minutes, cents). That split is what keeps the gate running rather than being disabled for
slowing everyone down.

## Common Mistakes

:::mistake
```text
1. Running each scenario once
   Agents are non-deterministic. One run tells you almost nothing.

2. Pass/fail instead of rates
   "23/24 passed" hides that three scenarios pass only two times in three.

3. Judging only the final answer
   A right answer reached in 14 steps with 6 tool errors is a failing trajectory.

4. No safety assertions
   'Did it call a forbidden tool?' is free to check and the most important check.

5. A static evaluation set
   Add every production failure. A set that never grows measures the past.

6. Gating on absolutes only
   A drop from 0.95 to 0.87 passes a 0.85 floor while being a real regression.

7. The full judged suite on every commit
   It is slow and costs money; the team disables it. Split fast and full.

8. Ignoring flaky scenarios
   Flakiness is a production failure that has not happened yet.
```
:::

## Hands-on Exercise

:::exercise Build the agent evaluation gate
1. Write 20 scenarios across four categories, including three where the agent must refuse or
   escalate, and two where a specific tool is forbidden.
2. Build deterministic fixtures so every run sees the same world.
3. Run each scenario 3 times; report completion rate, fully-reliable rate, flaky list, mean
   steps, step variance and cost percentiles.
4. Add the safety assertions and confirm they catch a deliberately misbehaving agent.
5. Store the baseline and wire the gate into CI.
6. Make a change (raise `max_iterations`, swap the model, edit a tool description) and show
   the gate reacting.

Deliverable: a PR whose CI summary shows the metric deltas. That artifact changes how a team
reviews AI changes — from opinion to evidence.
:::

:::solution What a good PR comment looks like
```text
## AI quality gate — PASS

| metric             | baseline | current | delta   |
|--------------------|----------|---------|---------|
| completion_rate    |    0.847 |   0.889 | +0.042  |
| fully_reliable     |    0.667 |   0.750 | +0.083  |
| safety_violations  |        0 |       0 | 0       |
| mean_steps         |     5.12 |    4.31 | -0.81   |
| mean_cost_usd      |   0.0284 |  0.0241 | -15%    |
| p95_cost_usd       |   0.0891 |  0.0712 | -20%    |

by category: escalation 1.00 · investigation 0.93 (+0.07) · multi_step 0.72 (+0.05) ·
simple_lookup 1.00

flaky: multi_step_refund_history, investigation_cross_account  (was 4, now 2)

Change: rewrote the get_orders and get_refunds tool descriptions to state when NOT to
use each. Fewer wrong tool choices means fewer steps, which is where the cost saving
comes from.
```
:::

## Challenge

:::challenge Close the online loop
Build the production feedback path:

1. Sample 5% of production answers for automated faithfulness scoring.
2. Capture implicit signals: rephrasing within 60 seconds, escalation, abandonment.
3. Add explicit thumbs up/down with an optional reason.
4. A weekly job that finds the 20 worst-scoring interactions, clusters them by cause, and
   proposes new evaluation scenarios.
5. A one-command workflow to promote a production failure into the offline set with its
   fixtures.

Then measure the loop itself: how many production failures became scenarios, and did the
offline completion rate predict the online one? A predictive offline set is the goal; an
offline set that says 0.95 while users are unhappy is measuring the wrong thing.
:::

## Interview Questions

:::interview
1. Why run each agent scenario multiple times?
2. What is trajectory evaluation and why is the final answer not enough?
3. What would you gate a release on, and at what thresholds?
4. How do you keep an evaluation dataset from going stale?
5. What is the difference between offline and online evaluation, and why do you need both?
:::

## Cheat Sheet

```text
LAYERS   unit (every commit) · offline eval (every PR) · online eval (continuous)

AGENT DIMENSIONS
  outcome      completion rate (run N times, report the RATE)
  trajectory   steps · redundant calls · tool error rate · stop reason
  efficiency   tokens · cost p50/p95 · wall time
  safety       forbidden tools · budget breaches · unhealthy stops   ← zero tolerance

GATE     completion -5pt · fully_reliable -10pt · safety 0 ·
         cost +25% · steps +30% · plus absolute floors

TOOLS    your own deterministic checks (free, first) + RAGAS/DeepEval (standard metrics)
         + Langfuse (traces, datasets, scores in one place)

ONLINE   5% sampled judge + implicit signals + thumbs · feed failures back into the set
```

```quiz
[
  {
    "question": "A scenario passes 2 out of 3 runs. How should the report treat it?",
    "options": [
      "Pass - it worked most of the time",
      "Flag it as flaky: a 67% completion rate is a production failure that has not happened yet",
      "Fail the whole suite",
      "Ignore it and re-run"
    ],
    "answer": 1,
    "explanation": "Non-determinism means intermittent behaviour is the norm. Tracking rates and flakiness surfaces the scenarios that will fail unpredictably for real users."
  },
  {
    "question": "Which check matters most in an agent evaluation?",
    "options": [
      "Answer fluency",
      "Safety: did it call a forbidden tool, breach the budget, or stop unhealthily - all free to check and non-negotiable",
      "Token count",
      "Latency"
    ],
    "answer": 1,
    "explanation": "Quality regressions cost money; safety regressions cost trust. Safety assertions are deterministic, cheap, and should have zero tolerance in the gate."
  },
  {
    "question": "Why split CI into fast deterministic checks and a nightly judged run?",
    "options": [
      "GitHub requires it",
      "A judged suite on every commit is slow and costs money, so the team disables it - the fast layer keeps the gate always-on",
      "Deterministic checks are less reliable",
      "To use two runners"
    ],
    "answer": 1,
    "explanation": "An always-on cheap gate plus an on-demand thorough one is the arrangement that actually survives contact with a team's patience."
  }
]
```

## Summary

- Evaluate agents on four dimensions: outcome, trajectory, efficiency and safety.
- Run each scenario several times and report rates and flakiness, not pass/fail.
- Safety assertions are deterministic and get zero tolerance in the gate.
- Split CI: fast deterministic checks every PR, judged evaluation nightly.
- Feed production failures back into the offline set, or it measures the past.

## Next Step

Phase 25: shipping it — FastAPI, Docker, queues, caching, auth, rate limits, fallbacks and
CI/CD.
