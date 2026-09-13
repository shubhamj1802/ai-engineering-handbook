---
title: The Production Checklist
order: 3
difficulty: Production
duration: 12
badges: ["Reference", "Production", "Security"]
summary: "Everything to verify before an AI system meets real users — correctness, safety, cost, operations and compliance — with the specific test or measurement that proves each item."
prereqs: ["Production AI Systems — FastAPI, Docker, Queues and Resilience"]
keyConcepts: ["readiness review", "safety verification", "cost control", "incident response"]
---

## How to use this

Each item names **the artifact that proves it**. "We have guardrails" is not evidence; "here
is the red-team suite output showing 18/20 blocked and a 5% false-positive rate" is. Work
through it before launch, and again before each significant expansion of scope.

Items marked **BLOCKING** should stop a launch.

## 1. Correctness and quality

```text
[ ] BLOCKING  An evaluation set exists with at least 50 cases
              → evals/dataset.jsonl, categories documented

[ ] BLOCKING  The set includes unanswerable cases (≥10%)
              → correct-refusal rate reported

[ ] The set includes adversarial cases (injection, social engineering)
              → red-team results with a block rate and a false-positive rate

[ ] Retrieval and generation are measured separately
              → recall@k and faithfulness reported independently

[ ] BLOCKING  Citation validity is 100% on the evaluation set
              → a fabricated citation blocks the answer, with a test proving it

[ ] Results are broken down by category
              → per-category table; no category is catastrophically worse

[ ] A baseline is recorded and a regression gate runs in CI
              → evals/baseline.json plus a CI job that fails on regression

[ ] Agent scenarios run multiple times, and flakiness is tracked
              → completion RATE, not pass/fail

[ ] A human has read 30 real outputs end to end
              → notes; automated metrics miss tone, formatting and usefulness
```

## 2. Safety and guardrails

```text
[ ] BLOCKING  Every irreversible action is gated or absent
              → the tool list the model actually receives, plus a test

[ ] BLOCKING  Approval identity comes from the authenticated session
              → not from the model, not from the request body

[ ] Preconditions are re-validated after an approval resumes
              → test: approve, change the world, resume, assert it aborts

[ ] Input validated: schema, size, control characters
              → Pydantic models on every endpoint

[ ] Output validated: schema, citations, policy
              → guardrail suite with unit tests

[ ] Guardrail false-positive rate measured on legitimate traffic
              → < 2% on a 20-case legitimate set

[ ] Tools are least-privilege; read-only by default
              → permission classification per tool

[ ] Non-idempotent writes are never auto-retried
              → idempotency keys on every write tool

[ ] Prompt injection contained architecturally, not by detection alone
              → a written note on which layer stops which attack

[ ] Loop, cost and time caps enforced in code
              → tests that force each limit
```

## 3. Security and privacy

```text
[ ] BLOCKING  Secrets in a secret manager; none in code, images or logs
              → gitleaks in CI; grep the image layers

[ ] BLOCKING  Tenant isolation enforced in every data store
              → a randomised isolation test at volume (10k+ queries)

[ ] Another tenant's resource returns 404, not 403
              → existence must not leak

[ ] Access control enforced at retrieval, not in the prompt
              → tests with a user who should see nothing

[ ] Refusals do not reveal that a forbidden document exists
              → byte-identical refusals, with a test

[ ] Traces and logs contain metadata only
              → grep the trace store for an email and an API key: zero hits

[ ] PII detected and redacted before storage or third-party transmission
              → redaction unit tests

[ ] Data deletion reaches every store including vector indexes and caches
              → a deletion test asserting zero across all layers

[ ] Retention policy defined and enforced by a job
              → the policy, plus the job's last run

[ ] Container runs as non-root, image scanned, base image pinned
              → scan output in CI

[ ] Dependencies audited; lock file committed
              → uv.lock plus an audit step
```

## 4. Cost control

```text
[ ] BLOCKING  Per-user or per-tenant daily spend cap
              → enforced in Redis or the database, with a test

[ ] Cost measured and attributed per request, feature and tenant
              → a dashboard, not an estimate

[ ] Prompt caching verified working
              → cache_read_input_tokens > 0 in production traces

[ ] Answer caching with invalidation on content change
              → hit rate reported; staleness test

[ ] Model tier chosen per request class
              → routing rules, with measured quality impact

[ ] Cost alert at 50%, 80% and 100% of the monthly budget
              → alert configuration

[ ] The most expensive 1% of requests investigated
              → they are usually failures; often the largest saving

[ ] Batch API used for non-interactive work
              → ingestion and evaluation jobs
```

## 5. Reliability and operations

```text
[ ] BLOCKING  /health/live and /health/ready are separate
              → liveness must not touch dependencies

[ ] Every external call has a timeout
              → grep for calls without one

[ ] Retries only on retryable errors, with backoff and jitter
              → the retry policy, with a test on a 400

[ ] A written and tested degraded mode for every dependency
              → a test per dependency forcing the failure

[ ] Fallback model configured and exercised
              → a test that disables the primary

[ ] Graceful shutdown drains in-flight requests
              → preStop hook plus a drain test

[ ] Long work runs in background jobs, not the request path
              → 202 plus a job status endpoint

[ ] Queue depth and worker lag monitored
              → alerts

[ ] Load tested at 3× expected peak
              → p50/p95/p99, error rate, cost per 1k at that load

[ ] Rollback rehearsed, and its duration measured
              → the number, in the runbook
```

## 6. Observability

```text
[ ] BLOCKING  Every request has an id, returned to the caller and in every log line

[ ] Traces capture per-stage timing, tokens, cost and outcome
              → one example trace in the runbook

[ ] Percentile latency dashboards (p50, p95, p99), not means

[ ] Outcome distribution tracked (answered / no context / escalated / blocked)

[ ] Alerts defined with runbook entries
              → each alert names what to check and what to do

[ ] Prompt and model versions recorded per request
              → so a quality change maps to a specific change

[ ] Approval decisions audited, including rejections

[ ] Online evaluation sampling a fraction of production answers
              → the sample rate and where the scores land
```

## 7. Documentation and process

```text
[ ] README: what it does, how to run it, how to evaluate it
[ ] Architecture diagram, current and dated
[ ] Model card or assistant card: purpose, limits, intended use, known failure modes
[ ] Runbook: each alert, what it means, what to do
[ ] On-call rotation, and someone who knows the system is on it
[ ] A rollout plan with stages and go/no-go criteria at each
[ ] A clear statement of what the system must not be used for
[ ] An incident template, and one dry run
```

## 8. Human factors

```text
[ ] Users are told they are talking to an AI system
[ ] There is an obvious way to reach a human
[ ] The system's confidence is communicated honestly (citations, caveats, refusals)
[ ] Feedback (thumbs, reasons) is collected and actually read
[ ] Reviewer load estimated and sustainable (< 1 hour/day at expected volume)
[ ] Affected teams know it is launching, and what it will do
```

## The pre-launch review

Run this as a meeting with the checklist on screen. Four questions, answered with artifacts:

```text
1. WHAT DOES IT DO WRONG?
   Show the evaluation failures by category. A team that cannot answer this has
   not measured enough.

2. WHAT IS THE WORST THING IT CAN DO?
   Walk the tool list and the approval gates. Name the blast radius of each
   irreversible action.

3. WHAT HAPPENS WHEN IT BREAKS?
   Demonstrate a degraded mode. Show the rollback time.

4. WHAT WILL IT COST?
   Cost per request times expected volume, plus the cap that stops a runaway.
```

## The first-week watch list

```text
day 1   error rate · p95 · outcome distribution · cost per 1k
        read 20 real conversations end to end
day 2   guardrail firing rate and false positives
        approval queue latency
day 3   containment or answered rate versus the evaluation prediction
        the gap between them tells you how representative your dataset is
day 7   cost trend · cache hit rate · the 1% most expensive requests
        add every production failure to the evaluation set
```

The day-3 item is the most valuable: **if the offline evaluation said 0.93 and production
says 0.71, the dataset is wrong, not the system.** Fixing the dataset is the highest-value
work available at that moment.

## Summary

- Every checklist item names the artifact that proves it; claims without artifacts are not
  evidence.
- Blocking items concentrate on irreversible actions, tenant isolation, secrets, spend caps
  and citation validity.
- Rehearse the rollback and the degraded modes before launch, not during the incident.
- In week one, compare offline predictions with production reality and fix the dataset.

## Next Step

The cheat sheet and what to do after finishing the handbook.
