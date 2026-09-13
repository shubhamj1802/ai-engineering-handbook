---
title: Interview Preparation for AI Engineering
order: 2
difficulty: Production
duration: 16
badges: ["Reference", "Read once, refer often"]
summary: "What gets asked in AI engineering interviews at each level — with model answers, the system-design frameworks, and the questions you should ask them."
prereqs: ["Roadmaps — Learning, Dependencies, Skills and Projects"]
keyConcepts: ["interview structure", "system design", "trade-off reasoning", "portfolio"]
---

## The shape of the process

| Stage | Focus | How to prepare |
| --- | --- | --- |
| Screen (30 min) | vocabulary, experience, honesty | be precise about what you built versus used |
| Coding (60 min) | Python, data structures, testing | Phases 1–2; practise writing tests first |
| ML/AI depth (60 min) | RAG, agents, evaluation, trade-offs | Phases 10–15, 24 |
| System design (60 min) | architecture, scale, cost, safety | this page's framework |
| Practical/take-home | building something small and complete | ship it with a README and an evaluation |
| Behavioural | incidents, disagreements, judgement | have three specific stories ready |

The single strongest signal across all of them: **you can say what you measured**. Candidates
who report numbers — recall@5, cost per request, p95, containment rate — stand out
immediately from candidates who report vibes.

## The questions that separate candidates

### 1. "Walk me through a RAG system"

Weak answers describe the diagram. Strong answers describe the **decisions and their
evidence**:

> "Ingest is structure-aware chunking at around 500 tokens with 15% overlap — we measured
> recall@5 across four chunk sizes and 512 with heading context headers was best at 0.93. At
> query time we retrieve 12 candidates with hybrid search, because BM25 took identifier
> recall from 0.40 to 0.93, then rerank to 5 with a cross-encoder, which moved MRR from 0.70
> to 0.87 for 44 ms. Generation is constrained: context only, citations required, refusal
> permitted. Then we verify in code that every cited id was actually retrieved — a
> fabricated citation blocks the answer. We measure retrieval and generation separately:
> recall@5 at 0.94 and faithfulness at 0.95, gated in CI against a stored baseline."

Every claim has a number, and the numbers explain the design.

### 2. "When would you not use an agent?"

> "Whenever I can draw the flowchart before the request arrives. An agent costs 10–30× a
> workflow in latency and money and adds failure modes — loops, drift, partial completion.
> On a support system we measured a routed workflow resolving 89% at $0.006 and 2.4 s
> against an agent resolving 95% at $0.12 and 18.7 s. We shipped the workflow with agent
> escalation for the 11% it could not resolve: 96% resolution at $0.014. Making the expensive
> path rarer beat making it better."

### 3. "How do you know your system is any good?"

> "Retrieval and generation separately, then end to end. Deterministic checks first because
> they are free — citation validity, correct refusal on unanswerable questions, required
> terms, latency. Then a claim-level faithfulness judge on what code cannot check, calibrated
> against human labels — ours agrees at kappa 0.71. 120 cases across six categories including
> unanswerable, adversarial and access-control. It runs in CI: fast deterministic checks on
> every PR, judged suite nightly, and the gate blocks a merge if pass rate drops more than
> three points or citation validity ever leaves 1.0."

### 4. "How do you stop prompt injection?"

The answer that fails is "input filtering". The answer that succeeds is architectural:

> "You do not stop it, you contain it. Detection catches the obvious cases and is telemetry,
> not a control. The controls are: least privilege, so the agent has no tool that can cause
> the harm; trust separation, so retrieved content never shares a context with privileged
> tools; deterministic output validation; and human approval for irreversible actions. In our
> red-team suite 18 of 20 attacks were blocked at the input layer, and the two that got
> through — a base64-encoded instruction and one split across two chunks — could not do
> anything, because the agent's tool list was read-only and the output was citation-verified."

### 5. "Design a customer support assistant for 10,000 tickets a day"

Use this framework, in this order:

```text
1. CLARIFY      what counts as success? what is the cost of a wrong answer?
                what may it do autonomously? what data exists?
2. METRICS      containment rate · wrong-action rate · p95 · cost per conversation
3. DATA         documents, account APIs, historical tickets, permissions
4. ARCHITECTURE route → RAG / tools / action / escalate, with the safety boundary drawn
5. SAFETY       what is irreversible? where is the approval gate? what tools exist at all?
6. EVALUATION   what dataset, what categories, what gate
7. SCALE        10k/day = 7/min; peaks 3x; cost = 10k x $0.03 = $300/day
8. FAILURE      degraded mode per dependency; what the user sees
9. ROLLOUT      shadow mode → question intent → account → actions
```

Spending the first five minutes on clarification and metrics, before drawing anything, is
what senior candidates do and junior candidates skip.

## Technical questions by area

:::interview Python and engineering
1. What happens with a mutable default argument, and why?
2. `is` versus `==`, and when each is correct.
3. What makes a function easy to unit test?
4. Explain async/await. When does it not help?
5. How do you test code that calls an external API?
6. What is dependency injection and why does it matter here?
7. Why pin framework versions in an AI project?
:::

:::interview Data and ML
1. Give three concrete examples of data leakage.
2. Why is accuracy misleading on imbalanced data? What do you use?
3. How do you choose a classification threshold?
4. Explain the bias–variance trade-off using train and test error.
5. When does classical ML beat an LLM in production?
6. What does `groupby().transform()` do that `agg()` does not?
:::

:::interview LLMs and RAG
1. What comes back from an LLM call besides the text?
2. Why is constrained structured output better than asking for JSON?
3. How does prompt caching work and what silently defeats it?
4. Why does a bigger context window not automatically help?
5. How do you choose a chunk size?
6. Why combine BM25 with vector search?
7. What is reciprocal rank fusion and why not normalise scores instead?
8. An answer is wrong — how do you tell whether retrieval or generation failed?
:::

:::interview Agents and production
1. Define an agent without using the word "autonomous".
2. What are the stop conditions of an agent loop?
3. Why must tool errors be returned rather than raised?
4. How do you guarantee an agent never takes an irreversible action unsupervised?
5. What is a reducer in a state graph and when do you need one?
6. What does `interrupt()` do, and what must you re-check on resume?
7. Why is self-reflection ineffective for factual errors?
8. What would you measure to decide between a pipeline and a supervisor?
9. What must never appear in a trace?
10. What is your degraded mode when the vector store is down?
:::

## Coding exercises you should be able to do

| Exercise | Tests | Phase |
| --- | --- | --- |
| Chunk text with overlap, handling edge cases | string handling, boundaries | 1 |
| Implement cosine similarity and top-k without libraries | NumPy, algorithms | 3 |
| Reciprocal rank fusion of two ranked lists | dicts, sorting | 13 |
| A retry decorator with exponential backoff and jitter | decorators, error taxonomy | 2 |
| Parse model output that may be wrapped in markdown fences | robust parsing | 10 |
| A bounded agent loop with stop conditions | control flow, safety | 14 |
| Verify citations against a retrieved set | sets, guardrails | 12 |
| A token-budget context assembler | greedy selection, testing | 10 |

Practise writing the **tests first**. In a live exercise, starting with two test cases before
the implementation reads as senior, and it usually produces a better solution.

## The take-home

If you get one, the differentiators are almost never the model work:

```text
[ ] a README: problem, approach, results, limitations, how to run it
[ ] an evaluation with numbers, including what did not work
[ ] tests, including one for a failure case
[ ] error handling and a timeout on every external call
[ ] no hard-coded secrets; a .env.example
[ ] cost and latency measured and stated
[ ] an honest "what I would do with another week"
```

A modest system with an evaluation table beats an ambitious one without. Reviewers are
looking for judgement, not scope.

## Questions to ask them

These tell you whether the role is real, and signal that you know what matters:

1. How do you evaluate your AI features today? Is it in CI?
2. What is your cost per request, and who watches it?
3. What happens when the model provider has an outage?
4. Which decisions can the system take without a human?
5. How do you handle prompt injection from user-supplied documents?
6. What is the split between AI work and ordinary backend work in this role?
7. What was your last AI-related incident, and what changed afterwards?

Question 7 is the most revealing. A team with a good answer has been running something real.

## Behavioural stories to prepare

Have three specific, measured stories ready:

| Prompt | What to include |
| --- | --- |
| "A system you built that failed" | the failure, how you detected it, the fix, what changed structurally |
| "A disagreement about technical direction" | the trade-off, how you resolved it with data, what you conceded |
| "Something you shipped that you would build differently" | the specific decision, the cost it caused, what you learned |

Use numbers. "It was slow" is weak; "p95 was 8 seconds because we were retrieving 40 chunks;
reranking to 5 took it to 2.3 seconds with better answers" is a story.

## Red flags to avoid

```text
1. "We use LangChain" as an architecture description
   Say what the system does and why the framework was the right fit.

2. No numbers anywhere
   The single clearest separator between levels.

3. Claiming an agent for what was a workflow
   Precision here signals that you understand the trade-off.

4. "We tested it manually"
   For an AI system that means untested.

5. Defending a technology choice on preference
   Every choice should have a trade-off you can state.

6. No opinion on cost
   Production AI is a cost-engineering discipline.
```

## Summary

- Numbers are the differentiator: recall, faithfulness, p95, cost, containment.
- Clarify and define metrics before designing; state safety boundaries explicitly.
- For injection, permissions and guardrails — architecture, not filters.
- In take-homes, the README, the evaluation and the honest limitations matter more than
  scope.
- Ask what happens during an outage and what their last incident was.

## Next Step

The production checklist: everything to verify before an AI system meets real users.
