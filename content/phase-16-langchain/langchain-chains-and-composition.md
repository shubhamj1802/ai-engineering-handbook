---
title: "LangChain — Chains, Runnables and Composition"
order: 3
difficulty: Expert
duration: 20
badges: ["Hands-on", "Deep dive"]
summary: The pipe operator explained properly — what a Runnable is, how prompt | model | parser works, running steps in parallel, fallbacks and retries, and when a plain function is the better answer.
prereqs: ["LangChain — Models, Messages, Structured Output and LCEL"]
keyConcepts: ["Runnable", "LCEL", "RunnableParallel", "RunnablePassthrough", "with_fallbacks", "batch"]
---

:::note In one line
**Everything in LangChain is a Runnable, and Runnables pipe into each other.** Learn one
interface — `invoke`, `stream`, `batch` — and every piece composes with every other piece.
:::

:::warning Versions used on this page
`langchain` **1.4.0** · `langchain-core` **1.6.3**
:::

## Why this matters

You will read a lot of LangChain code that looks like this:

```python
chain = prompt | model | parser
```

That pipe is not magic syntax the framework invented for prompts. It is one protocol applied
consistently, and once you see it, the whole library stops looking like a pile of unrelated
classes.

The payoff is practical: anything that is a Runnable gets **streaming, batching, async and
retries for free**, without you writing them.

## The Runnable protocol

A Runnable is any object with these methods:

| Method | What it does |
| --- | --- |
| `invoke(input)` | one input, one output |
| `stream(input)` | one input, output in pieces |
| `batch([inputs])` | many inputs, run concurrently |
| `ainvoke` / `astream` / `abatch` | the async versions of all three |

Models are Runnables. Prompts are Runnables. Parsers, retrievers and **your own functions**
can all be Runnables. That is why they compose.

<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Diagram: a prompt template, a model and an output parser each implement the same Runnable interface, so piping them together produces a new Runnable with that same interface.">
  <defs>
    <marker id="rn-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <rect x="14" y="40" width="126" height="64" rx="9" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.7"/>
  <text class="dg-label" x="77" y="62" text-anchor="middle" fill="var(--accent-3)">prompt</text>
  <text class="dg-sub"   x="77" y="80" text-anchor="middle">dict in</text>
  <text class="dg-sub"   x="77" y="95" text-anchor="middle">messages out</text>
  <rect x="180" y="40" width="126" height="64" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.9"/>
  <text class="dg-label" x="243" y="62" text-anchor="middle" fill="var(--accent)">model</text>
  <text class="dg-sub"   x="243" y="80" text-anchor="middle">messages in</text>
  <text class="dg-sub"   x="243" y="95" text-anchor="middle">AIMessage out</text>
  <rect x="346" y="40" width="126" height="64" rx="9" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.7"/>
  <text class="dg-label" x="409" y="62" text-anchor="middle" fill="var(--accent-2)">parser</text>
  <text class="dg-sub"   x="409" y="80" text-anchor="middle">AIMessage in</text>
  <text class="dg-sub"   x="409" y="95" text-anchor="middle">your object out</text>
  <text class="dg-label" x="158" y="78" fill="var(--accent)">|</text>
  <text class="dg-label" x="324" y="78" fill="var(--accent)">|</text>
  <rect x="512" y="40" width="134" height="64" rx="9" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="2"/>
  <text class="dg-label" x="579" y="62" text-anchor="middle" fill="var(--ok)">one Runnable</text>
  <text class="dg-sub"   x="579" y="80" text-anchor="middle">same interface again</text>
  <text class="dg-sub"   x="579" y="95" text-anchor="middle">so it pipes further</text>
  <path class="dg-arrow" d="M472,72 L506,72" marker-end="url(#rn-a)"/>
  <rect x="14" y="132" width="632" height="44" rx="8" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-sub" x="30" y="152">Each output type must match the next input type. That is the only rule, and it is where</text>
  <text class="dg-sub" x="30" y="168">almost every confusing LCEL error comes from: a shape mismatch between two links.</text>
  <text class="dg-sub" x="14" y="202" fill="var(--ok)">Because the result is itself a Runnable, the composed chain gets invoke, stream, batch and async for free.</text>
  <text class="dg-sub" x="14" y="224">You did not implement streaming. You inherited it.</text>
</svg>
<figcaption>
<strong>The pipe just means "feed the output of the left into the right".</strong> The value
is that the result has the same interface, so it keeps composing — and gains batching and
streaming without extra work.
</figcaption>
</figure>

## A real chain, end to end

```python title="src/lc/chain_basic.py"
"""prompt | model | structured output - the shape you will write most often."""
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    """A support ticket, classified."""
    urgency: str = Field(description="one of: low, normal, high, critical")
    category: str = Field(description="billing, technical, account or other")
    summary: str = Field(max_length=120)
    needs_human: bool = Field(description="true if a person must look at this")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You classify support tickets for a {product} team. Be conservative: "
               "when unsure, mark needs_human as true."),
    ("human", "{ticket_text}"),
])

model = init_chat_model("anthropic:claude-sonnet-5")

# with_structured_output makes the model return a validated Ticket, not a string
chain = prompt | model.with_structured_output(Ticket)

result = chain.invoke({
    "product": "billing",
    "ticket_text": "I was charged twice this month and nobody has replied to my email.",
})

print(type(result).__name__)     # Ticket
print(result.urgency)            # high
print(result.needs_human)        # True
```

Notice what you did **not** write: no JSON parsing, no `try/except` around
`json.loads`, no checking that the fields exist. The Pydantic model is the contract and
LangChain enforces it.

:::tip Field descriptions are prompt engineering
`Field(description="one of: low, normal, high, critical")` is sent to the model as part of
the schema. Vague descriptions produce vague output. This is the highest-leverage place to
spend your wording effort — more than the system prompt.
:::

## Batching: the free speed-up

Because the chain is a Runnable, it can run many inputs concurrently with no extra code.

```python title="src/lc/batching.py"
tickets = [
    {"product": "billing", "ticket_text": "Charged twice."},
    {"product": "billing", "ticket_text": "How do I change my plan?"},
    {"product": "billing", "ticket_text": "URGENT: service is completely down."},
]

results = chain.batch(tickets, config={"max_concurrency": 5})

for ticket, result in zip(tickets, results, strict=True):
    print(f"{result.urgency:<10} {result.category:<12} {ticket['ticket_text'][:40]}")
```

```text
high       billing      Charged twice.
low        billing      How do I change my plan?
critical   technical    URGENT: service is completely down.
```

Three sequential calls at ~1.2 s each would be 3.6 s. Batched, it is about 1.3 s.

:::warning Always cap max_concurrency
Without it, `batch` will fire every request at once. Give it a thousand inputs and you will
hit the provider's rate limit, get a wall of 429s, and the retries will make it worse.
Pick a number your rate limit can absorb.
:::

## Running steps in parallel

`RunnableParallel` runs several branches on the same input and collects the results into a
dict. It is how you fan out without writing async code.

```python title="src/lc/parallel.py"
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model("anthropic:claude-sonnet-5")

summarise = ChatPromptTemplate.from_template("Summarise in one line:\n\n{text}") | model
sentiment = ChatPromptTemplate.from_template("One word - sentiment of this:\n\n{text}") | model
actions = ChatPromptTemplate.from_template("List any action items, or 'none':\n\n{text}") | model

analyse = RunnableParallel(
    summary=summarise,
    sentiment=sentiment,
    actions=actions,
)

out = analyse.invoke({"text": "The deployment failed twice last night. Customers noticed."})

print(out["summary"].text)
print(out["sentiment"].text)
print(out["actions"].text)
```

All three calls go out together, so the whole thing takes about as long as the slowest one
rather than the sum of all three.

:::performance Parallel is not always cheaper
It is faster, not cheaper. You still pay for three calls. If two of the three branches are
only occasionally needed, run them conditionally instead — and consider whether one prompt
asking for all three fields in a structured output would do the job for a third of the cost.
:::

## Passing the input through

Chains often need the original input **and** something derived from it. That is what
`RunnablePassthrough` is for.

```python title="src/lc/passthrough.py"
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# retriever is any Runnable that takes a string and returns documents
prepared = {
    "context": retriever,                    # the question goes to the retriever
    "question": RunnablePassthrough(),       # and also straight through, untouched
}

answer_prompt = ChatPromptTemplate.from_template(
    "Answer using only this context. If it is not there, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

rag_chain = prepared | answer_prompt | model

print(rag_chain.invoke("What is our refund window?").text)
```

That dict is the RAG pattern in three lines. The question is used twice: once to search,
once in the prompt.

## Your own functions in a chain

Any function can join a chain. Wrap it in `RunnableLambda`, or just use a plain function
where a Runnable is expected and LangChain coerces it.

```python title="src/lc/custom_step.py"
from langchain_core.runnables import RunnableLambda

def trim_context(inputs: dict) -> dict:
    """Keep the context under a token budget. Deterministic - no model needed."""
    docs = inputs["context"]
    budget, kept = 3000, []
    for doc in docs:
        cost = len(doc.page_content) // 4        # rough tokens
        if budget - cost < 0:
            break
        budget -= cost
        kept.append(doc)
    return {**inputs, "context": "\n\n".join(d.page_content for d in kept)}

rag_chain = prepared | RunnableLambda(trim_context) | answer_prompt | model
```

:::production Put deterministic work in plain functions
Trimming, formatting, filtering, deduplicating — none of that needs a model. Doing it in a
plain Python function is free, instant, testable and never wrong. A surprising amount of
"AI pipeline" work is like this.
:::

## Fallbacks and retries

Providers have bad days. Two lines make your chain survive them.

```python title="src/lc/resilience.py"
strong = init_chat_model("anthropic:claude-opus-5")
cheap = init_chat_model("anthropic:claude-haiku-4-5")

# if the first fails, try the next - transparently
resilient = strong.with_fallbacks([cheap])

# and retry transient errors before giving up on a model
robust = strong.with_retry(stop_after_attempt=3).with_fallbacks([cheap])

chain = prompt | robust.with_structured_output(Ticket)
```

| Wrapper | Handles |
| --- | --- |
| `.with_retry(...)` | transient failures of the **same** model — timeouts, 429, 503 |
| `.with_fallbacks([...])` | that model being unusable — try a different one |

:::danger A fallback changes your output quality silently
If `claude-opus-5` fails and `claude-haiku-4-5` answers, the request succeeds — at a
different quality level, and nothing in your response says so. Log which model actually
answered, or your evaluation numbers will drift for reasons you cannot explain.
:::

## Configuration at call time

`with_config` attaches metadata and limits that flow through the whole chain — invaluable
for tracing.

```python title="src/lc/config.py"
result = chain.invoke(
    {"product": "billing", "ticket_text": text},
    config={
        "run_name": "classify_ticket",            # shows up in traces
        "tags": ["classification", "v3"],
        "metadata": {"request_id": request_id, "tenant": tenant_id},
        "max_concurrency": 5,
    },
)
```

Every trace line for this request now carries your request id, which is the difference
between debugging in minutes and debugging in hours.

## When a chain is the wrong tool

:::mistake Piping things together to look clever
```python
# Three model calls where one would do
chain = classify | RunnableLambda(pick_template) | model | parser
```
If the steps are fixed and you control the branching, **a plain Python function is often
clearer**:

```python
def classify_ticket(text: str) -> Ticket:
    if len(text) < 20:                              # no model needed
        return Ticket(urgency="low", category="other", summary=text, needs_human=False)
    return (prompt | model.with_structured_output(Ticket)).invoke({"ticket_text": text})
```
You keep structured output where it earns its place and drop the ceremony everywhere else.
The pipe is a tool, not a style requirement.
:::

## Debugging a chain

When a chain misbehaves, find out what each link actually received.

```python title="src/lc/debug_chain.py"
# 1. Run the links separately
messages = prompt.invoke({"product": "billing", "ticket_text": text})
print(messages)                        # is the prompt what you expected?
raw = model.invoke(messages)
print(raw.text)                        # did the model answer sensibly?

# 2. See the whole structure
print(chain.get_graph().draw_ascii())

# 3. Watch events as they happen
for event in chain.stream_events({"product": "billing", "ticket_text": text}, version="v2"):
    if event["event"] in {"on_chain_start", "on_chat_model_end"}:
        print(event["event"], event.get("name"))
```

:::tip Nine times in ten it is a shape mismatch
`KeyError: 'context'` means the previous step did not produce a `context` key. Print the
intermediate value rather than re-reading the chain definition — the error message names the
key, and the print tells you what you actually had.
:::

## Hands-on Exercise

:::exercise Build a resilient parallel analyser
Write `src/lc/exercise_chain.py` that takes one support email and produces a single
validated object containing:

- a one-line summary
- a classification (urgency, category) as a Pydantic model
- a suggested reply draft

Requirements:

1. The classification and the reply draft must run **in parallel** with `RunnableParallel`
2. The classification must use `with_structured_output`
3. The whole chain must have a retry and a fallback to a cheaper model
4. Pass a `request_id` through `config["metadata"]`
5. Run it over a **batch** of five emails with `max_concurrency=3`, and print total wall
   time plus total tokens

Then answer with numbers: how much faster was the parallel version than doing the two calls
one after another, and how much did it cost?
:::

:::solution A working version, and the numbers
```python title="src/lc/exercise_chain.py"
import time
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field

class Classification(BaseModel):
    urgency: str = Field(description="low, normal, high or critical")
    category: str = Field(description="billing, technical, account or other")

strong = init_chat_model("anthropic:claude-sonnet-5")
cheap = init_chat_model("anthropic:claude-haiku-4-5")
model = strong.with_retry(stop_after_attempt=3).with_fallbacks([cheap])

classify = (
    ChatPromptTemplate.from_template("Classify this support email:\n\n{email}")
    | model.with_structured_output(Classification)
)
summarise = ChatPromptTemplate.from_template("One line summary:\n\n{email}") | model
draft = ChatPromptTemplate.from_template(
    "Write a two-sentence reply. Do not promise refunds.\n\n{email}"
) | model

analyse = RunnableParallel(classification=classify, summary=summarise, draft=draft)

EMAILS = [...]   # five realistic emails

started = time.perf_counter()
results = analyse.batch(
    [{"email": e} for e in EMAILS],
    config={"max_concurrency": 3, "metadata": {"request_id": "demo-1"}},
)
elapsed = time.perf_counter() - started

tokens = sum(
    r["summary"].usage_metadata["total_tokens"] + r["draft"].usage_metadata["total_tokens"]
    for r in results
)
print(f"{len(EMAILS)} emails in {elapsed:.1f}s, {tokens} tokens")
```

```text
measured on 5 emails, 3 calls each = 15 model calls

sequential, one call at a time        18.4s
parallel branches, no batching         7.1s
parallel + batch(max_concurrency=3)    4.2s

tokens: 6,240 total  (~$0.02 at sonnet pricing)

The speed-up is real and the cost is identical - you made the same 15 calls either way.
```

The honest follow-up: those three branches could be **one** call returning a single
structured object with all three fields. That would be roughly a third of the tokens and
one round trip. Parallel branches are the right answer when the tasks genuinely need
different prompts or different models — not as a default.
:::

## Challenge

:::challenge Measure whether the abstraction costs you anything
Build the same ticket classifier twice: once as `prompt | model.with_structured_output(...)`,
and once with the raw Anthropic SDK plus your own Pydantic validation.

Measure three things over 50 tickets: p50 and p95 latency, total tokens, and lines of code
you had to write and maintain.

Then introduce a failure — point one version at a bad model name, and send one ticket that
provokes malformed output — and compare what each version does. Which fails more clearly?
Which would you rather debug at 3am?

Write up the comparison as a short recommendation for a team deciding whether to adopt
LangChain. The interesting answer is rarely "always" or "never"; it is a list of conditions.
:::

## Interview Questions

:::interview
1. What makes something a Runnable, and why does that matter?
2. What does `chain.batch()` give you that a `for` loop does not?
3. What is `RunnablePassthrough` for? Give a concrete use.
4. Difference between `with_retry` and `with_fallbacks`?
5. Why is a silent fallback to a cheaper model dangerous?
6. When would you write a plain function instead of a chain?
:::

## Cheat Sheet

```python
# compose
chain = prompt | model | parser
chain = prompt | model.with_structured_output(Schema)

# the four methods every Runnable has
chain.invoke(x); chain.stream(x); chain.batch([x], config={"max_concurrency": 5})
await chain.ainvoke(x)

# fan out on the same input
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
RunnableParallel(a=chain_a, b=chain_b)
{"context": retriever, "question": RunnablePassthrough()}
RunnableLambda(my_function)

# resilience
model.with_retry(stop_after_attempt=3).with_fallbacks([cheaper_model])

# tracing
chain.invoke(x, config={"run_name": "...", "metadata": {"request_id": rid}})

# debugging
print(chain.get_graph().draw_ascii())
for ev in chain.stream_events(x, version="v2"): ...
```

## Summary

- Everything is a Runnable; Runnables give you `invoke`, `stream`, `batch` and async.
- `|` feeds the left output into the right input — shape mismatches cause most errors.
- `batch` with a concurrency cap is the cheapest performance win available.
- `RunnableParallel` fans out on one input; it saves time, not money.
- `with_retry` handles flakiness, `with_fallbacks` handles a model being unusable — and log
  which one answered.
- Deterministic work belongs in plain functions, not model calls.

## Next Step

Next: retrieval, agents and middleware — putting documents behind the model and letting it
choose tools, with the guardrails that make that safe.
