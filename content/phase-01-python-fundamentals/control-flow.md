---
title: Control Flow — if, for, while
order: 6
difficulty: Beginner
duration: 14
badges: ["Hands-on"]
summary: Conditions, loops and the iteration helpers — range, enumerate, zip, break, continue and else — written the way Python programmers actually write them.
prereqs: ["Numbers, Booleans, None and Operators", "Lists and Tuples"]
keyConcepts: ["if/elif/else", "for", "while", "enumerate", "zip", "break"]
---

:::note In one line
**Three tools, and that is all:** do something `if` a condition holds, repeat `for` each item, repeat `while` something stays true. Everything else is a variation on these.
:::

## Why this matters

Control flow is where your program makes decisions: retry or fail, escalate or answer,
include this chunk or skip it. Python's loop idioms are also where beginners most visibly
write "C code in Python" — indexing with `range(len(x))` instead of iterating directly —
and that style produces slower, buggier code.

## Mental Model

Python gives you three tools for deciding what happens, and that is genuinely all of them.

<figure class="lesson-figure">
<svg viewBox="0 0 660 230" role="img" aria-label="Diagram of three control-flow tools: if chooses between branches once, for repeats once per item in a collection, and while repeats until a condition becomes false.">
  <defs>
    <marker id="cf-a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
    <marker id="cf-c" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>

  <text class="dg-mono" x="14" y="26" style="font-size:14px" fill="var(--accent)">if</text>
  <text class="dg-sub"  x="42" y="26">choose one path, once</text>

  <path d="M120,58 L152,44 L184,58 L152,72 z" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.4"/>
  <text class="dg-sub" x="152" y="62" text-anchor="middle">yes?</text>
  <rect x="222" y="30" width="96" height="28" rx="6" class="dg-box"/>
  <text class="dg-sub" x="270" y="49" text-anchor="middle">do this</text>
  <rect x="222" y="62" width="96" height="28" rx="6" class="dg-box"/>
  <text class="dg-sub" x="270" y="81" text-anchor="middle">else that</text>
  <path class="dg-arrow" d="M184,52 L216,45" marker-end="url(#cf-a)"/>
  <path class="dg-arrow" d="M184,64 L216,73" marker-end="url(#cf-a)"/>

  <text class="dg-mono" x="380" y="26" style="font-size:14px" fill="var(--accent)">for</text>
  <text class="dg-sub"  x="414" y="26">once per item</text>
  <rect x="380" y="40" width="44" height="30" rx="6" class="dg-box"/>
  <rect x="432" y="40" width="44" height="30" rx="6" class="dg-box"/>
  <rect x="484" y="40" width="44" height="30" rx="6" class="dg-box"/>
  <rect x="536" y="40" width="44" height="30" rx="6" class="dg-box"/>
  <text class="dg-sub" x="596" y="60">done</text>
  <path class="dg-arrow" d="M424,55 L430,55"/>
  <path class="dg-arrow" d="M476,55 L482,55"/>
  <path class="dg-arrow" d="M528,55 L534,55"/>
  <path class="dg-arrow" d="M580,55 L586,55" marker-end="url(#cf-a)"/>
  <text class="dg-sub" x="380" y="88">you know how many rounds</text>

  <line x1="14" y1="112" x2="646" y2="112" stroke="var(--border)" stroke-width="1"/>

  <text class="dg-mono" x="14" y="142" style="font-size:14px" fill="var(--accent)">while</text>
  <text class="dg-sub"  x="70" y="142">keep going until something changes</text>

  <rect x="150" y="156" width="130" height="44" rx="8" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-sub" x="215" y="182" text-anchor="middle" fill="var(--accent)">do the work</text>
  <path d="M280,178 Q330,178 330,150 Q330,134 215,134 L215,150" stroke="var(--accent)" stroke-width="1.8" fill="none" marker-end="url(#cf-c)"/>
  <text class="dg-sub" x="330" y="128" text-anchor="middle" fill="var(--accent)">still true? go again</text>

  <rect x="430" y="156" width="150" height="44" rx="8" fill="var(--panel)" stroke="var(--danger)" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text class="dg-sub" x="505" y="176" text-anchor="middle" fill="var(--danger)">always add a limit</text>
  <text class="dg-sub" x="505" y="192" text-anchor="middle">or it may never stop</text>

  <text class="dg-sub" x="150" y="220">you do not know how many rounds</text>
</svg>
<figcaption>
<strong>Pick by whether you know the number of rounds.</strong> Iterating a list? Use
<code>for</code>. Waiting for something to become true? Use <code>while</code> — and give it
a maximum, because an agent loop is a <code>while</code> loop and a runaway one costs real
money.
</figcaption>
</figure>

```python
# for: you have a collection, visit each item
for message in messages:
    print(message["role"])

# while: you do not know how many rounds - so cap it
steps = 0
while not done and steps < 10:        # the cap is not optional
    done = take_one_step()
    steps += 1
```

:::tip This is the shape of every agent
That `while` loop with a step cap **is** the agent loop from Phase 14. You are already
looking at it. The only thing that changes later is that a model decides whether `done` is
true.
:::

## Core Concepts

### if / elif / else

```python
score = 0.72

if score >= 0.85:
    verdict = "high confidence"
elif score >= 0.6:
    verdict = "medium confidence"
elif score >= 0.4:
    verdict = "low confidence"
else:
    verdict = "escalate"
```

Only the **first** matching branch runs. Indentation (4 spaces) defines the block — there
are no braces, and mixing tabs and spaces is an error.

The conditional expression, for simple two-way choices:

```python
label = "pass" if score >= 0.6 else "fail"
```

Python 3.10+ also has structural pattern matching, ideal for routing:

```python
match event:
    case {"type": "tool_call", "name": name}:
        run_tool(name)
    case {"type": "text", "content": str(content)} if content.strip():
        emit(content)
    case {"type": "error", "code": 429}:
        backoff()
    case _:
        raise ValueError(f"unknown event: {event}")
```

### for loops

```python
chunks = ["intro", "methods", "results"]

for chunk in chunks:                        # iterate the items, not the indices
    print(chunk)

for i, chunk in enumerate(chunks, start=1): # when you need position too
    print(f"{i}. {chunk}")

scores = [0.9, 0.7, 0.5]
for chunk, score in zip(chunks, scores):    # walk two sequences together
    print(f"{chunk}: {score}")

for key, value in config.items():           # dictionaries
    print(key, value)

for i in range(3):          # 0, 1, 2
for i in range(1, 4):       # 1, 2, 3
for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
for i in range(3, 0, -1):   # 3, 2, 1
```

:::mistake `for i in range(len(items))` is almost always wrong
```python
for i in range(len(chunks)):        # C-style, error-prone
    print(chunks[i])

for chunk in chunks:                # Pythonic
    print(chunk)

for i, chunk in enumerate(chunks):  # when the index is genuinely needed
    print(i, chunk)
```
:::

:::warning `zip` stops at the shortest sequence
```python
list(zip([1, 2, 3], ["a", "b"]))           # [(1, 'a'), (2, 'b')] - 3 silently dropped
list(zip([1, 2, 3], ["a", "b"], strict=True))   # ValueError - use this (3.10+)
```
Silent truncation has ruined many evaluation scripts where predictions and labels drifted
out of sync.
:::

### while loops

```python
attempt = 0
max_attempts = 5

while attempt < max_attempts:
    attempt += 1
    result = try_call()
    if result is not None:
        break
else:
    # runs only if the loop ended WITHOUT break - i.e. all attempts failed
    raise RuntimeError(f"gave up after {max_attempts} attempts")
```

The `for`/`while`-`else` clause is unusual and genuinely useful: "else" means "no `break`
happened". Read it as `nobreak`.

### break, continue, pass

```python
for chunk in chunks:
    if chunk.is_empty:
        continue            # skip this item, keep looping
    if budget_exhausted:
        break               # stop the loop entirely
    process(chunk)

def todo():
    pass                    # syntactically required placeholder, does nothing
```

## Minimal Example

```python title="route.py"
def route(similarity: float, has_citation: bool) -> str:
    if not has_citation:
        return "escalate"
    if similarity >= 0.82:
        return "answer"
    if similarity >= 0.60:
        return "answer_with_caveat"
    return "escalate"


cases = [(0.91, True), (0.70, True), (0.95, False), (0.30, True)]
for similarity, cited in cases:
    print(f"sim={similarity:.2f} cited={cited!s:<5} -> {route(similarity, cited)}")
```

```text
sim=0.91 cited=True  -> answer
sim=0.70 cited=True  -> answer_with_caveat
sim=0.95 cited=False -> escalate
sim=0.30 cited=True  -> escalate
```

Notice the **guard clause** style: handle the exceptional case first and return early
rather than nesting. Deeply nested `if`s are the main readability killer in beginner code.

## Real-World Example

The agent loop — a `while` loop with three termination conditions. You will build the real
version in Phase 14; the control flow is exactly this.

```python title="src/agent/loop.py"
"""A bounded tool-calling loop.

Three independent stop conditions, because an unbounded agent loop is the most
expensive bug in AI engineering:
  1. the model produced a final answer
  2. the iteration cap was reached
  3. the token/cost budget was exhausted
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopResult:
    answer: str | None
    iterations: int
    stop_reason: str
    tokens_used: int


def run_agent(
    question: str,
    *,
    max_iterations: int = 8,
    token_budget: int = 20_000,
) -> LoopResult:
    messages: list[dict] = [{"role": "user", "content": question}]
    tokens_used = 0
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        if tokens_used >= token_budget:
            return LoopResult(None, iteration, "budget_exhausted", tokens_used)

        response = call_model(messages)          # pretend this is the provider call
        tokens_used += response["tokens"]
        messages.append(response["message"])

        if response["type"] == "final":
            return LoopResult(response["text"], iteration, "final_answer", tokens_used)

        if response["type"] != "tool_call":
            continue                              # ignore anything unexpected, try again

        try:
            observation = run_tool(response["tool"], response["args"])
        except ToolError as exc:
            observation = f"Tool failed: {exc}. Try a different approach."

        messages.append({"role": "tool", "content": str(observation)[:4_000]})

    return LoopResult(None, iteration, "max_iterations", tokens_used)
```

Read the structure, not the fake helpers: **one loop, bounded, with an explicit
`stop_reason` on every exit path**. Never write an agent loop whose only exit is the model
deciding to stop.

## Common Mistakes

:::mistake
```python
# 1. Infinite while loop - the counter is never advanced
while attempt < 3:
    try_call()                # attempt never changes

# 2. Mutating the collection you are iterating
for c in chunks:
    if bad(c):
        chunks.remove(c)      # skips items

# 3. Off-by-one with range
for i in range(1, len(items)):    # skips items[0]

# 4. Catching the loop variable after the loop
for row in rows:
    ...
print(row)                    # works, but leaks the last value - confusing

# 5. Deep nesting instead of guard clauses
if a:
    if b:
        if c:
            do()              # invert the conditions and return early instead
```
:::

## Debugging

```python
for i, item in enumerate(items):
    if i < 3:
        print(f"[{i}] {item!r}")      # peek at the first few
    ...

# Count what actually happened
processed = skipped = failed = 0
...
print(f"processed={processed} skipped={skipped} failed={failed}")
```

Counters at the end of a batch loop turn "it seemed to work" into evidence, and they are
the beginning of the metrics you will emit in Phase 24.

## Best Practices

1. Iterate objects directly; use `enumerate` only when you need the index.
2. Guard clauses over nesting: handle the exit cases first.
3. Every `while` loop gets an explicit bound — iterations, time, or budget.
4. Use `zip(..., strict=True)` so mismatched lengths raise instead of truncating.
5. Name your booleans so conditions read like sentences: `if not has_citation:`.

## Performance Considerations

- Loops in Python are slow relative to vectorised operations. For numeric work over
  thousands of elements, NumPy (Phase 3) and Pandas (Phase 4) replace the loop entirely and
  run 10–100× faster.
- Hoist invariant work out of loops: compile regexes, build sets, open connections once.
- `any()` and `all()` short-circuit; `any(x > 0.9 for x in scores)` stops at the first hit.

## Hands-on Exercise

:::exercise Batch processor with retry
Write `process_batch(items, max_retries=2)` that:

1. iterates over items,
2. calls `handle(item)` which randomly raises `TransientError` about a third of the time,
3. retries each failing item up to `max_retries` times,
4. collects successes and permanent failures separately,
5. prints a summary line: `processed=N failed=M retries=R`.

Use a `for` loop over items with an inner `while` for retries, and a `for`/`else` or an
explicit flag to detect permanent failure.
:::

:::solution Solution
```python title="batch.py"
import random


class TransientError(RuntimeError):
    pass


def handle(item: str) -> str:
    if random.random() < 0.34:
        raise TransientError(f"temporary failure on {item}")
    return item.upper()


def process_batch(items: list[str], *, max_retries: int = 2) -> dict[str, object]:
    succeeded: list[str] = []
    failed: list[str] = []
    retries = 0

    for item in items:
        attempt = 0
        while True:
            try:
                succeeded.append(handle(item))
                break
            except TransientError:
                attempt += 1
                retries += 1
                if attempt > max_retries:
                    failed.append(item)
                    break

    print(f"processed={len(succeeded)} failed={len(failed)} retries={retries}")
    return {"succeeded": succeeded, "failed": failed, "retries": retries}


if __name__ == "__main__":
    random.seed(3)
    process_batch([f"doc-{i}" for i in range(10)])
```

```text
processed=10 failed=0 retries=4
```

The `while True` with explicit `break`s is clearer here than a counted loop, because there
are two distinct exit conditions.
:::

## Challenge

:::challenge Streaming token accumulator
Simulate a streaming LLM response: given a list of string deltas, loop over them and

- accumulate the full text,
- stop early (`break`) as soon as the accumulated text contains a stop sequence such as
  `"</answer>"`, discarding anything after it,
- count how many deltas you consumed,
- and detect the case where the stream ends without the stop sequence, reporting
  `stop_reason="incomplete"`.

This is the exact logic inside a streaming client (Phase 10).
:::

## Interview Questions

:::interview
1. When do you use `while` rather than `for`?
2. What does the `else` clause on a loop mean?
3. Why is `for i in range(len(items))` discouraged?
4. What happens when `zip` receives sequences of different lengths?
5. How do you guarantee an agent loop terminates?
:::

## Cheat Sheet

```python
if cond: ... elif cond: ... else: ...
value = a if cond else b
match value:
    case {"type": "x", "id": i}: ...
    case _: ...

for item in seq: ...
for i, item in enumerate(seq, start=1): ...
for a, b in zip(xs, ys, strict=True): ...
for k, v in d.items(): ...
range(stop) range(start, stop) range(start, stop, step)

while cond: ...
break      # exit the loop
continue   # next iteration
else:      # runs when the loop finished without break

any(pred(x) for x in seq)   all(...)   sum(...)   min(seq, key=...)
```

```quiz
[
  {
    "question": "What does a for/else clause do?",
    "options": [
      "Runs when the loop body raised an exception",
      "Runs when the loop completed without hitting break",
      "Runs on the last iteration",
      "Runs if the sequence was empty"
    ],
    "answer": 1,
    "explanation": "Read `else` as `nobreak`. It is the idiomatic way to express 'searched everything and found nothing'."
  },
  {
    "question": "Which guarantees an agent loop cannot run forever?",
    "options": [
      "A good system prompt telling it to finish quickly",
      "A max_iterations counter checked by the loop condition",
      "A larger context window",
      "Setting temperature to 0"
    ],
    "answer": 1,
    "explanation": "Termination must be enforced by code. Prompts are advisory; a counter (plus a budget check) is a guarantee."
  },
  {
    "question": "zip(predictions, labels) where predictions has 100 items and labels has 98. What happens?",
    "options": [
      "ValueError",
      "The last two predictions are silently ignored",
      "The last two labels repeat",
      "None is used for the missing labels"
    ],
    "answer": 1,
    "explanation": "zip truncates to the shortest input, silently corrupting evaluations. Pass strict=True so it raises instead."
  }
]
```

## Summary

- `for` iterates known collections, `while` repeats until a condition changes — and every
  `while` in production needs a bound.
- `enumerate` and `zip` replace index arithmetic; `zip(strict=True)` prevents silent
  truncation.
- Guard clauses beat nesting; `match` handles structured routing cleanly.
- Never mutate a collection you are iterating.

## Next Step

Comprehensions: the compact, fast way to transform and filter collections that you will see
in every Python codebase you ever read.
