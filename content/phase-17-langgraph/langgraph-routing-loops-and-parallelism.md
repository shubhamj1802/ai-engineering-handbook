---
title: "LangGraph — Routing, Loops and Parallel Branches"
order: 3
difficulty: Expert
duration: 22
badges: ["Hands-on", "Deep dive"]
summary: Conditional edges that branch, loops with hard caps, Command for combining an update with a jump, and Send for fanning one item out into many parallel workers.
prereqs: ["LangGraph — State, Nodes, Edges and Reducers"]
keyConcepts: ["add_conditional_edges", "Command", "Send", "map-reduce", "recursion_limit", "loop cap"]
---

:::note In one line
**Three control-flow tools: a conditional edge to branch, a loop with a counter to retry,
and `Send` to fan out.** Every graph you build is a combination of those three.
:::

:::warning Versions used on this page
`langgraph` **1.2.11** · `langchain` **1.4.0**
:::

## Why this matters

A straight line of nodes is a chain with extra typing. Graphs earn their keep the moment the
path depends on what happened.

Three shapes cover nearly everything:

<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Diagram of three control-flow shapes: a conditional edge branching to one of several nodes, a loop returning to an earlier node with an attempt counter, and a fan-out where one node sends many items to parallel workers that are then gathered.">
  <defs>
    <marker id="cf2-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
    <marker id="cf2-w" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--warn)"/>
    </marker>
  </defs>
  <text class="dg-label" x="14" y="20" fill="var(--accent)">1 · branch</text>
  <polygon points="24,60 66,42 108,60 66,78" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.7"/>
  <text class="dg-sub" x="66" y="64" text-anchor="middle" fill="var(--accent)">which?</text>
  <rect x="132" y="32" width="66" height="24" rx="5" class="dg-box"/>
  <text class="dg-sub" x="165" y="49" text-anchor="middle">a</text>
  <rect x="132" y="62" width="66" height="24" rx="5" class="dg-box"/>
  <text class="dg-sub" x="165" y="79" text-anchor="middle">b</text>
  <path class="dg-arrow" d="M108,54 L126,46" marker-end="url(#cf2-a)"/>
  <path class="dg-arrow" d="M108,66 L126,72" marker-end="url(#cf2-a)"/>
  <text class="dg-mono" x="14" y="106" style="font-size:10px">add_conditional_edges</text>
  <line x1="222" y1="14" x2="222" y2="118" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-label" x="244" y="20" fill="var(--warn)">2 · loop</text>
  <rect x="244" y="40" width="82" height="34" rx="6" class="dg-box"/>
  <text class="dg-sub" x="285" y="61" text-anchor="middle">work</text>
  <rect x="354" y="40" width="82" height="34" rx="6" fill="var(--panel-2)" stroke="var(--warn)" stroke-width="1.6"/>
  <text class="dg-sub" x="395" y="61" text-anchor="middle" fill="var(--warn)">good?</text>
  <path class="dg-arrow" d="M326,57 L348,57" marker-end="url(#cf2-w)"/>
  <path d="M395,74 Q395,100 285,100 L285,78" stroke="var(--warn)" stroke-width="1.6" fill="none" marker-end="url(#cf2-w)"/>
  <text class="dg-sub" x="290" y="114" fill="var(--danger)">must have a max attempts counter</text>
  <line x1="460" y1="14" x2="460" y2="118" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-label" x="482" y="20" fill="var(--accent-2)">3 · fan out</text>
  <rect x="482" y="34" width="52" height="26" rx="5" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.6"/>
  <text class="dg-sub" x="508" y="52" text-anchor="middle" fill="var(--accent-2)">split</text>
  <rect x="566" y="22" width="44" height="20" rx="4" class="dg-box"/>
  <rect x="566" y="46" width="44" height="20" rx="4" class="dg-box"/>
  <rect x="566" y="70" width="44" height="20" rx="4" class="dg-box"/>
  <path class="dg-arrow" d="M534,44 L560,32"/>
  <path class="dg-arrow" d="M534,47 L560,56"/>
  <path class="dg-arrow" d="M534,50 L560,80"/>
  <text class="dg-mono" x="482" y="106" style="font-size:10px">Send()</text>
  <text class="dg-sub" x="482" y="120">run together, gather after</text>
  <rect x="14" y="140" width="632" height="88" rx="10" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.4"/>
  <text class="dg-label" x="30" y="162">Pick by the question you are asking</text>
  <text class="dg-sub" x="30" y="184">"which one path?" → a conditional edge returning a node name</text>
  <text class="dg-sub" x="30" y="202">"again?" → an edge pointing backwards, plus an attempts counter in the state</text>
  <text class="dg-sub" x="30" y="220">"all of these, at once?" → Send, one per item, gathered by a reducer</text>
</svg>
<figcaption>
<strong>Branch, loop, fan out.</strong> The loop is the one that needs discipline: without a
counter in the state, a graph can spin until your budget is gone.
</figcaption>
</figure>

## Branching with a conditional edge

A conditional edge is an ordinary function that **returns the name of the next node**.

```python title="src/lg/routing.py"
"""Route a support request to the right handler."""
from typing import Annotated, Literal, TypedDict
from operator import add

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

model = init_chat_model("anthropic:claude-sonnet-5")

class Route(BaseModel):
    destination: Literal["billing", "technical", "escalate"] = Field(
        description="billing for charges and invoices, technical for errors and outages, "
                    "escalate for legal, press or anything threatening"
    )
    reason: str = Field(max_length=120)

class State(TypedDict):
    request: str
    destination: str
    answer: str
    trace: Annotated[list[str], add]

def classify(state: State) -> dict:
    route = model.with_structured_output(Route).invoke(
        f"Route this support request.\n\n{state['request']}"
    )
    return {"destination": route.destination, "trace": [f"routed to {route.destination}: {route.reason}"]}

# This is the conditional edge function. It returns a NODE NAME.
def pick_handler(state: State) -> Literal["billing", "technical", "escalate"]:
    return state["destination"]

def billing(state: State) -> dict:
    return {"answer": "Billing team will review your charges.", "trace": ["handled by billing"]}

def technical(state: State) -> dict:
    return {"answer": "Engineering is investigating.", "trace": ["handled by technical"]}

def escalate(state: State) -> dict:
    return {"answer": "A manager will contact you within one hour.", "trace": ["escalated"]}

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing", billing)
builder.add_node("technical", technical)
builder.add_node("escalate", escalate)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    pick_handler,
    # map what the function returns -> which node to run
    {"billing": "billing", "technical": "technical", "escalate": "escalate"},
)
for handler in ("billing", "technical", "escalate"):
    builder.add_edge(handler, END)

graph = builder.compile()

out = graph.invoke({"request": "You charged me twice and I want a refund today.",
                    "destination": "", "answer": "", "trace": []})
print(out["destination"], "|", out["answer"])
print(out["trace"])
```

```text
billing | Billing team will review your charges.
['routed to billing: duplicate charge, refund requested', 'handled by billing']
```

:::tip Use Literal, not str, for a destination
`Literal["billing", "technical", "escalate"]` means the model is constrained to those three
values and your editor can check the mapping. With a bare `str`, a model that answers
`"Billing"` with a capital B routes nowhere and the graph raises at runtime.
:::

## Looping, safely

A loop is just an edge that points backwards. The only hard rule: **the state must carry a
counter, and the exit must check it.**

```python title="src/lg/loop.py"
"""Draft, check, fix - at most three attempts, then give up honestly."""
from typing import Annotated, Literal, TypedDict
from operator import add

MAX_ATTEMPTS = 3

class State(TypedDict):
    question: str
    draft: str
    issues: list[str]
    attempts: int
    trace: Annotated[list[str], add]

def draft(state: State) -> dict:
    prompt = f"Answer in under 60 words: {state['question']}"
    if state["issues"]:
        prompt += "\n\nYour previous attempt had these problems, fix them: " + "; ".join(state["issues"])
    reply = model.invoke(prompt)
    return {
        "draft": reply.text,
        "attempts": state["attempts"] + 1,
        "trace": [f"attempt {state['attempts'] + 1}"],
    }

def check(state: State) -> dict:
    """Deterministic checks only - free, instant, never wrong."""
    issues = []
    if len(state["draft"].split()) > 60:
        issues.append("too long, over 60 words")
    if "as an AI" in state["draft"].lower():
        issues.append("remove boilerplate about being an AI")
    return {"issues": issues, "trace": [f"check found {len(issues)} issue(s)"]}

def decide(state: State) -> Literal["draft", "finish"]:
    if not state["issues"]:
        return "finish"                      # good enough
    if state["attempts"] >= MAX_ATTEMPTS:
        return "finish"                      # out of attempts - stop anyway
    return "draft"                           # try again

def finish(state: State) -> dict:
    ok = not state["issues"]
    return {"trace": [f"finished {'clean' if ok else 'with unresolved issues'}"]}

builder = StateGraph(State)
builder.add_node("draft", draft)
builder.add_node("check", check)
builder.add_node("finish", finish)
builder.add_edge(START, "draft")
builder.add_edge("draft", "check")
builder.add_conditional_edges("check", decide, {"draft": "draft", "finish": "finish"})
builder.add_edge("finish", END)

graph = builder.compile()
```

```text
trace:
  attempt 1
  check found 1 issue(s)
  attempt 2
  check found 0 issue(s)
  finished clean
```

:::danger Two independent limits, and you want both
`decide` checks `attempts >= MAX_ATTEMPTS` — that is **your** limit, and it lets you end
gracefully with a sensible message.

LangGraph also has `recursion_limit` (default 25), which raises `GraphRecursionError`:

```python
graph.invoke(initial, config={"recursion_limit": 12})
```

That one is a **backstop against your own bugs**, not a design tool. Rely on your counter for
normal behaviour; keep the recursion limit so a mistake fails fast instead of billing you
for 500 model calls.
:::

## Command: update and jump in one step

Sometimes a node needs to decide where to go *and* write to the state. `Command` does both,
and removes the need for a separate conditional-edge function.

```python title="src/lg/command_routing.py"
from typing import Literal
from langgraph.types import Command

def classify(state: State) -> Command[Literal["billing", "technical", "escalate"]]:
    route = model.with_structured_output(Route).invoke(
        f"Route this request.\n\n{state['request']}"
    )
    return Command(
        update={"destination": route.destination, "trace": [f"routed: {route.reason}"]},
        goto=route.destination,        # where to go next, decided here
    )

builder.add_node("classify", classify)
builder.add_edge(START, "classify")
# No add_conditional_edges needed - the node said where it is going.
```

| Approach | Use when |
| --- | --- |
| `add_conditional_edges` | routing logic is separate and reusable, or the graph shape should be visible in the wiring |
| `Command(goto=...)` | the node that decides is the node that knows — fewer moving parts |

:::tip Command makes handoffs natural
In a multi-agent graph, `Command(goto="researcher", update={...})` reads exactly like
"I am done, pass this to the researcher". That is why the multi-agent lesson leans on it.
:::

## Fanning out with Send

`Send` runs the **same node many times, in parallel, each with its own input**. It is
map-reduce for graphs.

```python title="src/lg/fan_out.py"
"""Summarise 30 documents concurrently, then combine."""
from typing import Annotated, TypedDict
from operator import add
from langgraph.types import Send

class State(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], add]      # the reducer gathers the parallel writes
    final: str

class WorkerState(TypedDict):
    document: str                              # what ONE worker receives

def fan_out(state: State):
    """Return a list of Sends - one per document."""
    return [Send("summarise_one", {"document": doc}) for doc in state["documents"]]

def summarise_one(state: WorkerState) -> dict:
    reply = model.invoke(f"One sentence summary:\n\n{state['document'][:4000]}")
    return {"summaries": [reply.text]}         # a LIST, because the reducer appends

def combine(state: State) -> dict:
    joined = "\n".join(f"- {s}" for s in state["summaries"])
    reply = model.invoke(f"Write three bullet points covering all of these:\n\n{joined}")
    return {"final": reply.text}

builder = StateGraph(State)
builder.add_node("summarise_one", summarise_one)
builder.add_node("combine", combine)
builder.add_conditional_edges(START, fan_out, ["summarise_one"])
builder.add_edge("summarise_one", "combine")
builder.add_edge("combine", END)

graph = builder.compile()

result = graph.invoke({"documents": docs, "summaries": [], "final": ""})
```

```text
30 documents
  sequential:  41.2s
  with Send:    5.8s     (concurrency limited by your rate limiter, not the graph)
```

Two requirements that catch people:

1. The gathering key **must have a reducer** (`Annotated[list[str], add]`). Without it, 30
   parallel writes to `summaries` collide and you keep one.
2. Each worker returns a **list**, because the reducer appends lists.

:::warning Fan-out multiplies cost immediately
Thirty documents is thirty model calls, launched at once. Combine `Send` with a rate limiter
and a budget ceiling, or a user pasting a large folder becomes an unplanned bill.

Cap it explicitly:
```python
graph.invoke(payload, config={"max_concurrency": 5, "recursion_limit": 50})
```
:::

## Parallel branches that are different nodes

Two edges from one node run both branches concurrently:

```python
builder.add_edge("prepare", "check_policy")
builder.add_edge("prepare", "check_facts")      # both run, at the same time
builder.add_edge("check_policy", "decide")
builder.add_edge("check_facts", "decide")       # decide waits for BOTH
```

`decide` runs once, after both have finished. If both write the same key, that key needs a
reducer — otherwise one write is lost and which one is not guaranteed.

:::mistake Two parallel nodes writing the same key with no reducer
```python
class State(TypedDict):
    findings: list[str]        # NO reducer - two parallel writers will clash

class State(TypedDict):
    findings: Annotated[list[str], add]    # correct
```
This is the most common parallel bug in LangGraph, and it presents as "sometimes one of my
checks does not appear" — which looks like a model problem and is not.
:::

## Debugging control flow

```python title="src/lg/debug_flow.py"
# 1. See the shape, including conditional edges
print(graph.get_graph().draw_ascii())

# 2. Watch which nodes actually ran, in order
for step in graph.stream(initial, stream_mode="updates"):
    for node, update in step.items():
        print(f"-> {node}: {list(update)}")

# 3. When a loop misbehaves, print the counter every pass
def draft(state):
    print(f"[draft] attempt={state['attempts']} issues={state['issues']}")
    ...
```

If a loop runs away, the trace tells you why immediately: either the exit condition is never
true, or the counter is not being incremented — and both are visible in one pass of output.

## Hands-on Exercise

:::exercise A research graph with all three shapes
Build `src/lg/exercise_flow.py` that:

1. **Branches** — classify a question as `simple`, `research` or `refuse`
   - `simple` answers directly in one call
   - `refuse` returns a fixed message and stops
   - `research` continues to the next stage
2. **Fans out** — for `research`, generate three sub-questions and answer them in parallel
   with `Send`
3. **Loops** — a `verify` node checks the combined answer mentions all three sub-questions;
   if not, re-combine, up to 2 attempts
4. Every node appends to `trace`
5. Set `recursion_limit=20` and `max_concurrency=3`

Report the trace for one simple question and one research question, and the wall time of the
parallel stage versus doing the three sub-questions sequentially.
:::

:::solution The traces, and the thing that bites
```text
simple question: "what is a token?"
  classified as simple
  answered directly
  2 nodes, 1 model call, 1.3s

research question: "how should we choose a vector database?"
  classified as research
  generated 3 sub-questions
  answered sub-question (x3, in parallel)
  combined
  verify: all 3 covered
  7 nodes, 6 model calls, 4.1s

  sub-questions sequentially: 3.9s
  sub-questions with Send:    1.4s
```

The bug almost everyone hits first:

```text
KeyError / only one sub-answer present

class State(TypedDict):
    sub_answers: list[str]                    # WRONG with Send
    sub_answers: Annotated[list[str], add]    # RIGHT
```

Three parallel workers all write `sub_answers`. Without the reducer the graph keeps one
write and the combine step silently summarises a third of the research. Nothing errors —
the answer is just quietly worse, which is the worst kind of bug.
:::

## Challenge

:::challenge Give the loop a real quality signal
The loop above checks length and boilerplate — cheap, deterministic, and it cannot tell
whether the answer is *good*.

Replace `check` with a two-part gate: keep the deterministic checks, and add a model-based
judge that scores faithfulness to retrieved context on a 1–5 scale with a written reason.
Loop while the score is below 4, capped at three attempts.

Then measure the thing that decides whether this is worth shipping: over 40 questions, how
often does attempt 2 actually score higher than attempt 1? Report the distribution, the extra
cost per question, and the cases where looping made the answer *worse*.

Self-correction loops are widely recommended and frequently do very little. Finding out
which you have requires exactly this measurement, and the answer depends on whether your
judge has something real to check against — the same lesson as Lab 5 in Phase 27.
:::

## Interview Questions

:::interview
1. What must a conditional edge function return?
2. What are the two independent limits on a loop, and what is each for?
3. What does `Command` let a node do that a plain return cannot?
4. Why must a key gathered by `Send` have a reducer?
5. Two parallel nodes write the same key with no reducer. What happens?
6. When would you choose `add_conditional_edges` over `Command(goto=...)`?
:::

## Cheat Sheet

```python
# branch
def router(state) -> Literal["a", "b"]: return "a" if cond else "b"
builder.add_conditional_edges("node", router, {"a": "a", "b": "b"})

# loop (always with a counter in the state)
def decide(state) -> Literal["retry", "done"]:
    if ok(state): return "done"
    return "done" if state["attempts"] >= MAX else "retry"

# update + jump in one node
from langgraph.types import Command
return Command(update={...}, goto="next_node")

# fan out, gather with a reducer
from langgraph.types import Send
def fan_out(state): return [Send("worker", {"item": i}) for i in state["items"]]
class State(TypedDict):
    results: Annotated[list[str], add]        # required for Send

# limits
graph.invoke(x, config={"recursion_limit": 20, "max_concurrency": 5})

# debug
print(graph.get_graph().draw_ascii())
for step in graph.stream(x, stream_mode="updates"): print(step)
```

## Summary

- A conditional edge is a function returning the next node's name; use `Literal` for it.
- Every loop needs a counter in the state and an exit that checks it. `recursion_limit` is a
  backstop for bugs, not your design.
- `Command(update=..., goto=...)` combines a state write with a jump — ideal for handoffs.
- `Send` fans one item out into many parallel runs; the gathering key **must** have a reducer.
- Parallel branches writing the same key without a reducer lose writes silently.
- Fan-out multiplies cost instantly — pair it with concurrency caps and a budget.

## Next Step

Next: persistence and interrupts — saving state after every node so a graph can pause for
human approval and resume days later.
