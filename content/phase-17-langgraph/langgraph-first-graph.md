---
title: "LangGraph from Zero — Your First Graph"
order: 1
difficulty: Intermediate
duration: 16
badges: ["Start here", "Hands-on"]
summary: Build a three-node graph with no models in it at all, so the machinery is obvious — state, nodes, edges, compile, invoke — then add a model once the shape makes sense.
prereqs: ["LangChain from Zero — Your First Calls"]
keyConcepts: ["StateGraph", "node", "edge", "START", "END", "compile", "state update"]
---

:::note In one line
**A LangGraph node is a function that takes the state and returns the bits it wants to
change.** Edges decide what runs next. That is genuinely the whole model.
:::

:::warning Versions used on this page
`langgraph` **1.2.11** · `langchain` **1.4.0** · `langchain-core` **1.6.3**
:::

## Why this matters

LangChain chains run **forwards**. Input goes in one end, output comes out the other.

Plenty of real work is not like that:

- Retry the search with different words if the first attempt found nothing
- Loop until the answer passes a check, up to five attempts
- Stop and wait for a human to approve, possibly for two days
- Route to one of six specialists depending on what the request turns out to be

You can build those with `if` statements and `while` loops. What you cannot easily build is
**a version that survives the process dying halfway through** — and that is what LangGraph
adds.

## Install

```bash
uv add langgraph
```

## A graph with no AI in it

We will start with no model at all. Every confusing thing about LangGraph is about state and
edges, so let us look at those with nothing else in the way.

```python title="src/lg/first_graph.py"
"""A three-node graph that does arithmetic. No models, no API key, no cost."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. The state: one dict shape, shared by every node.
class State(TypedDict):
    value: int
    log: str

# 2. Nodes: plain functions. Take the state, return ONLY what changes.
def double(state: State) -> dict:
    return {"value": state["value"] * 2, "log": "doubled"}

def add_ten(state: State) -> dict:
    return {"value": state["value"] + 10, "log": "added ten"}

def describe(state: State) -> dict:
    return {"log": f"final value is {state['value']}"}

# 3. Wire it up.
builder = StateGraph(State)
builder.add_node("double", double)
builder.add_node("add_ten", add_ten)
builder.add_node("describe", describe)

builder.add_edge(START, "double")
builder.add_edge("double", "add_ten")
builder.add_edge("add_ten", "describe")
builder.add_edge("describe", END)

# 4. Compile, then run.
graph = builder.compile()

result = graph.invoke({"value": 5, "log": ""})
print(result)
```

```bash
uv run python src/lg/first_graph.py
```

```text
{'value': 20, 'log': 'final value is 20'}
```

`5 → 10 → 20`, and the log ends up holding only the last thing written to it. Both of those
facts matter, and the second one surprises people — we come back to it.

## What just happened

<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Diagram: state flows from START through three nodes to END. Each node receives the whole state and returns only the keys it changes, which are merged back into the state before the next node runs.">
  <defs>
    <marker id="g1-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <circle cx="36" cy="80" r="20" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.8"/>
  <text class="dg-sub" x="36" y="85" text-anchor="middle" fill="var(--ok)">START</text>
  <rect x="94" y="54" width="112" height="52" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-label" x="150" y="76" text-anchor="middle" fill="var(--accent)">double</text>
  <text class="dg-mono"  x="150" y="94" text-anchor="middle" style="font-size:10px">value: 10</text>
  <rect x="242" y="54" width="112" height="52" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-label" x="298" y="76" text-anchor="middle" fill="var(--accent)">add_ten</text>
  <text class="dg-mono"  x="298" y="94" text-anchor="middle" style="font-size:10px">value: 20</text>
  <rect x="390" y="54" width="112" height="52" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-label" x="446" y="76" text-anchor="middle" fill="var(--accent)">describe</text>
  <text class="dg-mono"  x="446" y="94" text-anchor="middle" style="font-size:10px">log: ...</text>
  <circle cx="556" cy="80" r="20" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.8"/>
  <text class="dg-sub" x="556" y="85" text-anchor="middle" fill="var(--ok)">END</text>
  <path class="dg-arrow" d="M56,80 L88,80" marker-end="url(#g1-a)"/>
  <path class="dg-arrow" d="M206,80 L236,80" marker-end="url(#g1-a)"/>
  <path class="dg-arrow" d="M354,80 L384,80" marker-end="url(#g1-a)"/>
  <path class="dg-arrow" d="M502,80 L534,80" marker-end="url(#g1-a)"/>
  <rect x="94" y="140" width="408" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.8"/>
  <text class="dg-label" x="298" y="162" text-anchor="middle" fill="var(--accent-2)">the state — one dict, shared</text>
  <text class="dg-sub"   x="298" y="182" text-anchor="middle">each node gets all of it, returns only the keys it changes</text>
  <path d="M150,106 L150,136" stroke="var(--accent-2)" stroke-width="1.3" stroke-dasharray="3 3"/>
  <path d="M298,106 L298,136" stroke="var(--accent-2)" stroke-width="1.3" stroke-dasharray="3 3"/>
  <path d="M446,106 L446,136" stroke="var(--accent-2)" stroke-width="1.3" stroke-dasharray="3 3"/>
  <text class="dg-sub" x="14" y="224">A node never receives arguments and never returns a whole new state. It reads the state and returns a partial update.</text>
</svg>
<figcaption>
<strong>Nodes do not call each other.</strong> They read the shared state and return a patch.
The graph decides who runs next, which is exactly why it can be paused and resumed.
</figcaption>
</figure>

The four pieces, named:

| Piece | What it is |
| --- | --- |
| **State** | a `TypedDict` — the shape every node reads and writes |
| **Node** | a function `state -> dict of changes` |
| **Edge** | "after this node, run that one" |
| **compile()** | turns the description into something runnable |

`START` and `END` are built-in markers, not nodes you write.

## Returning only what changes

This trips everyone up once:

```python
def add_ten(state: State) -> dict:
    return {"value": state["value"] + 10}     # `log` is untouched, and that is fine
```

You return a **partial** update. Keys you leave out keep their previous value. You never
have to copy the rest of the state forward.

:::mistake Mutating the state instead of returning a patch
```python
def bad(state: State) -> dict:
    state["value"] += 10        # DON'T - mutating in place
    return {}                   # the graph sees no change to record

def good(state: State) -> dict:
    return {"value": state["value"] + 10}
```
Mutation appears to work in simple cases and then breaks checkpointing, time travel and
parallel branches — because the graph tracks *the updates you return*, not the dict you
scribbled on.
:::

## Why the log only kept the last line

By default, **writing a key replaces it**. Three nodes each wrote `log`, so only the last
survived.

If you want values to accumulate, you say so with a reducer:

```python title="src/lg/accumulate.py"
from typing import Annotated, TypedDict
from operator import add

class State(TypedDict):
    value: int
    log: Annotated[list[str], add]      # `add` = append to the list instead of replacing

def double(state: State) -> dict:
    return {"value": state["value"] * 2, "log": ["doubled"]}     # note: a LIST

def add_ten(state: State) -> dict:
    return {"value": state["value"] + 10, "log": ["added ten"]}
```

```text
{'value': 20, 'log': ['doubled', 'added ten']}
```

That `Annotated[list[str], add]` is how LangGraph keeps a message history — the next lesson
uses `add_messages`, which is the same idea with extra handling for message ids.

:::tip Replace or accumulate — decide per key
`value: int` replaces. `log: Annotated[list, add]` accumulates. Most real state has both:
a current status that replaces, and a history that grows.
:::

## Now add a model

Same structure, with a model call inside one node.

```python title="src/lg/with_model.py"
"""Draft, then check, then finish - with a real model in the draft node."""
from typing import Annotated, TypedDict
from operator import add

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

model = init_chat_model("anthropic:claude-sonnet-5")

class State(TypedDict):
    question: str
    draft: str
    issues: Annotated[list[str], add]

def write_draft(state: State) -> dict:
    reply = model.invoke(
        f"Answer in at most two sentences: {state['question']}"
    )
    return {"draft": reply.text}

def check_draft(state: State) -> dict:
    """A deterministic check - no model needed, so it is free and never wrong."""
    issues = []
    if len(state["draft"]) > 400:
        issues.append("too long")
    if "as an AI" in state["draft"].lower():
        issues.append("boilerplate phrasing")
    return {"issues": issues}

builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("check_draft", check_draft)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "check_draft")
builder.add_edge("check_draft", END)

graph = builder.compile()

result = graph.invoke({"question": "What is a vector database?", "draft": "", "issues": []})
print(result["draft"])
print("issues:", result["issues"])
```

```text
A vector database stores text as numeric vectors so it can retrieve passages by meaning
rather than exact keywords. It is the storage layer behind most RAG systems.
issues: []
```

Notice `check_draft` uses no model. **Deterministic checks belong in plain Python** — free,
instant, and correct every time.

## Seeing the graph

```python
print(graph.get_graph().draw_ascii())
```

```text
        +-----------+
        | __start__ |
        +-----------+
              *
              v
      +-------------+
      | write_draft |
      +-------------+
              *
              v
      +-------------+
      | check_draft |
      +-------------+
              *
              v
         +---------+
         | __end__ |
         +---------+
```

Useful the moment your graph has branches, and the fastest way to spot an edge you forgot.

## Watching it run, step by step

`invoke` gives you the final state. `stream` shows each node as it finishes — which is how
you debug a graph.

```python title="src/lg/streaming_steps.py"
for step in graph.stream(
    {"question": "What is a vector database?", "draft": "", "issues": []},
    stream_mode="updates",         # what each node returned
):
    for node_name, update in step.items():
        print(f"{node_name}: {update}")
```

```text
write_draft: {'draft': 'A vector database stores text as numeric vectors...'}
check_draft: {'issues': []}
```

| `stream_mode` | You get |
| --- | --- |
| `"updates"` | what each node returned — best for debugging |
| `"values"` | the whole state after each step |
| `"messages"` | model tokens as they generate — for a UI |

## Common mistakes

:::mistake Forgetting an edge from START
```python
builder.add_node("work", work)
graph = builder.compile()            # error: no entry point
```
Every graph needs `builder.add_edge(START, "first_node")`. Without it there is nothing to
run first.
:::

:::mistake A key in the update that is not in the State
```python
class State(TypedDict):
    value: int

def node(state): return {"vaule": 1}     # typo - silently does nothing useful
```
LangGraph will complain about unknown channels in most configurations, but a typo in a key
name is still the classic half-hour bug. Keep the state small and let your editor complete
the names.
:::

:::mistake Reaching for a graph when a function would do
If your steps always run in the same order with no loop, no branch and no pause, a plain
function or a LangChain chain is simpler and faster. LangGraph earns its place when you need
branching, looping, or the ability to stop and resume.
:::

## Hands-on Exercise

:::exercise A graph that classifies and logs
Build `src/lg/exercise.py` with a four-node graph and no branching yet:

1. State: `text: str`, `category: str`, `word_count: int`, `trace: Annotated[list[str], add]`
2. `count_words` — deterministic, no model
3. `classify` — one model call returning a category
4. `summarise` — one model call producing a one-line summary
5. `report` — deterministic, builds a final line from the state

Every node must append to `trace`. Then:

- Print the final state and confirm `trace` has **four** entries, in order
- Change `trace` from `Annotated[list[str], add]` to plain `list[str]`, re-run, and explain
  what you see
- Run it with `stream_mode="updates"` and paste the output
:::

:::solution What the reducer change reveals
```text
with  trace: Annotated[list[str], add]
  {'trace': ['counted 9 words', 'classified as technical',
             'summarised', 'report built']}       <- all four, in order

with  trace: list[str]
  {'trace': ['report built']}                     <- only the last node's write survived
```

That is the single most useful thing to internalise about LangGraph state: **the default is
replace.** A history key needs a reducer or it silently keeps only the final write.

```text
stream_mode="updates"
  count_words: {'word_count': 9, 'trace': ['counted 9 words']}
  classify:    {'category': 'technical', 'trace': ['classified as technical']}
  summarise:   {'summary': 'A short note about...', 'trace': ['summarised']}
  report:      {'trace': ['report built']}
```
Each line is one node's return value. When a graph misbehaves, this output tells you which
node produced the wrong shape — far faster than reading the graph definition.
:::

## Challenge

:::challenge Make the same thing twice and compare
Implement a "draft, check, fix if needed, up to 3 attempts" flow **twice**: once as a plain
Python `while` loop, and once as a LangGraph graph with a conditional edge.

Then kill the process halfway through each (a `sys.exit()` inside the fix step will do) and
restart it. Report what each version can recover, and how much code the recovery took.

The plain loop will lose everything, because its state lives in local variables. The graph
with a checkpointer can pick up where it stopped. That difference — not the syntax — is the
reason LangGraph exists, and the next lessons build on it.
:::

## Interview Questions

:::interview
1. What does a LangGraph node receive, and what should it return?
2. Why should a node not mutate the state dict in place?
3. What happens by default when two nodes write to the same state key?
4. What does a reducer like `Annotated[list, add]` change?
5. What are `START` and `END`?
6. When is a plain function a better choice than a graph?
:::

## Cheat Sheet

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int                            # replaced on write
    history: Annotated[list[str], add]    # appended on write

def node(state: State) -> dict:
    return {"value": state["value"] + 1, "history": ["did a thing"]}

builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()

graph.invoke({"value": 0, "history": []})
print(graph.get_graph().draw_ascii())

for step in graph.stream(initial, stream_mode="updates"):
    print(step)
```

## Summary

- A node is `state -> dict of changes`. Return a patch, never mutate.
- Edges say what runs next; `START` and `END` are built-in markers.
- Writing a key **replaces** it unless you give that key a reducer.
- Deterministic checks belong in plain Python nodes, with no model.
- `stream_mode="updates"` is the debugging tool; `draw_ascii()` is the map.
- Use a graph for branching, looping or pausing — not for a straight line.

## Next Step

Next: the state model in depth — reducers, `add_messages`, and how parallel branches write
to the same state without clobbering each other.
