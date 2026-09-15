---
title: "LangGraph — Subgraphs, Supervisors and Deployment"
order: 5
difficulty: Architect
duration: 22
badges: ["Production", "Deep dive"]
summary: Composing graphs out of graphs, the supervisor pattern with Command handoffs, streaming to a real UI, and what it takes to run a stateful graph behind an API.
prereqs: ["LangGraph — Persistence, Interrupts and Multi-Agent Graphs"]
keyConcepts: ["subgraph", "supervisor", "Command handoff", "state isolation", "streaming", "deployment"]
---

:::note In one line
**A compiled graph is itself a node.** That one fact gives you subgraphs, and subgraphs are
how a large system stays understandable.
:::

:::warning Versions used on this page
`langgraph` **1.2.11** · `langgraph-checkpoint-sqlite` **3.1.1** · `langchain` **1.4.0**
:::

## Why this matters

A graph with thirty nodes is as hard to read as a function with thirty branches. Subgraphs
are the same fix you already use in ordinary code: give a chunk a name and a boundary.

They also give you something a plain function does not — each subgraph keeps its own state
shape, so a retrieval pipeline does not need to know about the fields a review pipeline uses.

## A compiled graph is a node

```python title="src/lg/subgraph_basic.py"
"""The retrieval pipeline is its own graph, used as one node in the parent."""
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

# ---- the child graph: its own state shape -------------------------------
class RetrievalState(TypedDict):
    question: str
    documents: Annotated[list[str], add]
    rewritten: str

def rewrite(state: RetrievalState) -> dict:
    return {"rewritten": improve_query(state["question"])}

def search(state: RetrievalState) -> dict:
    return {"documents": vector_search(state["rewritten"], k=8)}

def rerank(state: RetrievalState) -> dict:
    return {"documents": top_by_cross_encoder(state["rewritten"], state["documents"], k=4)}

retrieval_builder = StateGraph(RetrievalState)
retrieval_builder.add_node("rewrite", rewrite)
retrieval_builder.add_node("search", search)
retrieval_builder.add_node("rerank", rerank)
retrieval_builder.add_edge(START, "rewrite")
retrieval_builder.add_edge("rewrite", "search")
retrieval_builder.add_edge("search", "rerank")
retrieval_builder.add_edge("rerank", END)

retrieval_graph = retrieval_builder.compile()        # <- now usable as a node

# ---- the parent graph ---------------------------------------------------
class AppState(TypedDict):
    question: str
    documents: Annotated[list[str], add]
    answer: str

def answer(state: AppState) -> dict:
    context = "\n\n".join(state["documents"])
    reply = model.invoke(f"Answer using only this context.\n\n{context}\n\nQ: {state['question']}")
    return {"answer": reply.text}

app_builder = StateGraph(AppState)
app_builder.add_node("retrieve", retrieval_graph)     # the child graph, as one node
app_builder.add_node("answer", answer)
app_builder.add_edge(START, "retrieve")
app_builder.add_edge("retrieve", "answer")
app_builder.add_edge("answer", END)

app = app_builder.compile()
```

The parent knows nothing about `rewritten`. That field exists only inside the child.

:::tip Shared keys are the interface
The child and parent communicate through **keys they both declare** — here `question` and
`documents`. Everything else is private to the child. Treat those shared keys as a deliberate
API, and keep them few.
:::

### When the state shapes do not match

If the child expects different field names, wrap it in a function that translates:

```python title="src/lg/subgraph_adapter.py"
def retrieve_adapter(state: AppState) -> dict:
    """Translate parent state -> child state -> parent update."""
    child_out = retrieval_graph.invoke({
        "question": state["question"],
        "documents": [],
        "rewritten": "",
    })
    return {"documents": child_out["documents"]}

app_builder.add_node("retrieve", retrieve_adapter)
```

This is the more common case in real systems, and it is better than forcing every subgraph to
share one enormous state shape.

## The supervisor pattern

One graph decides who works next; the workers report back. `Command` makes the handoff read
like a sentence.

<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Diagram: a supervisor node routes to one of three worker subgraphs using Command goto, each worker returns to the supervisor, and the supervisor eventually routes to a finish node.">
  <defs>
    <marker id="sp-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
    <marker id="sp-b" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="246" y="26" width="168" height="56" rx="10" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2.2"/>
  <text class="dg-label" x="330" y="48" text-anchor="middle" fill="var(--accent)">supervisor</text>
  <text class="dg-sub"   x="330" y="66" text-anchor="middle">who works next?</text>
  <rect x="24" y="130" width="160" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.7"/>
  <text class="dg-label" x="104" y="152" text-anchor="middle" fill="var(--accent-3)">researcher</text>
  <text class="dg-sub"   x="104" y="170" text-anchor="middle">its own subgraph</text>
  <rect x="250" y="130" width="160" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.7"/>
  <text class="dg-label" x="330" y="152" text-anchor="middle" fill="var(--accent-3)">analyst</text>
  <text class="dg-sub"   x="330" y="170" text-anchor="middle">its own subgraph</text>
  <rect x="476" y="130" width="160" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.7"/>
  <text class="dg-label" x="556" y="152" text-anchor="middle" fill="var(--accent-3)">writer</text>
  <text class="dg-sub"   x="556" y="170" text-anchor="middle">its own subgraph</text>
  <path class="dg-arrow" d="M264,82 L130,126" marker-end="url(#sp-a)"/>
  <path class="dg-arrow" d="M330,82 L330,126" marker-end="url(#sp-a)"/>
  <path class="dg-arrow" d="M396,82 L530,126" marker-end="url(#sp-a)"/>
  <path d="M154,130 Q200,104 262,84" stroke="var(--text-muted)" stroke-width="1.3" fill="none" stroke-dasharray="4 3" marker-end="url(#sp-b)"/>
  <path d="M360,130 Q380,108 372,84" stroke="var(--text-muted)" stroke-width="1.3" fill="none" stroke-dasharray="4 3" marker-end="url(#sp-b)"/>
  <path d="M506,130 Q460,104 398,84" stroke="var(--text-muted)" stroke-width="1.3" fill="none" stroke-dasharray="4 3" marker-end="url(#sp-b)"/>
  <text class="dg-mono" x="330" y="108" text-anchor="middle" fill="var(--accent)" style="font-size:10px">Command(goto=...)</text>
  <rect x="250" y="206" width="160" height="36" rx="8" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.8"/>
  <text class="dg-sub" x="330" y="229" text-anchor="middle" fill="var(--ok)">finish</text>
  <text class="dg-sub" x="24" y="216">workers always return</text>
  <text class="dg-sub" x="24" y="232">to the supervisor</text>
  <text class="dg-sub" x="440" y="216">the supervisor also needs</text>
  <text class="dg-sub" x="440" y="232" fill="var(--danger)">a step cap, or it loops forever</text>
</svg>
<figcaption>
<strong>One decision-maker, several specialists.</strong> Workers never talk to each other,
which keeps the number of possible paths small enough to reason about — the main argument
for a supervisor over peer-to-peer.
</figcaption>
</figure>

```python title="src/lg/supervisor.py"
"""Supervisor with handoffs, a step cap and a clear exit."""
from typing import Annotated, Literal, TypedDict
from operator import add

from langchain.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel, Field

MAX_STEPS = 8

class Decision(BaseModel):
    next: Literal["researcher", "analyst", "writer", "finish"]
    instruction: str = Field(max_length=200, description="what that worker should do")
    reason: str = Field(max_length=160)

class TeamState(TypedDict):
    task: str
    findings: Annotated[list[str], add]
    draft: str
    steps: int
    trace: Annotated[list[str], add]

def supervisor(state: TeamState) -> Command[Literal["researcher", "analyst", "writer", "finish"]]:
    # Hard cap first: never let the model decide whether to stop looping.
    if state["steps"] >= MAX_STEPS:
        return Command(update={"trace": ["step cap reached, finishing"]}, goto="finish")

    decision = model.with_structured_output(Decision).invoke(
        f"Task: {state['task']}\n"
        f"Findings so far: {state['findings'] or 'none'}\n"
        f"Draft exists: {bool(state['draft'])}\n"
        f"Steps used: {state['steps']} of {MAX_STEPS}\n\n"
        "Choose the next worker, or finish if the task is complete."
    )
    return Command(
        update={"steps": state["steps"] + 1,
                "trace": [f"-> {decision.next}: {decision.reason}"]},
        goto=decision.next,
    )

def researcher(state: TeamState) -> Command[Literal["supervisor"]]:
    found = search_tool(state["task"])
    return Command(update={"findings": [found], "trace": ["researched"]}, goto="supervisor")

def analyst(state: TeamState) -> Command[Literal["supervisor"]]:
    reply = model.invoke(f"Analyse these findings:\n{state['findings']}")
    return Command(update={"findings": [reply.text], "trace": ["analysed"]}, goto="supervisor")

def writer(state: TeamState) -> Command[Literal["supervisor"]]:
    reply = model.invoke(f"Write the final answer from:\n{state['findings']}")
    return Command(update={"draft": reply.text, "trace": ["drafted"]}, goto="supervisor")

def finish(state: TeamState) -> dict:
    return {"trace": ["done"]}

builder = StateGraph(TeamState)
for name, fn in [("supervisor", supervisor), ("researcher", researcher),
                 ("analyst", analyst), ("writer", writer), ("finish", finish)]:
    builder.add_node(name, fn)
builder.add_edge(START, "supervisor")
builder.add_edge("finish", END)

team = builder.compile()

result = team.invoke(
    {"task": "Compare pgvector and Qdrant for a 2M-vector workload",
     "findings": [], "draft": "", "steps": 0, "trace": []},
    config={"recursion_limit": 40},
)
```

:::danger The supervisor decides who works, never whether to stop
The `steps >= MAX_STEPS` check runs **before** the model is consulted. If you let the model
own the exit condition, a confused supervisor will ping-pong between two workers until the
recursion limit fires — and you pay for every hop.

Same rule as Lab 6 in Phase 27: the allocator should be arithmetic you can audit, not
judgement you have to trust.
:::

## Streaming a graph to a UI

Users need to see progress, especially when a graph takes twenty seconds.

```python title="src/lg/stream_ui.py"
"""Two useful streams at once: node progress and model tokens."""
async def run_with_progress(payload: dict, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    async for mode, chunk in graph.astream(
        payload, config=config, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node, update in chunk.items():
                yield {"type": "step", "node": node, "keys": list(update)}
        elif mode == "messages":
            message_chunk, metadata = chunk
            if text := getattr(message_chunk, "text", ""):
                yield {"type": "token", "text": text, "node": metadata.get("langgraph_node")}
```

| Mode | Shows | Good for |
| --- | --- | --- |
| `"updates"` | what each node returned | a progress list — "searching", "drafting" |
| `"values"` | the whole state each step | debugging, time travel |
| `"messages"` | model tokens as generated | the typing effect |
| `"custom"` | whatever you emit | your own progress events |

:::tip Progress beats a spinner
A twenty-second spinner feels broken. The same twenty seconds showing "rewriting query →
searching 8 documents → reranking → drafting" feels like work being done. The
`"updates"` stream gives you that for free.
:::

## Running it behind an API

A stateful graph needs three things a stateless chain does not: a durable checkpointer, a
thread id per conversation, and a way to resume an interrupted run.

```python title="src/lg/service.py"
"""FastAPI in front of a checkpointed graph."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

GRAPH = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # One checkpointer for the process, opened once.
    async with AsyncSqliteSaver.from_conn_string(".data/graph.sqlite") as checkpointer:
        GRAPH["app"] = builder.compile(checkpointer=checkpointer)
        yield
    GRAPH.clear()

app = FastAPI(lifespan=lifespan)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(min_length=1, max_length=200)

@app.post("/ask")
async def ask(request: AskRequest):
    config = {"configurable": {"thread_id": request.thread_id},
              "recursion_limit": 30}
    result = await GRAPH["app"].ainvoke({"question": request.question}, config=config)

    if interrupts := result.get("__interrupt__"):
        # The graph paused for approval. Tell the client what it is waiting on.
        return {"status": "needs_approval", "payload": interrupts[0].value,
                "thread_id": request.thread_id}
    return {"status": "complete", "answer": result["answer"]}

class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool

@app.post("/resume")
async def resume(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    result = await GRAPH["app"].ainvoke(Command(resume=request.approved), config=config)
    return {"status": "complete", "answer": result.get("answer")}
```

For production, swap SQLite for Postgres:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
    await checkpointer.setup()           # creates the tables, first run only
    graph = builder.compile(checkpointer=checkpointer)
```

:::production Deployment checklist for a stateful graph
| Check | Why |
| --- | --- |
| Durable checkpointer (Postgres), not in-memory | in-memory loses every paused run on deploy |
| `thread_id` from your own ids, validated | it is a database key; treat it like one |
| `recursion_limit` on every invoke | a bug must fail fast, not bill you |
| Step cap inside supervisors | the model must not own the exit |
| Budget ceiling per thread | one runaway conversation cannot drain the account |
| Interrupted runs visible somewhere | a pause nobody sees is a lost request |
| A cleanup job for stale threads | checkpoints accumulate forever otherwise |
:::

:::warning Paused threads are state you now own
Once a graph can wait for approval, you have a queue of half-finished work. Somebody must be
able to list it, and something must expire it. A graph paused for three weeks with a stale
draft is worse than a failure, because nobody knows it is there.
:::

## Testing a graph

```python title="tests/test_graph.py"
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain.messages import AIMessage

def test_router_sends_billing_to_billing():
    """Test the control flow, not the model."""
    graph = build_graph(model=FakeMessagesListChatModel(responses=[AIMessage("billing")]))
    visited = [list(step)[0] for step in graph.stream(SAMPLE, stream_mode="updates")]
    assert visited == ["classify", "billing", "finish"]

def test_loop_respects_cap(monkeypatch):
    """A check that never passes must still terminate."""
    monkeypatch.setattr("src.lg.loop.check", lambda s: {"issues": ["always bad"]})
    result = graph.invoke(SAMPLE)
    assert result["attempts"] == MAX_ATTEMPTS      # stopped at the cap, did not raise

def test_interrupt_pauses_before_side_effect(tmp_path):
    sent = []
    graph = build_graph(send_email=lambda *a: sent.append(a))
    result = graph.invoke(SAMPLE, config={"configurable": {"thread_id": "t1"}})
    assert result.get("__interrupt__")             # paused
    assert sent == []                             # and crucially, nothing was sent yet
```

That third test is the one that matters most. An approval gate that fires *after* the email
has gone out is not a gate, and this test is how you prove it is in the right place.

## Hands-on Exercise

:::exercise A supervisor with a subgraph and a real pause
Build a graph that:

1. Uses a **subgraph** for retrieval (rewrite → search → rerank) with its own state
2. Has a **supervisor** routing between `retrieve`, `draft` and `review`, with a step cap
   of 6 enforced before the model is asked
3. **Interrupts** before publishing, showing the draft for approval
4. Uses `AsyncSqliteSaver` so the pause survives a process restart
5. Streams `["updates", "messages"]` so you see both progress and tokens

Then prove the persistence: start a run, let it pause, **kill the process**, restart it, and
resume the same `thread_id` to completion.

Report the trace across the restart, and write the three tests above for your graph.
:::

:::solution What the restart proves
```text
process 1
  -> retrieve: rewrote, searched 8, reranked to 4
  -> draft: 180 words
  -> review: flagged 1 issue
  -> draft: 164 words
  -> review: clean
  interrupt: awaiting approval for thread support-4471
  [process killed]

process 2 (fresh start, same sqlite file)
  resume(Command(resume=True), thread_id="support-4471")
  -> publish: sent
  complete

total: 6 nodes across two processes, no work repeated
```

The important detail: process 2 did **not** re-run retrieval or drafting. The checkpointer
had the state after every node, so resuming picked up exactly at the paused node.

```text
tests
  test_router_sends_billing_to_billing   0.02s   passed
  test_loop_respects_cap                 0.01s   passed
  test_interrupt_pauses_before_publish   0.03s   passed  <- and it caught a real bug

The third test failed on the first attempt: the interrupt was placed AFTER the publish
node started, so the email went out and then asked for approval. Moving the interrupt
one node earlier fixed it. No amount of prompt work would have found that.
```
:::

## Challenge

:::challenge Compare a supervisor against plain routing, honestly
Build the same task two ways: a supervisor loop that can call workers repeatedly, and a
single-pass router that picks one worker and returns.

Over 50 realistic tasks, measure: success rate, mean model calls, p95 latency, cost per task,
and how many supervisor runs hit the step cap without finishing.

The result is usually uncomfortable — the supervisor wins on a small number of genuinely
multi-step tasks and loses badly on everything else, because it spends calls deciding rather
than working. Find the fraction of your traffic where it pays, and then decide whether to
route simple tasks straight past it.

That hybrid — cheap router in front, supervisor only for the hard minority — is what most
production systems converge on, and you will have the numbers to justify it.
:::

## Interview Questions

:::interview
1. What makes a compiled graph usable as a node?
2. How do a parent graph and a subgraph share data, and what stays private?
3. Why must a step cap be checked before asking the supervisor model?
4. What does `stream_mode=["updates", "messages"]` give you, and who is each for?
5. Why is an in-memory checkpointer unsuitable for a service with approval gates?
6. What operational problem do paused threads create?
:::

## Cheat Sheet

```python
# a compiled graph is a node
child = child_builder.compile()
parent_builder.add_node("child", child)              # shared keys are the interface

# or adapt when shapes differ
def adapter(state): return {"docs": child.invoke({...})["documents"]}

# supervisor handoff
from langgraph.types import Command
return Command(update={"steps": n + 1}, goto=decision.next)

# streaming both progress and tokens
async for mode, chunk in graph.astream(x, config=cfg, stream_mode=["updates", "messages"]):
    ...

# durable checkpointing
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string(url) as cp:
    await cp.setup()
    graph = builder.compile(checkpointer=cp)

# resume a paused thread
await graph.ainvoke(Command(resume=True), config={"configurable": {"thread_id": tid}})
```

## Summary

- Compiled graphs are nodes; subgraphs keep their own state and expose only shared keys.
- A supervisor routes with `Command(goto=...)`, and the step cap is enforced in code first.
- Stream `updates` for progress and `messages` for tokens — both, usually.
- Behind an API you need a durable checkpointer, validated thread ids and a resume endpoint.
- Paused threads are real state: list them, expire them, alert on them.
- Test the control flow with a fake model, and test that gates fire *before* side effects.

## Next Step

Next: a full project — a support workflow with retrieval, an approval gate and a resume
endpoint, built on everything in this phase.
