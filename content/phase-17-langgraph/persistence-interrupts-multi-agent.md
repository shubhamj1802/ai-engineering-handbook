---
title: "LangGraph — Persistence, Interrupts and Multi-Agent Graphs"
order: 2
difficulty: Expert
duration: 20
badges: ["Hands-on", "Production"]
summary: "Checkpointers, threads and time travel; interrupt/Command for human approval; subgraphs and supervisor topologies — the features that make an agent operable in production."
prereqs: ["LangGraph — State, Nodes, Edges and Reducers"]
keyConcepts: ["checkpointer", "thread_id", "interrupt", "Command", "subgraph", "supervisor"]
---

## Why this matters

An agent that cannot be paused cannot ask a human for permission. An agent that cannot be
resumed loses everything when the process restarts. An agent with no state history cannot be
debugged after the fact. Persistence and interrupts are what turn a demo into something you
can put in front of customers and auditors.

## Mental Model

```text
CHECKPOINTER   saves the full state after every node
THREAD         a conversation, identified by thread_id; its checkpoints are its history
INTERRUPT      a node pauses the graph and returns control to your code
COMMAND        how you resume: Command(resume=value) or Command(goto="node")

  invoke(state, config={"configurable": {"thread_id": "t1"}})
      ↓ runs, saving a checkpoint after each node
      ↓ hits interrupt("approve this refund?")
      ↓ RETURNS to your application - the process can now exit entirely
      ...hours later, possibly a different process...
  invoke(Command(resume="approved"), config={"configurable": {"thread_id": "t1"}})
      ↓ resumes from exactly that node with the full state restored
```

```mermaid
flowchart LR
  A["agent"] --> P{"needs approval?"}
  P -->|no| E["execute"]
  P -->|yes| I["interrupt()"]
  I -.->|graph pauses<br/>state persisted| HUMAN["human reviews<br/>minutes or days later"]
  HUMAN -.->|Command(resume=...)| R["resume at the same node"]
  R --> E
  E --> END([END])
```

## Core Concepts

### Checkpointers

```python
from langgraph.checkpoint.memory import InMemorySaver          # tests and development
from langgraph.checkpoint.sqlite import SqliteSaver            # single-process production
from langgraph.checkpoint.postgres import PostgresSaver        # real production

graph = builder.compile(checkpointer=InMemorySaver())
```

```bash
uv add langgraph-checkpoint-sqlite      # or langgraph-checkpoint-postgres
```

```python
with SqliteSaver.from_conn_string(".data/checkpoints.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke(state, config={"configurable": {"thread_id": "customer-881"}})
```

Once a checkpointer is attached, **every invocation needs a `thread_id`**, and state persists
between calls under that id:

```python
config = {"configurable": {"thread_id": "customer-881"}}

graph.invoke({"messages": [HumanMessage("How much is Pro?")]}, config)
graph.invoke({"messages": [HumanMessage("And Enterprise?")]}, config)   # remembers
```

:::danger `thread_id` is a security boundary
A `thread_id` collision means one user sees another's conversation. Derive it from
authenticated identity — `f"{user_id}:{conversation_id}"` — never from anything a client
supplies unvalidated.
:::

### Inspecting and time-travelling

```python
snapshot = graph.get_state(config)
snapshot.values          # the full state right now
snapshot.next            # which node(s) run next ('()' means finished)
snapshot.tasks           # pending tasks, including interrupts

for checkpoint in graph.get_state_history(config):       # newest first
    print(checkpoint.config["configurable"]["checkpoint_id"], checkpoint.next)

# rewind: resume from an earlier checkpoint
past = list(graph.get_state_history(config))[3]
graph.invoke(None, config=past.config)                   # re-runs from that point

# edit state, then continue (useful for correcting a wrong retrieval)
graph.update_state(config, {"question": "corrected question"})
graph.invoke(None, config)
```

State history is the debugging superpower: when a user reports a bad answer, you can replay
the exact run, inspect the state at every node, change one value and see what would have
happened.

### Interrupts — pausing for a human

```python
from langgraph.types import Command, interrupt


def approval_node(state: State) -> dict:
    decision = interrupt({                       # the graph STOPS here
        "action": "issue_refund",
        "amount": state["refund_amount"],
        "customer": state["customer_id"],
        "reason": state["justification"],
    })
    # execution resumes HERE when Command(resume=...) is supplied
    if decision.get("approved"):
        return {"approved": True, "approver": decision.get("approver")}
    return {"approved": False, "rejection_reason": decision.get("reason", "not approved")}
```

```python
result = graph.invoke(initial_state, config)

if "__interrupt__" in result:
    request = result["__interrupt__"][0].value
    print(f"approval needed: {request}")          # show it to a human, then stop

    # ... the process may exit here; the state is safely persisted ...

    graph.invoke(
        Command(resume={"approved": True, "approver": "alice@acme.com"}),
        config,
    )
```

`interrupt()` is not a callback — it genuinely suspends the run. The state is checkpointed,
your process returns, and hours later a different process can resume the same thread.

You can also interrupt **before or after** specific nodes without modifying them:

```python
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["execute_tools"])
```

### `Command` — dynamic routing from inside a node

```python
from langgraph.types import Command
from typing import Literal


def supervisor(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
    decision = router_model.invoke(state["messages"])
    return Command(
        goto=decision.next_agent,                      # where to go
        update={"messages": [decision.instruction]},   # and what to change
    )
```

`Command` lets a node decide both the update and the destination, which removes many
conditional edges and keeps related logic together.

### Subgraphs

A compiled graph is itself a node:

```python
research_graph = build_research_graph()          # its own state, nodes and edges

parent = StateGraph(ParentState)
parent.add_node("research", research_graph)      # a whole graph as one node
parent.add_node("write", write_node)
```

Subgraphs share state keys with the same names; anything else is private to the subgraph.
Use them for genuinely reusable units — a retrieval pipeline, a verification loop — not to
subdivide a simple flow.

## Real-World Example

A support graph with persistence, a human approval gate on refunds, and a supervisor
delegating to two specialist subgraphs.

```python title="src/graphs/support_graph.py"
"""Customer support graph with persistence and human approval.

Demonstrates the three production features together:
  - a checkpointer, so conversations survive process restarts
  - an interrupt, so irreversible actions wait for a person
  - a supervisor routing to specialist subgraphs
"""
from __future__ import annotations

import logging
from typing import Annotated, Literal
from operator import add

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

REFUND_APPROVAL_THRESHOLD_USD = 50.0


class SupportState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    intent: str
    answer: str
    refund_amount: float
    refund_charge_id: str
    approved: bool
    approver: str
    audit: Annotated[list[dict], add]
    escalated: bool


class Intent(BaseModel):
    """What the customer is asking for."""
    category: Literal["question", "refund", "technical", "other"]
    confidence: float = Field(ge=0.0, le=1.0)
    refund_amount: float = Field(default=0.0, ge=0.0)
    charge_id: str = ""
    reason: str = Field(max_length=200)


model = init_chat_model("anthropic:claude-opus-5", max_tokens=1_024)
cheap_model = init_chat_model("anthropic:claude-haiku-4-5", max_tokens=512)


# --- nodes ---------------------------------------------------------------------
def classify(state: SupportState) -> Command[Literal["answer_question", "prepare_refund",
                                                     "technical", "escalate"]]:
    """Classify, then route - Command carries both the update and the destination."""
    last = state["messages"][-1].content
    intent = cheap_model.with_structured_output(Intent).invoke(
        f"Classify this customer message:\n\n{last}"
    )

    if intent.confidence < 0.6:
        return Command(goto="escalate",
                       update={"intent": "unclear",
                               "audit": [{"node": "classify", "confidence": intent.confidence}]})

    destination = {
        "question": "answer_question",
        "refund": "prepare_refund",
        "technical": "technical",
        "other": "escalate",
    }[intent.category]

    return Command(
        goto=destination,
        update={
            "intent": intent.category,
            "refund_amount": intent.refund_amount,
            "refund_charge_id": intent.charge_id,
            "audit": [{"node": "classify", "intent": intent.category,
                       "confidence": round(intent.confidence, 3)}],
        },
    )


def answer_question(state: SupportState) -> dict:
    """Delegated to the RAG subgraph in the full system; simplified here."""
    answer = rag_graph.invoke({"question": state["messages"][-1].content,
                               "tenant_id": state["customer_id"]})["answer"]
    return {"answer": answer,
            "messages": [AIMessage(answer)],
            "audit": [{"node": "answer_question"}]}


def prepare_refund(state: SupportState) -> dict:
    """Gather the facts a human approver needs. No action taken yet."""
    charges = billing.get_charge(state.get("refund_charge_id", ""))
    return {
        "refund_amount": float(charges.get("amount", state.get("refund_amount", 0.0))),
        "audit": [{"node": "prepare_refund", "charge": charges.get("id")}],
    }


def request_approval(state: SupportState) -> dict:
    """PAUSE. The graph stops here; the state is persisted; the process may exit."""
    amount = state.get("refund_amount", 0.0)

    decision = interrupt({
        "type": "refund_approval",
        "customer_id": state["customer_id"],
        "amount": amount,
        "charge_id": state.get("refund_charge_id"),
        "conversation": [m.content for m in state["messages"][-4:]],
        "policy_note": f"Refunds over ${REFUND_APPROVAL_THRESHOLD_USD:.0f} require approval.",
    })

    # ← resumes here on Command(resume=...)
    approved = bool(decision.get("approved"))
    logger.info("refund %s by %s", "approved" if approved else "rejected",
                decision.get("approver", "unknown"))

    return {
        "approved": approved,
        "approver": decision.get("approver", ""),
        "audit": [{"node": "request_approval", "approved": approved,
                   "approver": decision.get("approver", ""),
                   "reason": decision.get("reason", "")}],
    }


def execute_refund(state: SupportState) -> dict:
    receipt = billing.refund(state["refund_charge_id"], state["refund_amount"])
    message = (f"I have refunded ${state['refund_amount']:.2f} to your original payment "
               f"method. It should appear within 5 business days. Reference {receipt['id']}.")
    return {"answer": message, "messages": [AIMessage(message)],
            "audit": [{"node": "execute_refund", "receipt": receipt["id"]}]}


def decline_refund(state: SupportState) -> dict:
    message = ("I am not able to process this refund automatically. A member of the "
               "billing team will review it and contact you within one business day.")
    return {"answer": message, "messages": [AIMessage(message)],
            "audit": [{"node": "decline_refund"}]}


def technical(state: SupportState) -> dict:
    answer = model.invoke(state["messages"]).text
    return {"answer": answer, "messages": [AIMessage(answer)],
            "audit": [{"node": "technical"}]}


def escalate(state: SupportState) -> dict:
    message = "I am handing this to a human colleague who will reply shortly."
    return {"answer": message, "escalated": True, "messages": [AIMessage(message)],
            "audit": [{"node": "escalate"}]}


# --- routing ---------------------------------------------------------------------
def needs_approval(state: SupportState) -> Literal["request_approval", "execute_refund"]:
    """Small refunds proceed automatically; larger ones wait for a person."""
    return ("request_approval"
            if state.get("refund_amount", 0.0) > REFUND_APPROVAL_THRESHOLD_USD
            else "execute_refund")


def after_approval(state: SupportState) -> Literal["execute_refund", "decline_refund"]:
    return "execute_refund" if state.get("approved") else "decline_refund"


# --- assembly ----------------------------------------------------------------------
def build_support_graph(checkpointer):
    builder = StateGraph(SupportState)

    builder.add_node("classify", classify)
    builder.add_node("answer_question", answer_question)
    builder.add_node("prepare_refund", prepare_refund)
    builder.add_node("request_approval", request_approval)
    builder.add_node("execute_refund", execute_refund)
    builder.add_node("decline_refund", decline_refund)
    builder.add_node("technical", technical)
    builder.add_node("escalate", escalate)

    builder.add_edge(START, "classify")
    # classify uses Command(goto=...), so it needs no outgoing edges
    builder.add_conditional_edges("prepare_refund", needs_approval,
                                  {"request_approval": "request_approval",
                                   "execute_refund": "execute_refund"})
    builder.add_conditional_edges("request_approval", after_approval,
                                  {"execute_refund": "execute_refund",
                                   "decline_refund": "decline_refund"})

    for terminal in ("answer_question", "execute_refund", "decline_refund",
                     "technical", "escalate"):
        builder.add_edge(terminal, END)

    return builder.compile(checkpointer=checkpointer)


# --- the application side ------------------------------------------------------------
def handle_message(graph, *, user_id: str, conversation_id: str, text: str) -> dict:
    """One turn. Returns either an answer or a pending approval request."""
    config = {"configurable": {"thread_id": f"{user_id}:{conversation_id}"}}   # authenticated ids

    result = graph.invoke({"messages": [HumanMessage(text)], "customer_id": user_id}, config)

    if "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        approvals.create(thread_id=config["configurable"]["thread_id"], request=request)
        return {"status": "pending_approval", "request": request,
                "reply": "I need a colleague to approve this. I will update you shortly."}

    return {"status": "answered", "reply": result["answer"], "audit": result.get("audit", [])}


def resume_after_approval(graph, *, thread_id: str, approved: bool, approver: str,
                          reason: str = "") -> dict:
    """Called from the approval UI - possibly days later, in a different process."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        Command(resume={"approved": approved, "approver": approver, "reason": reason}),
        config,
    )
    return {"status": "completed", "reply": result["answer"], "audit": result.get("audit", [])}


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

    with SqliteSaver.from_conn_string(".data/support.sqlite") as checkpointer:
        graph = build_support_graph(checkpointer)
        print(graph.get_graph().draw_ascii(), "\n")

        # --- turn 1: a small refund proceeds automatically ---------------------
        print(handle_message(graph, user_id="c_881", conversation_id="conv_1",
                             text="I was double charged $12 for API overage, please refund it."))

        # --- turn 2: a large refund pauses for approval -------------------------
        pending = handle_message(graph, user_id="c_902", conversation_id="conv_2",
                                 text="Please refund my $490 annual plan, I was charged twice.")
        print(pending)

        # the process could exit here entirely; state is in SQLite

        resumed = resume_after_approval(graph, thread_id="c_902:conv_2",
                                        approved=True, approver="alice@acme.com")
        print(resumed)
```

```text
INFO refund approved by alice@acme.com

{'status': 'answered', 'reply': 'I have refunded $12.00 to your original payment method...',
 'audit': [{'node': 'classify', 'intent': 'refund', 'confidence': 0.94},
           {'node': 'prepare_refund', 'charge': 'ch_3'},
           {'node': 'execute_refund', 'receipt': 'rf_8821'}]}

{'status': 'pending_approval',
 'request': {'type': 'refund_approval', 'customer_id': 'c_902', 'amount': 490.0,
             'policy_note': 'Refunds over $50 require approval.'},
 'reply': 'I need a colleague to approve this. I will update you shortly.'}

{'status': 'completed', 'reply': 'I have refunded $490.00 to your original payment method...',
 'audit': [..., {'node': 'request_approval', 'approved': True, 'approver': 'alice@acme.com'},
           {'node': 'execute_refund', 'receipt': 'rf_8822'}]}
```

The `audit` list is a complete, ordered record of what happened and who approved it —
produced by the graph itself because every node appends to a reduced field. That is what an
auditor will ask for.

### Supervisor with subgraphs

```python title="src/graphs/supervisor.py"
"""A supervisor routing to specialist subgraphs.

Use this only when a single agent genuinely cannot cope (Phase 15): distinct skills,
tool-list bloat, or parallelisable subtasks.
"""
from __future__ import annotations

from typing import Annotated, Literal
from operator import add

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class TeamState(TypedDict, total=False):
    task: str
    findings: Annotated[list[dict], add]
    draft: str
    final: str
    delegations: Annotated[list[str], add]
    rounds: int


class Delegation(BaseModel):
    """Which specialist should work next, and on what."""
    agent: Literal["researcher", "analyst", "writer", "done"]
    instruction: str = Field(max_length=300)
    rationale: str = Field(max_length=200)


MAX_ROUNDS = 6


def supervisor(state: TeamState) -> Command[Literal["researcher", "analyst", "writer", "__end__"]]:
    if state.get("rounds", 0) >= MAX_ROUNDS:                 # the loop cap, as always
        return Command(goto=END, update={"final": state.get("draft", "")
                                         or "Could not complete within the round budget."})

    decision = supervisor_model.with_structured_output(Delegation).invoke(
        f"Task: {state['task']}\n"
        f"Findings so far: {len(state.get('findings', []))}\n"
        f"Draft written: {bool(state.get('draft'))}\n"
        f"Previous delegations: {state.get('delegations', [])}\n\n"
        f"Who should work next?"
    )

    if decision.agent == "done":
        return Command(goto=END, update={"final": state.get("draft", "")})

    return Command(
        goto=decision.agent,
        update={"rounds": state.get("rounds", 0) + 1,
                "delegations": [f"{decision.agent}: {decision.instruction}"]},
    )


def researcher(state: TeamState) -> dict:
    findings = research_subgraph.invoke({"query": state["delegations"][-1]})["findings"]
    return {"findings": findings}


def analyst(state: TeamState) -> dict:
    analysis = analysis_subgraph.invoke({"findings": state.get("findings", [])})["analysis"]
    return {"findings": [{"type": "analysis", "content": analysis}]}


def writer(state: TeamState) -> dict:
    draft = writer_model.invoke(
        f"Task: {state['task']}\n\nFindings:\n{state.get('findings', [])}\n\n"
        f"Write the deliverable. Cite findings by index."
    ).text
    return {"draft": draft}


def build_team():
    builder = StateGraph(TeamState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher)
    builder.add_node("analyst", analyst)
    builder.add_node("writer", writer)

    builder.add_edge(START, "supervisor")
    for specialist in ("researcher", "analyst", "writer"):
        builder.add_edge(specialist, "supervisor")           # always report back

    return builder.compile()
```

```text
supervisor → researcher → supervisor → analyst → supervisor → writer → supervisor → END
delegations:
  researcher: find the 2026 retention policy and any superseding documents
  analyst: compare the 2024 and 2026 retention limits and identify what changed
  writer: draft a two-paragraph summary citing both documents
rounds: 3 · cost: $0.164 · latency: 41s
```

Compare that with a single agent holding all the tools: on this task the single agent
finished in 18 seconds for $0.048. The supervisor version is better only when the specialists
genuinely need different tools, prompts or models — which is the Phase 23 discussion.

## Common Mistakes

:::mistake
```python
# 1. No thread_id with a checkpointer
graph.invoke(state)                                   # raises
graph.invoke(state, config={"configurable": {"thread_id": tid}})

# 2. Client-supplied thread_id
thread_id = request.json["thread_id"]                 # one user can read another's thread
thread_id = f"{authenticated_user.id}:{conversation_id}"

# 3. Treating interrupt() as a callback
decision = interrupt(payload)                          # the graph STOPS; nothing below runs
                                                       # until Command(resume=...) arrives

# 4. Losing the interrupt payload
if "__interrupt__" in result: ...                      # you must persist this for the UI

# 5. InMemorySaver in production
# process restart = every conversation lost. Use SQLite or Postgres.

# 6. Huge state with a checkpointer
# the whole state is serialised after EVERY node; store big blobs by reference

# 7. A supervisor with no round cap
# supervisor → agent → supervisor → agent → ... forever

# 8. Subgraphs for trivial decomposition
# adds state-mapping complexity for no benefit; use plain nodes
```
:::

## Security Considerations

:::security Approval gates are only as strong as their boundary
1. **Identity comes from your auth layer, never from the model.** The approver is the
   authenticated user of the approval UI, not a name the agent produced.
2. **Re-validate on resume.** Between interrupt and resume, the refund may already have been
   issued, the customer may have churned, the policy may have changed. Check preconditions
   again in the node after the interrupt.
3. **Make the payload human-readable.** An approver shown raw JSON approves everything.
   Include the amount, the customer, the reason and the relevant conversation.
4. **Audit both decisions.** Rejections matter as much as approvals.
5. **Expire pending approvals.** A thread interrupted for 30 days should not be resumable
   without a fresh decision.
6. **Rate-limit resume endpoints.** They execute privileged actions by construction.
:::

## Performance Considerations

- Checkpointing writes state after **every node**: keep state lean, store documents by
  reference, and prefer many small nodes only where you need the granularity.
- `PostgresSaver` with connection pooling for production; SQLite is fine for a single process
  and excellent for local development.
- `stream_mode="updates"` avoids serialising full state per step to the client.
- Thread history grows without bound — add a retention policy that deletes checkpoints for
  threads older than N days, or your checkpoint table becomes your largest one.
- Subgraph invocation adds a state-mapping cost; it is small, but do not nest three deep
  without reason.

## Hands-on Exercise

:::exercise Build an approval workflow
Extend your RAG graph with a human gate:

1. Add a `sensitive_action` node that triggers `interrupt()` when the answer would include
   pricing commitments, legal statements or a customer-specific promise.
2. Use `SqliteSaver` so the state survives a process restart — prove it by exiting Python
   between interrupt and resume.
3. Build a small CLI approval tool: list pending threads, show each payload, approve or
   reject with a reason.
4. On resume, re-validate that the conditions still hold before acting.
5. Write tests for: approval, rejection, resume after restart, and a resume attempt on a
   thread that was never interrupted.
6. Add a 24-hour expiry: resuming an older interrupt requires a fresh decision.

The restart test is the one that matters. If it passes, you have a genuinely durable
workflow.
:::

:::solution Key parts
```python title="approval_cli.py"
import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

PENDING = Path(".data/pending_approvals.jsonl")


def list_pending() -> list[dict]:
    if not PENDING.exists():
        return []
    return [json.loads(line) for line in PENDING.read_text(encoding="utf-8").splitlines() if line]


def approve(thread_id: str, approver: str, approved: bool, reason: str = "") -> None:
    with SqliteSaver.from_conn_string(".data/support.sqlite") as checkpointer:
        graph = build_support_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        snapshot = graph.get_state(config)
        if not snapshot.tasks or not any(t.interrupts for t in snapshot.tasks):
            raise ValueError(f"thread {thread_id} is not waiting for approval")

        result = graph.invoke(
            Command(resume={"approved": approved, "approver": approver, "reason": reason}),
            config,
        )
        print(result["answer"])
```

```python title="tests/test_approval.py"
def test_state_survives_a_process_restart(tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "t1"}}

    # "process 1"
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        graph = build_support_graph(checkpointer)
        result = graph.invoke(refund_request_state, config)
        assert "__interrupt__" in result

    # "process 2": entirely new objects, same database
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        graph = build_support_graph(checkpointer)
        snapshot = graph.get_state(config)
        assert snapshot.tasks and snapshot.tasks[0].interrupts      # still pending

        result = graph.invoke(
            Command(resume={"approved": True, "approver": "alice@acme.com"}), config
        )
        assert "refunded" in result["answer"].lower()
```

```text
4 passed in 0.31s
```
:::

## Challenge

:::challenge Time-travel debugging
Build a debugging tool over `get_state_history`:

1. Render every checkpoint of a thread: node name, state diff from the previous checkpoint,
   and elapsed time.
2. Let the operator pick a checkpoint, edit one state field, and re-run from there.
3. Produce a side-by-side comparison of the original and alternate outcomes.

Use it on a real failure: find the checkpoint where the run went wrong, correct the state
(for example, replace a bad retrieval), and show that the outcome changes. This is how you
turn "the agent gave a wrong answer last Tuesday" into a specific, reproducible bug with a
known cause.
:::

## Interview Questions

:::interview
1. What does a checkpointer store, and what does `thread_id` identify?
2. How does `interrupt()` differ from a callback?
3. How do you resume a graph after a human decision, possibly in a different process?
4. What must you re-check after resuming an approval, and why?
5. When is a subgraph the right decomposition, and when is it overhead?
:::

## Cheat Sheet

```python
from langgraph.checkpoint.sqlite import SqliteSaver      # or postgres / memory
from langgraph.types import Command, interrupt

graph = builder.compile(checkpointer=checkpointer,
                        interrupt_before=["execute_tools"])   # optional static gates
config = {"configurable": {"thread_id": f"{user_id}:{conversation_id}"}}

graph.invoke(state, config)
graph.get_state(config).values / .next / .tasks
graph.get_state_history(config)                # newest first: time travel
graph.update_state(config, {"field": "value"}) # edit and continue

decision = interrupt(payload)                   # pauses; returns to your application
graph.invoke(Command(resume=decision_value), config)
return Command(goto="node", update={...})       # route and update from inside a node

builder.add_node("sub", compiled_subgraph)      # a graph as a node

SECURITY  thread_id from authenticated identity · re-validate after resume
          human-readable approval payloads · audit approvals AND rejections · expire old ones
```

```quiz
[
  {
    "question": "What does interrupt() actually do?",
    "options": [
      "Calls a callback function and continues",
      "Suspends the graph, persists state, and returns control to your application until Command(resume=...) is supplied",
      "Raises an exception",
      "Pauses for a fixed timeout"
    ],
    "answer": 1,
    "explanation": "The run genuinely stops. The process can exit entirely; a different process can resume the same thread hours later from the checkpoint."
  },
  {
    "question": "Where should thread_id come from?",
    "options": [
      "A client-supplied request parameter",
      "Authenticated identity, e.g. f'{user_id}:{conversation_id}'",
      "A random UUID per request",
      "The model's output"
    ],
    "answer": 1,
    "explanation": "thread_id selects whose conversation state is loaded. Accepting it from the client lets one user read and resume another's thread."
  },
  {
    "question": "After a human approves a refund three days later, what must the resuming node do first?",
    "options": [
      "Execute the refund immediately",
      "Re-validate preconditions - the charge may already be refunded, the policy may have changed, the customer may have churned",
      "Ask the model to confirm",
      "Create a new thread"
    ],
    "answer": 1,
    "explanation": "State was captured at interrupt time; the world moved on. Re-checking preconditions after resume is what prevents double refunds."
  }
]
```

## Summary

- Checkpointers persist state per thread, making conversations durable and runs replayable.
- `interrupt()` suspends the graph for a human decision; `Command(resume=...)` continues it,
  possibly in another process days later.
- `Command(goto=..., update=...)` lets a node route and update together.
- Subgraphs compose graphs as nodes; supervisors coordinate specialists — with a round cap.
- Approval gates need authenticated identity, readable payloads, re-validation and an audit
  trail.

## Next Step

Phase 18: CrewAI — the role-based approach to multi-agent systems, and an honest comparison
with LangGraph.
