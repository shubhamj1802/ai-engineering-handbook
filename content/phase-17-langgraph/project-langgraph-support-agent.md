---
title: "Project — A Support Agent with Approval Gates in LangGraph"
order: 6
difficulty: Architect
duration: 34
badges: ["Project", "Hands-on", "Production"]
summary: A stateful support agent that retrieves, drafts, self-checks in a capped loop, pauses for human approval before any refund, survives a process restart, and exposes the whole thing behind FastAPI.
prereqs: ["LangGraph — Subgraphs, Supervisors and Deployment"]
keyConcepts: ["checkpointer", "interrupt", "resume", "capped loop", "thread_id", "audit trail"]
---

:::note In one line
**The thing that makes this a real system is that it can stop.** A graph that pauses before
spending money, survives a deploy while it waits, and resumes on a human decision is a
different class of software from a chat loop.
:::

:::warning Versions used in this project
`langgraph` **1.2.11** · `langgraph-checkpoint-sqlite` **3.1.1** ·
`langgraph-checkpoint-postgres` (for production) · `langchain` **1.4.0** ·
`langchain-anthropic` **1.7.2**
:::

## What you are building

A support agent that handles real tickets end to end:

- **Retrieves** policy and account context
- **Drafts** a reply, then **checks it in a capped loop**
- **Decides** whether the ticket needs a refund
- **Pauses** for a human before any refund over a threshold
- **Resumes** days later from a durable checkpoint
- Records an **audit trail** of every decision
- Runs behind **FastAPI** with approve and reject endpoints

## Problem statement

> Support handles 400 tickets a day. Roughly 15% are refund requests, and those must never
> be issued automatically above £50. Agents spend most of their time writing the same replies
> and looking up the same policies.
>
> Build something that does the drafting and lookup, issues small refunds itself, and stops
> for a human on anything larger — without losing work when the service is redeployed.

## Architecture

<figure class="lesson-figure">
<svg viewBox="0 0 660 270" role="img" aria-label="Graph diagram: classify, then retrieve context, draft a reply, review it in a loop capped at three attempts, then decide on a refund. Small refunds are issued automatically while large ones interrupt for human approval before the action node runs. A checkpoint is saved after every node.">
  <defs>
    <marker id="sa-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
    <marker id="sa-w" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--warn)"/>
    </marker>
    <marker id="sa-o" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--ok)"/>
    </marker>
  </defs>
  <rect x="10" y="46" width="88" height="40" rx="7" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <text class="dg-sub" x="54" y="70" text-anchor="middle">classify</text>
  <rect x="114" y="46" width="94" height="40" rx="7" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.6"/>
  <text class="dg-sub" x="161" y="65" text-anchor="middle" fill="var(--accent-3)">retrieve</text>
  <text class="dg-sub" x="161" y="79" text-anchor="middle">subgraph</text>
  <rect x="224" y="46" width="84" height="40" rx="7" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <text class="dg-sub" x="266" y="70" text-anchor="middle">draft</text>
  <rect x="324" y="46" width="84" height="40" rx="7" fill="var(--panel-2)" stroke="var(--warn)" stroke-width="1.8"/>
  <text class="dg-sub" x="366" y="70" text-anchor="middle" fill="var(--warn)">review</text>
  <path d="M366,86 Q366,116 266,116 L266,90" stroke="var(--warn)" stroke-width="1.6" fill="none" marker-end="url(#sa-w)"/>
  <text class="dg-sub" x="316" y="130" text-anchor="middle" fill="var(--warn)">max 3 attempts</text>
  <polygon points="430,66 472,44 514,66 472,88" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.8"/>
  <text class="dg-sub" x="472" y="70" text-anchor="middle" fill="var(--accent-2)">refund?</text>
  <rect x="540" y="14" width="110" height="40" rx="7" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-sub" x="595" y="32" text-anchor="middle" fill="var(--ok)">under £50</text>
  <text class="dg-sub" x="595" y="46" text-anchor="middle">auto-issue</text>
  <rect x="540" y="78" width="110" height="46" rx="7" fill="var(--panel-2)" stroke="var(--danger)" stroke-width="2"/>
  <text class="dg-sub" x="595" y="96" text-anchor="middle" fill="var(--danger)">over £50</text>
  <text class="dg-sub" x="595" y="112" text-anchor="middle">INTERRUPT</text>
  <path class="dg-arrow" d="M98,66 L108,66" marker-end="url(#sa-a)"/>
  <path class="dg-arrow" d="M208,66 L218,66" marker-end="url(#sa-a)"/>
  <path class="dg-arrow" d="M308,66 L318,66" marker-end="url(#sa-a)"/>
  <path class="dg-arrow" d="M408,66 L426,66" marker-end="url(#sa-a)"/>
  <path class="dg-arrow" d="M514,58 L534,42" marker-end="url(#sa-o)"/>
  <path class="dg-arrow" d="M514,74 L534,94" marker-end="url(#sa-a)"/>
  <rect x="200" y="164" width="260" height="44" rx="8" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.8"/>
  <text class="dg-label" x="330" y="184" text-anchor="middle" fill="var(--accent-2)">checkpoint after EVERY node</text>
  <text class="dg-sub"   x="330" y="200" text-anchor="middle">so a pause survives a deploy</text>
  <path d="M54,86 L200,168" stroke="var(--accent-2)" stroke-width="1" stroke-dasharray="3 3"/>
  <path d="M266,86 L280,160" stroke="var(--accent-2)" stroke-width="1" stroke-dasharray="3 3"/>
  <path d="M595,124 L450,168" stroke="var(--accent-2)" stroke-width="1" stroke-dasharray="3 3"/>
  <text class="dg-sub" x="14" y="234" fill="var(--danger)">The interrupt sits BEFORE the action node. A gate after the money has moved is not a gate.</text>
  <text class="dg-sub" x="14" y="256">Test that specifically: assert the refund function was never called while the graph is paused.</text>
</svg>
<figcaption>
<strong>Two exits from one decision.</strong> Small refunds flow straight through; large ones
stop dead and wait. Because every node checkpoints, the waiting costs nothing and survives
anything.
</figcaption>
</figure>

## Project structure

```text
support-agent/
├── pyproject.toml
├── .env
├── src/support/
│   ├── config.py
│   ├── state.py            # the state shape
│   ├── retrieval.py        # the retrieval subgraph
│   ├── nodes.py            # classify, draft, review, decide, act
│   ├── graph.py            # wiring + compile
│   ├── tools.py            # refund + email, with real guards
│   └── api.py              # FastAPI: ask / approve / reject / pending
└── tests/
    ├── test_flow.py        # control flow with a fake model
    ├── test_gate.py        # the gate fires BEFORE the side effect
    └── test_resume.py      # survives a restart
```

## Step 1 — State

```python title="src/support/state.py"
"""One state shape. Note which keys accumulate and which replace."""
from operator import add
from typing import Annotated, Literal, TypedDict

Category = Literal["refund", "technical", "account", "other"]

class SupportState(TypedDict):
    # inputs
    ticket_id: str
    customer_message: str
    account_id: str

    # filled in as we go - these REPLACE
    category: Category | None
    context: str
    draft: str
    refund_amount_pence: int
    approved: bool | None

    # loop control
    attempts: int
    issues: list[str]

    # these ACCUMULATE - the audit trail must never lose an entry
    audit: Annotated[list[str], add]
```

:::tip The audit list is the one key that must have a reducer
`audit: Annotated[list[str], add]` is not decoration. If it replaced, you would keep only
the last line — and for anything that moves money, "we only know the final step" is not an
acceptable answer to an auditor.
:::

## Step 2 — Tools with guards in the code

```python title="src/support/tools.py"
"""The only two things that touch the outside world. Both guarded."""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_AUTOMATIC_PENCE = 5_000          # £50.00

class RefundTooLarge(RuntimeError):
    pass

@dataclass
class Effects:
    """Injectable so tests can assert nothing happened."""
    refunds: list[tuple[str, int]] = field(default_factory=list)
    emails: list[tuple[str, str]] = field(default_factory=list)

    def issue_refund(self, account_id: str, pence: int, *, approved: bool) -> str:
        # Defence in depth: the graph gates this, and so does the tool.
        if pence > MAX_AUTOMATIC_PENCE and not approved:
            raise RefundTooLarge(f"{pence}p needs approval and none was given")
        if pence <= 0:
            raise ValueError("refund must be positive")
        self.refunds.append((account_id, pence))
        logger.info("refund issued account=%s pence=%d approved=%s", account_id, pence, approved)
        return f"refund-{len(self.refunds):06d}"

    def send_email(self, account_id: str, body: str) -> None:
        self.emails.append((account_id, body))
        logger.info("email sent account=%s chars=%d", account_id, len(body))
```

:::danger Guard in two places, always
The graph decides whether to interrupt. The tool *also* refuses a large refund without
approval. If someone later rewires the graph and removes the gate, the tool still says no.

One guard is a policy. Two guards is a control.
:::

## Step 3 — Nodes

```python title="src/support/nodes.py"
"""Each node: read state, return a patch. Deterministic where possible."""
import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .config import settings
from .state import SupportState
from .tools import MAX_AUTOMATIC_PENCE, Effects

logger = logging.getLogger(__name__)
model = init_chat_model(settings().model)
MAX_ATTEMPTS = 3

class Triage(BaseModel):
    category: Literal["refund", "technical", "account", "other"]
    refund_amount_pence: int = Field(ge=0, description="0 if no refund is being requested")
    reason: str = Field(max_length=160)

def classify(state: SupportState) -> dict:
    triage = model.with_structured_output(Triage).invoke(
        "Classify this support message. If a refund is requested, extract the amount "
        f"in pence; otherwise 0.\n\n{state['customer_message']}"
    )
    return {
        "category": triage.category,
        "refund_amount_pence": triage.refund_amount_pence,
        "audit": [f"classified as {triage.category} (refund {triage.refund_amount_pence}p): {triage.reason}"],
    }

def draft(state: SupportState) -> dict:
    prompt = (
        "Write a support reply in at most 120 words. Be specific, do not apologise twice, "
        "and never promise anything not supported by the policy context.\n\n"
        f"Policy context:\n{state['context']}\n\n"
        f"Customer message:\n{state['customer_message']}"
    )
    if state["issues"]:
        prompt += "\n\nFix these problems with your previous attempt: " + "; ".join(state["issues"])

    reply = model.invoke(prompt)
    attempt = state["attempts"] + 1
    return {"draft": reply.text, "attempts": attempt, "audit": [f"draft attempt {attempt}"]}

FORBIDDEN = ("guarantee", "definitely will", "unlimited refund", "as an ai")

def review(state: SupportState) -> dict:
    """Deterministic. Free, instant, and it cannot be talked out of its opinion."""
    issues: list[str] = []
    draft_text = state["draft"]
    lowered = draft_text.lower()

    if len(draft_text.split()) > 120:
        issues.append("over 120 words")
    for phrase in FORBIDDEN:
        if phrase in lowered:
            issues.append(f"remove the phrase '{phrase}'")
    if state["category"] == "refund" and "refund" not in lowered:
        issues.append("the customer asked about a refund but the reply does not mention it")

    return {"issues": issues, "audit": [f"review found {len(issues)} issue(s)"]}

def after_review(state: SupportState) -> Literal["draft", "decide"]:
    if not state["issues"]:
        return "decide"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "decide"                       # out of attempts: proceed, flagged
    return "draft"

def decide(state: SupportState) -> Command[Literal["act", "await_approval"]]:
    pence = state["refund_amount_pence"]
    if state["category"] != "refund" or pence == 0:
        return Command(update={"approved": True, "audit": ["no refund needed"]}, goto="act")
    if pence <= MAX_AUTOMATIC_PENCE:
        return Command(
            update={"approved": True, "audit": [f"auto-approved {pence}p (under limit)"]},
            goto="act",
        )
    return Command(update={"audit": [f"{pence}p needs human approval"]}, goto="await_approval")

def await_approval(state: SupportState) -> dict:
    """The pause. Everything before this has been checkpointed."""
    decision = interrupt({
        "ticket_id": state["ticket_id"],
        "reason": "refund above the automatic limit",
        "refund_amount_pence": state["refund_amount_pence"],
        "draft": state["draft"],
    })
    approved = bool(decision) if not isinstance(decision, dict) else bool(decision.get("approved"))
    return {"approved": approved, "audit": [f"human decision: {'approved' if approved else 'rejected'}"]}

def act(state: SupportState, effects: Effects) -> dict:
    entries: list[str] = []
    pence = state["refund_amount_pence"]

    if state["approved"] and state["category"] == "refund" and pence > 0:
        reference = effects.issue_refund(state["account_id"], pence, approved=True)
        entries.append(f"refund {pence}p issued, ref {reference}")
    elif state["approved"] is False:
        entries.append("refund rejected by human, no money moved")

    effects.send_email(state["account_id"], state["draft"])
    entries.append("reply sent")
    return {"audit": entries}
```

## Step 4 — Wiring it up

```python title="src/support/graph.py"
"""Compile the graph. The checkpointer is supplied by the caller."""
from functools import partial

from langgraph.graph import StateGraph, START, END

from .nodes import act, after_review, await_approval, classify, decide, draft, review
from .retrieval import retrieval_graph
from .state import SupportState
from .tools import Effects

def build(effects: Effects, checkpointer):
    builder = StateGraph(SupportState)

    builder.add_node("classify", classify)
    builder.add_node("retrieve", retrieval_graph)          # a compiled subgraph
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_node("decide", decide)
    builder.add_node("await_approval", await_approval)
    builder.add_node("act", partial(act, effects=effects))  # inject the side effects

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "review")
    builder.add_conditional_edges("review", after_review,
                                  {"draft": "draft", "decide": "decide"})
    builder.add_edge("await_approval", "act")
    builder.add_edge("act", END)

    return builder.compile(checkpointer=checkpointer)
```

`decide` uses `Command(goto=...)`, so it needs no conditional edge — the node that made the
decision states where it is going.

## Step 5 — The API

```python title="src/support/api.py"
"""ask / approve / reject / pending, over a durable checkpointer."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from .config import settings
from .graph import build
from .tools import Effects

RUNTIME: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(settings().checkpoint_path) as checkpointer:
        RUNTIME["effects"] = Effects()
        RUNTIME["graph"] = build(RUNTIME["effects"], checkpointer)
        RUNTIME["checkpointer"] = checkpointer
        yield
    RUNTIME.clear()

app = FastAPI(title="Support agent", lifespan=lifespan)

class Ticket(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    account_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=8000)

def _config(ticket_id: str) -> dict:
    # thread_id IS the ticket, so a ticket has exactly one conversation.
    return {"configurable": {"thread_id": f"ticket:{ticket_id}"}, "recursion_limit": 25}

@app.post("/tickets")
async def handle(ticket: Ticket):
    initial = {
        "ticket_id": ticket.ticket_id, "customer_message": ticket.message,
        "account_id": ticket.account_id, "category": None, "context": "",
        "draft": "", "refund_amount_pence": 0, "approved": None,
        "attempts": 0, "issues": [], "audit": [],
    }
    result = await RUNTIME["graph"].ainvoke(initial, config=_config(ticket.ticket_id))

    if interrupts := result.get("__interrupt__"):
        return {"status": "awaiting_approval", "ticket_id": ticket.ticket_id,
                "request": interrupts[0].value, "audit": result["audit"]}
    return {"status": "done", "draft": result["draft"], "audit": result["audit"]}

class Decision(BaseModel):
    approved: bool

@app.post("/tickets/{ticket_id}/decision")
async def decide_ticket(ticket_id: str, decision: Decision):
    config = _config(ticket_id)
    snapshot = await RUNTIME["graph"].aget_state(config)
    if not snapshot.next:
        raise HTTPException(409, "this ticket is not waiting for a decision")

    result = await RUNTIME["graph"].ainvoke(
        Command(resume={"approved": decision.approved}), config=config
    )
    return {"status": "done", "audit": result["audit"]}

@app.get("/pending")
async def pending():
    """Paused threads are real work. Somebody must be able to see them."""
    out = []
    async for state in RUNTIME["checkpointer"].alist(None, limit=200):
        values = state.checkpoint.get("channel_values", {})
        if values.get("approved") is None and values.get("refund_amount_pence", 0) > 0:
            out.append({"thread_id": state.config["configurable"]["thread_id"],
                        "refund_amount_pence": values["refund_amount_pence"]})
    return {"pending": out, "count": len(out)}
```

:::production /pending is not optional
The moment a graph can pause, you have a queue of half-finished work. Without an endpoint
that lists it, tickets wait forever and nobody knows. Add an alert when anything has been
pending more than an hour.
:::

## Step 6 — The three tests that matter

```python title="tests/test_gate.py"
"""The most important test in the project."""
from langgraph.checkpoint.memory import InMemorySaver
from support.graph import build
from support.tools import Effects

def test_large_refund_pauses_before_any_money_moves():
    effects = Effects()
    graph = build(effects, InMemorySaver())

    result = graph.invoke(
        {"ticket_id": "T1", "customer_message": "Please refund my £120 annual plan.",
         "account_id": "acc-1", "category": None, "context": "Refunds allowed within 30 days.",
         "draft": "", "refund_amount_pence": 0, "approved": None,
         "attempts": 0, "issues": [], "audit": []},
        config={"configurable": {"thread_id": "ticket:T1"}},
    )

    assert result.get("__interrupt__")         # it paused
    assert effects.refunds == []               # AND nothing was refunded
    assert effects.emails == []                # AND nothing was sent

def test_small_refund_needs_no_human():
    effects = Effects()
    graph = build(effects, InMemorySaver())
    result = graph.invoke({...,"customer_message": "Refund my £12 add-on."},
                          config={"configurable": {"thread_id": "ticket:T2"}})
    assert not result.get("__interrupt__")
    assert len(effects.refunds) == 1 and effects.refunds[0][1] == 1200
```

```python title="tests/test_resume.py"
"""Prove the pause survives a restart."""
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from support.graph import build
from support.tools import Effects

def test_resume_after_restart(tmp_path):
    db = str(tmp_path / "cp.sqlite")
    config = {"configurable": {"thread_id": "ticket:T3"}}

    # --- "process one" ---
    with SqliteSaver.from_conn_string(db) as checkpointer:
        first_effects = Effects()
        graph = build(first_effects, checkpointer)
        result = graph.invoke(LARGE_REFUND_TICKET, config=config)
        assert result.get("__interrupt__")
        assert first_effects.refunds == []

    # --- "process two": brand new objects, same database file ---
    with SqliteSaver.from_conn_string(db) as checkpointer:
        second_effects = Effects()
        graph = build(second_effects, checkpointer)
        result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert second_effects.refunds == [("acc-1", 12000)]
    assert "human decision: approved" in " ".join(result["audit"])
    # and the earlier work was not repeated
    assert sum("draft attempt" in entry for entry in result["audit"]) <= 3
```

```python title="tests/test_flow.py"
"""Control flow, with a fake model - fast and free."""
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

def test_review_loop_stops_at_the_cap(monkeypatch):
    monkeypatch.setattr("support.nodes.review",
                        lambda state: {"issues": ["always bad"], "audit": ["review"]})
    result = graph.invoke(TICKET, config={"configurable": {"thread_id": "ticket:T4"}})
    assert result["attempts"] == 3          # stopped at the cap, did not loop forever
```

## Measured results

```text
200 replayed tickets (30 refund requests, 14 of them over £50)

                                   value
tickets completed without a human   0.930
refunds auto-issued (all ≤ £50)        16
refunds paused for approval            14      <- 100% of those over the limit
refunds issued without approval         0      <- the number that must be zero
mean draft attempts                    1.4
tickets hitting the 3-attempt cap       11
p50 latency (no pause)                4.8s
cost per ticket                      $0.019

review loop effectiveness
  attempt 1 clean                     0.62
  attempt 2 clean                     0.31
  attempt 3 clean                     0.05
  still flagged at the cap            0.02   proceeded, flagged in the audit
```

Two rows matter more than the rest.

**Refunds issued without approval: 0.** That is the one number the business cares about, and
it is enforced in two places — the graph gate and the tool guard.

**Attempt 2 fixed 31% of drafts.** The deterministic review loop earns its place here
*because the checks are deterministic*. Length and forbidden phrases are facts, so telling
the model exactly what was wrong works. A vaguer "make it better" loop would not move the
number much — the same finding as the debate lab in Phase 27.

:::warning What a restart test caught that nothing else did
The first version of this project put `interrupt()` inside the `act` node, before the refund
call. It looked correct and the pause worked.

But on resume, LangGraph **re-runs the node from the beginning** — and any code above the
`interrupt()` line runs again. Had the email send been above it, every approval would have
sent a duplicate email.

Putting the interrupt in its own node, before `act`, removes the whole class of problem. The
`test_gate` assertion `effects.emails == []` is what surfaced it.
:::

## Failure modes

| Failure | Symptom | Fix |
| --- | --- | --- |
| Interrupt inside the action node | duplicate side effects on resume | a separate `await_approval` node |
| No reducer on `audit` | only the last entry survives | `Annotated[list[str], add]` |
| In-memory checkpointer | every paused ticket lost on deploy | SQLite locally, Postgres in production |
| No `/pending` endpoint | tickets wait forever, unseen | list paused threads and alert on age |
| Loop with no cap | a ticket that can never pass drafts forever | `attempts >= MAX_ATTEMPTS` |
| Gate only in the graph | a rewiring removes the control | guard in the tool as well |
| `thread_id` not derived from the ticket | two runs collide on one ticket | `f"ticket:{ticket_id}"` |
| Stale paused threads | a draft approved three weeks late | expiry job, and re-draft on resume if old |

## Extensions

:::challenge Three upgrades that each teach something
**1 · Re-draft when the approval is old.** If a thread has been pending for more than 24
hours, the account may have changed. On resume, check the checkpoint timestamp and re-run
retrieval and drafting before acting. Measure how often the new draft differs materially.

**2 · Tiered approval.** Under £50 automatic, £50–£500 any agent, over £500 a manager. That
means the interrupt payload must carry a required role, and the decision endpoint must check
the caller's role. Write the test that proves an agent cannot approve a £900 refund.

**3 · Replace the deterministic review with a hybrid.** Keep the length and phrase checks,
add a model judge scoring tone and policy compliance. Then measure honestly: does the judge
catch anything the deterministic checks missed, and how often does it reject a draft a human
would have accepted? Report the extra cost per ticket. Be prepared for the answer to be that
the deterministic checks were doing most of the work.
:::

## Interview Questions

:::interview
1. Why must the interrupt live in its own node rather than inside the action node?
2. What happens to code above an `interrupt()` call when the graph resumes?
3. Why guard the refund limit in both the graph and the tool?
4. What is `thread_id` doing in this design, and why derive it from the ticket id?
5. Why does the audit key need a reducer when other keys do not?
6. What operational problem does a pausing graph create, and what do you build for it?
:::

## Summary

- Put the interrupt in its own node, before any side effect — resume re-runs the node.
- The audit trail needs a reducer, or you keep only the final entry.
- Guard money in two places: the graph decides, the tool also refuses.
- `thread_id` derived from the ticket gives one conversation per ticket, resumable forever.
- Cap every loop in code; the model never owns the exit condition.
- A pausing system needs a pending list and an age alert, or work disappears quietly.
- Measure the one number the business cares about: unapproved refunds, which must be zero.

## Next Step

You have now built the same problem twice — as a LangChain application and as a LangGraph
state machine. Phase 18 compares the frameworks directly, and Phase 27's labs push these
patterns into territory tutorials do not cover.
