---
title: Human-in-the-Loop Workflows
order: 1
difficulty: Production
duration: 16
badges: ["Production", "Hands-on"]
summary: "Approval gates, escalation, confidence thresholds and interrupt/resume — designing the points where a person decides, without destroying the product's usefulness."
prereqs: ["Guardrails, Prompt Injection and Safety", "LangGraph — Persistence, Interrupts and Multi-Agent Graphs"]
keyConcepts: ["approval gate", "escalation", "confidence threshold", "interrupt/resume", "review queue"]
---

## Why this matters

Human-in-the-loop is what makes an AI system deployable in domains where mistakes are
expensive. It is also the feature most often designed badly: gate everything and nobody uses
the product; gate nothing and one bad decision ends the project. The engineering is in
choosing *which* decisions get a human, and making that review fast enough to happen.

## Mental Model

```text
Gate on EXPECTED COST OF BEING WRONG, not on how uncertain the model feels.

            reversible?  ─────────────────────────────────
                │                                        │
               yes                                       no
                │                                        │
        cost of error?                          ALWAYS review
        │            │                          (or forbid entirely)
      low          high
        │            │
   auto-execute   review if confidence < threshold
```

| Action | Reversible | Blast radius | Gate |
| --- | --- | --- | --- |
| Answer a question | yes | one user | none |
| Draft an email | yes | none until sent | none |
| **Send** an email | no | recipient | review |
| Update a CRM note | yes | one record | sample review |
| Refund $8 | yes (charge again) | small | auto below a limit |
| Refund $4,000 | practically no | large | **always review** |
| Delete an account | no | catastrophic | **review + confirmation** |
| Deploy to production | rollback exists | large | review |

## Core Concepts

### The four patterns

**1. Approval gate** — the run pauses; a human approves or rejects a specific action.
Strongest, slowest. Use for irreversible actions.

**2. Review queue** — the system acts, and a human reviews a sample afterwards. Fast, catches
systematic problems rather than individual ones. Use for reversible, high-volume actions.

**3. Escalation** — the system hands the whole task to a human when it cannot proceed.
Use when the model recognises it is out of scope.

**4. Confidence threshold** — auto-execute above a score, review below it. The dial that
turns the other three into a spectrum.

### Confidence, honestly

A model's stated confidence is weakly calibrated. Build a composite signal instead:

```python
def confidence(*, retrieval_score: float, citation_count: int, model_confidence: float,
               answer_length: int, contradictions: int) -> float:
    """A composite signal. Each component is independently measurable."""
    signals = {
        "retrieval": min(retrieval_score / 0.8, 1.0),          # top chunk similarity
        "grounding": min(citation_count / 2.0, 1.0),           # claims have sources
        "model": model_confidence,                              # weakest signal
        "specificity": 1.0 if 20 <= answer_length <= 250 else 0.6,
        "consistency": 0.0 if contradictions else 1.0,
    }
    weights = {"retrieval": 0.3, "grounding": 0.3, "model": 0.1,
               "specificity": 0.1, "consistency": 0.2}
    return round(sum(signals[k] * weights[k] for k in signals), 3)
```

Then **calibrate the threshold against outcomes**: label 200 answers as good or bad, plot
error rate against confidence, and pick the threshold where the error rate crosses what you
can tolerate. A threshold chosen by intuition is a guess.

### Making review fast enough to happen

A review that takes three minutes will not be done at volume. Design the review payload:

```python
{
    "action": "issue_refund",
    "summary": "Refund $490 to c_902 — duplicate annual charge (INC-204)",   # the headline
    "amount": 490.00,
    "evidence": [
        "Charge ch_88 on 2026-03-01: $490.00 annual plan",
        "Charge ch_89 on 2026-03-01: $490.00 annual plan (duplicate)",
        "Known incident INC-204: migration job duplicated subscriptions",
    ],
    "policy": "Refunds over $50 require approval (policy 4.2)",
    "risk": "low — duplicate confirmed by two independent records",
    "recommended": "approve",
    "conversation_url": "/threads/c_902:conv_2",
    "expires_at": "2026-03-05T10:00:00Z",
}
```

An approver should reach a decision in **under fifteen seconds**: headline, evidence, policy,
recommendation, and a link to the full context for the rare case that needs it.

### The escalation contract

When escalating, hand over work, not a problem:

```text
BAD   "I could not help with this."
GOOD  "Customer asks about 2024 retention policy. I found the 2026 policy (30 days,
       p.4) but no 2024 version in the index. The customer says their contract
       specifies 90 days. Likely a superseded policy document not yet ingested.
       Suggested: check the contracts drive for policy-2024-v2.pdf."
```

The second version saves the human the five minutes the agent already spent.

## Real-World Example

An approval service with queues, expiry, audit and metrics.

```python title="src/hitl/approvals.py"
"""Approval workflow service.

Requirements this satisfies:
  - approvals survive process restarts (they live in a database)
  - every decision is audited with who, when and why
  - stale approvals expire rather than executing on ancient context
  - preconditions are re-validated at resume time
  - reviewer load is measurable
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalRequest:
    id: str
    thread_id: str
    action: str
    summary: str
    payload: dict
    evidence: list[str]
    policy_reference: str
    risk: RiskLevel
    recommendation: str
    requested_at: str
    expires_at: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str = ""
    decided_at: str = ""
    decision_reason: str = ""

    @property
    def expired(self) -> bool:
        return (self.status is ApprovalStatus.PENDING
                and datetime.now(timezone.utc) > datetime.fromisoformat(self.expires_at))

    def render(self) -> str:
        """What the reviewer sees. Optimised for a 15-second decision."""
        lines = [
            f"[{self.risk.upper()}] {self.summary}",
            f"action: {self.action}",
            "evidence:",
            *[f"  - {item}" for item in self.evidence],
            f"policy: {self.policy_reference}",
            f"recommendation: {self.recommendation}",
            f"expires: {self.expires_at}",
        ]
        return "\n".join(lines)


class ApprovalStore:
    """SQLite-backed. Swap for Postgres in production; the interface is the same."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS approvals (
        id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, action TEXT NOT NULL,
        summary TEXT, payload TEXT, evidence TEXT, policy_reference TEXT,
        risk TEXT, recommendation TEXT, requested_at TEXT, expires_at TEXT,
        status TEXT NOT NULL, decided_by TEXT, decided_at TEXT, decision_reason TEXT
    );
    CREATE INDEX IF NOT EXISTS approvals_status_idx ON approvals (status, expires_at);
    CREATE INDEX IF NOT EXISTS approvals_thread_idx ON approvals (thread_id);
    """

    def __init__(self, path: Path = Path(".data/approvals.sqlite")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(self.SCHEMA)

    def create(self, request: ApprovalRequest) -> ApprovalRequest:
        row = asdict(request)
        row["payload"] = json.dumps(row["payload"])
        row["evidence"] = json.dumps(row["evidence"])
        row["risk"] = str(request.risk)
        row["status"] = str(request.status)
        self._connection.execute(
            f"INSERT INTO approvals ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
            tuple(row.values()),
        )
        self._connection.commit()
        logger.info("approval requested", extra={"id": request.id, "action": request.action,
                                                 "risk": str(request.risk)})
        return request

    def get(self, approval_id: str) -> ApprovalRequest | None:
        row = self._connection.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        return self._to_request(row) if row else None

    def pending(self, *, risk: RiskLevel | None = None, limit: int = 50) -> list[ApprovalRequest]:
        query = "SELECT * FROM approvals WHERE status = 'pending'"
        params: list[object] = []
        if risk:
            query += " AND risk = ?"
            params.append(str(risk))
        query += " ORDER BY CASE risk WHEN 'critical' THEN 0 WHEN 'high' THEN 1 " \
                 "WHEN 'medium' THEN 2 ELSE 3 END, requested_at LIMIT ?"
        params.append(limit)

        rows = self._connection.execute(query, params).fetchall()
        return [r for r in (self._to_request(row) for row in rows) if not r.expired]

    def decide(self, approval_id: str, *, approved: bool, approver: str,
               reason: str = "") -> ApprovalRequest:
        request = self.get(approval_id)
        if request is None:
            raise KeyError(f"unknown approval: {approval_id}")
        if request.status is not ApprovalStatus.PENDING:
            raise ValueError(f"approval {approval_id} is already {request.status}")
        if request.expired:
            self._set_status(approval_id, ApprovalStatus.EXPIRED, "", "expired before decision")
            raise ValueError(f"approval {approval_id} expired at {request.expires_at}")

        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        self._set_status(approval_id, status, approver, reason)
        logger.info("approval decided", extra={"id": approval_id, "status": str(status),
                                               "approver": approver})
        return self.get(approval_id)

    def expire_stale(self) -> int:
        cursor = self._connection.execute(
            "UPDATE approvals SET status = 'expired' "
            "WHERE status = 'pending' AND expires_at < ?",
            (datetime.now(timezone.utc).isoformat(),),
        )
        self._connection.commit()
        return cursor.rowcount

    def metrics(self, *, hours: int = 24) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self._connection.execute(
            "SELECT status, risk, COUNT(*) n, "
            "AVG(julianday(decided_at) - julianday(requested_at)) * 86400 AS mean_seconds "
            "FROM approvals WHERE requested_at > ? GROUP BY status, risk", (since,)
        ).fetchall()

        return {
            "window_hours": hours,
            "by_status": {f"{r['risk']}/{r['status']}": r["n"] for r in rows},
            "mean_decision_seconds": {
                f"{r['risk']}": round(r["mean_seconds"] or 0)
                for r in rows if r["status"] in ("approved", "rejected")
            },
            "pending": len(self.pending()),
        }

    def _set_status(self, approval_id: str, status: ApprovalStatus,
                    approver: str, reason: str) -> None:
        self._connection.execute(
            "UPDATE approvals SET status=?, decided_by=?, decided_at=?, decision_reason=? "
            "WHERE id=?",
            (str(status), approver, datetime.now(timezone.utc).isoformat(), reason, approval_id),
        )
        self._connection.commit()

    @staticmethod
    def _to_request(row: sqlite3.Row) -> ApprovalRequest:
        data = dict(row)
        data["payload"] = json.loads(data["payload"])
        data["evidence"] = json.loads(data["evidence"])
        data["risk"] = RiskLevel(data["risk"])
        data["status"] = ApprovalStatus(data["status"])
        return ApprovalRequest(**data)


# --- policy ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """Which actions need a human, and at what threshold. Configuration, not code."""

    auto_approve_below: dict[str, float] = field(default_factory=lambda: {
        "issue_refund": 50.0,
        "apply_credit": 100.0,
    })
    always_review: frozenset[str] = frozenset({
        "delete_account", "change_plan", "send_external_email", "grant_access",
    })
    forbidden: frozenset[str] = frozenset({"delete_all_data", "modify_audit_log"})
    confidence_threshold: float = 0.7
    expiry_hours: int = 24

    def decide(self, action: str, *, amount: float = 0.0,
               confidence: float = 1.0) -> tuple[bool, RiskLevel, str]:
        """Returns (needs_human, risk, reason)."""
        if action in self.forbidden:
            return True, RiskLevel.CRITICAL, f"{action} is never permitted automatically"
        if action in self.always_review:
            return True, RiskLevel.HIGH, f"{action} always requires review"

        limit = self.auto_approve_below.get(action)
        if limit is not None and amount > limit:
            return True, RiskLevel.MEDIUM, f"${amount:.2f} exceeds the ${limit:.2f} auto limit"
        if confidence < self.confidence_threshold:
            return True, RiskLevel.MEDIUM, f"confidence {confidence:.2f} below threshold"

        return False, RiskLevel.LOW, "within automatic limits"


# --- the service -------------------------------------------------------------------
@dataclass
class ApprovalService:
    store: ApprovalStore
    policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)

    def request_if_needed(
        self, *, thread_id: str, action: str, payload: dict, summary: str,
        evidence: list[str], amount: float = 0.0, confidence: float = 1.0,
    ) -> ApprovalRequest | None:
        """Returns None when the action may proceed automatically."""
        needs_human, risk, reason = self.policy.decide(
            action, amount=amount, confidence=confidence
        )
        if not needs_human:
            logger.info("auto-approved", extra={"action": action, "amount": amount,
                                                "confidence": confidence})
            return None

        now = datetime.now(timezone.utc)
        return self.store.create(ApprovalRequest(
            id=f"apr_{uuid.uuid4().hex[:12]}",
            thread_id=thread_id, action=action, summary=summary, payload=payload,
            evidence=evidence, policy_reference=reason, risk=risk,
            recommendation="approve" if confidence > 0.8 else "review carefully",
            requested_at=now.isoformat(),
            expires_at=(now + timedelta(hours=self.policy.expiry_hours)).isoformat(),
        ))

    def resume(self, approval_id: str, *, graph, revalidate) -> dict:
        """Resume the paused graph - after re-checking that the world has not moved on."""
        from langgraph.types import Command

        request = self.store.get(approval_id)
        if request is None or request.status is ApprovalStatus.PENDING:
            raise ValueError("approval is missing or still pending")

        if request.status is ApprovalStatus.APPROVED:
            problems = revalidate(request.action, request.payload)
            if problems:
                logger.warning("preconditions changed since approval",
                               extra={"id": approval_id, "problems": problems})
                return graph.invoke(
                    Command(resume={"approved": False,
                                    "reason": f"preconditions changed: {problems}"}),
                    {"configurable": {"thread_id": request.thread_id}},
                )

        return graph.invoke(
            Command(resume={
                "approved": request.status is ApprovalStatus.APPROVED,
                "approver": request.decided_by,
                "reason": request.decision_reason,
            }),
            {"configurable": {"thread_id": request.thread_id}},
        )
```

```python title="src/hitl/cli.py"
"""Reviewer CLI. A real deployment would use Slack buttons or a web queue."""
from __future__ import annotations

import sys

from .approvals import ApprovalService, ApprovalStore


def review_loop(service: ApprovalService, approver: str) -> None:
    expired = service.store.expire_stale()
    if expired:
        print(f"({expired} stale approvals expired)")

    pending = service.store.pending()
    if not pending:
        print("nothing to review")
        return

    print(f"{len(pending)} pending approvals\n")
    for request in pending:
        print("─" * 72)
        print(request.render())
        choice = input("\n[a]pprove / [r]eject / [s]kip / [q]uit: ").strip().lower()

        if choice == "q":
            return
        if choice == "s":
            continue

        reason = input("reason (optional): ").strip()
        decided = service.store.decide(
            request.id, approved=choice == "a", approver=approver, reason=reason
        )
        print(f"→ {decided.status}\n")


if __name__ == "__main__":
    service = ApprovalService(ApprovalStore())
    if len(sys.argv) > 1 and sys.argv[1] == "metrics":
        print(service.store.metrics())
    else:
        review_loop(service, approver=sys.argv[1] if len(sys.argv) > 1 else "reviewer@acme.com")
```

```text
2 pending approvals

────────────────────────────────────────────────────────────────────────
[MEDIUM] Refund $490 to c_902 — duplicate annual charge
action: issue_refund
evidence:
  - Charge ch_88 on 2026-03-01: $490.00 annual plan
  - Charge ch_89 on 2026-03-01: $490.00 annual plan (duplicate)
  - Known incident INC-204: migration job duplicated subscriptions
policy: $490.00 exceeds the $50.00 auto limit
recommendation: approve
expires: 2026-03-05T10:00:00+00:00

[a]pprove / [r]eject / [s]kip / [q]uit: a
reason (optional): confirmed duplicate, matches INC-204
→ approved
```

```python
service.store.metrics(hours=24)
```

```text
{'window_hours': 24,
 'by_status': {'low/auto_approved': 184, 'medium/approved': 12, 'medium/rejected': 3,
               'high/approved': 2, 'high/pending': 1, 'medium/expired': 2},
 'mean_decision_seconds': {'medium': 412, 'high': 1_840},
 'pending': 1}
```

Those metrics drive the design: 184 auto-approvals to 17 reviews means the thresholds are
roughly right. Two expired approvals means the queue is not being watched closely enough, and
a mean decision time of 31 minutes for high-risk items tells you whether your SLA is real.

## Common Mistakes

:::mistake
```text
1. Gating everything
   Reviewers stop reading and approve by reflex. The gate becomes theatre.

2. Gating on model confidence alone
   Poorly calibrated. Use a composite signal and calibrate against outcomes.

3. Approval payloads that are raw JSON
   A reviewer who cannot decide in 15 seconds will rubber-stamp.

4. No expiry
   An approval granted on three-day-old context executes on a world that moved.

5. No re-validation at resume
   The refund may already have been issued by someone else.

6. Approver identity from the model or the request body
   It must come from your authenticated session.

7. Auditing approvals but not rejections
   Rejections are the signal that your thresholds are wrong.

8. No metrics on review latency
   You cannot tell that the queue has silently stopped being watched.
```
:::

## Hands-on Exercise

:::exercise Design the gates for a real system
For a system you know (internal tool, customer support, code deployment):

1. List every action it can take.
2. For each: reversible? blast radius? cost of being wrong? frequency?
3. Assign a gate: none, sample review, threshold review, always review, or forbidden.
4. Design the approval payload for the two highest-risk actions — aim for a 15-second
   decision.
5. Estimate the reviewer load per day at expected volume. If it exceeds one hour of a
   person's time, adjust the thresholds and re-estimate.
6. Define the expiry policy and what happens on expiry.

The load estimate is the step people skip, and it is the one that determines whether the
design survives contact with a real team.
:::

:::solution Reference design
```text
action                  reversible  radius   freq/day  gate
answer question         yes         1 user      1,200  none
draft reply             yes         none          400  none
send reply to customer  no          1 customer    400  sample 5% + all low-confidence
create ticket           yes         internal      120  none
apply credit < $100     yes         1 customer     40  auto
refund < $50            yes         1 customer     60  auto
refund >= $50           hard        1 customer     14  always review
change plan             yes         1 customer      8  always review
delete account          no          catastrophic    2  review + typed confirmation
bulk email              no          all customers   0  forbidden (no tool)

reviewer load: 14 refunds + 8 plan changes + 2 deletions + ~20 sampled replies
             = 44 reviews/day x 20s = 15 minutes. Sustainable.

Raising the refund auto-limit to $100 would cut reviews to 9/day but expose ~$600/day
of additional auto-refund. Not worth it at 15 minutes of reviewer time.
```
:::

## Challenge

:::challenge Calibrate a confidence threshold with data
Collect 200 answered requests with outcomes (correct/incorrect, or thumbs up/down).

1. Compute the composite confidence for each.
2. Plot error rate against confidence in ten buckets.
3. Find the threshold where the error rate falls below your tolerance (say 5%).
4. Compute the review load at that threshold, and the errors that would still slip through.
5. Produce the trade-off curve: threshold → review load → residual error rate.

Bring that curve to the decision instead of a number. "At 0.72 we review 12% of traffic and
let through 4% errors; at 0.80 we review 24% and let through 2%" is a conversation the
business can actually have.
:::

## Interview Questions

:::interview
1. How do you decide which actions need human approval?
2. Why is model confidence a poor gating signal on its own?
3. What must be re-checked when resuming after an approval?
4. How do you keep reviewers from rubber-stamping?
5. What do you measure about your approval workflow?
:::

## Cheat Sheet

```text
GATE ON     expected cost of being wrong = irreversibility x blast radius x error rate
PATTERNS    approval gate · sample review · escalation · confidence threshold
CONFIDENCE  composite (retrieval + grounding + consistency), calibrated against outcomes
PAYLOAD     headline · evidence · policy reference · recommendation · link · expiry
RESUME      re-validate preconditions · identity from auth · audit approve AND reject
MEASURE     auto/review ratio · decision latency · expiry rate · reviewer load per day
```

```quiz
[
  {
    "question": "Which action most needs an approval gate?",
    "options": [
      "Answering a documentation question",
      "Sending an email to an external customer",
      "Creating an internal ticket",
      "Drafting a reply for review"
    ],
    "answer": 1,
    "explanation": "Gate on irreversibility and blast radius. A sent email cannot be recalled and reaches someone outside your organisation; a draft harms nothing."
  },
  {
    "question": "An approval is granted three days after it was requested. What must happen before executing?",
    "options": [
      "Execute immediately - it was approved",
      "Re-validate preconditions: the charge may already be refunded, the policy may have changed, the customer may have churned",
      "Ask the model to confirm",
      "Extend the expiry"
    ],
    "answer": 1,
    "explanation": "The approval was made on a snapshot. Stale approvals executing on moved-on state is how double refunds happen - hence expiry plus re-validation."
  },
  {
    "question": "Your approval queue shows 340 pending items and a mean decision time of 4 hours. What is the problem?",
    "options": [
      "Reviewers are too slow",
      "The thresholds are too strict: at this volume reviewers will rubber-stamp, making the gate worthless",
      "The model is inaccurate",
      "The queue needs more storage"
    ],
    "answer": 1,
    "explanation": "A gate nobody can keep up with provides no safety. Raise auto-approval limits for low-risk actions, or add reviewers - but do not leave a queue that is approved without reading."
  }
]
```

## Summary

- Gate on the expected cost of being wrong: irreversibility × blast radius, not model
  feelings.
- Four patterns: approval gate, sample review, escalation, confidence threshold.
- Design the review payload for a 15-second decision, or reviewers will rubber-stamp.
- Expire stale approvals and re-validate preconditions at resume.
- Measure auto/review ratio, decision latency and reviewer load — and adjust thresholds with
  those numbers.

## Next Step

Memory and state: what an agent should remember, for how long, and what it must never store.
