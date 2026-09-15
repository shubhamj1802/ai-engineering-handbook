---
title: "Production AI Systems — FastAPI, Docker, Queues and Resilience"
order: 1
difficulty: Architect
duration: 22
badges: ["Production", "Project", "Architect"]
summary: "The reference architecture and the code to ship it: FastAPI with streaming and auth, background jobs, caching, rate limits, fallbacks, Docker, and a deployment checklist."
prereqs: ["Evaluating Agents and Gating Releases", "Logging, Configuration, HTTP Clients and CLIs"]
keyConcepts: ["FastAPI", "streaming", "background jobs", "caching", "rate limiting", "Docker"]
---

:::note In one line
**Everything from the whole handbook, assembled and deployable.** API, queue, cache, limits, fallbacks and cost control.
:::

## Why this matters

Everything so far produces answers. This phase produces a **service**: something with an SLO,
an on-call rotation, a bill, and users who notice when it breaks. The AI-specific parts are a
small fraction of it — which is exactly why the engineering discipline of Phase 2 mattered.

## Mental Model

Everything from the handbook, assembled. The parts that are not the model are most of the
work.
<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Deployment diagram: clients hit an API behind rate limits and authentication, fast requests answer inline while slow ones go on a queue to workers, with a cache in front of the model and Postgres and the vector store behind.">
  <defs>
    <marker id="pr-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="14" y="98" width="76" height="44" rx="8" class="dg-box"/>
  <text class="dg-sub" x="52" y="125" text-anchor="middle">clients</text>
  <rect x="106" y="92" width="110" height="56" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.9"/>
  <text class="dg-label" x="161" y="114" text-anchor="middle" fill="var(--accent)">API</text>
  <text class="dg-sub"   x="161" y="132" text-anchor="middle">auth, rate limit</text>
  <rect x="240" y="44" width="120" height="46" rx="8" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.6"/>
  <text class="dg-sub" x="300" y="64" text-anchor="middle" fill="var(--ok)">quick path</text>
  <text class="dg-sub" x="300" y="80" text-anchor="middle">answer inline</text>
  <rect x="240" y="150" width="120" height="46" rx="8" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.6"/>
  <text class="dg-sub" x="300" y="170" text-anchor="middle" fill="var(--accent-2)">queue</text>
  <text class="dg-sub" x="300" y="186" text-anchor="middle">slow jobs, workers</text>
  <rect x="392" y="92" width="104" height="56" rx="9" fill="var(--panel-2)" stroke="var(--warn)" stroke-width="1.8"/>
  <text class="dg-label" x="444" y="114" text-anchor="middle" fill="var(--warn)">cache</text>
  <text class="dg-sub"   x="444" y="132" text-anchor="middle">skip repeat calls</text>
  <rect x="524" y="44" width="122" height="46" rx="8" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-sub" x="585" y="64" text-anchor="middle" fill="var(--accent)">model</text>
  <text class="dg-sub" x="585" y="80" text-anchor="middle">+ a fallback one</text>
  <rect x="524" y="104" width="122" height="40" rx="8" class="dg-box"/>
  <text class="dg-sub" x="585" y="129" text-anchor="middle">Postgres</text>
  <rect x="524" y="156" width="122" height="40" rx="8" class="dg-box"/>
  <text class="dg-sub" x="585" y="181" text-anchor="middle">vector store</text>
  <path class="dg-arrow" d="M90,120 L100,120" marker-end="url(#pr-a)"/>
  <path class="dg-arrow" d="M216,108 L234,80" marker-end="url(#pr-a)"/>
  <path class="dg-arrow" d="M216,132 L234,166" marker-end="url(#pr-a)"/>
  <path class="dg-arrow" d="M360,72 L390,104" marker-end="url(#pr-a)"/>
  <path class="dg-arrow" d="M360,170 L390,140" marker-end="url(#pr-a)"/>
  <path class="dg-arrow" d="M496,108 L518,80" marker-end="url(#pr-a)"/>
  <path class="dg-arrow" d="M496,120 L518,124" marker-end="url(#pr-a)"/>
  <path class="dg-arrow" d="M496,132 L518,168" marker-end="url(#pr-a)"/>
  <text class="dg-sub" x="14" y="224">Only one box here is the model. The other seven are why production work takes the time it does.</text>
</svg>
<figcaption>
<strong>The queue is the part people skip and regret.</strong> A request that takes ninety
seconds cannot be answered inline — it needs a job id the client can poll, or the first
traffic spike takes the whole service down.
</figcaption>
</figure>

```mermaid
flowchart TB
  C["Client<br/>web · Slack · API"] --> LB["Load balancer / ingress"]
  LB --> API["FastAPI<br/>auth · rate limit · request id · validation"]
  API --> CACHE[("Cache (Redis)<br/>answers · embeddings")]
  API --> SVC["AI service<br/>guardrails → graph → tools"]
  API -.long jobs.-> Q[("Queue<br/>Redis / SQS")]
  Q --> W["Workers<br/>ingestion · batch evals"]
  SVC --> LLM["Model providers<br/>primary + fallback"]
  SVC --> VDB[("Vector DB")]
  SVC --> DB[("Postgres<br/>state · approvals · audit")]
  SVC --> OBS["Traces · metrics · evals"]
  W --> VDB
  W --> DB
```

Three rules that shape everything below:

1. **Nothing slow happens in the request path.** Ingestion, batch evaluation and re-indexing
   are jobs, not endpoints.
2. **Every external dependency can fail.** Each needs a timeout, a retry policy and a
   degraded mode.
3. **Everything is attributable.** One request id threads through API, service, tools, traces
   and logs.

## Core Concepts

### The API layer

```python title="src/service/api.py"
"""FastAPI application.

Nothing here is AI-specific except what it calls - and that is the point. An AI
feature needs the same operational discipline as any other endpoint.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .auth import Principal, require_principal
from .config import get_settings
from .deps import get_cache, get_pipeline, get_rate_limiter
from .observability import configure_logging, trace

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    conversation_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    outcome: str
    request_id: str
    trace_id: str
    cached: bool = False
    latency_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything expensive once, and fail fast if it is not ready."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)

    app.state.pipeline = await build_pipeline(settings)
    app.state.cache = await build_cache(settings)
    app.state.ready = True
    logger.info("service started", extra={"environment": settings.environment})

    yield

    app.state.ready = False
    await app.state.pipeline.aclose()
    await app.state.cache.aclose()
    logger.info("service stopped")


app = FastAPI(title="AI Assistant", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error", extra={"request_id": request_id,
                                                   "path": request.url.path})
        return JSONResponse(
            {"error": "internal error", "request_id": request_id}, status_code=500
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"

    logger.info("request", extra={
        "request_id": request_id, "method": request.method, "path": request.url.path,
        "status": response.status_code, "ms": round(elapsed_ms),
    })
    return response


# --- health and readiness: two different questions --------------------------
@app.get("/health/live")
async def liveness() -> dict:
    """Is the process alive? Never touches dependencies."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness() -> JSONResponse:
    """Can it serve traffic? Checks dependencies with short timeouts."""
    checks: dict[str, bool] = {}
    try:
        async with asyncio.timeout(2.0):
            checks["vector_store"] = await app.state.pipeline.store.health()
            checks["cache"] = await app.state.cache.ping()
            checks["database"] = await app.state.pipeline.db.ping()
    except (TimeoutError, Exception) as exc:
        logger.warning("readiness check failed: %s", exc)

    ready = app.state.ready and all(checks.values())
    return JSONResponse({"status": "ready" if ready else "not_ready", "checks": checks},
                        status_code=200 if ready else 503)


# --- the main endpoint ---------------------------------------------------------
@app.post("/v1/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
    rate_limiter=Depends(get_rate_limiter),
    cache=Depends(get_cache),
    pipeline=Depends(get_pipeline),
) -> AskResponse:
    started = time.perf_counter()

    allowed, retry_after = await rate_limiter.check(principal.id)
    if not allowed:
        raise HTTPException(429, detail="rate limit exceeded",
                            headers={"Retry-After": str(retry_after)})

    cache_key = cache.key(principal.tenant_id, payload.question)
    if cached := await cache.get(cache_key):
        return AskResponse(**cached, request_id=request.state.request_id, cached=True,
                           latency_ms=int((time.perf_counter() - started) * 1000))

    with trace("ask", user_id=principal.id, session_id=payload.conversation_id or "") as t:
        try:
            async with asyncio.timeout(get_settings().request_timeout_s):
                result = await pipeline.answer(
                    payload.question,
                    tenant_id=principal.tenant_id,
                    conversation_id=payload.conversation_id,
                )
        except TimeoutError:
            logger.warning("request timed out", extra={"request_id": request.state.request_id})
            raise HTTPException(504, detail="the request took too long; please retry")

    response = AskResponse(
        answer=result.text, citations=[c.to_api() for c in result.citations],
        outcome=str(result.outcome), request_id=request.state.request_id,
        trace_id=t.trace_id, latency_ms=int((time.perf_counter() - started) * 1000),
    )

    if result.outcome == "answered":
        await cache.set(cache_key, response.model_dump(exclude={"request_id", "latency_ms"}),
                        ttl=600)
    return response


# --- streaming ------------------------------------------------------------------
@app.post("/v1/ask/stream")
async def ask_stream(
    payload: AskRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
    pipeline=Depends(get_pipeline),
) -> StreamingResponse:
    """Server-sent events. Stage events first, then tokens - progress beats a spinner."""

    async def events() -> AsyncIterator[str]:
        request_id = request.state.request_id
        yield f"event: start\ndata: {{\"request_id\": \"{request_id}\"}}\n\n"

        try:
            async for chunk in pipeline.astream(payload.question,
                                                tenant_id=principal.tenant_id):
                if chunk["type"] == "stage":
                    yield f"event: stage\ndata: {json.dumps(chunk)}\n\n"
                elif chunk["type"] == "token":
                    yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
                elif chunk["type"] == "citations":
                    yield f"event: citations\ndata: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            logger.exception("stream failed", extra={"request_id": request_id})
            yield f"event: error\ndata: {json.dumps({'error': str(exc)[:200]})}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",     # nginx must not buffer
    })


# --- long work goes to a queue ------------------------------------------------------
class IngestRequest(BaseModel):
    source_uri: str
    document_type: str = "auto"


@app.post("/v1/documents/ingest", status_code=202)
async def ingest(
    payload: IngestRequest,
    principal: Annotated[Principal, Depends(require_principal)],
    queue=Depends(get_queue),
) -> dict:
    """202 Accepted. Ingestion takes minutes; it never blocks a request."""
    job_id = await queue.enqueue("ingest_document", {
        "source_uri": payload.source_uri, "tenant_id": principal.tenant_id,
        "requested_by": principal.id,
    })
    return {"job_id": job_id, "status": "queued",
            "status_url": f"/v1/jobs/{job_id}"}


@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: str, principal=Depends(require_principal), queue=Depends(get_queue)):
    job = await queue.get(job_id)
    if job is None or job["tenant_id"] != principal.tenant_id:
        raise HTTPException(404, detail="job not found")
    return job
```

### Authentication and multi-tenancy

```python title="src/service/auth.py"
"""Authentication and tenant isolation.

The tenant id comes from the verified token and NEVER from the request body.
Every downstream query is scoped by it.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException

from .config import get_settings


@dataclass(frozen=True, slots=True)
class Principal:
    id: str
    tenant_id: str
    scopes: frozenset[str]
    plan: str = "free"

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise HTTPException(403, detail=f"missing scope: {scope}")


async def require_principal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> Principal:
    settings = get_settings()

    if authorization and authorization.startswith("Bearer "):
        try:
            claims = jwt.decode(
                authorization.removeprefix("Bearer "),
                settings.jwt_public_key, algorithms=["RS256"],
                audience=settings.jwt_audience, options={"require": ["exp", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(401, detail="invalid token") from exc

        return Principal(id=claims["sub"], tenant_id=claims["tenant"],
                         scopes=frozenset(claims.get("scopes", [])),
                         plan=claims.get("plan", "free"))

    if x_api_key:
        record = await api_keys.lookup(hashlib.sha256(x_api_key.encode()).hexdigest())
        if record is None or not hmac.compare_digest(record.key_hash,
                                                     hashlib.sha256(x_api_key.encode()).hexdigest()):
            raise HTTPException(401, detail="invalid api key")
        if record.expires_at and record.expires_at < time.time():
            raise HTTPException(401, detail="api key expired")
        return Principal(id=record.principal_id, tenant_id=record.tenant_id,
                         scopes=frozenset(record.scopes), plan=record.plan)

    raise HTTPException(401, detail="authentication required")
```

### Rate limiting and budgets

```python title="src/service/limits.py"
"""Per-principal rate limits and spend caps.

Two different things: requests per minute protects the service; dollars per day
protects the budget. Both are needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

PLAN_LIMITS = {
    "free":       {"rpm": 10,  "daily_usd": 0.50,  "burst": 5},
    "pro":        {"rpm": 60,  "daily_usd": 10.0,  "burst": 20},
    "enterprise": {"rpm": 600, "daily_usd": 200.0, "burst": 100},
}


@dataclass
class RedisRateLimiter:
    """Sliding window in Redis, so limits hold across replicas."""

    redis: object

    async def check(self, principal_id: str, *, plan: str = "free") -> tuple[bool, int]:
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
        now = time.time()
        key = f"rl:{principal_id}"

        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(key, 0, now - 60)     # drop entries older than a minute
        pipeline.zcard(key)
        pipeline.zadd(key, {f"{now}:{id(now)}": now})
        pipeline.expire(key, 120)
        _, count, _, _ = await pipeline.execute()

        if count >= limits["rpm"]:
            return False, 60
        return True, 0

    async def check_budget(self, principal_id: str, *, plan: str = "free") -> tuple[bool, float]:
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
        key = f"spend:{principal_id}:{time.strftime('%Y-%m-%d')}"
        spent = float(await self.redis.get(key) or 0.0)
        return spent < limits["daily_usd"], spent

    async def record_spend(self, principal_id: str, cost_usd: float) -> None:
        key = f"spend:{principal_id}:{time.strftime('%Y-%m-%d')}"
        await self.redis.incrbyfloat(key, cost_usd)
        await self.redis.expire(key, 172_800)          # two days
```

### Caching

```python title="src/service/cache.py"
"""Two caches with very different hit rates and value."""
from __future__ import annotations

import hashlib
import json
import re


class AnswerCache:
    """Exact-question cache. 10-25% hit rate on a support assistant; free latency."""

    def __init__(self, redis, *, ttl: int = 600) -> None:
        self.redis = redis
        self.ttl = ttl

    @staticmethod
    def key(tenant_id: str, question: str) -> str:
        normalised = re.sub(r"\s+", " ", question.strip().lower())
        digest = hashlib.sha256(f"{tenant_id}\0{normalised}".encode()).hexdigest()[:32]
        return f"answer:{digest}"

    async def get(self, key: str) -> dict | None:
        raw = await self.redis.get(key)
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict, *, ttl: int | None = None) -> None:
        await self.redis.setex(key, ttl or self.ttl, json.dumps(value, default=str))

    async def invalidate_tenant(self, tenant_id: str) -> int:
        """Call this after ingestion: stale answers are worse than no cache."""
        count = 0
        async for key in self.redis.scan_iter(match="answer:*"):
            count += await self.redis.delete(key)
        return count
```

:::warning Cache invalidation is the hard part
An answer cached before a document was updated is now wrong, cited and confident. Invalidate
on ingestion, keep the TTL short (5–15 minutes), and never cache anything the user could
interpret as a commitment.
:::

### Fallbacks and degraded modes

```python title="src/service/resilience.py"
"""Every dependency gets a degraded mode. Decide it now, not during the incident."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEGRADED_MODES = {
    "vector_store_down": "answer from the cache only; otherwise say documentation search "
                         "is temporarily unavailable and offer to create a ticket",
    "primary_model_down": "fall back to the secondary model, and flag the answer",
    "all_models_down": "return a maintenance message with a ticket link",
    "cache_down": "serve normally; log and continue (the cache is an optimisation)",
    "database_down": "read-only mode: answer questions, refuse anything that writes",
}


async def answer_with_fallbacks(pipeline, question: str, *, tenant_id: str) -> dict:
    try:
        return await pipeline.answer(question, tenant_id=tenant_id)

    except VectorStoreError:
        logger.error("vector store unavailable; degrading")
        return {"answer": ("I cannot search the documentation right now. I can create a "
                           "ticket for you, or you can try again in a few minutes."),
                "degraded": True, "reason": "retrieval_unavailable"}

    except ModelUnavailable:
        logger.warning("primary model unavailable; using the fallback")
        try:
            return await pipeline.answer(question, tenant_id=tenant_id, model="fallback")
        except ModelUnavailable:
            return {"answer": "The assistant is temporarily unavailable. Please try again "
                              "shortly or contact support.",
                    "degraded": True, "reason": "all_models_unavailable"}
```

The discipline: **write the degraded behaviour into code before the incident.** Anything not
decided in advance gets decided badly at 3am.

### Docker and deployment

```dockerfile title="Dockerfile"
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

# dependencies first: this layer stays cached while source changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--timeout-keep-alive", "65", "--access-log"]
```

```yaml title="docker-compose.yml"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://app:${DB_PASSWORD}@postgres:5432/app
    env_file: [.env]
    depends_on:
      redis: { condition: service_healthy }
      postgres: { condition: service_healthy }
    deploy:
      resources:
        limits: { cpus: "2", memory: 2G }

  worker:
    build: .
    command: ["python", "-m", "service.worker"]
    environment:
      - REDIS_URL=redis://redis:6379/0
    env_file: [.env]
    depends_on: [redis, postgres]
    deploy:
      replicas: 2

  redis:
    image: redis:7-alpine
    command: ["redis-server", "--maxmemory", "512mb", "--maxmemory-policy", "allkeys-lru"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=app
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app"]
      interval: 10s

volumes:
  pgdata:
```

### The deployment checklist

```text
BEFORE FIRST DEPLOY
  [ ] secrets in a secret manager, not in the image or the repo
  [ ] non-root container, pinned base image, vulnerability scan in CI
  [ ] /health/live and /health/ready, with ready checking dependencies
  [ ] structured logs with a request id; no prompts or PII
  [ ] tracing with cost per request
  [ ] per-principal rate limits AND daily spend caps
  [ ] request timeout shorter than the load balancer's
  [ ] degraded mode written for every dependency
  [ ] evaluation gate in CI with a stored baseline
  [ ] graceful shutdown draining in-flight requests
  [ ] database migrations run separately from the app start
  [ ] rollback tested, not assumed

BEFORE SCALING
  [ ] p95 latency and cost per request dashboards with alerts
  [ ] cache hit rate monitored (a drop is a cost incident)
  [ ] queue depth and worker lag alerts
  [ ] load test at 3x expected peak
  [ ] provider rate limits known, and your concurrency set below them
  [ ] a spend alert at 50%, 80% and 100% of the monthly budget
  [ ] on-call runbook: what each alert means and what to do
```

## Common Mistakes

:::mistake
```text
1. Long work in the request path
   Ingestion in an endpoint: timeouts, retries that re-ingest, unhappy load balancer.

2. One health endpoint
   Liveness must not check dependencies, or a slow database restarts your pods.

3. No request timeout
   A hung provider call holds a worker until the process is restarted.

4. Rate limits without spend caps
   10 requests/minute of a 200k-token agent run is still a very large bill.

5. Caching without invalidation
   Confident, cited, stale answers after a document update.

6. Tenant id from the request body
   Trivial cross-tenant data access. It comes from the verified token, always.

7. No degraded mode
   The whole feature returns 500 because one dependency is slow.

8. Secrets baked into the image
   They are in every layer, every registry copy, and every developer's laptop.
```
:::

## Performance Considerations

| Lever | Typical effect |
| --- | --- |
| Streaming | large perceived-latency win, no cost change |
| Answer cache | 10–25% of requests at ~0 ms and ~$0 |
| Prompt caching | 50–90% of input cost in agent loops |
| Model routing | 2–5× cost reduction when routed well |
| Batching in workers | 50% discount on non-interactive work |
| Connection pooling | 20–50 ms per request |
| `--workers N` | CPU parallelism; size to cores, not to hope |

Scaling order: cache first (free), then route by difficulty, then optimise prompts, then add
replicas. Adding replicas to a cost problem makes the bill worse.

## Hands-on Exercise

:::exercise Ship the service
Take your RAG or agent system and productionise it:

1. FastAPI app: `/v1/ask`, `/v1/ask/stream`, `/v1/documents/ingest` (202 + job status),
   `/health/live`, `/health/ready`.
2. JWT or API-key auth with a tenant id that reaches every query.
3. Redis rate limits and daily spend caps per plan.
4. Answer caching with invalidation on ingestion.
5. A worker process consuming an ingestion queue.
6. Multi-stage Dockerfile (non-root) and a compose file with Redis and Postgres.
7. Degraded modes for vector store, model and cache failures — with tests that force each.
8. A load test at 3× expected peak; report p50/p95/p99, error rate and cost per 1,000
   requests.

The degraded-mode tests are the ones that matter. A fallback that has never been executed is
a comment.
:::

:::solution What the load test should show
```text
load test: 50 concurrent users, 10 minutes, 3x expected peak

requests            18,420
success rate         99.7%
p50 latency          1,210 ms
p95 latency          3,040 ms
p99 latency          6,880 ms
cache hit rate        22.4%
cost per 1k         $11.40
errors                  54  (48 rate-limited by design, 6 provider 529s → fell back)

degraded-mode tests
  vector store down  → 200 with "documentation search unavailable" + ticket offer
  primary model down → 200 via the fallback model, degraded=true
  cache down         → 200, normal latency +8ms, warning logged
  database down      → 200 for reads, 503 for writes with a clear message

p99 is above the 5s SLO under 3x load. Root cause from traces: agent-path requests with
6+ tool calls. Action: lower the agent iteration cap from 8 to 6 and escalate earlier -
the evaluation showed no completion-rate loss at 6.
```
:::

## Challenge

:::challenge Blue/green with an evaluation gate
Build a deployment pipeline where a release only proceeds if quality holds:

1. Deploy the new version alongside the old, taking no traffic.
2. Run the full evaluation suite against the new version's endpoint.
3. Shift 5% of traffic; compare online metrics (answered rate, latency, cost, thumbs) for 30
   minutes.
4. Promote to 100% only if no metric regressed beyond its budget; otherwise roll back
   automatically.
5. Keep the old version warm for one rollback window.

Then practise a rollback under load and measure how long it takes. A rollback you have
executed is a safety net; one you have only designed is a hope.
:::

## Interview Questions

:::interview
1. Why separate liveness from readiness?
2. What belongs in a background job rather than the request path?
3. How do you prevent one tenant from exhausting your model budget?
4. What is your degraded mode when the vector store is unavailable?
5. What would you cache in an AI service, and how do you invalidate it?
:::

## Cheat Sheet

```text
API        auth → rate limit → validate → cache → service → trace → respond
           request id in every log, header and response
HEALTH     /health/live (no dependencies) · /health/ready (dependencies, short timeouts)
ASYNC      202 + job id for anything over ~2 seconds
LIMITS     rpm per principal AND dollars per day; both in Redis so they span replicas
CACHE      answers (short TTL, invalidate on ingest) · embeddings (long TTL)
FALLBACK   every dependency has a written degraded mode, with a test that forces it
DOCKER     multi-stage · deps before source · --no-dev · non-root · healthcheck
DEPLOY     migrations separate · graceful drain · blue/green · tested rollback
SCALE      cache → route → prompt size → replicas (in that order)
```

```quiz
[
  {
    "question": "Why must /health/live not check the database?",
    "options": [
      "It is slower",
      "A slow dependency would fail liveness and make the orchestrator restart healthy pods, turning a degradation into an outage",
      "Databases do not support health checks",
      "It is a security risk"
    ],
    "answer": 1,
    "explanation": "Liveness answers 'is the process wedged?'. Readiness answers 'should it receive traffic?'. Conflating them causes restart storms during dependency slowdowns."
  },
  {
    "question": "A user on the free plan sends 10 requests per minute, each triggering a 20-step agent run. Rate limits are satisfied. What is missing?",
    "options": [
      "Nothing",
      "A spend cap: request-rate limits do not bound cost per request, so a compliant user can still generate an enormous bill",
      "A bigger rate limit",
      "More workers"
    ],
    "answer": 1,
    "explanation": "Requests per minute and dollars per day are independent controls. AI systems need both, because per-request cost varies by orders of magnitude."
  },
  {
    "question": "Your vector database becomes unavailable. What should the API do?",
    "options": [
      "Return 500 for every request",
      "Degrade: serve from cache where possible, otherwise say documentation search is unavailable and offer an alternative such as a ticket",
      "Retry until it recovers",
      "Answer from the model's own knowledge"
    ],
    "answer": 1,
    "explanation": "A written, tested degraded mode keeps the product partially useful and honest. Answering from model memory would silently drop grounding - the worst option."
  }
]
```

## Summary

- Production AI is ordinary service engineering plus cost control and graceful degradation.
- Keep slow work out of the request path; use queues and job status endpoints.
- Auth supplies the tenant id; rate limits and spend caps are separate controls.
- Cache answers with short TTLs and invalidate on ingestion.
- Write and test a degraded mode for every dependency, and rehearse the rollback.

## Next Step

Phase 26: five capstone projects that combine everything — from a document-intelligence RAG
service to a full agent platform.
