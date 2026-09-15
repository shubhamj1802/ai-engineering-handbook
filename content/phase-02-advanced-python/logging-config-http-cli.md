---
title: Logging, Configuration, HTTP Clients and CLIs
order: 9
difficulty: Intermediate
duration: 15
badges: ["Hands-on", "Production"]
summary: "The operational plumbing every service needs — structured logging, layered settings with Pydantic, a reusable httpx client, and a CLI people can actually use."
prereqs: ["Errors and Exceptions", "Dataclasses, Properties and Enums", "Async Python — asyncio, Concurrency and Parallelism"]
keyConcepts: ["logging", "structured logs", "pydantic-settings", "httpx", "argparse", "correlation id"]
---

:::note In one line
**Never use `print` for anything you want to see in production.** Logging gives you levels, timestamps and somewhere for the output to go.
:::

## Why this matters

This is the boring layer that decides whether your system is operable at 3am. When a user
says "the assistant gave a wrong answer at about 14:20", you need a request id, a structured
log line, the retrieved chunk ids and the model used. When you move from laptop to
container, configuration must come from the environment without a code change. None of this
is AI-specific, and all of it is assumed by every later phase.

## Mental Model

`print` writes a string to the screen and forgets it. A **logger** attaches a level, a
timestamp and a source to every message, then lets you decide where it goes — without
touching the code that wrote it.

<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Diagram: print sends text only to the terminal, while a logger tags each message with a level and sends it to the console, a file or a log service depending on configuration.">
  <defs>
    <marker id="lg-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="14" y="26" width="120" height="44" rx="8" fill="var(--panel-2)" stroke="var(--danger)" stroke-width="1.7"/>
  <text class="dg-mono" x="74" y="47" text-anchor="middle" fill="var(--danger)" style="font-size:11.5px">print()</text>
  <text class="dg-sub"  x="74" y="63" text-anchor="middle">no level, no time</text>
  <rect x="216" y="26" width="150" height="44" rx="8" class="dg-box"/>
  <text class="dg-sub" x="291" y="44" text-anchor="middle">the terminal only</text>
  <text class="dg-sub" x="291" y="61" text-anchor="middle">nowhere else, ever</text>
  <path class="dg-arrow" d="M134,48 L210,48" marker-end="url(#lg-a)"/>
  <line x1="14" y1="88" x2="646" y2="88" stroke="var(--border)" stroke-width="1"/>
  <rect x="14" y="116" width="120" height="58" rx="8" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-mono" x="74" y="138" text-anchor="middle" fill="var(--accent)" style="font-size:11px">logger.info()</text>
  <text class="dg-sub"  x="74" y="155" text-anchor="middle">level + time</text>
  <text class="dg-sub"  x="74" y="169" text-anchor="middle">+ which module</text>
  <rect x="196" y="104" width="130" height="34" rx="6" class="dg-box"/>
  <text class="dg-sub" x="261" y="125" text-anchor="middle">the console</text>
  <rect x="196" y="146" width="130" height="34" rx="6" class="dg-box"/>
  <text class="dg-sub" x="261" y="167" text-anchor="middle">a file</text>
  <rect x="196" y="188" width="130" height="34" rx="6" class="dg-box"/>
  <text class="dg-sub" x="261" y="209" text-anchor="middle">a log service</text>
  <path class="dg-arrow" d="M134,138 Q170,138 190,121" marker-end="url(#lg-a)"/>
  <path class="dg-arrow" d="M134,145 L190,161" marker-end="url(#lg-a)"/>
  <path class="dg-arrow" d="M134,152 Q170,152 190,199" marker-end="url(#lg-a)"/>
  <rect x="380" y="116" width="266" height="86" rx="9" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-label" x="513" y="140" text-anchor="middle" fill="var(--ok)">and you can filter</text>
  <text class="dg-sub"   x="513" y="162" text-anchor="middle">DEBUG while developing</text>
  <text class="dg-sub"   x="513" y="182" text-anchor="middle">WARNING in production</text>
  <path class="dg-arrow" d="M326,161 L374,161" marker-end="url(#lg-a)"/>
</svg>
<figcaption>
<strong>Same call, configurable destination.</strong> The code that logs never needs to know
where the message ends up — which is exactly why you can turn the noise up while debugging
and down in production.
</figcaption>
</figure>

The whole setup is two lines per module:

```python
import logging

logger = logging.getLogger(__name__)     # __name__ = this module's name, automatically

logger.debug("retrieved %d chunks", len(chunks))     # detail, usually hidden
logger.info("request finished in %.0fms", elapsed)   # normal operation
logger.warning("falling back to keyword search")     # something to look at
logger.error("provider call failed", exc_info=True)  # include the traceback
```

| Level | Use it when | Visible in production? |
| --- | --- | --- |
| `debug` | tracing your own logic | no |
| `info` | normal, notable events | usually |
| `warning` | recovered from something bad | yes |
| `error` | the request failed | yes, and someone gets paged |

:::mistake Never put secrets or user data in a log
```python
logger.info("calling API with key %s", api_key)     # now your key is in the log file
logger.info("user asked: %s", question)             # and possibly personal data
```
Logs get shipped to third-party services, kept for months and read by lots of people. Log
*identifiers*, not contents: `logger.info("request %s from tenant %s", request_id, tenant_id)`.
:::

## Core Concepts

### Logging, properly

```python
import logging

logger = logging.getLogger(__name__)      # module-level, named after the module

logger.debug("chunk scores: %s", scores)          # lazy %-formatting, not f-strings
logger.info("retrieved %d chunks in %dms", len(chunks), elapsed_ms)
logger.warning("similarity below threshold: %.3f", score)
logger.error("vector store unavailable")
logger.exception("request failed")                 # inside except: adds the traceback
```

:::mistake Three logging mistakes
```python
print("retrieved", len(chunks))         # no level, no timestamp, no module, unfilterable
logger.info(f"scores: {expensive()}")   # formats even when INFO is disabled
logging.info("...")                     # root logger: no module name in the output
```
Use `logger = logging.getLogger(__name__)` and `%s` placeholders.
:::

Configure once, at the entry point — never in a library module:

```python title="src/service/logging_config.py"
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """One JSON object per line - what log aggregators want."""

    REDACT = {"api_key", "authorization", "password", "token", "email"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # anything passed via logger.info("msg", extra={...}) lands on the record
        for key, value in record.__dict__.items():
            if key in logging.LogRecord("", 0, "", 0, "", (), None).__dict__:
                continue
            if key in {"message", "asctime"}:
                continue
            payload[key] = "[redacted]" if key.lower() in self.REDACT else value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # third-party noise
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
```

### Correlation ids with `contextvars`

```python title="src/service/context.py"
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    request_id = uuid.uuid4().hex[:12]
    _request_id.set(request_id)
    return request_id


def current_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        return True
```

`ContextVar` works correctly with asyncio: each task sees its own value, so concurrent
requests do not overwrite each other's id — a global variable would.

### Configuration with pydantic-settings

```bash
uv add pydantic-settings
```

```python title="src/service/config.py"
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings.

    Precedence: constructor args > environment variables > .env > defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore"
    )

    # --- app
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = True

    # --- model provider
    llm_api_key: SecretStr = Field(..., alias="LLM_API_KEY")
    llm_base_url: str = "https://api.example.com/v1"
    llm_model: str = "small"
    llm_timeout_s: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- retrieval
    retrieval_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- budgets
    max_cost_per_request_usd: float = Field(default=0.10, gt=0)

    @field_validator("llm_base_url")
    @classmethod
    def must_be_https_in_prod(cls, value: str, info) -> str:
        if info.data.get("environment") == "prod" and not value.startswith("https://"):
            raise ValueError("llm_base_url must use https in production")
        return value

    @property
    def debug(self) -> bool:
        return self.environment == "dev"


@lru_cache
def get_settings() -> Settings:
    """Cached: parsed and validated exactly once per process."""
    return Settings()          # raises ValidationError listing EVERY problem at once
```

`SecretStr` prevents the key appearing in `repr()`, logs or tracebacks — you must call
`.get_secret_value()` deliberately.

### A reusable HTTP client

```python title="src/service/adapters/http.py"
import httpx

from ..config import get_settings


def build_client() -> httpx.AsyncClient:
    settings = get_settings()
    return httpx.AsyncClient(
        base_url=settings.llm_base_url,
        headers={
            "authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
            "user-agent": "ai-handbook-service/1.0",
        },
        timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        follow_redirects=False,
    )
```

One client per process. Creating one per request discards connection pooling and can
exhaust file descriptors under load.

### A CLI with argparse

The standard library is enough for most tools; `typer` and `click` are nicer for larger
ones.

```python title="src/service/cli.py"
"""Command line interface.

    uv run python -m service.cli ingest data/corpus --batch-size 32
    uv run python -m service.cli ask "What is HNSW?" --k 8 --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import get_settings
from .context import new_request_id
from .logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="service",
        description="RAG service operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log-level", default=None, help="override the configured log level")
    parser.add_argument("--json", action="store_true", help="machine-readable output")

    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a directory of documents")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--batch-size", type=int, default=64)
    ingest.add_argument("--dry-run", action="store_true", help="parse and chunk, but do not write")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=None, help="chunks to retrieve")

    sub.add_parser("health", help="check configuration and dependencies")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = get_settings()
    except Exception as exc:                       # pydantic ValidationError
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 78                                   # EX_CONFIG

    configure_logging(args.log_level or settings.log_level, json_logs=settings.json_logs)
    request_id = new_request_id()

    try:
        if args.command == "health":
            result = {"ok": True, "environment": settings.environment, "model": settings.llm_model}
        elif args.command == "ingest":
            if not args.path.exists():
                print(f"no such path: {args.path}", file=sys.stderr)
                return 2
            result = {"indexed": 0, "dry_run": args.dry_run, "path": str(args.path)}
        elif args.command == "ask":
            result = {"answer": "…", "citations": [], "k": args.k or settings.retrieval_k}
        else:                                       # pragma: no cover - argparse guarantees
            return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error [{request_id}]: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({**result, "request_id": request_id}))
    else:
        for key, value in result.items():
            print(f"{key:>12}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Register it as a command in `pyproject.toml`:

```toml title="pyproject.toml"
[project.scripts]
service = "service.cli:main"
```

After `uv sync`, `uv run service health` works — that is how a folder of modules becomes a
tool.

:::tip Exit codes are an API
`0` success, `1` general failure, `2` usage error, `78` configuration error, `130`
interrupted. Scripts and CI depend on them; a tool that always exits `0` cannot be
automated.
:::

## Real-World Example

Putting it together — one request, fully instrumented:

```python title="src/service/pipeline.py"
from __future__ import annotations

import logging
import time

from .config import get_settings
from .context import current_request_id, new_request_id

logger = logging.getLogger(__name__)


async def handle_question(question: str, *, user_id: str) -> dict:
    settings = get_settings()
    request_id = new_request_id()
    started = time.perf_counter()

    logger.info(
        "request received",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "question_chars": len(question),      # length, not content: PII discipline
            "model": settings.llm_model,
        },
    )

    try:
        chunks = await retrieve(question, k=settings.retrieval_k)
        logger.info(
            "retrieval complete",
            extra={
                "request_id": request_id,
                "chunk_ids": [c.id for c in chunks],
                "top_score": round(chunks[0].score, 3) if chunks else None,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

        answer = await generate(question, chunks)

    except Exception:
        logger.exception("request failed", extra={"request_id": request_id})
        raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "request complete",
        extra={
            "request_id": request_id,
            "elapsed_ms": elapsed_ms,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "cost_usd": round(answer.cost_usd, 6),
            "citations": len(answer.citations),
        },
    )
    return {**answer.to_api(), "request_id": request_id}
```

Output:

```json
{"ts":"2026-03-04T14:20:11.512Z","level":"INFO","logger":"service.pipeline","message":"retrieval complete","request_id":"9f2c1ab83e40","chunk_ids":["c17","c04","c22"],"top_score":0.883,"elapsed_ms":142}
{"ts":"2026-03-04T14:20:13.004Z","level":"INFO","logger":"service.pipeline","message":"request complete","request_id":"9f2c1ab83e40","elapsed_ms":1634,"input_tokens":2140,"output_tokens":181,"cost_usd":0.003357,"citations":3}
```

Those two lines answer "why did this request cost that much, take that long, and cite those
documents?" — which is exactly what you need at 3am.

## Common Mistakes

:::mistake
```python
# 1. print() instead of logging          → unfilterable, no level, no context
# 2. Logging secrets or full prompts     → your log store becomes a PII store
# 3. configure_logging() in a library    → hijacks the application's configuration
# 4. os.getenv scattered everywhere      → nobody can tell what the app requires
# 5. A new httpx.Client per request      → no connection reuse; fd exhaustion under load
# 6. Config validated lazily on first use → the pod starts healthy, then fails on traffic
# 7. Logging at INFO inside a hot loop   → gigabytes of logs and real slowdown
```
:::

## Security Considerations

:::security What must never reach a log
- API keys, tokens, cookies, authorisation headers.
- Full prompts and full user messages by default (log lengths, hashes and ids instead;
  enable content logging only with explicit consent and a retention policy).
- Customer identifiers beyond an opaque user id.
- Retrieved document *content* — log chunk ids, not chunk text.

Use `SecretStr`, a redacting formatter, and a `repr=False` on secret dataclass fields.
Assume every log line will eventually be read by someone who should not see customer data.
:::

## Performance Considerations

- Guard expensive debug logging: `if logger.isEnabledFor(logging.DEBUG): logger.debug(...)`.
- JSON formatting costs ~10 µs per record — irrelevant per request, noticeable in a loop
  running a million times.
- `lru_cache` on `get_settings()` avoids re-parsing `.env` on every call.
- Reusing one `AsyncClient` typically removes 20–50 ms of handshake per request.

## Hands-on Exercise

:::exercise Instrument the weather CLI
Take the weather client from Phase 0 and upgrade it:

1. Replace the hand-rolled config with `pydantic-settings`, including a `SecretStr` key,
   a validated `timeout_s`, and an `environment` field.
2. Add JSON logging with a request id, and log one line per attempt including status code
   and latency — never the key.
3. Reuse a single `httpx.Client` across calls.
4. Turn it into a proper CLI with subcommands `now <city>` and `health`, a `--json` flag and
   correct exit codes.
5. Prove it: run with a missing key and confirm exit code 78 with a message naming the
   missing variable.
:::

:::solution Key parts of the solution
```python title="src/weather/config.py"
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    weather_api_key: SecretStr = Field(..., alias="OPENWEATHER_API_KEY")
    base_url: str = "https://api.openweathermap.org/data/2.5"
    timeout_s: float = Field(default=10.0, gt=0, le=60)


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python title="src/weather/client.py (excerpt)"
import logging
import time

import httpx

from .config import get_settings
from .context import current_request_id

logger = logging.getLogger(__name__)
_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        settings = get_settings()
        _client = httpx.Client(base_url=settings.base_url, timeout=settings.timeout_s)
    return _client


def get_weather(city: str, *, max_attempts: int = 3) -> dict:
    settings = get_settings()
    params = {"q": city, "appid": settings.weather_api_key.get_secret_value(), "units": "metric"}

    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        response = get_client().get("/weather", params=params)
        logger.info(
            "weather api call",
            extra={
                "request_id": current_request_id(),
                "city": city,
                "status": response.status_code,
                "attempt": attempt,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code not in {429, 500, 502, 503, 504}:
            raise WeatherError(f"status {response.status_code}")
        time.sleep(2 ** (attempt - 1))
    raise WeatherError(f"failed after {max_attempts} attempts")
```

```bash
uv run weather now London --json
```

```text
{"ts":"2026-03-04T09:02:11.104Z","level":"INFO","logger":"weather.client","message":"weather api call","request_id":"3b71d0aa19c4","city":"London","status":200,"attempt":1,"elapsed_ms":184}
{"city":"London","description":"light rain","temperature_c":11.3,"request_id":"3b71d0aa19c4"}
```

With the key removed:

```text
configuration error:
1 validation error for Settings
OPENWEATHER_API_KEY
  Field required [type=missing, input_value={...}, input_type=dict]
exit code: 78
```
:::

## Challenge

:::challenge Config precedence test suite
Write tests proving the precedence chain: defaults < `.env` < environment variable <
explicit constructor argument. Use `monkeypatch.setenv` and `tmp_path` to write a temporary
`.env`. Then add a `settings.redacted_dict()` method that returns every setting with secrets
masked, and assert that no test output ever contains the fake key value. Teams that skip
this discover the precedence only during an incident, when an environment variable turns out
to be silently ignored.
:::

## Interview Questions

:::interview
1. Why `logging` instead of `print`?
2. What is structured logging and why does it matter?
3. How do you thread a request id through an async application?
4. Where should configuration be read and validated?
5. What must never appear in logs, and how do you enforce that?
:::

## Cheat Sheet

```python
logger = logging.getLogger(__name__)
logger.info("event", extra={"request_id": rid, "k": 5})    # %s args, structured extras
logger.exception("failed")                                  # inside except

configure_logging(level, json_logs=True)    # once, at the entry point only

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    api_key: SecretStr = Field(..., alias="API_KEY")
    k: int = Field(default=5, ge=1, le=50)

@lru_cache
def get_settings() -> Settings: return Settings()

client = httpx.AsyncClient(base_url=..., timeout=httpx.Timeout(30, connect=5),
                           limits=httpx.Limits(max_connections=50))

parser.add_subparsers(dest="command", required=True)
[project.scripts] name = "pkg.cli:main"
exit codes: 0 ok · 1 error · 2 usage · 78 config · 130 interrupted
```

```quiz
[
  {
    "question": "Which log call is correct?",
    "options": [
      "logger.info(f'retrieved {len(chunks)} chunks')",
      "logger.info('retrieved %d chunks', len(chunks))",
      "print('retrieved', len(chunks))",
      "logging.info('retrieved chunks')"
    ],
    "answer": 1,
    "explanation": "%-style args are formatted only if the level is enabled, and a module-level logger records where the line came from."
  },
  {
    "question": "Why validate settings at startup rather than on first use?",
    "options": [
      "It is faster",
      "The process fails immediately and visibly instead of after it has started serving traffic",
      "Pydantic requires it",
      "It reduces memory"
    ],
    "answer": 1,
    "explanation": "Fail-fast turns a misconfiguration into a failed deploy rather than a partially broken service and a confusing incident."
  },
  {
    "question": "What should you log about a user's question?",
    "options": [
      "The full question text, always",
      "Length, a hash, and ids - content only under an explicit policy",
      "Nothing at all",
      "The question and the API key for debugging"
    ],
    "answer": 1,
    "explanation": "Logs are widely readable and long-lived. Log metadata by default; content logging needs consent, redaction and retention rules."
  }
]
```

## Summary

- One logger per module, structured `extra` fields, configuration only at the entry point.
- Correlation ids via `contextvars` survive async concurrency; globals do not.
- Settings are validated once at startup with pydantic-settings; secrets use `SecretStr`.
- One HTTP client per process; subcommands and honest exit codes make a CLI automatable.

## Next Step

Packaging and distribution — the last step before Phase 3, where we start working with data
at scale.
