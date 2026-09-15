---
title: "Project 2 — Latency Regression and Serving a Model"
order: 2
difficulty: Advanced
duration: 20
badges: ["Project", "Hands-on", "Production"]
summary: "A regression project end to end, then wrap both models in a FastAPI service with validation, versioning, monitoring and drift detection."
prereqs: ["Project 1 — Support Ticket Escalation Classifier", "Logging, Configuration, HTTP Clients and CLIs"]
keyConcepts: ["regression", "quantile loss", "FastAPI", "model serving", "drift"]
---

:::note In one line
**Training a model is half the job; serving it behind an API is the other half.** Here you do both.
:::

## Why this matters

<figure class="lesson-figure">
<svg viewBox="0 0 660 210" role="img" aria-label="Diagram: an offline training job produces a saved model file, which a web service loads once at startup and then uses to answer prediction requests over HTTP.">
  <defs>
    <marker id="sv2-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="14" y="26" width="290" height="80" rx="10" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.7"/>
  <text class="dg-label" x="30" y="48" fill="var(--accent-3)">Offline — runs occasionally</text>
  <text class="dg-sub"   x="30" y="70">train on history, evaluate, save</text>
  <text class="dg-mono"  x="30" y="92" style="font-size:10.5px">joblib.dump(pipeline, "model.joblib")</text>
  <rect x="330" y="46" width="112" height="44" rx="8" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="2"/>
  <text class="dg-sub" x="386" y="66" text-anchor="middle" fill="var(--ok)">model.joblib</text>
  <text class="dg-sub" x="386" y="82" text-anchor="middle">the whole pipeline</text>
  <path class="dg-arrow" d="M304,68 L324,68" marker-end="url(#sv2-a)"/>
  <rect x="14" y="124" width="632" height="76" rx="10" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-label" x="30" y="146" fill="var(--accent)">Online — runs on every request</text>
  <rect x="30" y="156" width="130" height="34" rx="6" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <text class="dg-sub" x="95" y="177" text-anchor="middle">load ONCE at startup</text>
  <rect x="180" y="156" width="120" height="34" rx="6" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <text class="dg-sub" x="240" y="177" text-anchor="middle">validate input</text>
  <rect x="320" y="156" width="120" height="34" rx="6" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <text class="dg-sub" x="380" y="177" text-anchor="middle">predict</text>
  <rect x="460" y="156" width="170" height="34" rx="6" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <text class="dg-sub" x="545" y="177" text-anchor="middle">return JSON + log it</text>
  <path class="dg-arrow" d="M160,173 L176,173"/>
  <path class="dg-arrow" d="M300,173 L316,173"/>
  <path class="dg-arrow" d="M440,173 L456,173"/>
  <path class="dg-arrow" d="M386,90 Q386,112 200,112 L95,112 L95,150" stroke="var(--ok)" stroke-width="1.5" fill="none" stroke-dasharray="4 3" marker-end="url(#sv2-a)"/>
  <text class="dg-sub" x="470" y="114" fill="var(--danger)">Loading the model per request is the classic mistake — it makes every call far slower.</text>
</svg>
<figcaption>
<strong>Save the whole pipeline, not just the model.</strong> The scaler and encoders must
travel with it, or serving applies different transformations than training did — and the
predictions quietly go wrong.
</figcaption>
</figure>

A model in a notebook has produced no value. This lesson closes the loop: a second project
that predicts a *number* (with the different metrics and failure modes that implies), and
then the service that makes both models usable — with input validation, versioning, latency
budgets and drift monitoring. The FastAPI patterns here reappear in Phases 25 and 26.

## Part 1 — Predicting request latency

### Problem statement

> Our RAG service has a 3-second p95 SLO. **Predict the latency of a request before running
> it**, from features known at request time (question length, k, model tier, index size,
> time of day, recent load). If predicted latency exceeds the budget, degrade gracefully:
> fewer chunks, a smaller model, or an immediate "this will take a moment" response.
>
> We care about *not under-predicting*: a request we promised in 2s that takes 8s is worse
> than one we conservatively estimated at 5s.

That last sentence changes the loss function. Symmetric error is not what we want.

### Metrics for regression

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y, pred)                     # "average error in ms" - interpretable
rmse = mean_squared_error(y, pred) ** 0.5              # punishes big misses
r2 = r2_score(y, pred)                                 # share of variance explained

# The one that matters here: how often do we under-promise?
under_rate = (pred < y).mean()
p90_under = np.percentile(np.maximum(y - pred, 0), 90)  # worst-case under-prediction
```

:::tip Asymmetric costs need an asymmetric loss
Squared error treats "50ms too slow" and "50ms too fast" identically. Quantile regression
does not: fitting the 0.9 quantile produces predictions that are above the true value 90% of
the time — exactly the conservative estimate this problem asked for.
:::

### Implementation

```python title="src/latency/train.py"
"""Latency regression with a quantile objective.

We fit three models:
  - median (q=0.5)  the honest central estimate
  - q=0.9           the conservative estimate used for admission control
  - mean (squared)  for comparison, to show what symmetric loss gives us
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["question_chars", "k", "index_size_k", "hour", "concurrent_requests",
           "history_turns", "reranker_enabled"]
CATEGORICAL = ["model_tier", "retrieval_strategy"]
FEATURES = NUMERIC + CATEGORICAL


def build(quantile: float | None) -> Pipeline:
    """quantile=None → squared error (predicts the mean)."""
    estimator = HistGradientBoostingRegressor(
        loss="quantile" if quantile is not None else "squared_error",
        quantile=quantile,
        max_iter=400,
        learning_rate=0.06,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",            # boosting needs no scaling
        verbose_feature_names_out=False,
    )
    return Pipeline([("prep", preprocessor), ("reg", estimator)])


@dataclass(frozen=True, slots=True)
class RegressionReport:
    name: str
    mae: float
    rmse: float
    r2: float
    under_prediction_rate: float
    p90_under_ms: float
    mean_over_ms: float

    def row(self) -> dict[str, float | str]:
        return {
            "model": self.name,
            "mae_ms": round(self.mae, 1),
            "rmse_ms": round(self.rmse, 1),
            "r2": round(self.r2, 3),
            "under_rate": round(self.under_prediction_rate, 3),
            "p90_under_ms": round(self.p90_under_ms, 1),
            "mean_over_ms": round(self.mean_over_ms, 1),
        }


def report(name: str, y_true: np.ndarray, predicted: np.ndarray) -> RegressionReport:
    under = np.maximum(y_true - predicted, 0)      # how much we under-promised
    over = np.maximum(predicted - y_true, 0)       # wasted headroom
    return RegressionReport(
        name=name,
        mae=float(mean_absolute_error(y_true, predicted)),
        rmse=float(mean_squared_error(y_true, predicted) ** 0.5),
        r2=float(r2_score(y_true, predicted)),
        under_prediction_rate=float((predicted < y_true).mean()),
        p90_under_ms=float(np.percentile(under, 90)),
        mean_over_ms=float(over.mean()),
    )


def train_all(train_df: pd.DataFrame, test_df: pd.DataFrame, *, label: str = "latency_ms"):
    X_train, y_train = train_df[FEATURES], train_df[label]
    X_test, y_test = test_df[FEATURES], test_df[label]

    models = {
        "mean (squared error)": build(None),
        "median (q=0.5)": build(0.5),
        "conservative (q=0.9)": build(0.9),
    }

    reports, fitted = [], {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        reports.append(report(name, y_test.to_numpy(), pipeline.predict(X_test)))
        fitted[name] = pipeline

    return fitted, pd.DataFrame(r.row() for r in reports)


def save(models: dict[str, Pipeline], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "latency_v1.joblib"
    joblib.dump({
        "median": models["median (q=0.5)"],
        "conservative": models["conservative (q=0.9)"],
        "features": FEATURES,
        "schema_version": 1,
    }, path)
    return path


def synthetic_requests(n: int = 25_000, seed: int = 7) -> pd.DataFrame:
    """Latency has structure: model tier dominates, load and k add on top,
    and there is a heavy right tail - exactly like the real thing."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "question_chars": rng.integers(20, 800, n),
        "k": rng.integers(3, 20, n),
        "index_size_k": rng.choice([50, 200, 800], n, p=[0.3, 0.5, 0.2]),
        "hour": rng.integers(0, 24, n),
        "concurrent_requests": rng.poisson(6, n),
        "history_turns": rng.integers(0, 12, n),
        "reranker_enabled": rng.random(n) < 0.4,
        "model_tier": rng.choice(["small", "medium", "large"], n, p=[0.5, 0.35, 0.15]),
        "retrieval_strategy": rng.choice(["vector", "hybrid"], n, p=[0.6, 0.4]),
    })

    base = df["model_tier"].map({"small": 420, "medium": 950, "large": 2_100})
    latency = (
        base
        + df["k"] * 18
        + df["question_chars"] * 0.6
        + df["history_turns"] * 40
        + df["reranker_enabled"] * 320
        + (df["retrieval_strategy"] == "hybrid") * 180
        + df["index_size_k"] * 0.35
        + df["concurrent_requests"] ** 1.4 * 12
    )
    # heavy tail: 4% of requests hit a retry or a cold cache
    latency *= rng.lognormal(0, 0.22, n)
    latency += (rng.random(n) < 0.04) * rng.gamma(2, 900, n)

    df["latency_ms"] = latency.round()
    df["reranker_enabled"] = df["reranker_enabled"].astype(int)
    return df


if __name__ == "__main__":
    df = synthetic_requests()
    cut = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    models, table = train_all(train_df, test_df)
    print(table.to_string(index=False))

    budget_ms = 3_000
    conservative = models["conservative (q=0.9)"].predict(test_df[FEATURES])
    would_degrade = conservative > budget_ms
    actually_slow = test_df["latency_ms"] > budget_ms

    print(f"\nadmission control at {budget_ms}ms budget:")
    print(f"  requests we would degrade : {would_degrade.mean():.1%}")
    print(f"  requests actually over SLO: {actually_slow.mean():.1%}")
    print(f"  SLO breaches we would catch: "
          f"{(would_degrade & actually_slow).sum() / max(actually_slow.sum(), 1):.1%}")
    print(f"\nsaved: {save(models, Path('models'))}")
```

```text
               model  mae_ms  rmse_ms     r2  under_rate  p90_under_ms  mean_over_ms
mean (squared error)   228.4    412.7  0.884       0.497         418.2         113.9
      median (q=0.5)   221.6    424.1  0.878       0.501         421.6         110.3
conservative (q=0.9)   412.8    611.2  0.741       0.101         118.4         398.1

admission control at 3000ms budget:
  requests we would degrade : 9.8%
  requests actually over SLO: 7.1%
  SLO breaches we would catch: 82.4%
```

Read the `under_rate` column. The mean model under-predicts half the time — useless for a
promise. The q=0.9 model under-predicts 10% of the time, has worse MAE, and is the **right**
model for this job. Choosing the model with the best MAE here would be choosing the wrong
one, which is why the metric must come from the problem statement.

## Part 2 — Serving both models

### Service structure

```text
ml-service/
├── src/ml_service/
│   ├── __init__.py
│   ├── config.py        settings (model paths, thresholds, limits)
│   ├── schemas.py       Pydantic request/response models
│   ├── registry.py      loads and versions model artifacts
│   ├── monitoring.py    prediction log + drift detection
│   └── api.py           FastAPI app
├── tests/test_api.py
├── models/
└── Dockerfile
```

```bash
uv add fastapi uvicorn pydantic pydantic-settings joblib scikit-learn pandas
uv add --dev pytest httpx
```

### Schemas — validation at the boundary

```python title="src/ml_service/schemas.py"
"""Request and response contracts.

Pydantic validates at runtime, generates the OpenAPI schema, and rejects malformed
input with a 422 before any model code runs.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TicketRequest(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=64)
    customer_id: int = Field(ge=1)
    message: str = Field(min_length=1, max_length=10_000)
    product: Literal["core", "analytics", "addon", "legacy"]
    plan: Literal["free", "pro", "enterprise"]
    channel: Literal["web", "email", "slack"] = "web"
    region: Literal["eu", "us", "apac"] = "eu"
    customer_prior_escalations: int = Field(default=0, ge=0, le=1_000)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be blank")
        return cleaned


class EscalationResponse(BaseModel):
    ticket_id: str
    probability: float = Field(ge=0.0, le=1.0)
    flagged: bool
    threshold: float
    model_version: str
    latency_ms: int


class LatencyRequest(BaseModel):
    question_chars: int = Field(ge=1, le=100_000)
    k: int = Field(default=5, ge=1, le=50)
    index_size_k: int = Field(default=200, ge=1, le=10_000)
    hour: int = Field(ge=0, le=23)
    concurrent_requests: int = Field(default=0, ge=0, le=10_000)
    history_turns: int = Field(default=0, ge=0, le=100)
    reranker_enabled: bool = False
    model_tier: Literal["small", "medium", "large"] = "medium"
    retrieval_strategy: Literal["vector", "hybrid"] = "vector"


class LatencyResponse(BaseModel):
    expected_ms: float
    conservative_ms: float
    within_budget: bool
    budget_ms: float
    recommendation: Literal["proceed", "reduce_k", "smaller_model", "queue"]
    model_version: str
```

### Registry — load once, version explicitly

```python title="src/ml_service/registry.py"
"""Model registry.

Artifacts are loaded once at startup. A model that fails to load must prevent the
service from reporting itself healthy - serving with a missing model is worse than
not serving.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    name: str
    version: str
    model: Any
    features: list[str]
    threshold: float | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}

    def load(self, name: str, path: Path) -> LoadedModel:
        if not path.exists():
            raise FileNotFoundError(f"model artifact not found: {path}")

        payload = joblib.load(path)
        entry = LoadedModel(
            name=name,
            version=path.stem,
            model=payload["model"] if "model" in payload else payload,
            features=payload["features"],
            threshold=payload.get("threshold"),
        )
        self._models[name] = entry
        logger.info("loaded model", extra={"model": name, "version": entry.version,
                                           "features": len(entry.features)})
        return entry

    def get(self, name: str) -> LoadedModel:
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"model {name!r} is not loaded; available: {sorted(self._models)}") from exc

    @property
    def ready(self) -> bool:
        return bool(self._models)

    def versions(self) -> dict[str, str]:
        return {name: entry.version for name, entry in self._models.items()}


registry = ModelRegistry()
```

### Monitoring and drift

```python title="src/ml_service/monitoring.py"
"""Prediction logging and drift detection.

Two things every served model needs and most lack:
  1. a log of every prediction, so you can evaluate it later against outcomes
  2. a drift signal, so you know when the input distribution has moved
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PredictionLog:
    """Append-only JSONL of predictions, for later joining with outcomes."""

    path: Path
    buffer_size: int = 100
    _buffer: list[dict] = field(default_factory=list)

    def record(self, *, model: str, version: str, request_id: str,
               features: dict, prediction: float, extra: dict | None = None) -> None:
        self._buffer.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "model": model,
            "version": version,
            "request_id": request_id,
            "prediction": prediction,
            "features": features,          # numeric features only: no message text
            **(extra or {}),
        })
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in self._buffer:
                handle.write(json.dumps(row, default=str) + "\n")
        self._buffer.clear()


@dataclass
class DriftMonitor:
    """Population Stability Index against a training reference.

    PSI < 0.1  no meaningful shift
    0.1 - 0.25 moderate shift, investigate
    > 0.25     significant shift, consider retraining
    """

    reference: dict[str, list[float]]           # feature -> bin edges from training data
    window: int = 2_000
    _recent: dict[str, deque] = field(default_factory=dict)

    def observe(self, features: dict[str, float]) -> None:
        for name, value in features.items():
            if name not in self.reference:
                continue
            self._recent.setdefault(name, deque(maxlen=self.window)).append(float(value))

    def psi(self, feature: str) -> float | None:
        edges = self.reference.get(feature)
        recent = self._recent.get(feature)
        if not edges or not recent or len(recent) < 100:
            return None

        expected = [1.0 / (len(edges) - 1)] * (len(edges) - 1)    # uniform by construction
        counts = [0] * (len(edges) - 1)
        for value in recent:
            for i in range(len(edges) - 1):
                if edges[i] <= value < edges[i + 1] or (i == len(edges) - 2 and value >= edges[i]):
                    counts[i] += 1
                    break

        total = sum(counts) or 1
        score = 0.0
        for observed_count, expected_share in zip(counts, expected, strict=True):
            observed_share = max(observed_count / total, 1e-6)
            expected_share = max(expected_share, 1e-6)
            score += (observed_share - expected_share) * math.log(observed_share / expected_share)
        return round(score, 4)

    def report(self) -> dict[str, dict[str, float | str | None]]:
        out: dict[str, dict[str, float | str | None]] = {}
        for feature in self.reference:
            value = self.psi(feature)
            out[feature] = {
                "psi": value,
                "status": (
                    "insufficient data" if value is None
                    else "ok" if value < 0.1
                    else "moderate drift" if value < 0.25
                    else "significant drift"
                ),
            }
        return out
```

### The API

```python title="src/ml_service/api.py"
"""FastAPI service exposing both models."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .monitoring import DriftMonitor, PredictionLog
from .registry import registry
from .schemas import EscalationResponse, LatencyRequest, LatencyResponse, TicketRequest

logger = logging.getLogger(__name__)

LATENCY_BUDGET_MS = 3_000.0
prediction_log = PredictionLog(Path("data/predictions.jsonl"))
drift = DriftMonitor(reference={
    "question_chars": [0, 100, 250, 500, 1_000, 100_000],
    "k": [1, 5, 8, 12, 51],
    "concurrent_requests": [0, 3, 6, 10, 10_001],
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup; fail fast if an artifact is missing."""
    registry.load("escalation", Path("models/escalation_latest.joblib"))
    registry.load("latency", Path("models/latency_v1.joblib"))
    logger.info("service ready", extra={"models": registry.versions()})
    yield
    prediction_log.flush()


app = FastAPI(title="ML Service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.get("/health")
async def health() -> dict:
    """Liveness + readiness. Not ready without models is a real failure state."""
    if not registry.ready:
        return JSONResponse({"status": "not_ready", "models": {}}, status_code=503)
    return {"status": "ok", "models": registry.versions()}


@app.get("/metrics/drift")
async def drift_report() -> dict:
    return drift.report()


@app.post("/predict/escalation", response_model=EscalationResponse)
async def predict_escalation(payload: TicketRequest, request: Request) -> EscalationResponse:
    started = time.perf_counter()
    entry = registry.get("escalation")

    frame = _ticket_to_frame(payload, entry.features)
    try:
        probability = float(entry.model.predict_proba(frame)[0, 1])
    except Exception as exc:                       # a model failure is a 503, not a 500
        logger.exception("escalation model failed", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=503, detail="model unavailable") from exc

    threshold = entry.threshold or 0.5
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    prediction_log.record(
        model="escalation", version=entry.version, request_id=request.state.request_id,
        features={"msg_chars": len(payload.message), "plan": payload.plan},
        prediction=probability, extra={"flagged": probability >= threshold},
    )

    return EscalationResponse(
        ticket_id=payload.ticket_id,
        probability=round(probability, 4),
        flagged=probability >= threshold,
        threshold=threshold,
        model_version=entry.version,
        latency_ms=elapsed_ms,
    )


@app.post("/predict/latency", response_model=LatencyResponse)
async def predict_latency(payload: LatencyRequest, request: Request) -> LatencyResponse:
    entry = registry.get("latency")
    frame = pd.DataFrame([payload.model_dump()]).astype({"reranker_enabled": int})

    bundle = entry.model
    expected = float(bundle["median"].predict(frame[entry.features])[0])
    conservative = float(bundle["conservative"].predict(frame[entry.features])[0])

    drift.observe({"question_chars": payload.question_chars, "k": payload.k,
                   "concurrent_requests": payload.concurrent_requests})

    within = conservative <= LATENCY_BUDGET_MS
    recommendation = (
        "proceed" if within
        else "reduce_k" if payload.k > 6
        else "smaller_model" if payload.model_tier != "small"
        else "queue"
    )

    return LatencyResponse(
        expected_ms=round(expected, 1),
        conservative_ms=round(conservative, 1),
        within_budget=within,
        budget_ms=LATENCY_BUDGET_MS,
        recommendation=recommendation,
        model_version=entry.version,
    )


def _ticket_to_frame(payload: TicketRequest, features: list[str]) -> pd.DataFrame:
    """Build the feature row the model expects.

    Features must be computed EXACTLY as in training - any divergence is
    training/serving skew, the most common cause of 'it was better offline'.
    """
    from escalation.features import add_text_features, add_time_features

    frame = pd.DataFrame([{
        "message": payload.message,
        "created_at": pd.Timestamp.now(tz="UTC"),
        "product": payload.product,
        "plan": payload.plan,
        "channel": payload.channel,
        "region": payload.region,
        "customer_prior_escalations": payload.customer_prior_escalations,
        "customer_prior_escalation_rate": 0.0,
        "customer_ticket_number": payload.customer_prior_escalations,
        "days_since_last_ticket": -1.0,
        "tickets_last_hour": 0,
    }])
    frame = add_time_features(add_text_features(frame))

    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"feature contract violation: {missing}")
    return frame[features]
```

```bash
uv run uvicorn ml_service.api:app --reload --port 8000
```

```bash
curl -s localhost:8000/predict/escalation -H 'content-type: application/json' -d '{
  "ticket_id": "T1", "customer_id": 42,
  "message": "Production outage - URGENT - customers affected immediately",
  "product": "legacy", "plan": "enterprise"
}' | jq
```

```json
{
  "ticket_id": "T1",
  "probability": 0.4812,
  "flagged": true,
  "threshold": 0.121,
  "model_version": "escalation_2026-06-28",
  "latency_ms": 7
}
```

Seven milliseconds. Worth remembering the next time someone proposes an LLM call to make the
same decision.

### Tests

```python title="tests/test_api.py"
from fastapi.testclient import TestClient

from ml_service.api import app

client = TestClient(app)


def test_health_reports_model_versions():
    response = client.get("/health")
    assert response.status_code == 200
    assert "escalation" in response.json()["models"]


def test_rejects_blank_message():
    response = client.post("/predict/escalation", json={
        "ticket_id": "T1", "customer_id": 1, "message": "   ",
        "product": "core", "plan": "free",
    })
    assert response.status_code == 422          # validation, before any model runs


def test_rejects_unknown_plan():
    response = client.post("/predict/escalation", json={
        "ticket_id": "T1", "customer_id": 1, "message": "hello",
        "product": "core", "plan": "platinum",
    })
    assert response.status_code == 422


def test_urgent_ticket_scores_higher_than_calm_one():
    def score(message: str) -> float:
        return client.post("/predict/escalation", json={
            "ticket_id": "T", "customer_id": 1, "message": message,
            "product": "legacy", "plan": "enterprise",
        }).json()["probability"]

    assert score("URGENT outage, customers affected, escalate now!") > score(
        "Quick question about the export format, no rush."
    )


def test_latency_recommendation_degrades_when_over_budget():
    response = client.post("/predict/latency", json={
        "question_chars": 700, "k": 20, "index_size_k": 800, "hour": 14,
        "concurrent_requests": 40, "history_turns": 10,
        "reranker_enabled": True, "model_tier": "large", "retrieval_strategy": "hybrid",
    })
    body = response.json()
    assert body["conservative_ms"] > body["expected_ms"]
    assert body["recommendation"] != "proceed"
```

## Common Mistakes

:::mistake
```text
1. Different feature code in training and serving   → training/serving skew; share the module
2. Loading the model per request                    → 200ms of joblib.load on every call
3. No input validation                              → the model receives garbage and returns a number
4. Returning 500 for a model failure                → use 503 so load balancers stop routing to it
5. No prediction log                                → you can never evaluate the model in production
6. No versioning in the response                    → an incident cannot be traced to a model version
7. Optimising the wrong metric                      → best MAE, worst under-prediction rate
```
:::

## Performance Considerations

- Load models at startup in `lifespan`, never per request.
- Batch where possible: `predict_proba` on 100 rows costs barely more than on one.
- Keep feature construction vectorised; per-request DataFrame creation is the main cost at
  this scale (~1–3 ms).
- Run `uvicorn --workers N` behind a reverse proxy; scikit-learn models are CPU-bound and do
  not release the GIL usefully in-process.
- Sub-10ms inference means the network and serialisation dominate — measure end to end, not
  just `model.predict`.

## Hands-on Exercise

:::exercise Complete and harden the service
1. Add a `/predict/escalation/batch` endpoint accepting up to 100 tickets, returning results
   in input order, with a per-request cap enforced by Pydantic.
2. Add a `/metrics` endpoint reporting request count, p50/p95 latency and flag rate over the
   last 1,000 predictions (use a `deque`).
3. Add graceful degradation: if the escalation model is unavailable, fall back to a keyword
   rule and set `"fallback": true` in the response.
4. Write a test proving the fallback path returns 200 with `fallback=true` when the model is
   missing.
5. Add a `Dockerfile` (multi-stage, non-root) and confirm the container starts and passes
   `/health`.
:::

:::solution Key pieces
```python title="graceful degradation"
URGENT_WORDS = ("urgent", "outage", "critical", "escalate", "manager", "asap")

def keyword_fallback(message: str) -> float:
    hits = sum(word in message.lower() for word in URGENT_WORDS)
    return min(0.15 + 0.2 * hits, 0.95)


@app.post("/predict/escalation", response_model=EscalationResponse)
async def predict_escalation(payload: TicketRequest, request: Request) -> EscalationResponse:
    try:
        entry = registry.get("escalation")
        probability = float(entry.model.predict_proba(_ticket_to_frame(payload, entry.features))[0, 1])
        version, fallback = entry.version, False
    except (KeyError, FileNotFoundError):
        logger.warning("escalation model unavailable, using keyword fallback",
                       extra={"request_id": request.state.request_id})
        probability, version, fallback = keyword_fallback(payload.message), "fallback-rules", True
    ...
```

```dockerfile title="Dockerfile"
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app models/ models/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health',timeout=2).json()['status']=='ok' else 1)"
CMD ["uvicorn", "ml_service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

A fallback that is tested is a fallback that works. An untested one is a second outage
waiting inside the first.
:::

## Challenge

:::challenge Close the feedback loop
Add `POST /feedback` recording the actual outcome for a `request_id`, then write a nightly
job that joins `predictions.jsonl` with `feedback.jsonl` and reports rolling 7-day PR-AUC,
precision at the deployed threshold, and per-segment recall — failing loudly if PR-AUC drops
more than 15% below the model card's baseline.

That job is the difference between a model you deployed and a model you *operate*. It is
also exactly the structure of the LLM evaluation pipeline you will build in Phase 24.
:::

## Interview Questions

:::interview
1. Why would you optimise quantile loss instead of squared error?
2. What is training/serving skew and how do you prevent it?
3. Why should a failed model return 503 rather than 500?
4. What do you log for every prediction, and what must you not log?
5. How do you detect that a deployed model has gone stale?
:::

## Summary

- The loss function comes from the problem: asymmetric costs need quantile regression, not
  the best MAE.
- Serve with validated schemas, models loaded once, explicit versions and request ids.
- Share the feature code between training and serving — divergence is the classic failure.
- Log every prediction and monitor input drift; a model without a feedback loop silently
  decays.

## Next Step

Phase 8: how the field got from symbolic AI to foundation models, and the vocabulary of
generative AI.
