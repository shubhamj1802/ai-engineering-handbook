---
title: Matplotlib, Seaborn and Chart Design
order: 1
difficulty: Intermediate
duration: 14
badges: ["Hands-on"]
summary: "The figure/axes model, the six chart types that cover most needs, and the design rules that make a chart answer a question instead of decorating a slide."
prereqs: ["groupby, Joins, Reshaping and Time Series"]
keyConcepts: ["figure", "axes", "line/bar/scatter/hist", "subplots", "annotation"]
---

:::note In one line
**A chart should answer one question.** If you cannot say which question, the chart is decoration.
:::

## Why this matters

You will use charts three ways in this handbook: to understand a dataset before modelling
(Phase 6), to diagnose what a model got wrong (Phase 7), and to show whether an AI system
regressed after a change (Phase 24). A clear chart makes a decision obvious in two seconds;
a cluttered one hides it entirely.

## Mental Model

Before touching Matplotlib, answer one question: **what question is this chart answering?**
The answer picks the chart type for you.
<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Diagram matching four questions to four chart types: comparing categories uses a bar chart, change over time uses a line chart, the shape of a spread uses a histogram, and the relationship between two numbers uses a scatter plot.">
  <rect x="14" y="26" width="150" height="126" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.7"/>
  <text class="dg-sub" x="26" y="46">"which is biggest?"</text>
  <rect x="34" y="96" width="18" height="40" fill="var(--accent)" opacity="0.8"/>
  <rect x="60" y="76" width="18" height="60" fill="var(--accent)" opacity="0.8"/>
  <rect x="86" y="110" width="18" height="26" fill="var(--accent)" opacity="0.8"/>
  <rect x="112" y="88" width="18" height="48" fill="var(--accent)" opacity="0.8"/>
  <line x1="28" y1="136" x2="140" y2="136" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-label" x="26" y="66" fill="var(--accent)">bar chart</text>
  <rect x="178" y="26" width="150" height="126" rx="9" fill="var(--panel-2)" stroke="var(--accent-3)" stroke-width="1.7"/>
  <text class="dg-sub" x="190" y="46">"is it going up?"</text>
  <polyline points="192,128 220,110 248,118 276,86 304,92" fill="none" stroke="var(--accent-3)" stroke-width="2.2"/>
  <line x1="192" y1="136" x2="304" y2="136" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-label" x="190" y="66" fill="var(--accent-3)">line chart</text>
  <rect x="342" y="26" width="150" height="126" rx="9" fill="var(--panel-2)" stroke="var(--accent-2)" stroke-width="1.7"/>
  <text class="dg-sub" x="354" y="46">"what is typical?"</text>
  <rect x="356" y="120" width="16" height="16" fill="var(--accent-2)" opacity="0.8"/>
  <rect x="374" y="104" width="16" height="32" fill="var(--accent-2)" opacity="0.8"/>
  <rect x="392" y="84" width="16" height="52" fill="var(--accent-2)" opacity="0.8"/>
  <rect x="410" y="96" width="16" height="40" fill="var(--accent-2)" opacity="0.8"/>
  <rect x="428" y="116" width="16" height="20" fill="var(--accent-2)" opacity="0.8"/>
  <line x1="352" y1="136" x2="468" y2="136" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-label" x="354" y="66" fill="var(--accent-2)">histogram</text>
  <rect x="506" y="26" width="140" height="126" rx="9" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-sub" x="518" y="46">"are they linked?"</text>
  <circle cx="530" cy="126" r="3.5" fill="var(--ok)"/>
  <circle cx="548" cy="114" r="3.5" fill="var(--ok)"/>
  <circle cx="566" cy="118" r="3.5" fill="var(--ok)"/>
  <circle cx="584" cy="98" r="3.5" fill="var(--ok)"/>
  <circle cx="602" cy="88" r="3.5" fill="var(--ok)"/>
  <circle cx="620" cy="94" r="3.5" fill="var(--ok)"/>
  <line x1="524" y1="136" x2="634" y2="136" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-label" x="518" y="66" fill="var(--ok)">scatter plot</text>
  <rect x="14" y="170" width="632" height="60" rx="9" fill="var(--panel)" stroke="var(--danger)" stroke-width="1.4" stroke-dasharray="5 4"/>
  <text class="dg-sub" x="330" y="192" text-anchor="middle" fill="var(--danger)">If you cannot say which question the chart answers, it is decoration.</text>
  <text class="dg-sub" x="330" y="212" text-anchor="middle">Put the answer in the title: "Leeds leads on revenue" beats "Revenue by city".</text>
</svg>
<figcaption>
<strong>Pick the chart from the question, not from the gallery.</strong> Four question shapes
cover almost everything you will plot at work.
</figcaption>
</figure>

```text
Figure          the canvas          fig, ax = plt.subplots()
 └── Axes       one plot area       ax.plot(...) ax.set_xlabel(...)
      ├── lines/bars/points
      ├── x-axis, y-axis, ticks
      └── title, legend, annotations

Use the OBJECT-ORIENTED interface (fig, ax) — not the stateful plt.plot(),
which silently draws on "whatever figure is current" and breaks in loops,
in notebooks and in any function that plots.
```

Every chart should answer one question. Write the question down first, then choose the
chart type that answers it.

| Question | Chart |
| --- | --- |
| How did it change over time? | line |
| How do categories compare? | horizontal bar |
| How is it distributed? | histogram or box/violin |
| Do two variables move together? | scatter |
| How do parts make a whole over time? | stacked area |
| Where are the hot spots in a matrix? | heatmap |

## Prerequisites

```bash
uv add matplotlib seaborn
```

## Core Concepts

### The basic figure

```python
import matplotlib
matplotlib.use("Agg")          # non-interactive backend: required for scripts/CI
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)

ax.plot(x, y, label="p95 latency", linewidth=2, color="#5eead4")
ax.set_xlabel("Date")
ax.set_ylabel("Latency (ms)")
ax.set_title("p95 latency by day")
ax.legend(frameon=False)
ax.grid(True, alpha=0.25, linestyle="--")

fig.tight_layout()
fig.savefig("reports/latency.png", bbox_inches="tight")
plt.close(fig)                 # free the memory - essential in loops and servers
```

:::warning Always `plt.close(fig)` in scripts and servers
Matplotlib keeps every unclosed figure in memory. A loop that plots 500 charts without
closing them will exhaust RAM, and a web process will leak until it is restarted.
:::

### The six charts

```python
# 1. line - trend over an ordered axis
ax.plot(daily["date"], daily["p95"], marker="o", markersize=3)

# 2. bar - compare categories (horizontal is easier to read with long labels)
ax.barh(summary["model"], summary["cost_usd"], color="#a78bfa")

# 3. scatter - relationship between two numeric variables
ax.scatter(df["top1_score"], df["thumbs_up_rate"], s=20, alpha=0.6)

# 4. histogram - distribution of one variable
ax.hist(df["latency_ms"], bins=50, color="#38bdf8", edgecolor="none")

# 5. box / violin - distribution across groups
ax.boxplot([g["latency_ms"].values for _, g in df.groupby("model")],
           tick_labels=list(df["model"].unique()))

# 6. heatmap - a matrix of values
im = ax.imshow(matrix, cmap="viridis", aspect="auto")
fig.colorbar(im, ax=ax, label="escalation rate")
```

### Subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(...)
axes[0, 1].hist(...)
axes[1, 0].scatter(...)
axes[1, 1].barh(...)

for ax in axes.flat:
    ax.grid(alpha=0.2)

fig.suptitle("Service overview", fontsize=14)
fig.tight_layout()
```

`sharex=True` / `sharey=True` lock axes together so panels are genuinely comparable — one of
the highest-value two-word additions in plotting.

### Annotation — where charts become useful

```python
ax.axhline(1500, color="#fb7185", linestyle="--", linewidth=1, label="SLO 1.5s")
ax.axvline(pd.Timestamp("2026-03-01"), color="#fbbf24", linestyle=":", label="prompt v2 deployed")

ax.annotate(
    "regression starts",
    xy=(regression_date, regression_value),
    xytext=(15, 25), textcoords="offset points",
    arrowprops={"arrowstyle": "->", "color": "#fb7185"},
    fontsize=9,
)

ax.fill_between(daily["date"], daily["p50"], daily["p95"], alpha=0.15, label="p50–p95 band")
```

A threshold line plus a deployment marker turns "here is some data" into "here is when it
broke and by how much".

### Formatting axes

```python
from matplotlib.ticker import FuncFormatter, PercentFormatter

ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.2f}"))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.set_ylim(0, None)                     # bar charts MUST start at zero
ax.set_yscale("log")                     # for values spanning orders of magnitude
fig.autofmt_xdate(rotation=30)           # readable date labels
```

### Seaborn for statistical charts

```python
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

sns.histplot(data=df, x="latency_ms", hue="model", bins=40, element="step", ax=ax)
sns.boxplot(data=df, x="model", y="latency_ms", ax=ax)
sns.violinplot(data=df, x="model", y="top_score", inner="quartile", ax=ax)
sns.scatterplot(data=df, x="tokens", y="latency_ms", hue="model", size="cost_usd", ax=ax)
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="rocket_r", ax=ax)
sns.lineplot(data=daily, x="date", y="p95", hue="model", errorbar=("ci", 95), ax=ax)
```

Seaborn takes long-format DataFrames and handles grouping, colouring and legends. Use it for
distribution and relationship charts; drop to raw Matplotlib when you need precise control.

Plotly is the third option, for interactive HTML charts you want to hand to someone:

```python
import plotly.express as px
fig = px.scatter(df, x="tokens", y="latency_ms", color="model", hover_data=["request_id"])
fig.write_html("reports/latency.html")
```

## Minimal Example

```python title="chart_demo.py"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
dates = pd.date_range("2026-02-01", periods=45, freq="D")
p95 = 900 + rng.normal(0, 60, 45).cumsum() * 0.4
p95[30:] += 450                                    # a regression on day 30

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
ax.plot(dates, p95, color="#38bdf8", linewidth=2, label="p95 latency")
ax.axhline(1500, color="#fb7185", linestyle="--", linewidth=1, label="SLO 1500ms")
ax.axvline(dates[30], color="#fbbf24", linestyle=":", linewidth=1.5, label="prompt v2")
ax.annotate("regression", xy=(dates[32], p95[32]), xytext=(12, 22),
            textcoords="offset points", fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#fb7185"})

ax.set_title("p95 latency regressed after the prompt v2 deploy")
ax.set_ylabel("milliseconds")
ax.set_ylim(0, None)
ax.grid(alpha=0.2, linestyle="--")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("reports/p95.png")
plt.close(fig)
print("wrote reports/p95.png")
```

The title states the finding, not the axes. "p95 latency by day" makes the reader work;
"p95 latency regressed after the prompt v2 deploy" tells them what you found.

## Real-World Example

A reusable reporting module that produces the standard dashboard for an LLM service.

```python title="src/analytics/charts.py"
"""Chart helpers for service reporting.

Every function takes a DataFrame and an output path, returns the path, and closes
its figure. That makes them safe to call in a loop, in a scheduled job or inside
a web request.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

PALETTE = {"small": "#5eead4", "medium": "#38bdf8", "large": "#a78bfa"}
GRID = {"alpha": 0.2, "linestyle": "--", "linewidth": 0.7}


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def latency_trend(daily: pd.DataFrame, path: Path, *, slo_ms: float = 1_500) -> Path:
    """p50 band and p95 line per model, against the SLO."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    for model, group in daily.groupby("model"):
        colour = PALETTE.get(str(model), "#94a3b8")
        ax.plot(group["date"], group["p95"], label=f"{model} p95", color=colour, linewidth=2)
        ax.fill_between(group["date"], group["p50"], group["p95"], color=colour, alpha=0.12)

    ax.axhline(slo_ms, color="#fb7185", linestyle="--", linewidth=1.2, label=f"SLO {slo_ms:.0f}ms")
    ax.set_title("Latency: p50–p95 band by model")
    ax.set_ylabel("milliseconds")
    ax.set_ylim(0, None)
    ax.grid(**GRID)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.autofmt_xdate(rotation=30)
    return _finish(fig, path)


def cost_breakdown(daily: pd.DataFrame, path: Path) -> Path:
    """Stacked area of daily cost per model, with a cumulative line."""
    wide = daily.pivot_table(index="date", columns="model", values="cost", aggfunc="sum").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.stackplot(
        wide.index,
        *[wide[c].to_numpy() for c in wide.columns],
        labels=list(wide.columns),
        colors=[PALETTE.get(str(c), "#94a3b8") for c in wide.columns],
        alpha=0.85,
    )

    twin = ax.twinx()
    twin.plot(wide.index, wide.sum(axis=1).cumsum(), color="#1e293b", linewidth=1.5,
              linestyle=":", label="cumulative")
    twin.set_ylabel("cumulative $")
    twin.grid(False)

    ax.set_title(f"Daily spend by model (total ${wide.to_numpy().sum():,.2f})")
    ax.set_ylabel("USD per day")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(**GRID)
    fig.autofmt_xdate(rotation=30)
    return _finish(fig, path)


def quality_vs_score(joined: pd.DataFrame, path: Path) -> Path:
    """Does better retrieval produce happier users? Bucketed, with sample sizes."""
    buckets = (
        joined.assign(bucket=pd.cut(joined["top1_score"], np.arange(0, 1.05, 0.1)))
        .groupby("bucket", observed=True)
        .agg(rate=("thumbs_up", "mean"), n=("thumbs_up", "count"))
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    centres = [interval.mid for interval in buckets.index]
    bars = ax.bar(centres, buckets["rate"], width=0.08, color="#38bdf8", alpha=0.9)

    for bar, count in zip(bars, buckets["n"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"n={count}", ha="center", fontsize=8, color="#475569")

    ax.set_title("User satisfaction rises with top-1 retrieval score")
    ax.set_xlabel("top-1 retrieval score")
    ax.set_ylabel("thumbs-up rate")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(**GRID, axis="y")
    return _finish(fig, path)


def escalation_heatmap(requests: pd.DataFrame, path: Path) -> Path:
    """Escalation rate by weekday and hour - where does the system struggle?"""
    frame = requests.assign(
        hour=requests["timestamp"].dt.hour,
        weekday=requests["timestamp"].dt.day_name().str[:3],
    )
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = (
        frame.pivot_table(index="weekday", columns="hour", values="escalated", aggfunc="mean")
        .reindex(order)
    )

    fig, ax = plt.subplots(figsize=(11, 4), dpi=120)
    image = ax.imshow(pivot.to_numpy(), cmap="rocket_r", aspect="auto", vmin=0)
    ax.set_xticks(range(0, 24, 2), labels=[f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(len(pivot.index)), labels=list(pivot.index))
    ax.set_title("Escalation rate by weekday and hour (UTC)")
    fig.colorbar(image, ax=ax, label="escalation rate", format=PercentFormatter(xmax=1.0))
    return _finish(fig, path)


def dashboard(daily: pd.DataFrame, joined: pd.DataFrame, requests: pd.DataFrame,
              out_dir: Path) -> list[Path]:
    return [
        latency_trend(daily, out_dir / "latency.png"),
        cost_breakdown(daily, out_dir / "cost.png"),
        quality_vs_score(joined, out_dir / "quality.png"),
        escalation_heatmap(requests, out_dir / "escalation.png"),
    ]


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    days = pd.date_range("2026-02-01", periods=40, freq="D")

    daily = pd.DataFrame([
        {
            "date": day,
            "model": model,
            "p50": base * rng.uniform(0.9, 1.1),
            "p95": base * rng.uniform(2.0, 2.6),
            "cost": rng.uniform(*cost_range),
        }
        for day in days
        for model, base, cost_range in [
            ("small", 320, (2, 6)), ("medium", 680, (8, 18)), ("large", 1400, (15, 40))
        ]
    ])

    n = 6_000
    joined = pd.DataFrame({
        "top1_score": rng.beta(5, 2, n),
    })
    joined["thumbs_up"] = (rng.random(n) < joined["top1_score"] * 0.85 + 0.05).astype(int)

    requests = pd.DataFrame({
        "timestamp": pd.date_range("2026-02-01", periods=8_000, freq="5min", tz="UTC"),
        "escalated": rng.random(8_000) < 0.06,
    })
    night = requests["timestamp"].dt.hour.isin([2, 3, 4])
    requests.loc[night, "escalated"] = rng.random(night.sum()) < 0.18     # a real pattern

    for path in dashboard(daily, joined, requests, Path("reports")):
        print("wrote", path)
```

```text
wrote reports/latency.png
wrote reports/cost.png
wrote reports/quality.png
wrote reports/escalation.png
```

The heatmap immediately shows escalations tripling between 02:00 and 05:00 UTC — a pattern
invisible in a daily average and instantly actionable once you see it.

## Common Mistakes

:::mistake
```python
# 1. The stateful interface inside functions
plt.plot(x, y); plt.title("…")      # draws on "the current figure", whatever that is
fig, ax = plt.subplots(); ax.plot(x, y)

# 2. Not closing figures
for group in groups: fig = make_chart(group)      # memory grows without bound
    ...; plt.close(fig)

# 3. Truncated bar axis
ax.set_ylim(90, 100)                 # exaggerates small differences - misleading
ax.set_ylim(0, None)                 # bars must start at zero

# 4. Mean where the distribution matters
ax.bar(models, mean_latency)         # hides the tail
ax.boxplot(...)  or  ax.hist(...)

# 5. Too many series
ax.plot(...)  # 12 lines in 12 colours: nobody can read it
# → small multiples (subplots) or highlight one and grey out the rest

# 6. Axis labels missing units
ax.set_ylabel("latency")             # ms? seconds?
ax.set_ylabel("latency (ms)")

# 7. Rainbow colormaps for continuous data
cmap="jet"                           # perceptually misleading
cmap="viridis" | "rocket_r"          # perceptually uniform
```
:::

## Best Practices

1. One question per chart; put the answer in the title.
2. Object-oriented API, always: `fig, ax = plt.subplots()`.
3. Label both axes with units; format currency and percentages.
4. Bars from zero; lines may use a truncated axis if you say so.
5. Annotate thresholds, deployments and incidents — context is the value.
6. Show sample sizes on aggregated charts (`n=…`) so the reader can judge reliability.
7. Use a colour-blind-safe palette; never encode meaning by colour alone.
8. Save as PNG at `dpi=120` for reports, SVG for documents that get zoomed.

## Hands-on Exercise

:::exercise Build a model comparison figure
Using the retrieval log from the Pandas phase, produce a single 2×2 figure that answers:

1. **Top-left**: latency distribution per model (box plot, log y-axis).
2. **Top-right**: cost per 1,000 tokens per model (horizontal bar, sorted, value labels).
3. **Bottom-left**: escalation rate over time per model (line, with a 7-day rolling mean and
   the raw values faint behind it).
4. **Bottom-right**: retrieval score versus latency (scatter, coloured by model, with an
   alpha low enough to show density).

Requirements: shared styling, a figure-level title stating the conclusion, every axis
labelled with units, and the figure saved and closed.
:::

:::solution Solution
```python title="model_comparison.py"
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PALETTE = {"small": "#5eead4", "medium": "#38bdf8", "large": "#a78bfa"}


def build(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    models = ["small", "medium", "large"]

    # 1. latency distribution
    ax = axes[0, 0]
    data = [df.loc[df["model"] == m, "latency_ms"].to_numpy() for m in models]
    box = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=False)
    for patch, model in zip(box["boxes"], models, strict=True):
        patch.set_facecolor(PALETTE[model])
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.set_title("Latency distribution")
    ax.set_ylabel("milliseconds (log scale)")
    ax.grid(alpha=0.2, linestyle="--", axis="y")

    # 2. cost efficiency
    ax = axes[0, 1]
    efficiency = (
        df.assign(cost_per_1k=df["cost_usd"] / (df["total_tokens"] / 1_000))
        .groupby("model")["cost_per_1k"].mean().reindex(models).sort_values()
    )
    bars = ax.barh(efficiency.index, efficiency.to_numpy(),
                   color=[PALETTE[m] for m in efficiency.index], alpha=0.9)
    for bar, value in zip(bars, efficiency, strict=True):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                f"${value:.4f}", va="center", fontsize=9)
    ax.set_title("Cost per 1,000 tokens")
    ax.set_xlabel("USD")
    ax.set_xlim(0, efficiency.max() * 1.25)
    ax.grid(alpha=0.2, linestyle="--", axis="x")

    # 3. escalation over time
    ax = axes[1, 0]
    daily = (
        df.set_index("timestamp")
        .groupby([pd.Grouper(freq="D"), "model"])["escalated"].mean()
        .reset_index()
    )
    for model in models:
        group = daily[daily["model"] == model]
        ax.plot(group["timestamp"], group["escalated"], color=PALETTE[model], alpha=0.25, linewidth=1)
        ax.plot(group["timestamp"], group["escalated"].rolling(7, min_periods=1).mean(),
                color=PALETTE[model], linewidth=2, label=model)
    ax.set_title("Escalation rate (7-day rolling mean)")
    ax.set_ylabel("share of requests escalated")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2, linestyle="--")
    ax.tick_params(axis="x", rotation=30)

    # 4. score vs latency
    ax = axes[1, 1]
    for model in models:
        subset = df[df["model"] == model].sample(min(1_500, (df["model"] == model).sum()),
                                                 random_state=0)
        ax.scatter(subset["top_score"], subset["latency_ms"], s=8, alpha=0.25,
                   color=PALETTE[model], label=model)
    ax.set_title("Retrieval score vs latency")
    ax.set_xlabel("top-1 retrieval score")
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, markerscale=2)
    ax.grid(alpha=0.2, linestyle="--")

    fig.suptitle(
        "The medium model is the best cost/latency trade-off; large adds cost without better retrieval",
        fontsize=13, y=0.995,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    n = 30_000
    model = rng.choice(["small", "medium", "large"], n, p=[0.5, 0.35, 0.15])
    base_latency = pd.Series(model).map({"small": 260, "medium": 620, "large": 1500}).to_numpy()

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-15", periods=n, freq="2min", tz="UTC"),
        "model": model,
        "latency_ms": (base_latency * rng.gamma(4, 0.25, n)).round(),
        "total_tokens": rng.integers(600, 5_000, n),
        "top_score": rng.beta(5, 2, n),
        "escalated": rng.random(n) < 0.06,
    })
    df["cost_usd"] = df["total_tokens"] / 1_000 * pd.Series(model).map(
        {"small": 0.0004, "medium": 0.0035, "large": 0.018}
    ).to_numpy()

    print("wrote", build(df, Path("reports/model_comparison.png")))
```

```text
wrote reports/model_comparison.png
```

The figure title carries the conclusion. That is the difference between a chart someone
looks at and a chart someone acts on.
:::

## Challenge

:::challenge An automated weekly report
Write `generate_report(df, out_dir)` that produces a self-contained HTML file embedding four
charts as base64 PNGs, plus a summary table and three automatically generated bullet points
("p95 latency rose 35% for the medium model", "escalations peak between 02:00 and 05:00
UTC", "the large model costs 5× more per token for no measurable quality gain"). Generate
the bullets from the data with explicit thresholds, not by hand.

An AI system nobody looks at is an AI system nobody trusts; a weekly report that writes
itself gets read.
:::

## Interview Questions

:::interview
1. Why use `fig, ax = plt.subplots()` rather than `plt.plot()`?
2. When must a bar chart's axis start at zero?
3. What does a box plot show that a bar of means does not?
4. Why avoid rainbow colormaps for continuous values?
5. What belongs in a chart title?
:::

## Cheat Sheet

```python
matplotlib.use("Agg")                        # scripts and servers
fig, ax = plt.subplots(figsize=(9,5), dpi=120)
fig, axes = plt.subplots(2, 2, sharex=True)

ax.plot(x, y, label=, color=, linewidth=)    ax.barh(cats, vals)
ax.scatter(x, y, s=, alpha=, c=)             ax.hist(v, bins=50)
ax.boxplot(list_of_arrays, tick_labels=)     ax.imshow(matrix, cmap="viridis")
ax.fill_between(x, lo, hi, alpha=0.15)       ax.stackplot(x, *series, labels=)

ax.set_title/xlabel/ylabel/xlim/ylim/yscale("log")
ax.axhline(v, ls="--")  ax.axvline(d, ls=":")  ax.annotate(text, xy=, xytext=, arrowprops=)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.legend(frameon=False)  ax.grid(alpha=0.2, linestyle="--")
fig.suptitle(...)  fig.autofmt_xdate()  fig.tight_layout()
fig.savefig(path, bbox_inches="tight");  plt.close(fig)

sns.set_theme(style="whitegrid")
sns.histplot/boxplot/violinplot/scatterplot/heatmap/lineplot(data=df, ..., ax=ax)
```

```quiz
[
  {
    "question": "Why is plt.plot() risky inside a function?",
    "options": [
      "It is slower",
      "It draws on whatever figure happens to be current, so output depends on global state",
      "It cannot save files",
      "It does not support colours"
    ],
    "answer": 1,
    "explanation": "The stateful API is fine for a quick look in a notebook, but in reusable code you must be explicit about which axes you are drawing on."
  },
  {
    "question": "You compare mean latency across three models with a bar chart starting at 800ms. What is wrong?",
    "options": [
      "Nothing",
      "Truncating the bar axis visually exaggerates differences, and a mean hides the tail",
      "Bars cannot show latency",
      "Three models is too few"
    ],
    "answer": 1,
    "explanation": "Bars encode magnitude by length, so they must start at zero; and latency needs a distribution view (box/percentiles) rather than a mean."
  },
  {
    "question": "Which title is most useful?",
    "options": [
      "Latency chart",
      "p95 latency by day and model",
      "p95 latency regressed 35% for the medium model after the v2 deploy",
      "Figure 3"
    ],
    "answer": 2,
    "explanation": "State the finding. The axes already say what is plotted; the title should say what the reader should conclude."
  }
]
```

## Summary

- Use the figure/axes API, label units, and close figures in scripts.
- Match the chart to the question: line for time, bar for categories, histogram/box for
  distribution, scatter for relationships, heatmap for matrices.
- Annotate thresholds and deploys; show sample sizes; start bars at zero.
- Seaborn for statistical charts over long-format frames; Plotly when interactivity helps.

## Next Step

Phase 6: machine learning from first principles — where these charts become your main
diagnostic tool.
