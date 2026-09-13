---
title: groupby, Joins, Reshaping and Time Series
order: 3
difficulty: Intermediate
duration: 16
badges: ["Hands-on"]
summary: "Split-apply-combine, merges that do not silently duplicate rows, pivots, and resampling — the operations that turn a log into an analysis."
prereqs: ["Cleaning, Missing Values and Transformation"]
keyConcepts: ["groupby", "agg", "transform", "merge", "pivot", "resample"]
---

## Why this matters

"What is our p95 latency per model per day?" "Which documents get retrieved but never
cited?" "Did last week's prompt change reduce escalations?" All three are groupby, join and
resample questions — and the answers drive real engineering decisions in Phases 24 and 25.

## Mental Model

```text
SPLIT–APPLY–COMBINE

  rows ──group by model──▶ [small] [medium] [large]
                              │        │        │
                            apply    apply    apply       mean, p95, count…
                              └────────┴────────┘
                                   combine → one row per group

agg()       group → ONE row per group        (summary)
transform() group → SAME shape as input      (broadcast the group value back)
filter()    group → keep/drop whole groups
apply()     group → anything (slowest)
```

## Core Concepts

### groupby and agg

```python
import pandas as pd

df.groupby("model")["latency_ms"].mean()
df.groupby("model")["latency_ms"].agg(["count", "mean", "median", "std"])

df.groupby(["model", "escalated"])["cost_usd"].sum()          # multi-key → MultiIndex

# named aggregations - the readable form, and the one to default to
summary = df.groupby("model").agg(
    requests=("request_id", "count"),
    mean_latency=("latency_ms", "mean"),
    p95_latency=("latency_ms", lambda s: s.quantile(0.95)),
    total_cost=("cost_usd", "sum"),
    escalation_rate=("escalated", "mean"),
).round(3).reset_index()
```

:::tip `as_index=False` or `.reset_index()`
By default the grouping keys become the index. Flattening back to columns keeps downstream
code (merges, plotting, `to_csv`) simple.
:::

### transform — group statistics broadcast back

```python
# add each row's deviation from its model's mean latency
df["model_mean"] = df.groupby("model")["latency_ms"].transform("mean")
df["latency_vs_model"] = df["latency_ms"] - df["model_mean"]

# group-aware imputation
df["score"] = df["score"].fillna(df.groupby("model")["score"].transform("median"))

# rank within a group
df["rank_in_doc"] = df.groupby("document")["score"].rank(ascending=False, method="dense")
```

`transform` returns something the same length as the input; `agg` collapses. Choosing the
wrong one is the most common groupby confusion.

### filter and apply on groups

```python
# keep only documents with at least 10 chunks
df.groupby("document").filter(lambda g: len(g) >= 10)

# top 3 chunks per document
df.sort_values("score", ascending=False).groupby("document").head(3)

# arbitrary per-group logic (slow; use only when nothing else fits)
df.groupby("document", group_keys=False).apply(lambda g: g.assign(pct=g["score"] / g["score"].sum()))
```

### Merging

```python
merged = questions.merge(
    answers,
    on="request_id",              # or left_on=..., right_on=...
    how="inner",                  # inner | left | right | outer
    validate="one_to_one",        # raises if the relationship is violated
    indicator=True,               # adds _merge: left_only / right_only / both
    suffixes=("_q", "_a"),
)
```

| how | Keeps |
| --- | --- |
| `inner` | only matching keys (default) |
| `left` | all left rows; unmatched right columns become NaN |
| `right` | all right rows |
| `outer` | everything from both |

:::danger Always check row counts around a merge
```python
before = len(left)
merged = left.merge(right, on="id", how="left")
assert len(merged) == before, f"merge changed row count: {before} → {len(merged)}"
```
If the right side has duplicate keys, a left join **multiplies** rows. This is the single
most common silent data corruption in analysis, and `validate="one_to_one"` /
`"many_to_one"` turns it into an immediate error instead of a wrong number in a report.
:::

```python
pd.concat([df1, df2], ignore_index=True)        # stack rows
pd.concat([df1, df2], axis=1)                   # side by side (aligns on index!)
```

### Reshaping

```python
# long → wide
wide = df.pivot_table(
    index="date", columns="model", values="latency_ms", aggfunc="mean", fill_value=0
)

# wide → long
long = wide.reset_index().melt(id_vars="date", var_name="model", value_name="latency_ms")

# frequency cross-tab
pd.crosstab(df["model"], df["escalated"], normalize="index")

df.stack() / df.unstack()                        # move between index and columns levels
```

Long format (one row per observation) is what plotting libraries and models want; wide is
what humans read in a spreadsheet.

### Time series

```python
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp").sort_index()

df.loc["2026-03"]                                   # partial-string slicing: all of March
df.loc["2026-03-01":"2026-03-07"]

daily = df.resample("D").agg(
    requests=("request_id", "count"),
    p95_latency=("latency_ms", lambda s: s.quantile(0.95)),
    cost=("cost_usd", "sum"),
)

daily["cost_7d_avg"] = daily["cost"].rolling(7, min_periods=1).mean()
daily["cost_change"] = daily["cost"].pct_change()
daily["cost_cumulative"] = daily["cost"].cumsum()

df.index.hour / df.index.dayofweek / df.index.day_name()
df.tz_convert("Europe/Berlin")
```

Resample frequencies: `"h"` hourly, `"D"` daily, `"W"` weekly, `"ME"` month-end, `"15min"`.

:::warning Always store timestamps in UTC
Convert to a local timezone only for display. Mixed-timezone or naive timestamps produce
daily aggregates that are wrong by a few hours near boundaries — and nobody notices until
a monthly report disagrees with the finance system.
:::

## Minimal Example

```python title="groupby_demo.py"
import pandas as pd

df = pd.DataFrame({
    "model": ["small", "small", "medium", "medium", "large"],
    "latency_ms": [320, 410, 890, 1250, 2100],
    "cost_usd": [0.0008, 0.0011, 0.0090, 0.0125, 0.0620],
    "escalated": [False, False, True, False, False],
})

summary = df.groupby("model", as_index=False).agg(
    requests=("model", "count"),
    mean_latency=("latency_ms", "mean"),
    total_cost=("cost_usd", "sum"),
    escalation_rate=("escalated", "mean"),
)

df["share_of_model_cost"] = (
    df["cost_usd"] / df.groupby("model")["cost_usd"].transform("sum")
).round(3)

print(summary.round(4).to_string(index=False))
print()
print(df[["model", "cost_usd", "share_of_model_cost"]].to_string(index=False))
```

```text
 model  requests  mean_latency  total_cost  escalation_rate
 large         1        2100.0      0.0620              0.0
medium         2        1070.0      0.0215              0.5
 small         2         365.0      0.0019              0.0

 model  cost_usd  share_of_model_cost
 small    0.0008                0.421
 small    0.0011                0.579
medium    0.0090                0.419
medium    0.0125                0.581
 large    0.0620                1.000
```

## Real-World Example

An operations report for an LLM service: daily metrics per model, retrieval quality joined
from a second table, and week-over-week comparison — the analysis you run before deciding
whether a change shipped safely.

```python title="src/analytics/report.py"
"""Operational analytics for an LLM service.

Answers the four questions you get asked every week:
  1. what did it cost, per model, per day?
  2. did latency regress?
  3. which documents actually drive answers?
  4. did last week's change help?
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_metrics(requests: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, model) with the metrics that matter operationally."""
    frame = requests.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    grouped = (
        frame
        .set_index("timestamp")
        .groupby([pd.Grouper(freq="D"), "model"])
        .agg(
            requests=("request_id", "count"),
            users=("user_id", "nunique"),
            p50_latency=("latency_ms", "median"),
            p95_latency=("latency_ms", lambda s: s.quantile(0.95)),
            mean_tokens=("total_tokens", "mean"),
            cost_usd=("cost_usd", "sum"),
            escalation_rate=("escalated", "mean"),
            mean_top_score=("top_score", "mean"),
        )
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )

    grouped["cost_per_request"] = grouped["cost_usd"] / grouped["requests"]
    return grouped.round(4)


def citation_coverage(retrievals: pd.DataFrame, citations: pd.DataFrame) -> pd.DataFrame:
    """Which retrieved chunks were actually cited?

    A chunk retrieved constantly but never cited is wasting context window -
    one of the highest-value findings a RAG system can surface about itself.
    """
    before = len(retrievals)
    merged = retrievals.merge(
        citations.assign(cited=True)[["request_id", "chunk_id", "cited"]],
        on=["request_id", "chunk_id"],
        how="left",
        validate="one_to_one",          # fails loudly if citations contain duplicates
    )
    assert len(merged) == before, "merge changed the row count"

    merged["cited"] = merged["cited"].fillna(False)

    return (
        merged.groupby("chunk_id", as_index=False)
        .agg(
            document=("document", "first"),
            retrieved=("request_id", "count"),
            cited=("cited", "sum"),
            mean_rank=("rank", "mean"),
            mean_score=("score", "mean"),
        )
        .assign(citation_rate=lambda d: (d["cited"] / d["retrieved"]).round(3))
        .sort_values("retrieved", ascending=False)
    )


def week_over_week(daily: pd.DataFrame, *, metric: str = "p95_latency") -> pd.DataFrame:
    """Compare the last 7 days with the 7 before, per model."""
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    latest = frame["date"].max()

    this_week = frame[frame["date"] > latest - pd.Timedelta(days=7)]
    last_week = frame[
        (frame["date"] <= latest - pd.Timedelta(days=7))
        & (frame["date"] > latest - pd.Timedelta(days=14))
    ]

    comparison = (
        this_week.groupby("model")[metric].mean().rename("this_week")
        .to_frame()
        .join(last_week.groupby("model")[metric].mean().rename("last_week"), how="outer")
    )
    comparison["change_pct"] = (
        (comparison["this_week"] - comparison["last_week"]) / comparison["last_week"] * 100
    ).round(1)
    comparison["verdict"] = np.where(
        comparison["change_pct"].abs() < 5, "no change",
        np.where(comparison["change_pct"] > 0, "regressed", "improved"),
    )
    return comparison.round(2).reset_index()


def cost_pivot(daily: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per day, one column per model - the format humans read."""
    return (
        daily.pivot_table(index="date", columns="model", values="cost_usd",
                          aggfunc="sum", fill_value=0.0)
        .assign(total=lambda d: d.sum(axis=1))
        .round(2)
    )


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    n = 20_000

    requests = pd.DataFrame({
        "request_id": [f"req_{i}" for i in range(n)],
        "timestamp": pd.date_range("2026-02-15", periods=n, freq="90s", tz="UTC"),
        "user_id": rng.integers(1, 400, n),
        "model": rng.choice(["small", "medium", "large"], n, p=[0.6, 0.3, 0.1]),
        "latency_ms": rng.gamma(4, 250, n).round(),
        "total_tokens": rng.integers(600, 5_000, n),
        "cost_usd": rng.random(n) * 0.02,
        "escalated": rng.random(n) < 0.06,
        "top_score": rng.beta(5, 2, n),
    })
    # simulate a latency regression in the final week for the medium model
    recent = requests["timestamp"] > requests["timestamp"].max() - pd.Timedelta(days=7)
    requests.loc[recent & (requests["model"] == "medium"), "latency_ms"] *= 1.35

    daily = daily_metrics(requests)
    print("daily metrics (tail):")
    print(daily.tail(6).to_string(index=False), "\n")

    print("week over week, p95 latency:")
    print(week_over_week(daily).to_string(index=False), "\n")

    print("daily cost by model (last 5 days):")
    print(cost_pivot(daily).tail(5).to_string(), "\n")

    # citation coverage
    retrievals = pd.DataFrame({
        "request_id": rng.choice(requests["request_id"], 40_000),
        "chunk_id": rng.choice([f"c{i}" for i in range(300)], 40_000),
        "document": rng.choice(["handbook", "faq", "policy"], 40_000),
        "rank": rng.integers(1, 6, 40_000),
        "score": rng.beta(5, 2, 40_000),
    }).drop_duplicates(subset=["request_id", "chunk_id"])

    citations = retrievals.sample(9_000, random_state=1)[["request_id", "chunk_id"]]

    coverage = citation_coverage(retrievals, citations)
    print("most retrieved chunks:")
    print(coverage.head(5).to_string(index=False))
    print("\nretrieved often but rarely cited (dead weight in the context window):")
    print(
        coverage.query("retrieved > 100 and citation_rate < 0.15")
        .head(5).to_string(index=False)
    )
```

```text
week over week, p95 latency:
 model  this_week  last_week  change_pct    verdict
 large    1789.60    1794.30        -0.3  no change
medium    2418.85    1791.20        35.0  regressed
 small    1793.05    1788.44         0.3  no change
```

The medium model's p95 is 35% worse this week — surfaced by a groupby and a subtraction,
and exactly the kind of regression that a mean would have hidden.

## Common Mistakes

:::mistake
```python
# 1. A merge that multiplies rows
merged = orders.merge(customers, on="customer_id", how="left")     # duplicate customers?
merged = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")

# 2. agg where transform was needed
df["mean"] = df.groupby("g")["x"].mean()          # shape mismatch → NaNs
df["mean"] = df.groupby("g")["x"].transform("mean")

# 3. Mean instead of a percentile for latency
"mean_latency"            # hides the tail users actually feel
"p95_latency"             # what your SLO is written against

# 4. groupby().apply() for something agg can do
df.groupby("g").apply(lambda x: x["a"].sum())     # 10-50x slower
df.groupby("g")["a"].sum()

# 5. Forgetting dropna=False
df.groupby("category").size()                     # silently omits NaN categories
df.groupby("category", dropna=False).size()

# 6. Naive timestamps
pd.to_datetime(s)                                 # local/naive: daily buckets shift
pd.to_datetime(s, utc=True)

# 7. concat(axis=1) with mismatched indexes
pd.concat([a, b], axis=1)                         # aligns on index → NaNs
```
:::

## Debugging

```python
# after every merge
print(merged["_merge"].value_counts())            # with indicator=True
print(f"{len(left)} + {len(right)} → {len(merged)}")

# check for duplicate keys BEFORE merging
print(right["id"].duplicated().sum())

# inspect one group
df.groupby("model").get_group("medium").head()
print(df.groupby("model").size())
```

`indicator=True` followed by `value_counts()` on `_merge` answers "how many rows failed to
match?" — the question you actually care about after a join.

## Performance Considerations

| Operation | Note |
| --- | --- |
| `groupby().agg("mean")` | fast, runs in C |
| `groupby().agg(lambda)` | Python per group — acceptable for hundreds of groups, not millions |
| `groupby().apply()` | slowest; avoid on large frames |
| `merge` on sorted, indexed keys | much faster; set the index or sort first |
| `category` dtype for group keys | large speed-up for repeated grouping |
| `observed=True` on categorical groupby | avoids generating every unused combination |

For frames over a few million rows with heavy grouping, DuckDB over the same Parquet files
is often 10× faster and uses far less memory:

```python
import duckdb
duckdb.sql("SELECT model, quantile_cont(latency_ms, 0.95) FROM 'logs/*.parquet' GROUP BY model").df()
```

## Hands-on Exercise

:::exercise Build a RAG quality report
Given three synthetic tables — `requests` (request_id, timestamp, user_id, model,
latency_ms, cost_usd, escalated), `retrievals` (request_id, chunk_id, document, rank, score)
and `feedback` (request_id, thumbs_up) — produce a single report answering:

1. Daily request volume, p50/p95 latency and total cost, per model.
2. The relationship between `mean top-1 retrieval score` and `thumbs_up rate`, bucketed into
   deciles of score.
3. The top 10 documents by retrieval volume, with their citation-free rate.
4. Escalation rate by hour of day, as a pivot table.
5. Week-over-week change in thumbs-up rate per model, with a verdict column.

Every join must assert that the row count is what you expect.
:::

:::solution Solution
```python title="rag_quality_report.py"
from __future__ import annotations

import numpy as np
import pandas as pd


def build_report(requests: pd.DataFrame, retrievals: pd.DataFrame, feedback: pd.DataFrame) -> dict[str, pd.DataFrame]:
    requests = requests.assign(timestamp=pd.to_datetime(requests["timestamp"], utc=True))

    # --- 1. daily operational metrics ------------------------------------
    daily = (
        requests.set_index("timestamp")
        .groupby([pd.Grouper(freq="D"), "model"])
        .agg(
            requests=("request_id", "count"),
            p50=("latency_ms", "median"),
            p95=("latency_ms", lambda s: s.quantile(0.95)),
            cost=("cost_usd", "sum"),
        )
        .reset_index()
        .round(2)
    )

    # --- 2. retrieval quality vs satisfaction -----------------------------
    top1 = (
        retrievals.sort_values("rank")
        .groupby("request_id", as_index=False)
        .first()[["request_id", "score"]]
        .rename(columns={"score": "top1_score"})
    )

    before = len(requests)
    joined = requests.merge(top1, on="request_id", how="left", validate="one_to_one")
    assert len(joined) == before, "top1 merge changed the row count"

    joined = joined.merge(feedback, on="request_id", how="left", validate="one_to_one")
    assert len(joined) == before, "feedback merge changed the row count"

    joined["score_decile"] = pd.qcut(joined["top1_score"], 10, labels=False, duplicates="drop")
    quality = (
        joined.dropna(subset=["thumbs_up"])
        .groupby("score_decile", as_index=False)
        .agg(
            mean_top1=("top1_score", "mean"),
            thumbs_up_rate=("thumbs_up", "mean"),
            n=("request_id", "count"),
        )
        .round(3)
    )

    # --- 3. document usage -------------------------------------------------
    cited_requests = set(feedback.loc[feedback["thumbs_up"] == 1, "request_id"])
    documents = (
        retrievals.assign(helpful=retrievals["request_id"].isin(cited_requests))
        .groupby("document", as_index=False)
        .agg(
            retrievals=("request_id", "count"),
            distinct_chunks=("chunk_id", "nunique"),
            mean_score=("score", "mean"),
            helpful_rate=("helpful", "mean"),
        )
        .sort_values("retrievals", ascending=False)
        .round(3)
    )

    # --- 4. escalation by hour --------------------------------------------
    hourly = requests.assign(hour=requests["timestamp"].dt.hour).pivot_table(
        index="hour", columns="model", values="escalated", aggfunc="mean", fill_value=0.0
    ).round(3)

    # --- 5. week over week satisfaction ------------------------------------
    latest = joined["timestamp"].max()
    joined["period"] = np.where(
        joined["timestamp"] > latest - pd.Timedelta(days=7), "this_week",
        np.where(joined["timestamp"] > latest - pd.Timedelta(days=14), "last_week", "older"),
    )
    wow = (
        joined[joined["period"] != "older"]
        .pivot_table(index="model", columns="period", values="thumbs_up", aggfunc="mean")
        .assign(change_pp=lambda d: ((d["this_week"] - d["last_week"]) * 100).round(1))
        .assign(verdict=lambda d: np.where(d["change_pp"].abs() < 2, "no change",
                                           np.where(d["change_pp"] > 0, "improved", "regressed")))
        .round(3)
        .reset_index()
    )

    return {"daily": daily, "quality": quality, "documents": documents,
            "hourly_escalation": hourly, "week_over_week": wow}


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    n = 12_000

    requests = pd.DataFrame({
        "request_id": [f"r{i}" for i in range(n)],
        "timestamp": pd.date_range("2026-02-01", periods=n, freq="2min", tz="UTC"),
        "user_id": rng.integers(1, 300, n),
        "model": rng.choice(["small", "medium", "large"], n, p=[0.5, 0.35, 0.15]),
        "latency_ms": rng.gamma(4, 260, n).round(),
        "cost_usd": (rng.random(n) * 0.03).round(5),
        "escalated": rng.random(n) < 0.07,
    })

    retrievals = pd.concat([
        pd.DataFrame({
            "request_id": requests["request_id"],
            "chunk_id": rng.choice([f"c{i}" for i in range(400)], n),
            "document": rng.choice(["handbook", "faq", "policy", "legacy"], n, p=[.4, .3, .2, .1]),
            "rank": rank,
            "score": rng.beta(6 - rank, 2, n).round(4),
        })
        for rank in (1, 2, 3)
    ], ignore_index=True)

    # satisfaction correlates with top-1 score, plus noise
    top1_scores = retrievals.query("rank == 1").set_index("request_id")["score"]
    probability = (top1_scores.reindex(requests["request_id"]).to_numpy() * 0.8 + 0.1)
    feedback = pd.DataFrame({
        "request_id": requests["request_id"],
        "thumbs_up": (rng.random(n) < probability).astype(int),
    }).sample(frac=0.45, random_state=2)

    report = build_report(requests, retrievals, feedback)

    print("retrieval quality vs satisfaction:")
    print(report["quality"].to_string(index=False), "\n")
    print("documents:")
    print(report["documents"].to_string(index=False), "\n")
    print("week over week satisfaction:")
    print(report["week_over_week"].to_string(index=False))
```

```text
retrieval quality vs satisfaction:
 score_decile  mean_top1  thumbs_up_rate    n
            0      0.436           0.443  540
            1      0.581           0.585  541
            ...
            9      0.921           0.850  541

documents:
 document  retrievals  distinct_chunks  mean_score  helpful_rate
 handbook       14401              400       0.630         0.309
      faq       10783              400       0.628         0.313
   policy        7218              400       0.631         0.309
   legacy        3598              399       0.629         0.307
```

The decile table is the important output: thumbs-up rate rises monotonically with top-1
retrieval score, which is *evidence* that retrieval quality drives user satisfaction — the
argument you need before investing in reranking (Phase 13).
:::

## Challenge

:::challenge Cohort retention
Using the requests table, compute a weekly cohort retention matrix: group users by the week
of their first request, then for each subsequent week compute the fraction of that cohort
that made at least one request. Render it as a triangular pivot table.

You will need a self-merge on `user_id`, a `Grouper`, and careful handling of the diagonal.
Retention is the metric that tells you whether an internal AI tool is genuinely useful or
was merely tried once.
:::

## Interview Questions

:::interview
1. What is the difference between `agg` and `transform`?
2. How can a left join increase your row count, and how do you prevent it?
3. When do you use `pivot_table` versus `groupby`?
4. Why report p95 rather than mean latency?
5. How do you compare two time periods correctly with mixed timezones?
:::

## Cheat Sheet

```python
df.groupby(k)[c].mean() / .sum() / .size() / .nunique()
df.groupby(k, as_index=False, dropna=False, observed=True).agg(
    n=("id", "count"), p95=("latency_ms", lambda s: s.quantile(0.95)))
df.groupby(k)[c].transform("mean" | "median" | "rank")
df.groupby(k).filter(lambda g: len(g) >= 10)
df.sort_values(c, ascending=False).groupby(k).head(3)

left.merge(right, on="id", how="left", validate="many_to_one", indicator=True)
pd.concat([a, b], ignore_index=True)

df.pivot_table(index=, columns=, values=, aggfunc=, fill_value=)
df.melt(id_vars=, var_name=, value_name=)   pd.crosstab(a, b, normalize="index")

df.set_index("ts").sort_index().resample("D").agg(...)
s.rolling(7, min_periods=1).mean()  s.pct_change()  s.cumsum()  s.shift(1)
pd.Grouper(freq="D")   df.index.hour   df.tz_convert("Europe/Berlin")
```

```quiz
[
  {
    "question": "A left join of 1,000 orders to a customers table returns 1,240 rows. What happened?",
    "options": [
      "Pandas added default rows",
      "The customers table has duplicate customer_id values, so matching rows multiplied",
      "The join was actually an outer join",
      "Some orders matched twice by chance"
    ],
    "answer": 1,
    "explanation": "Duplicate keys on the right side multiply left rows. validate='many_to_one' turns this silent corruption into an immediate error."
  },
  {
    "question": "You want each row to carry its group's mean. Which do you use?",
    "options": ["agg('mean')", "transform('mean')", "apply(mean)", "filter(mean)"],
    "answer": 1,
    "explanation": "transform returns a result aligned to the original rows; agg collapses each group to a single row."
  },
  {
    "question": "Why report p95 latency instead of the mean?",
    "options": [
      "It is easier to compute",
      "The mean hides the slow tail that users actually experience and SLOs are written against",
      "p95 is always lower",
      "Means cannot be grouped"
    ],
    "answer": 1,
    "explanation": "A few very slow requests barely move the mean but define the experience of 5% of your users - and in an agent system, tail latency compounds across steps."
  }
]
```

## Summary

- groupby splits, applies and combines; `agg` collapses while `transform` broadcasts back.
- Validate every merge with `validate=` and an assertion on row count.
- `pivot_table` for human-readable wide tables, `melt` back to long for plotting and models.
- Store timestamps in UTC, resample for time buckets, and report percentiles for latency.

## Next Step

Visualisation — turning these tables into charts that make a regression obvious at a glance.
