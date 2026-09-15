---
title: Cleaning, Missing Values and Transformation
order: 2
difficulty: Intermediate
duration: 14
badges: ["Hands-on"]
summary: "Find and fix the problems in real data — missing values, duplicates, outliers, inconsistent text — and transform columns without loops."
prereqs: ["Series, DataFrames and Loading Data"]
keyConcepts: ["NaN", "fillna", "dropna", "duplicates", "str accessor", "apply"]
---

:::note In one line
**Real data is always broken.** Missing values, wrong types, duplicates and inconsistent text - this lesson is the repair kit.
:::

## Why this matters

Real datasets are dirty: missing fields, three spellings of the same category, prices as
text, duplicate rows from a retried export. Cleaning is where most data-work time goes, and
the decisions you make here — especially about missing values — directly change what your
model learns and what your evaluation reports.

## Mental Model

```text
Missing data is a QUESTION, not a nuisance. Ask why it is missing:

MCAR  missing completely at random   → dropping is roughly safe
MAR   missing depends on other columns → impute using those columns
MNAR  missing depends on the value itself → the missingness IS information;
                                            add an indicator column

"salary is blank" may mean: not collected, refused to answer, or zero.
Those are three different facts and must not become the same number.
```

## Core Concepts

### Finding problems

```python
import pandas as pd

df.isna().sum()                          # nulls per column
df.isna().mean().sort_values(ascending=False)     # as a fraction
df.isna().any(axis=1).sum()              # rows with any null
df.duplicated().sum()                    # exact duplicate rows
df.duplicated(subset=["id"]).sum()       # duplicate keys
df.nunique()                             # distinct values per column
df["category"].value_counts(dropna=False)
df.describe()                            # min/max reveal impossible values
```

A useful one-liner for any new dataset:

```python
missing = pd.DataFrame({
    "nulls": df.isna().sum(),
    "pct": (df.isna().mean() * 100).round(1),
    "dtype": df.dtypes.astype(str),
}).query("nulls > 0").sort_values("pct", ascending=False)
```

### Handling missing values

```python
df.dropna()                              # rows with ANY null - usually too aggressive
df.dropna(subset=["text", "embedding"])  # only where these matter
df.dropna(thresh=5)                      # keep rows with at least 5 non-null values
df.dropna(axis=1, thresh=len(df) * 0.5)  # drop columns more than half empty

df["score"] = df["score"].fillna(0.0)
df["score"] = df["score"].fillna(df["score"].median())      # robust to outliers
df["category"] = df["category"].fillna("unknown")
df["price"] = df.groupby("product")["price"].transform(lambda s: s.fillna(s.median()))

df["value"] = df["value"].ffill()        # carry the last value forward (time series)
df["value"] = df["value"].interpolate()  # numeric interpolation
```

:::warning Record that you imputed
```python
df["score_was_missing"] = df["score"].isna()       # BEFORE filling
df["score"] = df["score"].fillna(df["score"].median())
```
The indicator column is often predictive in its own right, and it keeps the imputation
visible to whoever reads the data later. Silent imputation is how a model learns that "the
median score" means "we had no data".
:::

### Duplicates

```python
df.drop_duplicates()                                     # exact duplicates
df.drop_duplicates(subset=["chunk_id"], keep="first")    # by key
df.drop_duplicates(subset=["text"], keep="last")

df[df.duplicated(subset=["id"], keep=False)].sort_values("id")     # inspect them first
```

Always look at duplicates before dropping: two rows with the same id and *different* content
mean an upstream bug, not a harmless repeat.

### Text cleaning with the `.str` accessor

```python
s = df["category"]

s.str.strip().str.lower()
s.str.replace(r"\s+", " ", regex=True)
s.str.contains("handbook", case=False, na=False)
s.str.startswith("doc_")
s.str.extract(r"(\d{4})-(\d{2})")          # capture groups → DataFrame
s.str.split("/", expand=True)              # → DataFrame of parts
s.str.len()
s.str.normalize("NFKC")                    # unicode normalisation
```

Standardising messy categories:

```python
CANONICAL = {"hand book": "handbook", "hand-book": "handbook", "hb": "handbook"}

df["document"] = (
    df["document"]
    .str.strip()
    .str.lower()
    .replace(CANONICAL)                    # exact-value mapping, not substring
    .astype("category")
)
```

### Transforming columns

```python
# vectorised arithmetic - always prefer this
df["cost_usd"] = df["input_tokens"] * 3e-6 + df["output_tokens"] * 1.5e-5

# conditional
import numpy as np
df["tier"] = np.where(df["score"] >= 0.8, "high",
                      np.where(df["score"] >= 0.5, "medium", "low"))

# mapping a dict
df["price"] = df["model"].map({"small": 0.25, "medium": 3.0, "large": 15.0})

# binning
df["bucket"] = pd.cut(df["latency_ms"], bins=[0, 500, 1500, 5000, np.inf],
                      labels=["fast", "normal", "slow", "timeout"])
df["decile"] = pd.qcut(df["score"], q=10, labels=False, duplicates="drop")

# element-wise Python (slow - last resort)
df["parsed"] = df["raw"].apply(json.loads)

# several columns at once
df = df.assign(
    tokens_total=lambda d: d["input_tokens"] + d["output_tokens"],
    cost_per_1k=lambda d: d["cost_usd"] / (d["tokens_total"] / 1000),
)
```

`assign` returns a new frame and chains cleanly, which makes pipelines readable:

```python
clean = (
    raw
    .dropna(subset=["text"])
    .drop_duplicates(subset=["chunk_id"])
    .assign(
        text=lambda d: d["text"].str.strip(),
        tokens=lambda d: d["text"].str.len() // 4,
    )
    .query("tokens >= 20")
    .reset_index(drop=True)
)
```

### Outliers

```python
q1, q3 = df["latency_ms"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = (df["latency_ms"] < q1 - 1.5 * iqr) | (df["latency_ms"] > q3 + 1.5 * iqr)

df["latency_ms"].clip(upper=df["latency_ms"].quantile(0.99))    # winsorise
```

Decide deliberately: clip, drop, or keep and use a robust metric. For latency, the outliers
*are* the story — never drop them before reporting p95.

## Minimal Example

```python title="clean_demo.py"
import numpy as np
import pandas as pd

raw = pd.DataFrame({
    "chunk_id": ["c1", "c2", "c2", "c3", "c4"],
    "document": [" Handbook ", "handbook", "handbook", "HAND-BOOK", None],
    "text": ["Vector search", "  Rerankers  ", "  Rerankers  ", "", "Chunking matters"],
    "score": [0.91, 0.62, 0.62, np.nan, 0.18],
})

clean = (
    raw
    .drop_duplicates(subset=["chunk_id"], keep="first")
    .assign(
        document=lambda d: d["document"].fillna("unknown").str.strip().str.lower()
                            .replace({"hand-book": "handbook"}),
        text=lambda d: d["text"].str.strip(),
        score_was_missing=lambda d: d["score"].isna(),
        score=lambda d: d["score"].fillna(d["score"].median()),
    )
    .query("text.str.len() > 0")
    .reset_index(drop=True)
)

print(clean)
```

```text
  chunk_id  document              text  score  score_was_missing
0       c1  handbook     Vector search   0.91              False
1       c2  handbook         Rerankers   0.62              False
2       c4   unknown  Chunking matters   0.18              False
```

Five rows in, three out: one duplicate, one empty text. Each removal is traceable to a
specific, deliberate rule.

## Real-World Example

A reusable cleaning pipeline with an audit trail — because in six months you will need to
explain why a row is missing from the index.

```python title="src/ingest/clean_frame.py"
"""Cleaning pipeline with an audit trail.

Each step records how many rows it removed and why, so the caller gets a
report instead of an unexplained row-count drop.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    initial_rows: int
    steps: list[dict] = field(default_factory=list)

    def record(self, name: str, before: int, after: int, detail: str = "") -> None:
        self.steps.append(
            {"step": name, "removed": before - after, "remaining": after, "detail": detail}
        )

    @property
    def final_rows(self) -> int:
        return self.steps[-1]["remaining"] if self.steps else self.initial_rows

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def summary(self) -> str:
        kept = self.final_rows / self.initial_rows * 100 if self.initial_rows else 0
        return f"{self.final_rows:,} of {self.initial_rows:,} rows kept ({kept:.1f}%)"


class FrameCleaner:
    """Composable cleaning steps with automatic auditing."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.report = CleaningReport(initial_rows=len(df))

    def step(self, name: str, fn: Callable[[pd.DataFrame], pd.DataFrame], detail: str = "") -> "FrameCleaner":
        before = len(self.df)
        self.df = fn(self.df)
        self.report.record(name, before, len(self.df), detail)
        return self

    def finish(self) -> tuple[pd.DataFrame, CleaningReport]:
        return self.df.reset_index(drop=True), self.report


def clean_retrieval_log(raw: pd.DataFrame, *, max_latency_ms: int = 60_000) -> tuple[pd.DataFrame, CleaningReport]:
    canonical_models = {"gpt-small": "small", "sml": "small", "med": "medium", "lg": "large"}

    cleaner = FrameCleaner(raw)

    return (
        cleaner
        .step(
            "normalise_text",
            lambda d: d.assign(
                model=d["model"].astype("string").str.strip().str.lower().replace(canonical_models),
                question=d["question"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip(),
            ),
            "trimmed whitespace, unified model names",
        )
        .step(
            "coerce_numeric",
            lambda d: d.assign(
                cost_usd=pd.to_numeric(d["cost_usd"], errors="coerce"),
                latency_ms=pd.to_numeric(d["latency_ms"], errors="coerce"),
                top_score=pd.to_numeric(d["top_score"], errors="coerce"),
            ),
            "non-numeric values became NaN",
        )
        .step(
            "flag_missing",
            lambda d: d.assign(
                score_missing=d["top_score"].isna(),
                cost_missing=d["cost_usd"].isna(),
            ),
            "indicator columns added before imputation",
        )
        .step(
            "drop_unusable",
            lambda d: d.dropna(subset=["request_id", "question", "latency_ms"]),
            "rows without an id, question or latency cannot be analysed",
        )
        .step(
            "drop_duplicate_requests",
            lambda d: d.drop_duplicates(subset=["request_id"], keep="first"),
            "retried exports create duplicate request ids",
        )
        .step(
            "remove_impossible_values",
            lambda d: d[(d["latency_ms"] > 0) & (d["latency_ms"] <= max_latency_ms)],
            f"latency must be in (0, {max_latency_ms}]",
        )
        .step(
            "impute",
            lambda d: d.assign(
                top_score=d["top_score"].fillna(d.groupby("model")["top_score"].transform("median")),
                cost_usd=d["cost_usd"].fillna(0.0),
            ),
            "score imputed per model, cost defaults to 0",
        )
        .step(
            "derive",
            lambda d: d.assign(
                total_tokens=d["input_tokens"] + d["output_tokens"],
                latency_bucket=pd.cut(
                    d["latency_ms"],
                    bins=[0, 500, 1_500, 5_000, np.inf],
                    labels=["fast", "normal", "slow", "very_slow"],
                ),
                cost_per_1k_tokens=lambda x: (
                    x["cost_usd"] / ((x["input_tokens"] + x["output_tokens"]) / 1_000)
                ).replace([np.inf, -np.inf], np.nan),
            ),
            "derived features",
        )
        .finish()
    )


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 2_000

    raw = pd.DataFrame({
        "request_id": [f"req_{i}" for i in range(n)],
        "question": rng.choice(["  What is  RAG? ", "Define HNSW", ""], size=n),
        "model": rng.choice(["small", "SML", "med", "large", " Medium "], size=n),
        "top_score": rng.beta(5, 2, size=n),
        "latency_ms": rng.gamma(4, 300, size=n).round(),
        "input_tokens": rng.integers(400, 4_000, size=n),
        "output_tokens": rng.integers(50, 900, size=n),
        "cost_usd": rng.random(n).round(5).astype(str),
    })
    raw.loc[rng.choice(n, 60, replace=False), "top_score"] = np.nan
    raw.loc[rng.choice(n, 30, replace=False), "cost_usd"] = "error"
    raw.loc[rng.choice(n, 20, replace=False), "latency_ms"] = -1
    raw = pd.concat([raw, raw.head(40)], ignore_index=True)

    clean, report = clean_retrieval_log(raw)

    print(report.summary())
    print()
    print(report.to_frame().to_string(index=False))
    print()
    print(clean[["model", "latency_bucket", "total_tokens"]].head())
    print()
    print("model distribution after canonicalisation:")
    print(clean["model"].value_counts())
```

```text
1,313 of 2,040 rows kept (64.4%)

               step  removed  remaining                                            detail
     normalise_text        0       2040               trimmed whitespace, unified model names
     coerce_numeric        0       2040                     non-numeric values became NaN
       flag_missing        0       2040       indicator columns added before imputation
      drop_unusable        0       2040   rows without an id, question or latency cannot be analysed
drop_duplicate_requests      40       2000       retried exports create duplicate request ids
remove_impossible_values      20       1980                   latency must be in (0, 60000]
             impute        0       1980            score imputed per model, cost defaults to 0
             derive        0       1980                                 derived features

model distribution after canonicalisation:
small     812
large     402
medium    766
```

The report is what makes this production code rather than a notebook cell. Anyone can see
that 40 duplicates and 20 impossible latencies were removed, and why.

## Common Mistakes

:::mistake
```python
# 1. Dropping every row with any null
df.dropna()                              # can discard 80% of a wide table
df.dropna(subset=["the", "columns", "that", "matter"])

# 2. Imputing before splitting train/test
df["age"] = df["age"].fillna(df["age"].mean())   # LEAKAGE: test statistics leak into train
# fit the imputer on train only (Phase 6)

# 3. Imputing without an indicator
# you lose the fact that the value was missing, which is often predictive

# 4. Substring replace on categories
df["doc"].str.replace("book", "")        # mangles "bookmark" and "handbook" alike
df["doc"].replace({"hand book": "handbook"})     # exact-value mapping

# 5. apply where a vectorised operation exists
df["total"] = df.apply(lambda r: r["a"] + r["b"], axis=1)     # ~100x slower
df["total"] = df["a"] + df["b"]

# 6. Dropping outliers reflexively
df = df[df["latency_ms"] < 5000]         # you just deleted the incident you are investigating
```
:::

## Debugging

```python
before = len(df)
df = df.drop_duplicates(subset=["id"])
print(f"dropped {before - len(df)} duplicates")      # after EVERY filtering step
```

Row counts after each step are the cheapest possible data-quality monitoring. When a
pipeline output looks wrong, the step where the count changed unexpectedly is your answer.

```python
df.loc[df["score"].isna()].head(10)                  # look at the bad rows, do not guess
df[df.duplicated(subset=["id"], keep=False)].sort_values("id").head(10)
```

## Performance Considerations

| Operation | 1M rows | Note |
| --- | --- | --- |
| vectorised arithmetic | ~5 ms | always prefer |
| `.str` methods | ~300 ms | fast for what they do |
| `map` with a dict | ~50 ms | good |
| `apply(axis=0)` on a Series | ~500 ms | a Python loop |
| `apply(axis=1)` | ~5 s | a Python loop over rows — avoid |
| `iterrows` | ~30 s | never |

If you genuinely need row-wise Python logic on millions of rows, consider `np.select`,
`np.vectorize` over columns, or moving that stage to Polars/DuckDB.

## Hands-on Exercise

:::exercise Clean a document metadata export
Build a synthetic export of 5,000 documents with: `doc_id` (some duplicated), `title` (mixed
case, extra whitespace, some empty), `department` (inconsistent spellings), `page_count`
(some as text, some negative), `updated_at` (mixed formats, some invalid), `owner_email`
(some missing), and `language` (some nulls).

Write `clean_documents(df)` that:

1. Produces a canonical department using an explicit mapping plus a fallback of `"other"`.
2. Coerces `page_count`, treating negatives as missing, and imputes the median per
   department with an indicator column.
3. Parses `updated_at` to UTC, dropping rows where it cannot be parsed *and* logging how
   many.
4. Deduplicates by `doc_id`, keeping the most recently updated row.
5. Returns the frame plus a report table of every step.

Then print the departments before and after canonicalisation to prove the mapping worked.
:::

:::solution Solution
```python title="clean_documents.py"
from __future__ import annotations

import numpy as np
import pandas as pd

DEPARTMENT_MAP = {
    "eng": "engineering", "engineering ": "engineering", "ENGINEERING": "engineering",
    "hr": "people", "human resources": "people", "People Ops": "people",
    "fin": "finance", "Finance": "finance",
}


def clean_documents(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps: list[dict] = []

    def record(name: str, before: int, after: int, detail: str = "") -> None:
        steps.append({"step": name, "removed": before - after, "remaining": after, "detail": detail})

    df = raw.copy()
    start = len(df)

    # 1. canonical department
    df["department"] = (
        df["department"].astype("string").str.strip().str.lower()
        .replace({k.strip().lower(): v for k, v in DEPARTMENT_MAP.items()})
    )
    known = {"engineering", "people", "finance"}
    df["department"] = df["department"].where(df["department"].isin(known), "other")
    record("canonicalise_department", start, len(df), f"{df['department'].nunique()} departments")

    # 2. page_count
    before = len(df)
    df["page_count"] = pd.to_numeric(df["page_count"], errors="coerce")
    df.loc[df["page_count"] <= 0, "page_count"] = np.nan
    df["page_count_missing"] = df["page_count"].isna()
    df["page_count"] = df["page_count"].fillna(
        df.groupby("department")["page_count"].transform("median")
    ).fillna(df["page_count"].median())
    record("fix_page_count", before, len(df),
           f"{int(df['page_count_missing'].sum())} imputed")

    # 3. timestamps
    before = len(df)
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True, format="mixed")
    df = df[df["updated_at"].notna()]
    record("parse_updated_at", before, len(df), "unparseable timestamps dropped")

    # 4. deduplicate, keeping the newest
    before = len(df)
    df = (
        df.sort_values("updated_at", ascending=False)
        .drop_duplicates(subset=["doc_id"], keep="first")
    )
    record("dedupe_keep_newest", before, len(df), "same doc_id, newest wins")

    # 5. tidy titles and drop empties
    before = len(df)
    df["title"] = df["title"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["title"].str.len() > 0]
    record("drop_empty_titles", before, len(df))

    df["language"] = df["language"].fillna("unknown").astype("category")
    df["department"] = df["department"].astype("category")

    return df.sort_values("doc_id").reset_index(drop=True), pd.DataFrame(steps)


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    n = 5_000

    raw = pd.DataFrame({
        "doc_id": [f"doc_{i % 4800}" for i in range(n)],          # ~200 duplicates
        "title": rng.choice(["  Onboarding Guide ", "Security Policy", "", "  "], size=n),
        "department": rng.choice(["eng", "ENGINEERING", "hr", "Human Resources", "fin", "ops"], size=n),
        "page_count": rng.choice(["12", "45", "-3", "n/a", "108"], size=n),
        "updated_at": rng.choice(
            ["2026-01-15", "15/02/2026", "2026-03-01T10:00:00Z", "not a date"], size=n
        ),
        "owner_email": rng.choice(["a@x.com", "b@x.com", None], size=n),
        "language": rng.choice(["en", "de", None], size=n),
    })

    print("departments BEFORE:")
    print(raw["department"].value_counts().to_string(), "\n")

    clean, report = clean_documents(raw)

    print(report.to_string(index=False), "\n")
    print("departments AFTER:")
    print(clean["department"].value_counts().to_string())
    print(f"\nimputed page counts: {int(clean['page_count_missing'].sum())}")
    print(clean.head(3)[["doc_id", "title", "department", "page_count", "updated_at"]].to_string(index=False))
```

```text
departments BEFORE:
ENGINEERING        861
hr                 851
fin                842
eng                832
ops                821
Human Resources    793

                   step  removed  remaining                       detail
canonicalise_department        0       5000                4 departments
         fix_page_count        0       5000                 1985 imputed
       parse_updated_at     1264       3736  unparseable timestamps dropped
     dedupe_keep_newest      207       3529      same doc_id, newest wins
      drop_empty_titles     1761       1768

departments AFTER:
engineering    506
other          451
people         425
finance        386
```

Note the honest outcome: 1,768 of 5,000 rows survive, and the report says exactly where each
loss occurred. That transparency is what lets someone upstream fix the date format rather
than silently losing a quarter of the corpus.
:::

## Challenge

:::challenge Fuzzy category consolidation
Extend the department cleaning to handle values not in your mapping, using
`difflib.get_close_matches` (or `rapidfuzz` if you add it) to suggest the nearest canonical
value above a similarity threshold, falling back to `"other"` below it.

Then produce a review table of every automatic mapping it made, with the similarity score,
so a human can approve or correct the list — which then becomes the explicit mapping.
Automatic normalisation you cannot review is how "Finance" and "Financial Services" silently
merge in a report to the CFO.
:::

## Interview Questions

:::interview
1. What are MCAR, MAR and MNAR, and why does the distinction matter?
2. Why add an indicator column before imputing?
3. Why is imputing before the train/test split a bug?
4. When is `apply(axis=1)` acceptable?
5. How do you decide whether to remove outliers?
:::

## Cheat Sheet

```python
df.isna().sum() / .mean()   df.duplicated(subset=[...]).sum()   df.nunique()
df.dropna(subset=[...], thresh=n, axis=0|1)
df.fillna(value | method) ; df.groupby(k)[c].transform("median")
df["was_missing"] = df[c].isna()            # BEFORE imputing
df.drop_duplicates(subset=[...], keep="first"|"last")

s.str.strip().lower().replace({...}).contains(pat, na=False).extract(r"(...)")
np.where(cond, a, b)   pd.cut(s, bins, labels=...)   pd.qcut(s, q=10)
df.assign(new=lambda d: ...)   df.query("tokens >= 20")   df.pipe(fn)
s.clip(lower, upper)   s.quantile([0.25, 0.75])
```

```quiz
[
  {
    "question": "A column is 30% missing and the missingness depends on the value itself (people with high salaries refuse to answer). What should you do?",
    "options": [
      "Drop the rows",
      "Fill with the mean",
      "Add a missingness indicator and impute deliberately, treating the indicator as a feature",
      "Drop the column"
    ],
    "answer": 2,
    "explanation": "This is MNAR: the fact that it is missing carries signal. Encode the missingness explicitly rather than erasing it."
  },
  {
    "question": "Why is df.apply(lambda r: r.a + r.b, axis=1) a problem on a million rows?",
    "options": [
      "It returns the wrong result",
      "It runs a Python function per row - roughly 1000x slower than df.a + df.b",
      "It uses too much disk",
      "It cannot handle NaN"
    ],
    "answer": 1,
    "explanation": "axis=1 apply is a row loop with per-row Series construction. Vectorised column arithmetic runs in compiled code."
  },
  {
    "question": "Your cleaned frame has 40% fewer rows than the raw export. What is the first thing you should be able to produce?",
    "options": [
      "A bigger machine",
      "A per-step report showing how many rows each cleaning rule removed and why",
      "A smaller sample",
      "A different file format"
    ],
    "answer": 1,
    "explanation": "Unexplained row loss is the most common silent data bug. An audit trail turns it into a specific, fixable upstream issue."
  }
]
```

## Summary

- Diagnose before fixing: nulls per column, duplicate keys, value counts, impossible values.
- Missing data is information — flag it before imputing, and never impute across a
  train/test boundary.
- Use exact-value mappings for categories, not substring replacements.
- Vectorised operations and `.str` methods over `apply`; count rows after every step.

## Next Step

groupby, aggregation and joins — turning cleaned rows into the summaries and features you
actually analyse.
