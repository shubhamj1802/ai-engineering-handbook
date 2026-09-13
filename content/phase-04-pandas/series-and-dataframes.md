---
title: Series, DataFrames and Loading Data
order: 1
difficulty: Intermediate
duration: 14
badges: ["Hands-on", "Start here"]
summary: "The two Pandas objects, how the index works, and loading CSV/JSON/Parquet correctly — including the dtype choices that decide whether your pipeline fits in memory."
prereqs: ["Arrays, Shapes and dtypes", "Files, JSON and CSV"]
keyConcepts: ["Series", "DataFrame", "index", "dtype", "read_csv", "category"]
---

## Why this matters

Pandas is where your data actually lives before it becomes features, evaluation results or
a corpus manifest. It is also where the majority of data bugs happen: a numeric column read
as text, a silently dropped row, a join that duplicates records. Getting loading and dtypes
right at the start removes most of them.

## Mental Model

```text
Series    = one labelled column     (a NumPy array + an index)
DataFrame = a dict of Series sharing one index

        index    question              tokens   score
        ------   -------------------   ------   -----
   0    q1       "What is RAG?"           12     0.91
   1    q2       "Define HNSW"             9     0.72

df["score"]        → a Series
df[["question"]]   → a DataFrame (note the double brackets)
df.loc[0]          → a Series (that row)
```

The **index** is not just row numbers: it is a label for each row, used for alignment in
every operation. Two Series added together align on their index, not their position — the
most surprising and most useful behaviour in Pandas.

## Prerequisites

```bash
uv add pandas pyarrow          # pyarrow enables Parquet and better string dtypes
```

## Core Concepts

### Series

```python
import pandas as pd

scores = pd.Series([0.91, 0.34, 0.72], index=["c1", "c2", "c3"], name="score")

scores["c1"]          # 0.91           by label
scores.iloc[0]        # 0.91           by position
scores.mean()         # 0.657
scores[scores > 0.5]  # c1 and c3      boolean mask, same as NumPy
scores * 100          # vectorised
scores.index          # Index(['c1', 'c2', 'c3'])
```

Alignment in action:

```python
a = pd.Series([1, 2], index=["x", "y"])
b = pd.Series([10, 20], index=["y", "z"])
a + b
```

```text
x     NaN      ← only in a
y    12.0      ← in both
z     NaN      ← only in b
```

This is a feature — it stops you from silently adding mismatched data — and a trap if you
expected positional behaviour.

### DataFrames

```python
df = pd.DataFrame({
    "id": ["q1", "q2", "q3"],
    "question": ["What is RAG?", "Define HNSW", "Explain reranking"],
    "tokens": [12, 9, 14],
    "score": [0.91, 0.34, 0.72],
})

df.shape          # (3, 4)
df.columns        # Index(['id', 'question', 'tokens', 'score'])
df.dtypes         # per-column types
df.head(2) df.tail(2) df.sample(2, random_state=0)
df.info()         # dtypes, non-null counts, memory - ALWAYS run this first
df.describe()     # numeric summary
df.describe(include="object")   # categorical summary
```

### Selecting

```python
df["score"]                    # a Series
df[["id", "score"]]            # a DataFrame

df.loc[0]                      # row by LABEL
df.loc[0, "score"]             # single value
df.loc[df["score"] > 0.5, ["id", "score"]]     # mask + columns

df.iloc[0]                     # row by POSITION
df.iloc[0:2, 1:3]              # positional slice
df.at[0, "score"]              # fast scalar access by label
```

:::warning `loc` is inclusive of the end label, `iloc` is not
```python
df.loc[0:2]     # rows 0, 1 AND 2   (labels)
df.iloc[0:2]    # rows 0, 1          (positions)
```
:::

### Loading data

```python
df = pd.read_csv(
    "data/questions.csv",
    dtype={"id": "string", "tokens": "int32", "category": "category"},
    parse_dates=["created_at"],
    usecols=["id", "question", "tokens", "created_at", "category"],   # read less
    nrows=1000,                      # sample while exploring
    na_values=["", "NA", "null", "-"],
    encoding="utf-8",
)

pd.read_json("data/records.json")
pd.read_json("data/records.jsonl", lines=True)          # JSONL
pd.read_parquet("data/chunks.parquet")                  # fast, typed, compressed
pd.read_sql("SELECT * FROM queries WHERE created_at > %s", conn, params=[cutoff])
pd.read_excel("data/report.xlsx", sheet_name="results")
```

Writing:

```python
df.to_csv("out.csv", index=False)                  # index=False unless the index is data
df.to_parquet("out.parquet", index=False)          # preferred for anything you reload
df.to_json("out.jsonl", orient="records", lines=True)
```

:::tip Use Parquet for intermediate data
CSV loses every dtype, stores numbers as text, and cannot compress well. Parquet is
typically 5–10× smaller, 10–50× faster to read, and preserves dtypes exactly. Keep CSV for
human hand-off only.
:::

### dtypes and memory

```python
df.dtypes
df.memory_usage(deep=True).sum() / 1024**2      # MB, including string contents
```

| Pandas dtype | Notes |
| --- | --- |
| `int64` / `float64` | defaults; often oversized |
| `int32`, `float32` | half the memory, fine for most features |
| `Int64` (capital I) | nullable integer — keeps ints when NaNs are present |
| `string` | proper string dtype (needs pyarrow), better than `object` |
| `category` | for low-cardinality text: huge memory win |
| `bool` | 1 byte |
| `datetime64[ns]` | real timestamps, not strings |

```python
df["category"] = df["category"].astype("category")     # 10k rows, 5 distinct values:
                                                        # ~600 KB → ~11 KB
df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")   # bad values → NaN
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
```

## Minimal Example

```python title="pandas_basics.py"
import pandas as pd

df = pd.DataFrame({
    "chunk_id": ["c1", "c2", "c3", "c4"],
    "document": ["handbook", "handbook", "faq", "blog"],
    "tokens": [180, 240, 95, 310],
    "score": [0.91, 0.62, 0.44, 0.18],
})

print(df.info())
print()
print(df[df["score"] >= 0.5][["chunk_id", "document", "score"]])
print()
print("mean score by document:")
print(df.groupby("document")["score"].mean().round(3).sort_values(ascending=False))
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   chunk_id  4 non-null      object
 1   document  4 non-null      object
 2   tokens    4 non-null      int64
 3   score     4 non-null      float64
memory usage: 260.0+ bytes

  chunk_id  document  score
0       c1  handbook   0.91
1       c2  handbook   0.62

mean score by document:
document
handbook    0.765
faq         0.440
blog        0.180
```

## Real-World Example

A corpus manifest loader: read a chunk inventory, enforce a schema, optimise memory and
report data quality — the first step of every ingestion pipeline.

```python title="src/ingest/manifest.py"
"""Load and validate a corpus manifest.

Four things every loader in a real pipeline must do:
  1. declare the schema instead of trusting inference
  2. coerce and record failures rather than raising on the first bad row
  3. optimise dtypes so the frame fits comfortably in memory
  4. emit a data-quality report the caller can act on
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCHEMA: dict[str, str] = {
    "chunk_id": "string",
    "document": "category",
    "page": "Int32",            # nullable: some sources have no pages
    "text": "string",
    "tokens": "Int32",
    "language": "category",
}

REQUIRED = ["chunk_id", "document", "text"]


@dataclass(frozen=True, slots=True)
class LoadReport:
    rows_read: int
    rows_kept: int
    duplicates_dropped: int
    coercion_failures: dict[str, int]
    memory_mb: float

    @property
    def rows_dropped(self) -> int:
        return self.rows_read - self.rows_kept

    def summary(self) -> str:
        return (
            f"{self.rows_kept:,} kept of {self.rows_read:,} "
            f"({self.rows_dropped:,} dropped, {self.duplicates_dropped:,} duplicates), "
            f"{self.memory_mb:.1f} MB"
        )


def load_manifest(path: Path, *, min_tokens: int = 20) -> tuple[pd.DataFrame, LoadReport]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")

    reader = pd.read_parquet if path.suffix == ".parquet" else pd.read_csv
    df = reader(path)
    rows_read = len(df)

    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")

    # --- coerce types, counting failures instead of raising ------------------
    failures: dict[str, int] = {}
    for column, dtype in SCHEMA.items():
        if column not in df.columns:
            df[column] = pd.NA
        if dtype in {"Int32", "Int64"}:
            coerced = pd.to_numeric(df[column], errors="coerce").astype(dtype)
            failures[column] = int(coerced.isna().sum() - df[column].isna().sum())
            df[column] = coerced
        else:
            df[column] = df[column].astype(dtype)

    if "ingested_at" in df.columns:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce", utc=True)

    # --- quality filters -----------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["chunk_id"], keep="first")
    duplicates = before - len(df)

    df = df[df["text"].notna() & (df["text"].str.len() > 0)]
    df = df[df["tokens"].fillna(0) >= min_tokens]
    df = df.reset_index(drop=True)

    report = LoadReport(
        rows_read=rows_read,
        rows_kept=len(df),
        duplicates_dropped=duplicates,
        coercion_failures={k: v for k, v in failures.items() if v > 0},
        memory_mb=df.memory_usage(deep=True).sum() / 1024**2,
    )
    return df, report


def optimise_memory(df: pd.DataFrame, *, category_threshold: float = 0.5) -> pd.DataFrame:
    """Downcast numerics and categorise low-cardinality strings."""
    out = df.copy()
    for column in out.columns:
        kind = out[column].dtype.kind
        if kind == "i":
            out[column] = pd.to_numeric(out[column], downcast="integer")
        elif kind == "f":
            out[column] = pd.to_numeric(out[column], downcast="float")
        elif kind in {"O", "U"} or str(out[column].dtype) == "string":
            uniqueness = out[column].nunique(dropna=False) / max(len(out), 1)
            if uniqueness < category_threshold:
                out[column] = out[column].astype("category")
    return out


if __name__ == "__main__":
    demo = Path("data/manifest.csv")
    demo.parent.mkdir(parents=True, exist_ok=True)
    demo.write_text(
        "chunk_id,document,page,text,tokens,language\n"
        "c1,handbook,4,Vector search finds similar embeddings,180,en\n"
        "c2,handbook,5,Rerankers reorder candidates,240,en\n"
        "c2,handbook,5,Rerankers reorder candidates,240,en\n"      # duplicate
        "c3,faq,,Short,8,en\n"                                      # below min_tokens
        "c4,blog,1,Chunking decides what can be retrieved,not_a_number,de\n",
        encoding="utf-8",
    )

    df, report = load_manifest(demo, min_tokens=20)
    print(report.summary())
    print("coercion failures:", report.coercion_failures)
    print(df[["chunk_id", "document", "tokens", "language"]])
    print("\noptimised memory:",
          f"{optimise_memory(df).memory_usage(deep=True).sum() / 1024:.1f} KB")
```

```text
2 kept of 5 (3 dropped, 1 duplicates), 0.0 MB
coercion failures: {'tokens': 1}
  chunk_id  document  tokens language
0       c1  handbook     180       en
1       c2  handbook     240       en

optimised memory: 0.7 KB
```

The report is the point: it tells you that one row had an unparseable token count and one
was a duplicate. A loader that silently returns two rows out of five is how bad data reaches
an index.

## Common Mistakes

:::mistake
```python
# 1. Trusting dtype inference
df = pd.read_csv(path)                 # ids become int64 and lose leading zeros;
                                        # mixed columns become object
df = pd.read_csv(path, dtype={"id": "string"})

# 2. Chained assignment
df[df["score"] > 0.5]["flag"] = True   # SettingWithCopyWarning; writes to a copy
df.loc[df["score"] > 0.5, "flag"] = True

# 3. Writing the index by accident
df.to_csv("out.csv")                   # adds an unnamed index column
df.to_csv("out.csv", index=False)

# 4. iterrows in a loop
for _, row in df.iterrows():           # slow and copies each row into a Series
    ...
df["new"] = df["a"] * df["b"]          # vectorised

# 5. Ignoring alignment
df["ratio"] = series_a / series_b      # NaNs appear if the indexes differ
                                        # .reset_index(drop=True) or .to_numpy() first

# 6. object dtype strings
df["text"].str.len()                    # works, but object dtype is slow and memory-hungry
df["text"] = df["text"].astype("string")
```
:::

## Debugging

```python
df.info(memory_usage="deep")         # dtypes, nulls, real memory
df.head() / df.sample(5, random_state=0)
df.dtypes.value_counts()
df.isna().sum().sort_values(ascending=False)     # missing per column
df["column"].value_counts(dropna=False).head(10)
df.duplicated(subset=["id"]).sum()
df.describe(include="all").T
```

Run `info()` and `isna().sum()` on every dataset before you do anything else. They answer
"what am I actually holding?" in two lines.

## Performance Considerations

| Operation | Cost | Better |
| --- | --- | --- |
| `iterrows()` | ~100 µs/row | vectorised column operations |
| `apply(axis=1)` | ~10 µs/row | vectorised, or `np.where` |
| `df.append` in a loop | O(n²) | collect a list, then `pd.concat` once |
| CSV read | slow, untyped | Parquet |
| `object` strings | high memory | `string` or `category` |
| repeated `df.loc[...]` writes | slow | build columns, assign once |

For frames beyond a few gigabytes, move to Polars or DuckDB rather than fighting Pandas —
both read Parquet natively and Pandas can hand data to them zero-copy via Arrow.

## Hands-on Exercise

:::exercise Load and profile a retrieval log
Generate a synthetic retrieval log of 50,000 rows with columns `request_id`, `timestamp`,
`user_id`, `question`, `model`, `k`, `top_score`, `latency_ms`, `input_tokens`,
`output_tokens`, `cost_usd`, `escalated` — deliberately including some missing values,
duplicated request ids and a numeric column stored as text.

Then write `profile(df)` that reports, in one table: per column, the dtype, null count, null
percentage, number of distinct values, memory in KB, and a sample value. Finish by applying
dtype optimisation and printing the before/after memory.
:::

:::solution Solution
```python title="profile_log.py"
from __future__ import annotations

import numpy as np
import pandas as pd


def make_log(n: int = 50_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "request_id": [f"req_{i:06d}" for i in range(n)],
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="30s", tz="UTC"),
        "user_id": rng.integers(1, 500, size=n),
        "question": rng.choice(["What is RAG?", "Define HNSW", "Explain reranking"], size=n),
        "model": rng.choice(["small", "medium", "large"], size=n, p=[0.6, 0.3, 0.1]),
        "k": rng.integers(3, 12, size=n),
        "top_score": rng.beta(5, 2, size=n).round(4),
        "latency_ms": rng.gamma(shape=4, scale=300, size=n).round().astype(int),
        "input_tokens": rng.integers(500, 4000, size=n),
        "output_tokens": rng.integers(50, 800, size=n),
        "escalated": rng.random(n) < 0.07,
    })
    # cost stored as text, with some junk - the classic real-world column
    df["cost_usd"] = (df["input_tokens"] * 3e-6 + df["output_tokens"] * 1.5e-5).round(6).astype(str)
    df.loc[rng.choice(n, 200, replace=False), "cost_usd"] = "n/a"
    df.loc[rng.choice(n, 800, replace=False), "top_score"] = np.nan
    return pd.concat([df, df.head(120)], ignore_index=True)      # inject duplicates


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "nulls": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "distinct": int(series.nunique(dropna=True)),
            "memory_kb": round(series.memory_usage(deep=True) / 1024, 1),
            "sample": str(series.dropna().iloc[0])[:28] if series.notna().any() else "",
        })
    return pd.DataFrame(rows).set_index("column")


def optimise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_usd"] = pd.to_numeric(out["cost_usd"], errors="coerce").astype("float32")
    for column in ["model", "question"]:
        out[column] = out[column].astype("category")
    for column in ["user_id", "k", "latency_ms", "input_tokens", "output_tokens"]:
        out[column] = pd.to_numeric(out[column], downcast="integer")
    out["top_score"] = out["top_score"].astype("float32")
    out = out.drop_duplicates(subset=["request_id"], keep="first").reset_index(drop=True)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    log = make_log()

    print(profile(log))
    before = log.memory_usage(deep=True).sum() / 1024**2

    optimised = optimise(log)
    after = optimised.memory_usage(deep=True).sum() / 1024**2

    print(f"\nrows: {len(log):,} -> {len(optimised):,} (duplicates removed)")
    print(f"memory: {before:.1f} MB -> {after:.1f} MB ({(1 - after / before) * 100:.0f}% smaller)")
    print(f"unparseable costs: {optimised['cost_usd'].isna().sum()}")
```

```text
                       dtype  nulls  null_pct  distinct  memory_kb                    sample
column
request_id            object      0      0.00     50000     3306.4                req_000000
timestamp     datetime64[ns…      0      0.00     50000      391.5   2026-01-01 00:00:00+00:00
user_id                int64      0      0.00       499      391.5                       321
question              object      0      0.00         3     3197.3              What is RAG?
model                 object      0      0.00         3     3049.8                     small
k                      int64      0      0.00         9      391.5                         7
top_score            float64    802      1.60     45684      391.5                    0.7891
latency_ms             int64      0      0.00      3862      391.5                       842
input_tokens           int64      0      0.00      3500      391.5                      2317
output_tokens          int64      0      0.00       750      391.5                       412
escalated               bool      0      0.00         2       49.0                     False
cost_usd              object      0      0.00     47891     3441.0                  0.013128

rows: 50,120 -> 50,000 (duplicates removed)
memory: 14.6 MB -> 3.2 MB (78% smaller)
unparseable costs: 200
```

78% less memory, duplicates removed, and 200 bad cost values surfaced rather than silently
treated as text.
:::

## Challenge

:::challenge Schema contract with validation
Write a `validate(df, schema)` function where `schema` declares, per column: dtype, required,
allowed range or allowed values, and maximum null fraction. It returns a list of violations
(column, rule, observed) rather than raising, and a boolean `is_valid`. Then wire it into the
loader so a manifest that fails validation is rejected with a report naming every problem.

This is the Pandas equivalent of Pydantic validation, and it is what stops a corrupted export
from reaching your vector index — a failure mode that is very hard to debug downstream.
:::

## Interview Questions

:::interview
1. What is the difference between `loc` and `iloc`?
2. What does index alignment do when you add two Series?
3. When would you use the `category` dtype?
4. Why prefer Parquet over CSV for intermediate data?
5. What causes `SettingWithCopyWarning` and how do you fix it properly?
:::

## Cheat Sheet

```python
pd.read_csv(p, dtype={...}, parse_dates=[...], usecols=[...], nrows=1000)
pd.read_parquet(p) / pd.read_json(p, lines=True) / pd.read_sql(q, conn)
df.to_parquet(p, index=False) / df.to_csv(p, index=False)

df.shape df.columns df.dtypes df.info(memory_usage="deep") df.describe()
df.head() df.sample(5, random_state=0) df.memory_usage(deep=True).sum()

df["col"]  df[["a","b"]]  df.loc[mask, ["a"]]  df.iloc[0:2, 1:3]  df.at[0, "col"]
df.loc[mask, "col"] = value          # the only correct way to assign on a subset

df["col"].astype("category" | "string" | "float32" | "Int32")
pd.to_numeric(s, errors="coerce", downcast="integer")
pd.to_datetime(s, errors="coerce", utc=True)

df.isna().sum()  df.duplicated(subset=["id"]).sum()  df["c"].value_counts(dropna=False)
```

```quiz
[
  {
    "question": "df.loc[0:2] versus df.iloc[0:2] — how many rows does each return?",
    "options": ["Both 2", "Both 3", "loc 3, iloc 2", "loc 2, iloc 3"],
    "answer": 2,
    "explanation": "loc slices by label and includes the endpoint (0,1,2); iloc slices by position and excludes it (0,1)."
  },
  {
    "question": "Why does `df[df.score > 0.5]['flag'] = True` fail to change the DataFrame?",
    "options": [
      "Boolean masks are read-only",
      "The first selection may return a copy, so the assignment writes to a temporary",
      "flag must exist first",
      "True is not a valid value"
    ],
    "answer": 1,
    "explanation": "Chained indexing creates an intermediate object; assign through a single .loc call instead: df.loc[df.score > 0.5, 'flag'] = True."
  },
  {
    "question": "A column has 50,000 rows and 4 distinct string values. Which dtype?",
    "options": ["object", "string", "category", "int8"],
    "answer": 2,
    "explanation": "category stores the four values once plus small integer codes - typically a 50-100x memory reduction for low-cardinality text."
  }
]
```

## Summary

- A DataFrame is aligned Series sharing an index; alignment is by label, not position.
- Declare dtypes on load; coerce with `errors="coerce"` and count the failures.
- `category`, `string`, `Int32` and `float32` dramatically reduce memory.
- Parquet for machine-to-machine data; CSV only for humans.
- Assign with a single `.loc[mask, column] = value`.

## Next Step

Cleaning, missing values and transformation — turning a raw table into something you can
model or index.
