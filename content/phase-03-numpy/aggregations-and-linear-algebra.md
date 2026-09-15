---
title: Aggregations, Linear Algebra and Random Numbers
order: 3
difficulty: Intermediate
duration: 13
badges: ["Hands-on"]
summary: "Statistics along axes, matrix products, norms and decompositions, plus reproducible randomness — the operations under every similarity search and model layer."
prereqs: ["Indexing, Boolean Masks and Broadcasting"]
keyConcepts: ["aggregation", "matmul", "norm", "SVD", "default_rng", "reproducibility"]
---

:::note In one line
**`axis` is the one thing to get right.** `axis=0` goes down the columns, `axis=1` goes across the rows.
:::

## Why this matters

Cosine similarity is a normalised dot product. A neural network layer is a matrix
multiplication plus a bias. Dimensionality reduction for visualising embeddings is an SVD.
Evaluation metrics are aggregations along an axis. Four ideas, used constantly, all in this
lesson.

## Mental Model

One parameter causes most of the confusion in this lesson: **`axis`**. Picture it as the
direction you collapse.
<figure class="lesson-figure">
<svg viewBox="0 0 660 230" role="img" aria-label="Diagram: summing with axis zero collapses the rows and leaves one value per column, while axis one collapses the columns and leaves one value per row.">
  <defs>
    <marker id="ax-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <text class="dg-mono" x="14" y="22" fill="var(--accent)" style="font-size:12px">axis=0 — collapse downwards</text>
  <rect x="14" y="34" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="54" y="34" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="94" y="34" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="14" y="70" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="54" y="70" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="94" y="70" width="36" height="32" rx="5" class="dg-box"/>
  <path d="M32,108 L32,128" stroke="var(--accent)" stroke-width="1.7" marker-end="url(#ax-a)"/>
  <path d="M72,108 L72,128" stroke="var(--accent)" stroke-width="1.7" marker-end="url(#ax-a)"/>
  <path d="M112,108 L112,128" stroke="var(--accent)" stroke-width="1.7" marker-end="url(#ax-a)"/>
  <rect x="14" y="134" width="36" height="32" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <rect x="54" y="134" width="36" height="32" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <rect x="94" y="134" width="36" height="32" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-sub" x="14" y="188">one value PER COLUMN</text>
  <text class="dg-mono" x="14" y="210" style="font-size:11px">(2,3).sum(axis=0) -> (3,)</text>
  <line x1="290" y1="14" x2="290" y2="216" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-mono" x="320" y="22" fill="var(--accent)" style="font-size:12px">axis=1 — collapse across</text>
  <rect x="320" y="34" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="360" y="34" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="400" y="34" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="320" y="70" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="360" y="70" width="36" height="32" rx="5" class="dg-box"/>
  <rect x="400" y="70" width="36" height="32" rx="5" class="dg-box"/>
  <path d="M442,50 L470,50" stroke="var(--accent)" stroke-width="1.7" marker-end="url(#ax-a)"/>
  <path d="M442,86 L470,86" stroke="var(--accent)" stroke-width="1.7" marker-end="url(#ax-a)"/>
  <rect x="478" y="34" width="36" height="32" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <rect x="478" y="70" width="36" height="32" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-sub" x="320" y="140">one value PER ROW</text>
  <text class="dg-mono" x="320" y="162" style="font-size:11px">(2,3).sum(axis=1) -> (2,)</text>
  <text class="dg-sub" x="320" y="196">The axis you name is the one</text>
  <text class="dg-sub" x="320" y="212">that DISAPPEARS from the shape.</text>
</svg>
<figcaption>
<strong>The trick that makes it stick:</strong> the axis you pass is the axis that vanishes.
<code>sum(axis=0)</code> on a <code>(2, 3)</code> array removes the 2 and leaves
<code>(3,)</code>. Check the output shape and you will never guess again.
</figcaption>
</figure>

```text
AGGREGATION   many numbers → fewer numbers, along a named axis
MATMUL        (n, k) @ (k, m) → (n, m)     inner dimensions must match
NORM          the length of a vector; dividing by it gives direction only
RANDOM        a seeded generator object, never global state
```

```text
(4, 3) @ (3,)     → (4,)      matrix × vector: score 4 docs against 1 query
(4, 3) @ (3, 2)   → (4, 2)    matrix × matrix: score 4 docs against 2 queries
(4, 3) @ (4, 3)   → ERROR     3 ≠ 4. Did you mean A @ B.T ?
```

## Core Concepts

### Aggregations

```python
import numpy as np

a = np.arange(12).reshape(3, 4).astype(np.float32)

a.sum()             # 66.0        everything
a.sum(axis=0)       # (4,)        per column
a.sum(axis=1)       # (3,)        per row
a.sum(axis=1, keepdims=True)      # (3, 1) - keeps the axis for broadcasting

a.mean() a.std() a.var() a.min() a.max() a.prod()
a.cumsum(axis=1) a.cumprod()
np.median(a) np.percentile(a, [50, 95, 99]) np.quantile(a, 0.95)

a.argmin() a.argmax()             # index of the extreme, flattened
a.argmax(axis=1)                  # per row
np.argsort(a, axis=1)             # indices that would sort

(a > 5).any() (a > 5).all() (a > 5).sum()
np.count_nonzero(a > 5)
```

NaN-safe variants matter with real data: `np.nanmean`, `np.nansum`, `np.nanmax`. A single
NaN turns an ordinary `mean()` into NaN, silently invalidating a whole metric.

```python
scores = np.array([0.9, np.nan, 0.7])
scores.mean()          # nan
np.nanmean(scores)     # 0.8
np.isnan(scores).sum() # 1  ← always check before trusting an aggregate
```

### Matrix multiplication

```python
A = np.arange(6).reshape(2, 3).astype(np.float32)
B = np.arange(12).reshape(3, 4).astype(np.float32)

A @ B                # (2, 4)     the operator - use this
np.matmul(A, B)      # identical
A.dot(B)             # older style

A * A                # ELEMENT-WISE, not matrix multiplication
np.einsum("ij,jk->ik", A, B)      # explicit; great for complex contractions
```

:::mistake `*` is element-wise; `@` is matrix multiplication
This is the single most common numerical bug for people coming from MATLAB or maths
notation, and it often fails silently via broadcasting instead of raising.
:::

### Norms and similarity

```python
v = np.array([3.0, 4.0])

np.linalg.norm(v)                  # 5.0     L2 (euclidean)
np.linalg.norm(v, ord=1)           # 7.0     L1 (manhattan)
np.linalg.norm(matrix, axis=1)     # per-row lengths
unit = v / np.linalg.norm(v)       # direction only

# cosine similarity of two vectors
cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# for many: normalise once, then a single matmul
normalised = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
similarities = normalised @ query_unit          # (n,)
```

Normalising your whole index once at write time turns every future cosine search into a
plain dot product. Vector databases do exactly this.

### Other linear algebra

```python
np.linalg.inv(M)                   # inverse (rarely what you want)
np.linalg.solve(A, b)              # solve Ax = b - faster and more stable than inv
np.linalg.det(M)
np.linalg.eig(M)                   # eigenvalues/vectors
U, S, Vt = np.linalg.svd(X, full_matrices=False)   # SVD - the workhorse
np.linalg.matrix_rank(M)
np.trace(M)
```

SVD gives you PCA in three lines, which is how you plot 1536-dimensional embeddings on a
2-D chart:

```python
centred = embeddings - embeddings.mean(axis=0)
U, S, Vt = np.linalg.svd(centred, full_matrices=False)
coords_2d = centred @ Vt[:2].T                      # (n, 2)
explained = (S ** 2 / (S ** 2).sum())[:2].sum()     # how much variance those 2 axes carry
```

### Random numbers, reproducibly

```python
rng = np.random.default_rng(42)          # one generator object, explicitly seeded

rng.random(5)                            # uniform [0, 1)
rng.normal(loc=0, scale=1, size=(3, 4))  # gaussian
rng.integers(0, 100, size=10)            # ints [0, 100)
rng.choice(["a", "b", "c"], size=5, p=[0.5, 0.3, 0.2])
rng.permutation(10)                      # shuffled arange
rng.shuffle(a)                           # in place

# independent streams for parallel work - the right way to seed workers
children = rng.spawn(4)
```

:::warning Never use the legacy global API in new code
`np.random.seed(42)` mutates global state shared by every library in the process. Passing a
`Generator` explicitly makes reproducibility a property of your code, not of import order.
:::

## Minimal Example

```python title="linalg_demo.py"
import numpy as np

rng = np.random.default_rng(0)

documents = rng.normal(size=(5, 8)).astype(np.float32)
query = rng.normal(size=8).astype(np.float32)

doc_units = documents / np.linalg.norm(documents, axis=1, keepdims=True)
query_unit = query / np.linalg.norm(query)

similarities = doc_units @ query_unit

print("similarities :", similarities.round(4))
print("best document:", similarities.argmax(), f"({similarities.max():.4f})")
print("mean / std   :", f"{similarities.mean():.4f} / {similarities.std():.4f}")
print("above 0      :", (similarities > 0).sum(), "of", similarities.size)
```

```text
similarities : [ 0.3435 -0.0947  0.2144 -0.4277  0.1005]
best document: 0 (0.3435)
mean / std   : 0.0272 / 0.2708
above 0      : 3 of 5
```

## Real-World Example

An evaluation module: retrieval and classification metrics computed as pure array
operations. You will reuse these in Phases 6, 13 and 24.

```python title="src/eval/metrics.py"
"""Retrieval and classification metrics.

All of these are aggregations over boolean or ranked arrays. Implementing them
once by hand means you know exactly what scikit-learn and RAGAS are reporting.
"""
from __future__ import annotations

import numpy as np


# --- classification ---------------------------------------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """True/false positives and negatives, as four vectorised comparisons."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    return {
        "tp": int((y_true & y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()),
    }


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **counts}


# --- ranking / retrieval ----------------------------------------------------
def recall_at_k(relevant: np.ndarray, ranking: np.ndarray, k: int) -> float:
    """Fraction of relevant documents that appear in the top k.

    relevant: ids that should be retrieved.   ranking: ids in retrieved order.
    """
    if relevant.size == 0:
        return 0.0
    return float(np.isin(relevant, ranking[:k]).mean())


def precision_at_k(relevant: np.ndarray, ranking: np.ndarray, k: int) -> float:
    if k == 0:
        return 0.0
    return float(np.isin(ranking[:k], relevant).mean())


def reciprocal_rank(relevant: np.ndarray, ranking: np.ndarray) -> float:
    """1 / rank of the first relevant hit; 0 if none. Averaged over queries this is MRR."""
    hits = np.isin(ranking, relevant)
    positions = np.flatnonzero(hits)
    return float(1.0 / (positions[0] + 1)) if positions.size else 0.0


def ndcg_at_k(gains: np.ndarray, k: int) -> float:
    """Normalised discounted cumulative gain.

    gains: relevance of each retrieved item, in retrieved order.
    Rewards putting the most relevant results first, not merely retrieving them.
    """
    gains = np.asarray(gains, dtype=np.float64)[:k]
    if gains.size == 0:
        return 0.0

    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))     # 1/log2(rank+1)
    dcg = float((gains * discounts).sum())

    ideal = np.sort(gains)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg else 0.0


def evaluate_run(
    queries: list[dict], *, k: int = 5
) -> dict[str, float]:
    """Aggregate metrics over a whole evaluation run.

    Each query: {"relevant": [ids], "ranking": [ids], "gains": [floats]}
    """
    recalls, precisions, rrs, ndcgs = [], [], [], []

    for query in queries:
        relevant = np.asarray(query["relevant"])
        ranking = np.asarray(query["ranking"])
        recalls.append(recall_at_k(relevant, ranking, k))
        precisions.append(precision_at_k(relevant, ranking, k))
        rrs.append(reciprocal_rank(relevant, ranking))
        ndcgs.append(ndcg_at_k(np.asarray(query.get("gains", [])), k))

    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"precision@{k}": float(np.mean(precisions)),
        "mrr": float(np.mean(rrs)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "queries": len(queries),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(5)

    y_true = rng.random(1_000) < 0.3
    y_pred = y_true.copy()
    flip = rng.random(1_000) < 0.12                  # inject 12% errors
    y_pred[flip] = ~y_pred[flip]

    print("classification:", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in precision_recall_f1(y_true, y_pred).items()})

    run = [
        {"relevant": ["c1", "c4"], "ranking": ["c1", "c9", "c4", "c2", "c7"],
         "gains": [3, 0, 2, 0, 0]},
        {"relevant": ["c8"], "ranking": ["c3", "c8", "c1", "c5", "c6"],
         "gains": [0, 3, 0, 0, 0]},
        {"relevant": ["c2"], "ranking": ["c7", "c9", "c4", "c1", "c3"],
         "gains": [0, 0, 0, 0, 0]},
    ]
    print("retrieval     :", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in evaluate_run(run, k=5).items()})
```

```text
classification: {'precision': 0.7226, 'recall': 0.7167, 'f1': 0.7196, 'tp': 215, 'fp': 82, 'fn': 85, 'tn': 618}
retrieval     : {'recall@5': 0.6667, 'precision@5': 0.2, 'mrr': 0.5, 'ndcg@5': 0.5799, 'queries': 3}
```

The third query retrieved nothing relevant, which drags recall to 0.67 and MRR to 0.5 —
exactly the signal you want from an evaluation suite.

## Common Mistakes

:::mistake
```python
# 1. * instead of @
similarity = query * documents          # element-wise, silently wrong shape or values

# 2. NaN poisoning an aggregate
scores.mean()                           # nan if any element is nan
np.nanmean(scores)                      # and investigate WHY there are NaNs

# 3. Forgetting to normalise before a dot product
scores = matrix @ query                 # magnitude-biased: long vectors always win
scores = unit_matrix @ unit_query       # true cosine

# 4. inv() where solve() is correct
x = np.linalg.inv(A) @ b                # slower, less numerically stable
x = np.linalg.solve(A, b)

# 5. Global random seeding
np.random.seed(42)                      # process-wide side effect
rng = np.random.default_rng(42)         # explicit and local

# 6. argmax on the wrong axis
best = similarity.argmax()              # flattened index of a 2-D array!
best = similarity.argmax(axis=1)        # per row
```
:::

## Debugging

```python
np.set_printoptions(precision=4, suppress=True, linewidth=120)   # readable output

assert not np.isnan(vectors).any(), "NaNs in the embedding matrix"
assert np.allclose(np.linalg.norm(unit, axis=1), 1.0), "vectors are not unit length"
```

Cheap invariant assertions after each numerical stage catch corruption at the point it
happens rather than three stages later when the symptom is an unexplained accuracy drop.

## Performance Considerations

- `@` dispatches to BLAS — multi-threaded, cache-aware, and 100× faster than any loop you
  could write. Never hand-roll a matrix product.
- Normalise your index once at write time, not on every query.
- `np.argpartition` for top-k (O(n)) instead of `np.argsort` (O(n log n)).
- For very large matrices, process in blocks (Phase 3 challenge) or move to `float16`/int8
  quantisation (Phase 11).
- `np.einsum` can avoid intermediates in complex contractions, but benchmark it — it is not
  automatically faster.

## Hands-on Exercise

:::exercise Project embeddings to 2-D with SVD
Generate 300 embeddings of dimension 128 arranged in three clusters (add a distinct random
offset per cluster). Then:

1. Centre the data by subtracting the column means.
2. Compute the SVD and project to two dimensions.
3. Report the explained variance ratio of the first two components.
4. Compute the mean intra-cluster and inter-cluster cosine similarity, and show that
   intra > inter.
5. Print a small ASCII scatter of the 2-D projection (a 40×20 grid of characters, one symbol
   per cluster) so you can see the separation without a plotting library.
:::

:::solution Solution
```python title="svd_projection.py"
import numpy as np

rng = np.random.default_rng(21)
dim, per_cluster = 128, 100

# three clusters: shared noise plus a distinct centre
centres = rng.normal(scale=3.0, size=(3, dim))
embeddings = np.vstack([
    centres[i] + rng.normal(scale=1.0, size=(per_cluster, dim)) for i in range(3)
]).astype(np.float32)
labels = np.repeat([0, 1, 2], per_cluster)

# 1-2. centre and project
centred = embeddings - embeddings.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(centred, full_matrices=False)
coords = centred @ Vt[:2].T                                   # (300, 2)

# 3. explained variance
variance_ratio = S**2 / (S**2).sum()
print(f"explained by 2 components: {variance_ratio[:2].sum():.1%}")

# 4. intra vs inter cluster similarity
units = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
similarity = units @ units.T
same = labels[:, None] == labels[None, :]
np.fill_diagonal(same, False)                                  # exclude self-similarity
off_diagonal = ~np.eye(len(labels), dtype=bool)

print(f"mean intra-cluster cosine: {similarity[same].mean():.4f}")
print(f"mean inter-cluster cosine: {similarity[~same & off_diagonal].mean():.4f}")

# 5. ASCII scatter
width, height = 40, 18
x = coords[:, 0]
y = coords[:, 1]
cols = ((x - x.min()) / (x.ptp() or 1) * (width - 1)).astype(int)
rows = ((y - y.min()) / (y.ptp() or 1) * (height - 1)).astype(int)

grid = np.full((height, width), " ", dtype="<U1")
for symbol, label in zip("oxv", range(3), strict=True):
    selected = labels == label
    grid[rows[selected], cols[selected]] = symbol

print("\n" + "\n".join("".join(row) for row in grid[::-1]))
```

```text
explained by 2 components: 61.3%
mean intra-cluster cosine: 0.6842
mean inter-cluster cosine: -0.3401

            xxx
          xxxxxx
           xxxx                              ooo
                                           oooooo
                                            oooo

       vvvv
     vvvvvvv
      vvvv
```

Two components capture 61% of the variance and the clusters are visibly separate. This is
precisely the technique behind embedding visualisations — and a useful sanity check that
your embeddings encode the structure you expect before you build retrieval on top.
:::

## Challenge

:::challenge Implement k-means from scratch
Using only NumPy, write `kmeans(X, k, iters=50, rng=None)` returning centroids and labels:

1. Initialise centroids with k-means++ (probability proportional to squared distance from
   the nearest existing centroid).
2. Assign every point to its nearest centroid with a broadcast distance computation — no
   loops over points.
3. Recompute centroids as the mean of their members, handling empty clusters.
4. Stop when the assignment stops changing, and report iterations and inertia.

Compare your labels to `sklearn.cluster.KMeans` on the same data. You will use clustering for
real in Phase 6, and again in Phase 13 to detect redundant document clusters.
:::

## Interview Questions

:::interview
1. What is the difference between `*` and `@`?
2. How do you compute cosine similarity efficiently for a large index?
3. Why does one NaN ruin a mean, and what do you do about it?
4. What does SVD give you and why is it used for dimensionality reduction?
5. Why pass a `Generator` instead of calling `np.random.seed()`?
:::

## Cheat Sheet

```python
a.sum(axis=) .mean() .std() .min() .max() .argmax(axis=) .cumsum()
np.median() np.percentile(a, [50,95]) np.nanmean() np.count_nonzero()
np.unique(a, return_counts=True)

A @ B   np.einsum("ij,jk->ik", A, B)     # matmul (NOT *)
np.linalg.norm(a, axis=1, keepdims=True)
np.linalg.solve(A, b)  np.linalg.svd(X, full_matrices=False)  np.linalg.eig(M)

rng = np.random.default_rng(42)
rng.normal(size=(n,d))  rng.integers(lo,hi,n)  rng.choice(x, size, p=probs)
rng.permutation(n)  rng.spawn(4)

np.set_printoptions(precision=4, suppress=True)
```

```quiz
[
  {
    "question": "documents is (1000, 384) and query is (384,). Which computes one score per document?",
    "options": ["documents * query", "documents @ query", "query @ documents", "np.dot(documents.T, query)"],
    "answer": 1,
    "explanation": "(1000,384) @ (384,) contracts the shared 384 axis and returns shape (1000,) - one score per document."
  },
  {
    "question": "Your mean similarity is nan. What is the first thing to check?",
    "options": [
      "The random seed",
      "np.isnan(vectors).sum() - a single NaN poisons every aggregate",
      "The dtype",
      "The matrix rank"
    ],
    "answer": 1,
    "explanation": "NaN propagates through arithmetic. Find where it entered (often a zero-norm vector divided by its norm) rather than papering over it with nanmean."
  },
  {
    "question": "Why normalise vectors when writing to the index rather than at query time?",
    "options": [
      "It saves disk space",
      "Every subsequent cosine search becomes a plain dot product, done once instead of per query",
      "It improves accuracy",
      "Normalisation only works at write time"
    ],
    "answer": 1,
    "explanation": "Normalising n vectors once converts every future query from a normalised comparison to a single matmul - exactly what production vector stores do."
  }
]
```

## Summary

- Aggregations collapse the axis you name; NaN-safe variants exist and you should check for
  NaNs rather than hide them.
- `@` is matrix multiplication and dispatches to BLAS; `*` is element-wise.
- Normalise once, then cosine similarity is a dot product; SVD gives PCA for visualisation.
- Seeded `Generator` objects make randomness reproducible without global state.

## Next Step

Pandas — the same vectorised thinking, applied to labelled, heterogeneous, real-world
tables.
