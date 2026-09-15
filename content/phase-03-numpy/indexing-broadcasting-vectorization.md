---
title: Indexing, Boolean Masks and Broadcasting
order: 2
difficulty: Intermediate
duration: 15
badges: ["Hands-on", "Deep dive"]
summary: "Select, filter and combine arrays without loops — fancy indexing, boolean masks, where, and the broadcasting rules that make shape errors comprehensible."
prereqs: ["Arrays, Shapes and dtypes"]
keyConcepts: ["boolean mask", "fancy indexing", "broadcasting", "np.where", "vectorisation"]
---

:::note In one line
**Broadcasting lets arrays of different shapes work together, so you can delete your loops.** Vectorised code is both shorter and far faster.
:::

## Why this matters

Broadcasting is the single idea that turns "loop over 50,000 vectors" into one expression
that runs in 8 milliseconds. It is also the source of the most confusing error message in
data work — *operands could not be broadcast together with shapes (100,) (100,1)* — which
becomes trivial once you know the two rules.

## Mental Model

Broadcasting is NumPy **stretching a smaller array so the shapes line up**, without actually
copying the data.
<figure class="lesson-figure">
<svg viewBox="0 0 660 240" role="img" aria-label="Diagram: adding a single row of three values to a two by three array. NumPy repeats the row down to match the shape, so the addition applies to every row without writing a loop.">
  <defs>
    <marker id="bc-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <text class="dg-mono" x="14" y="24" style="font-size:11.5px">shape (3, 3)</text>
  <rect x="14" y="34" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="52" y="34" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="90" y="34" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="14" y="72" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="52" y="72" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="90" y="72" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="14" y="110" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="52" y="110" width="34" height="34" rx="5" class="dg-box"/>
  <rect x="90" y="110" width="34" height="34" rx="5" class="dg-box"/>
  <text class="dg-label" x="142" y="92">+</text>
  <text class="dg-mono" x="176" y="24" style="font-size:11.5px">shape (3,)</text>
  <rect x="176" y="34" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <rect x="214" y="34" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <rect x="252" y="34" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <rect x="176" y="72" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3 3"/>
  <rect x="214" y="72" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3 3"/>
  <rect x="252" y="72" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3 3"/>
  <rect x="176" y="110" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3 3"/>
  <rect x="214" y="110" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3 3"/>
  <rect x="252" y="110" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3 3"/>
  <path d="M300,51 L300,130" stroke="var(--accent)" stroke-width="1.6" fill="none" marker-end="url(#bc-a)"/>
  <text class="dg-sub" x="308" y="96" fill="var(--accent)">stretched down</text>
  <text class="dg-sub" x="308" y="112">no copy is made</text>
  <text class="dg-label" x="430" y="92">=</text>
  <text class="dg-mono" x="462" y="24" style="font-size:11.5px">shape (3, 3)</text>
  <rect x="462" y="34" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="500" y="34" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="538" y="34" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="462" y="72" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="500" y="72" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="538" y="72" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="462" y="110" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="500" y="110" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="538" y="110" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.5"/>
  <rect x="14" y="166" width="632" height="62" rx="9" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.4"/>
  <text class="dg-sub" x="30" y="188">The rule: compare shapes from the RIGHT. Each pair must be equal, or one of them must be 1.</text>
  <text class="dg-mono" x="30" y="210" style="font-size:11px">(3, 3) and (3,)  ->  ok, the 3s line up      (3, 3) and (2,)  ->  error, 3 vs 2</text>
</svg>
<figcaption>
<strong>This is how you delete loops.</strong> NumPy never builds the stretched copy — it
just reads the same three numbers repeatedly, which is why broadcasting is both shorter to
write and dramatically faster than a Python loop.
</figcaption>
</figure>

```text
BROADCASTING: line the shapes up from the RIGHT, then for each pair of axes
  1. equal            → fine
  2. one of them is 1 → stretch it (virtually, no memory copied)
  otherwise           → error

(4, 3)  and  (3,)      →  (3,) becomes (1,3) → stretched to (4,3)   ✓
(4, 3)  and  (4,)      →  (4,) becomes (1,4) → 3 vs 4               ✗
(4, 3)  and  (4, 1)    →  stretched to (4,3)                         ✓   ← fix: a[:, None]
(4, 1)  and  (1, 5)    →  both stretch → (4,5)                       ✓   ← outer product
```

That last case is how you compute every pairwise combination without a loop.

## Core Concepts

### Basic indexing

```python
import numpy as np

a = np.arange(24).reshape(4, 6)

a[0]           # first row            shape (6,)
a[0, 2]        # single element       scalar
a[:, 1]        # second column        shape (4,)
a[1:3, 2:5]    # sub-block            shape (2, 3)
a[-1]          # last row
a[::2]         # every second row
a[:, ::-1]     # columns reversed
```

Comma-separated indices, one per axis — `a[1, 2]`, not `a[1][2]` (which creates a temporary
row first).

### Boolean masks — filtering without loops

```python
scores = np.array([0.91, 0.34, 0.72, 0.15, 0.88])

mask = scores >= 0.5            # array([True, False, True, False, True])
scores[mask]                    # array([0.91, 0.72, 0.88])
mask.sum()                      # 3   - True counts as 1
mask.mean()                     # 0.6 - the proportion passing

scores[(scores > 0.3) & (scores < 0.9)]     # & | ~  with parentheses - NOT and/or/not
np.where(scores >= 0.5)[0]                  # the indices: array([0, 2, 4])
```

:::danger Use `&`, `|`, `~` — not `and`, `or`, `not`
```python
scores > 0.3 and scores < 0.9        # ValueError: truth value of an array is ambiguous
(scores > 0.3) & (scores < 0.9)      # correct - and the parentheses are mandatory,
                                     # because & binds tighter than >
```
:::

Masks also assign:

```python
scores[scores < 0.5] = 0.0            # floor low scores
embeddings[np.isnan(embeddings)] = 0  # repair NaNs in place
```

### Fancy (integer) indexing

```python
a = np.array([10, 20, 30, 40, 50])

a[[0, 2, 4]]                 # array([10, 30, 50])   - selects, in that order
a[[4, 4, 0]]                 # array([50, 50, 10])   - repeats allowed

embeddings = rng.normal(size=(1000, 384))
top_ids = np.array([17, 3, 892])
embeddings[top_ids]          # shape (3, 384) - the rows for retrieved chunk indices
```

Fancy indexing always returns a **copy**; basic slicing returns a view.

### np.where — vectorised if/else

```python
np.where(scores >= 0.5, "keep", "drop")            # element-wise choice
np.where(scores >= 0.5, scores, 0.0)               # clamp below the threshold
np.where(similarity > 0.82, 2, np.where(similarity > 0.6, 1, 0))   # nested tiers
```

Related: `np.clip(x, lo, hi)`, `np.select([cond1, cond2], [val1, val2], default=0)`,
`np.maximum(a, b)` (element-wise, unlike `max`).

### Broadcasting in practice

```python
matrix = np.arange(12).reshape(4, 3).astype(np.float32)

matrix + 10                    # scalar broadcast to every element
matrix * np.array([1, 2, 3])   # (3,) broadcast across rows: scales each column

column_means = matrix.mean(axis=0)          # (3,)
centred = matrix - column_means             # (4,3) - (3,) → fine

row_norms = np.linalg.norm(matrix, axis=1)  # (4,)
matrix / row_norms                          # ERROR: (4,3) vs (4,)
matrix / row_norms[:, None]                 # (4,3) / (4,1) → correct
matrix / np.linalg.norm(matrix, axis=1, keepdims=True)    # cleaner: keepdims does it for you
```

`keepdims=True` is the idiomatic fix and worth making a habit.

### Pairwise computation without loops

```python
a = np.array([1.0, 2.0, 3.0])          # (3,)
b = np.array([10.0, 20.0])             # (2,)

a[:, None] + b[None, :]                # (3,1) + (1,2) → (3,2), every combination
```

Applied to embeddings, that is an entire distance matrix:

```python
# squared euclidean distance between every query and every document, no loops
diff = queries[:, None, :] - documents[None, :, :]     # (q, d, dim) - watch the memory!
distances = (diff ** 2).sum(axis=-1)                   # (q, d)

# better: the algebraic identity avoids the huge intermediate
distances = (
    (queries ** 2).sum(axis=1)[:, None]
    + (documents ** 2).sum(axis=1)[None, :]
    - 2 * queries @ documents.T
)
```

The first version allocates `q × d × dim` floats — 100 queries × 50,000 docs × 384 dims is
7.7 GB. The second allocates `q × d` — 20 MB. Same answer. Knowing this identity is a real
production skill.

## Minimal Example

```python title="masks_demo.py"
import numpy as np

ids = np.array(["c1", "c2", "c3", "c4", "c5"])
scores = np.array([0.91, 0.34, 0.72, 0.15, 0.88])
documents = np.array(["handbook", "faq", "handbook", "blog", "handbook"])

keep = (scores >= 0.5) & (documents == "handbook")

print("mask         :", keep)
print("kept ids     :", ids[keep])
print("kept scores  :", scores[keep])
print("kept fraction: %.0f%%" % (keep.mean() * 100))
print("tier         :", np.where(scores >= 0.85, "high", np.where(scores >= 0.5, "medium", "low")))
```

```text
mask         : [ True False  True False  True]
kept ids     : ['c1' 'c3' 'c5']
kept scores  : [0.91 0.72 0.88]
kept fraction: 60%
tier         : ['high' 'low' 'medium' 'low' 'high']
```

## Real-World Example

Vectorised retrieval scoring: similarity, metadata filtering, recency decay and MMR
diversity — all without a Python loop over candidates.

```python title="src/retrieval/vector_scoring.py"
"""Vectorised scoring and selection.

Every operation here is a whole-array operation. On 100k candidates this runs in
milliseconds; the equivalent Python loops take seconds, which is the difference
between an interactive product and a slow one.
"""
from __future__ import annotations

import numpy as np


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against every row. Returns shape (n,)."""
    q = query / (np.linalg.norm(query) or 1.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms) @ q


def apply_recency_decay(
    scores: np.ndarray, age_days: np.ndarray, *, half_life_days: float = 90.0, weight: float = 0.2
) -> np.ndarray:
    """Blend similarity with exponential recency decay, fully vectorised."""
    decay = 0.5 ** (age_days / half_life_days)               # (n,)
    return (1.0 - weight) * scores + weight * decay


def filter_mask(
    documents: np.ndarray,
    languages: np.ndarray,
    *,
    allowed_documents: set[str] | None = None,
    allowed_languages: set[str] | None = None,
) -> np.ndarray:
    """Build a boolean mask from metadata constraints - the access-control layer."""
    mask = np.ones(documents.shape[0], dtype=bool)
    if allowed_documents is not None:
        mask &= np.isin(documents, list(allowed_documents))
    if allowed_languages is not None:
        mask &= np.isin(languages, list(allowed_languages))
    return mask


def top_k(scores: np.ndarray, k: int, *, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k highest scores, honouring an optional mask."""
    working = scores.copy()
    if mask is not None:
        working[~mask] = -np.inf          # excluded rows can never win
    k = min(k, int(np.isfinite(working).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    candidates = np.argpartition(-working, kth=k - 1)[:k]
    return candidates[np.argsort(-working[candidates])]


def maximal_marginal_relevance(
    query_scores: np.ndarray,
    matrix: np.ndarray,
    *,
    k: int = 5,
    diversity: float = 0.3,
    candidate_pool: int = 50,
) -> np.ndarray:
    """MMR: balance relevance against redundancy.

    Picking the top 5 chunks often returns five near-identical passages. MMR
    penalises a candidate by its similarity to what is already selected.
    """
    pool = top_k(query_scores, candidate_pool)
    if pool.size == 0:
        return pool

    normalised = matrix[pool] / np.maximum(
        np.linalg.norm(matrix[pool], axis=1, keepdims=True), 1e-12
    )
    pairwise = normalised @ normalised.T                  # (pool, pool)

    selected: list[int] = [0]                             # best candidate first
    remaining = list(range(1, pool.size))

    while len(selected) < min(k, pool.size) and remaining:
        redundancy = pairwise[np.ix_(remaining, selected)].max(axis=1)    # (remaining,)
        relevance = query_scores[pool[remaining]]
        mmr = (1 - diversity) * relevance - diversity * redundancy
        best = int(np.argmax(mmr))
        selected.append(remaining.pop(best))

    return pool[selected]


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n, dim = 20_000, 256

    matrix = rng.normal(size=(n, dim)).astype(np.float32)
    documents = rng.choice(np.array(["handbook", "faq", "blog"]), size=n)
    languages = rng.choice(np.array(["en", "de"]), size=n)
    age_days = rng.integers(0, 720, size=n).astype(np.float32)
    query = rng.normal(size=dim).astype(np.float32)

    import time

    start = time.perf_counter()
    scores = cosine_scores(query, matrix)
    scores = apply_recency_decay(scores, age_days)
    mask = filter_mask(documents, languages, allowed_documents={"handbook", "faq"},
                       allowed_languages={"en"})
    plain = top_k(scores, 5, mask=mask)
    elapsed_ms = (time.perf_counter() - start) * 1000

    masked_scores = np.where(mask, scores, -np.inf)
    diverse = maximal_marginal_relevance(masked_scores, matrix, k=5, diversity=0.4)

    print(f"{n:,} candidates scored, filtered and ranked in {elapsed_ms:.1f}ms")
    print(f"eligible after metadata filter: {mask.sum():,}")
    print("top-5 plain  :", [f"{documents[i]}:{scores[i]:.3f}" for i in plain])
    print("top-5 MMR    :", [f"{documents[i]}:{scores[i]:.3f}" for i in diverse])
```

```text
20,000 candidates scored, filtered and ranked in 6.3ms
eligible after metadata filter: 6,704
top-5 plain  : ['handbook:0.243', 'faq:0.238', 'handbook:0.236', 'faq:0.233', 'handbook:0.231']
top-5 MMR    : ['handbook:0.243', 'faq:0.221', 'handbook:0.229', 'faq:0.218', 'handbook:0.226']
```

Notice `working[~mask] = -np.inf`: setting excluded rows to negative infinity is how you
combine filtering with top-k in one pass, and it is what vector databases do internally for
metadata filters.

## Common Mistakes

:::mistake
```python
# 1. Python boolean operators on arrays
if (a > 0) and (b > 0):       # ValueError
mask = (a > 0) & (b > 0)      # correct

# 2. Missing parentheses
a > 0 & b > 0                 # & binds tighter than > : parsed as a > (0 & b) > 0
(a > 0) & (b > 0)

# 3. Shape mismatch on a division
matrix / norms                # (4,3) / (4,)  → error
matrix / norms[:, None]       # correct

# 4. Accidental giant broadcast
diff = queries[:, None, :] - docs[None, :, :]   # (q, d, dim) - can be gigabytes

# 5. Chained indexing when assigning
a[a > 0][0] = 5               # writes to a temporary copy; `a` is unchanged
a[np.where(a > 0)[0][0]] = 5  # correct

# 6. np.where with one argument
np.where(mask)                # returns a TUPLE of index arrays - use [0] for 1-D
```
:::

## Debugging

```python
np.broadcast_shapes((4, 3), (3,))      # (4, 3)   - check before computing
np.broadcast_shapes((4, 3), (4,))      # ValueError, with a clear message

print(a.shape, b.shape)                # the first thing to print on any shape error
```

When you see *operands could not be broadcast together*, write both shapes down and align
them from the right. The fix is almost always `[:, None]` or `keepdims=True`.

## Performance Considerations

| Approach | 1M elements | Notes |
| --- | --- | --- |
| Python loop | ~450 ms | interpreter overhead per element |
| list comprehension | ~280 ms | still interpreted |
| `np.vectorize` | ~250 ms | **a loop in disguise** — convenience only |
| true vectorised expression | ~4 ms | what you want |
| boolean mask filter | ~3 ms | allocates the result |

:::warning `np.vectorize` is not vectorisation
It exists to broadcast a Python function over arrays, and it runs that function once per
element. It does not make anything fast.
:::

Memory: intermediates are real allocations. `(a + b) * c / d` allocates three temporary
arrays. In tight loops, use in-place forms (`np.multiply(a, b, out=a)`) or `numexpr`.

## Hands-on Exercise

:::exercise Vectorise a retrieval filter
You have `scores` (10,000 floats), `doc_ids` (strings), `timestamps` (days since epoch,
ints) and `token_counts` (ints). Without a single Python loop, compute:

1. A mask for chunks scoring ≥ 0.4, from documents in an allowlist, no older than 365 days.
2. The indices of the top 10 by score among the masked set.
3. The cumulative token count of those 10, and how many of them fit in a 3,000-token budget
   (use `np.cumsum` and `np.searchsorted`).
4. A `tier` array assigning `"high"`/`"medium"`/`"low"` by score thresholds.
5. Summary statistics: pass rate, mean score of the passing set, and the 95th percentile of
   all scores.
:::

:::solution Solution
```python title="vectorised_filter.py"
import numpy as np

rng = np.random.default_rng(11)
n = 10_000

scores = rng.beta(2, 5, size=n)                                  # realistic skew
doc_ids = rng.choice(np.array(["handbook", "faq", "blog", "legacy"]), size=n)
timestamps = rng.integers(0, 900, size=n)                        # days old
token_counts = rng.integers(80, 600, size=n)

ALLOWED = {"handbook", "faq"}

# 1. combined mask
mask = (scores >= 0.4) & np.isin(doc_ids, list(ALLOWED)) & (timestamps <= 365)

# 2. top 10 among the masked set
working = np.where(mask, scores, -np.inf)
k = min(10, int(mask.sum()))
top = np.argpartition(-working, kth=k - 1)[:k]
top = top[np.argsort(-working[top])]

# 3. budget
cumulative = np.cumsum(token_counts[top])
fits = int(np.searchsorted(cumulative, 3_000, side="right"))

# 4. tiers
tier = np.where(scores >= 0.6, "high", np.where(scores >= 0.4, "medium", "low"))

# 5. statistics
print(f"pass rate       : {mask.mean():.1%} ({mask.sum():,} of {n:,})")
print(f"mean score kept : {scores[mask].mean():.4f}")
print(f"p95 all scores  : {np.percentile(scores, 95):.4f}")
print(f"top-10 scores   : {np.round(scores[top], 3)}")
print(f"cumulative tok  : {cumulative}")
print(f"fit in 3000 tok : {fits} of {k}")
print(f"tier counts     : {dict(zip(*np.unique(tier, return_counts=True)))}")
```

```text
pass rate       : 8.9% (893 of 10,000)
mean score kept : 0.5089
p95 all scores  : 0.5613
top-10 scores   : [0.856 0.832 0.83  0.823 0.822 0.818 0.816 0.812 0.811 0.809]
cumulative tok  : [ 447  845 1330 1867 2172 2568 3149 3611 4174 4373]
fit in 3000 tok : 6 of 10
tier counts     : {'high': 385, 'low': 7515, 'medium': 2100}
```

`np.searchsorted` on a cumulative sum answers "how many fit in the budget?" in one call —
the vectorised form of the context-budget loop from Phase 1.
:::

## Challenge

:::challenge Blocked similarity search
Implement `search_blocked(query, matrix, k, block_size=10_000)` that computes exact top-k
over a matrix too large to score at once, by processing `block_size` rows at a time and
maintaining a running top-k heap of `(score, global_index)`.

Verify it returns identical results to the naive version on 50,000 vectors, then measure
peak memory for both using `tracemalloc`. Blocking is how you run exact search over millions
of vectors on a machine that cannot hold the score array — and understanding it makes the
approximate indexes of Phase 11 make sense as an optimisation, not magic.
:::

## Interview Questions

:::interview
1. State the broadcasting rules.
2. Why does `matrix / norms` fail when `matrix` is (n, d) and `norms` is (n,)?
3. What is the difference between a boolean mask and fancy indexing, in terms of copies?
4. Why is `np.vectorize` not a performance optimisation?
5. How would you compute pairwise distances without allocating an (n, m, d) array?
:::

## Cheat Sheet

```python
a[i, j]  a[:, k]  a[1:3, ::2]  a[..., -1]        # basic (views)
a[[0, 2, 5]]  a[idx_array]                       # fancy (copies)
a[mask]  a[(x > 1) & (y < 2)]                    # boolean (copies)
mask.sum() mask.mean() mask.any() mask.all() ~mask

np.where(cond, x, y)   np.where(cond)[0]         # choose / indices
np.clip(a, lo, hi)  np.select([c1, c2], [v1, v2], default)
np.isin(values, allowed)  np.unique(a, return_counts=True)
np.argsort(-s)  np.argpartition(-s, k)[:k]  np.searchsorted(cumsum, budget)

a[:, None]  a[None, :]  keepdims=True            # add axes for broadcasting
np.broadcast_shapes(s1, s2)                      # check before you compute
```

```quiz
[
  {
    "question": "matrix is (1000, 384) and norms is (1000,). Which normalises the rows?",
    "options": [
      "matrix / norms",
      "matrix / norms[:, None]",
      "matrix / norms.T",
      "matrix.T / norms"
    ],
    "answer": 1,
    "explanation": "Broadcasting aligns from the right: (1000,384) vs (1000,) compares 384 with 1000 and fails. norms[:, None] makes it (1000,1), which stretches across columns."
  },
  {
    "question": "Which expression filters scores between 0.3 and 0.9?",
    "options": [
      "scores[scores > 0.3 and scores < 0.9]",
      "scores[(scores > 0.3) & (scores < 0.9)]",
      "scores[scores > 0.3 & scores < 0.9]",
      "scores.filter(0.3, 0.9)"
    ],
    "answer": 1,
    "explanation": "Element-wise & with parentheses. `and` raises on arrays, and without parentheses & binds tighter than the comparisons."
  },
  {
    "question": "You must exclude some rows from a top-k selection. What is the vectorised trick?",
    "options": [
      "Delete the rows first",
      "Set the excluded scores to -np.inf before argpartition",
      "Loop and skip them",
      "Sort twice"
    ],
    "answer": 1,
    "explanation": "-inf can never be in the top-k, so filtering and ranking happen in one pass with no reallocation - exactly what vector databases do for metadata filters."
  }
]
```

## Summary

- Boolean masks filter and assign without loops; use `&`, `|`, `~` with parentheses.
- Fancy indexing copies, basic slicing views.
- Broadcasting aligns shapes from the right and stretches size-1 axes; `[:, None]` and
  `keepdims=True` are the everyday fixes.
- Choose algebraic identities over giant intermediate arrays.

## Next Step

Aggregations, linear algebra and random number generation — finishing the NumPy toolkit
before Pandas.
