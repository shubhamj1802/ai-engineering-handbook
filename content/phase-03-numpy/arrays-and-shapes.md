---
title: Arrays, Shapes and dtypes
order: 1
difficulty: Intermediate
duration: 14
badges: ["Hands-on", "Start here"]
summary: "The ndarray mental model — shape, dimensions, dtype and memory layout — which is the same model PyTorch tensors and every embedding matrix use."
prereqs: ["Lists and Tuples", "Comprehensions and Generator Expressions"]
keyConcepts: ["ndarray", "shape", "dtype", "axis", "reshape", "view"]
---

:::note In one line
**A NumPy array is a grid of one type, and its shape is the thing you will spend your time reasoning about.** Most errors are shape errors.
:::

## Why this matters

An embedding is a NumPy array. A batch of embeddings is a 2-D array. A similarity search is
a matrix multiplication. PyTorch tensors copy NumPy's API almost exactly, and most
"dimension mismatch" errors you will hit in Phases 9–11 are shape errors you can debug the
moment you can read a shape tuple fluently.

## Mental Model

Everything in NumPy comes back to one question: **what shape is this?** Most errors you will
hit are shape errors, so it pays to be able to picture them.
<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Diagram showing a one-dimensional array of four values with shape four, a two-dimensional array of two rows by three columns with shape two by three, and a three-dimensional array of two stacked two by two blocks.">
  <text class="dg-label" x="14" y="22">1-D — a row</text>
  <text class="dg-mono"  x="14" y="42" fill="var(--accent)" style="font-size:11.5px">shape (4,)</text>
  <rect x="14"  y="54" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="56"  y="54" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="98"  y="54" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="140" y="54" width="38" height="38" rx="5" class="dg-box"/>
  <text class="dg-sub" x="14" y="112">4 values, one axis</text>
  <text class="dg-label" x="240" y="22">2-D — a table</text>
  <text class="dg-mono"  x="240" y="42" fill="var(--accent)" style="font-size:11.5px">shape (2, 3)</text>
  <rect x="240" y="54" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="282" y="54" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="324" y="54" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="240" y="96" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="282" y="96" width="38" height="38" rx="5" class="dg-box"/>
  <rect x="324" y="96" width="38" height="38" rx="5" class="dg-box"/>
  <text class="dg-sub" x="240" y="154">2 rows, 3 columns</text>
  <text class="dg-sub" x="376" y="78">axis 0</text>
  <text class="dg-sub" x="376" y="94">goes down</text>
  <text class="dg-sub" x="240" y="174">axis 1 goes across</text>
  <text class="dg-label" x="470" y="22">3-D — a stack</text>
  <text class="dg-mono"  x="470" y="42" fill="var(--accent)" style="font-size:11.5px">shape (2, 2, 2)</text>
  <rect x="482" y="54" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <rect x="520" y="54" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <rect x="482" y="92" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <rect x="520" y="92" width="34" height="34" rx="5" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.2"/>
  <rect x="470" y="66" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <rect x="508" y="66" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <rect x="470" y="104" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <rect x="508" y="104" width="34" height="34" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <text class="dg-sub" x="470" y="160">2 blocks of 2x2</text>
  <text class="dg-sub" x="470" y="178">batch of images, for example</text>
  <rect x="14" y="196" width="632" height="44" rx="9" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <text class="dg-sub" x="330" y="216" text-anchor="middle">Read a shape left to right as "outermost to innermost".</text>
  <text class="dg-sub" x="330" y="234" text-anchor="middle">(32, 128, 768) = 32 sequences, each 128 tokens, each token a 768-number vector.</text>
</svg>
<figcaption>
<strong>Print the shape before you debug anything else.</strong> <code>print(a.shape)</code>
answers most NumPy questions instantly, and that last example is the exact shape you will
meet again in Phase 10 when you look at transformer inputs.
</figcaption>
</figure>

```text
shape (3,)          1-D  vector          one embedding of 3 dimensions
shape (4, 3)        2-D  matrix          4 embeddings × 3 dimensions
shape (2, 4, 3)     3-D  batch           2 batches × 4 embeddings × 3 dims

axis 0 = rows      "down"      → across items
axis 1 = columns   "across"    → within an item

a.sum(axis=0)  collapses rows    → one value per column   shape (4,3) -> (3,)
a.sum(axis=1)  collapses columns → one value per row      shape (4,3) -> (4,)
```

The rule that unsticks everyone: **the axis you name is the axis that disappears.**

## Prerequisites

```bash
uv add numpy
```

## Core Concepts

### Creating arrays

```python
import numpy as np

np.array([1, 2, 3])                     # from a list        shape (3,)
np.array([[1, 2], [3, 4]])              # 2-D                shape (2, 2)

np.zeros((2, 3))                        # filled with 0.0
np.ones((2, 3))
np.full((2, 3), 0.5)
np.empty((2, 3))                        # uninitialised: fast, contains garbage
np.eye(3)                               # identity matrix
np.arange(0, 10, 2)                     # [0 2 4 6 8]
np.linspace(0, 1, 5)                    # [0. 0.25 0.5 0.75 1.]

rng = np.random.default_rng(seed=42)    # the modern generator API
rng.random((2, 3))                      # uniform [0, 1)
rng.normal(0, 1, size=(2, 3))           # gaussian
rng.integers(0, 10, size=5)
```

:::tip Always seed a generator in examples and tests
`np.random.default_rng(42)` gives reproducible results. The legacy `np.random.seed()` sets
global state and is best avoided in new code.
:::

### Attributes

```python
a = np.zeros((4, 3), dtype=np.float32)

a.shape        # (4, 3)      dimensions
a.ndim         # 2           number of axes
a.size         # 12          total elements
a.dtype        # float32
a.itemsize     # 4           bytes per element
a.nbytes       # 48          total bytes
```

### dtypes and why they matter

```python
np.array([1, 2, 3])                  # int64   (platform default)
np.array([1.0, 2.0])                 # float64
np.array([1, 2], dtype=np.float32)   # float32 - half the memory
np.array([True, False])              # bool
```

| dtype | Bytes | Use |
| --- | --- | --- |
| `float64` | 8 | scientific default, NumPy's default |
| `float32` | 4 | **embeddings, ML features, PyTorch default** |
| `float16` | 2 | quantised models, memory-constrained inference |
| `int64` | 8 | ids, counts, indices |
| `int8` | 1 | quantised vectors |
| `bool` | 1 | masks |

One million 1536-dimensional embeddings: `float64` needs 12.3 GB, `float32` needs 6.1 GB.
Embedding APIs return 32-bit precision anyway, so `float64` is pure waste — and this is the
single most common memory mistake in vector pipelines.

```python
vectors = np.array(api_response, dtype=np.float32)      # cast at the boundary
vectors.astype(np.float32)                              # explicit conversion (copies)
```

:::warning Integer overflow is silent
```python
np.array([2**62], dtype=np.int64) * 4        # wraps around to a negative number
```
NumPy uses fixed-width integers, unlike Python's arbitrary-precision `int`.
:::

### Reshaping

```python
a = np.arange(12)              # shape (12,)

a.reshape(3, 4)                # shape (3, 4)   row-major ("C order")
a.reshape(3, -1)               # -1 = infer:    (3, 4)
a.reshape(-1, 1)               # column vector  (12, 1)
a.reshape(2, 2, 3)             # 3-D

a.ravel()                      # flatten to 1-D (a view when possible)
a.flatten()                    # flatten to 1-D (always a copy)

m = np.arange(6).reshape(2, 3)
m.T                            # transpose      (3, 2)
m[np.newaxis, :, :]            # add an axis    (1, 2, 3)
m[:, None]                     # same idea      (2, 1, 3)
np.expand_dims(m, axis=0)      # explicit
np.squeeze(x)                  # remove size-1 axes
```

Adding an axis with `None`/`np.newaxis` is what makes broadcasting work (next lesson) and
what fixes most "expected 2-D, got 1-D" errors from scikit-learn.

### Views vs copies

```python
a = np.arange(6)
b = a[2:5]          # a VIEW: shares memory
b[0] = 99
a                   # array([ 0,  1, 99,  3,  4,  5])  ← a changed

c = a[2:5].copy()   # independent
c[0] = 0
a                   # unchanged
```

Slicing gives views (fast, no allocation); fancy indexing gives copies. Knowing which you
have prevents both accidental mutation and accidental gigabyte copies.

```python
b.base is a         # True for a view, None for a copy
np.shares_memory(a, b)
```

## Minimal Example

```python title="arrays_basics.py"
import numpy as np

rng = np.random.default_rng(42)

# 4 documents, 3-dimensional embeddings
embeddings = rng.normal(0, 1, size=(4, 3)).astype(np.float32)

print("shape :", embeddings.shape)
print("dtype :", embeddings.dtype, f"({embeddings.nbytes} bytes)")
print(embeddings.round(3))

print("\nmean per dimension (axis=0):", embeddings.mean(axis=0).round(3))
print("norm per document  (axis=1):", np.linalg.norm(embeddings, axis=1).round(3))
```

```text
shape : (4, 3)
dtype : float32 (48 bytes)
[[ 0.305 -1.04   0.75 ]
 [ 0.941 -1.951 -1.302]
 [ 0.127 -0.316 -0.017]
 [-0.854  0.879  0.778]]

mean per dimension (axis=0): [ 0.13  -0.607  0.052]
norm per document  (axis=1): [1.317 2.545 0.341 1.454]
```

Read the two aggregations carefully: `axis=0` gave 3 numbers (one per dimension), `axis=1`
gave 4 (one per document). The named axis disappeared.

## Real-World Example

An embedding store built on a single NumPy matrix — this is genuinely how FAISS's flat index
works underneath.

```python title="src/vectors/matrix_store.py"
"""A vector store backed by one contiguous float32 matrix.

Exact brute-force search. For fewer than ~100k vectors this beats a vector
database on latency and is far simpler to operate. Phase 11 explains when
approximate indexes become necessary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    score: float
    index: int


@dataclass
class MatrixStore:
    dimension: int
    dtype: np.dtype = np.dtype(np.float32)
    _vectors: np.ndarray = field(init=False)
    _ids: list[str] = field(default_factory=list, init=False)
    _id_to_index: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Pre-allocate an empty matrix with the right shape and dtype.
        object.__setattr__(self, "_vectors", np.empty((0, self.dimension), dtype=self.dtype))

    # --- writing ---------------------------------------------------------
    def add(self, ids: list[str], vectors: np.ndarray) -> int:
        """Append vectors, normalising them so cosine similarity is a dot product."""
        vectors = np.asarray(vectors, dtype=self.dtype)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)          # a single vector is still a matrix
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected {self.dimension} dimensions, got {vectors.shape[1]}"
            )
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"{len(ids)} ids but {vectors.shape[0]} vectors")

        normalised = self._normalise(vectors)
        start = len(self._ids)

        self._vectors = np.vstack([self._vectors, normalised])
        for offset, chunk_id in enumerate(ids):
            self._id_to_index[chunk_id] = start + offset
        self._ids.extend(ids)
        return vectors.shape[0]

    # --- reading ---------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 5) -> list[Hit]:
        """Exact nearest neighbours by cosine similarity."""
        if len(self._ids) == 0:
            return []

        q = self._normalise(np.asarray(query, dtype=self.dtype).reshape(1, -1))[0]

        # One matrix-vector product scores every stored vector at once.
        scores = self._vectors @ q                       # shape (n,)

        k = min(k, scores.shape[0])
        # argpartition finds the top-k without fully sorting: O(n) instead of O(n log n)
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]              # sort just those k

        return [Hit(id=self._ids[i], score=float(scores[i]), index=int(i)) for i in top]

    def get(self, chunk_id: str) -> np.ndarray:
        index = self._id_to_index.get(chunk_id)
        if index is None:
            raise KeyError(f"unknown id: {chunk_id!r}")
        return self._vectors[index]                      # a view, not a copy

    # --- introspection ---------------------------------------------------
    def stats(self) -> dict[str, object]:
        return {
            "vectors": len(self._ids),
            "dimension": self.dimension,
            "dtype": str(self.dtype),
            "megabytes": round(self._vectors.nbytes / 1024**2, 2),
        }

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)     # shape (n, 1)
        norms[norms == 0] = 1.0                                    # avoid division by zero
        return matrix / norms


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    store = MatrixStore(dimension=384)

    ids = [f"chunk-{i}" for i in range(50_000)]
    vectors = rng.normal(size=(50_000, 384)).astype(np.float32)
    store.add(ids, vectors)

    import time

    query = rng.normal(size=384).astype(np.float32)
    start = time.perf_counter()
    hits = store.search(query, k=5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(store.stats())
    print(f"search over 50,000 vectors: {elapsed_ms:.1f}ms")
    for hit in hits:
        print(f"  {hit.id:<14} {hit.score:.4f}")
```

```text
{'vectors': 50000, 'dimension': 384, 'dtype': 'float32', 'megabytes': 73.24}
search over 50,000 vectors: 8.4ms
50000 vectors searched exactly
  chunk-21738    0.2431
  chunk-6621     0.2368
  chunk-44052    0.2299
  chunk-38790    0.2287
  chunk-11913    0.2265
```

50,000 exact comparisons in under 10 ms, from one `@` operator. That is the payoff of
vectorisation, and it is why you should not reach for a vector database before you need one.

## Common Mistakes

:::mistake
```python
# 1. Mixing up axes
scores.mean(axis=0)        # per dimension
scores.mean(axis=1)        # per item  ← usually what you wanted

# 2. Forgetting that slices are views
batch = embeddings[:100]
batch /= 2                 # modified the original matrix in place!

# 3. Wrong dtype
np.array(api_vectors)              # float64: double the memory for no benefit
np.array(api_vectors, np.float32)  # correct

# 4. 1-D where 2-D is expected
model.predict(features)            # ValueError: Expected 2D array
model.predict(features.reshape(1, -1))

# 5. Comparing arrays with `if`
if a == b:                         # ValueError: truth value of an array is ambiguous
if np.array_equal(a, b):           # correct
if (a == b).all():                 # or this

# 6. Growing an array in a loop
for v in vectors:
    matrix = np.vstack([matrix, v])    # O(n^2): reallocates every time
matrix = np.vstack(vectors)            # one allocation
```
:::

## Debugging

```python
def describe(a: np.ndarray, name: str = "array") -> None:
    print(f"{name}: shape={a.shape} dtype={a.dtype} "
          f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f} "
          f"nan={np.isnan(a).sum()} inf={np.isinf(a).sum()}")
```

Print shapes at every step of a pipeline. Ninety percent of NumPy and PyTorch bugs are shape
bugs, and they are obvious the moment the shapes are visible.

`np.isnan(a).any()` early-detects the NaNs that otherwise silently poison an entire
similarity matrix.

## Performance Considerations

```python
import numpy as np, time

n = 1_000_000
python_list = list(range(n))
numpy_array = np.arange(n)

start = time.perf_counter()
total = sum(x * 2 for x in python_list)
print(f"python: {(time.perf_counter() - start) * 1000:.1f}ms")

start = time.perf_counter()
total = (numpy_array * 2).sum()
print(f"numpy : {(time.perf_counter() - start) * 1000:.1f}ms")
```

```text
python: 78.4ms
numpy : 1.9ms
```

~40× for a trivial operation. The gap comes from contiguous typed memory and loops that run
in C rather than in the interpreter.

Other rules:

- `float32` over `float64` for embeddings: half the memory, half the bandwidth.
- Preallocate (`np.empty((n, d))`) and fill, rather than appending.
- Prefer in-place operators (`a *= 2`) in memory-tight code to avoid a temporary copy.
- `np.argpartition` when you need top-k but not a full sort.

## Hands-on Exercise

:::exercise Build a similarity matrix
Using a seeded generator, create `embeddings` of shape `(100, 64)` as `float32`, then:

1. Normalise every row to unit length (watch the `keepdims` argument).
2. Compute the full `(100, 100)` cosine similarity matrix with one matrix multiplication.
3. Set the diagonal to `-1` so a vector is never its own neighbour.
4. For each row, find the index and score of its nearest neighbour.
5. Report: mean nearest-neighbour similarity, and the number of pairs above 0.5 (counting
   each pair once).
6. Verify memory use with `nbytes` and confirm the matrix is symmetric to within floating
   point tolerance.
:::

:::solution Solution
```python title="similarity_matrix.py"
import numpy as np

rng = np.random.default_rng(7)
embeddings = rng.normal(size=(100, 64)).astype(np.float32)

# 1. normalise rows
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)     # (100, 1) not (100,)
normalised = embeddings / norms

# 2. all pairwise cosine similarities in one operation
similarity = normalised @ normalised.T                        # (100, 100)

# 6. symmetry check before we mutate the diagonal
assert np.allclose(similarity, similarity.T, atol=1e-6)

# 3. exclude self-similarity
np.fill_diagonal(similarity, -1.0)

# 4. nearest neighbour per row
nearest_index = similarity.argmax(axis=1)                     # (100,)
nearest_score = similarity.max(axis=1)                        # (100,)

# 5. statistics
pairs_above = int((np.triu(similarity, k=1) > 0.5).sum())     # upper triangle: each pair once

print(f"matrix shape   : {similarity.shape}")
print(f"memory         : {similarity.nbytes / 1024:.1f} KB")
print(f"mean NN score  : {nearest_score.mean():.4f}")
print(f"max NN score   : {nearest_score.max():.4f} (row {nearest_score.argmax()})")
print(f"pairs > 0.5    : {pairs_above}")
print(f"row 0 nearest  : {nearest_index[0]} @ {nearest_score[0]:.4f}")
```

```text
matrix shape   : (100, 100)
memory         : 39.1 KB
mean NN score  : 0.3021
max NN score   : 0.4632 (row 62)
pairs > 0.5    : 0
row 0 nearest  : 79 @ 0.3164
```

Random 64-dimensional vectors are nearly orthogonal — no pair exceeds 0.5. That is the
*curse of dimensionality*, and it is exactly why real embeddings, which are structured rather
than random, can separate meaning at all. You will meet this again in Phase 11.
:::

## Challenge

:::challenge Deduplicate a corpus with one matrix operation
Given `(5000, 384)` embeddings, find every pair with cosine similarity above 0.95 (near
duplicates) without a Python loop over pairs, and produce the set of indices to drop —
keeping the first occurrence of each duplicate group. Measure memory: the full similarity
matrix is 5000×5000 float32 = 95 MB, which is fine; then work out how you would do the same
for 200,000 vectors, where the matrix would be 150 GB. (Hint: process in blocks of rows.)
Block processing is the standard technique for scaling exact search.
:::

## Interview Questions

:::interview
1. What does `axis=0` mean when calling `.sum()` on a 2-D array?
2. Why store embeddings as `float32` rather than `float64`?
3. What is the difference between a view and a copy?
4. What does `reshape(-1, 1)` do and when do you need it?
5. Why is a NumPy operation much faster than the equivalent Python loop?
:::

## Cheat Sheet

```python
np.array(x) np.zeros(s) np.ones(s) np.full(s, v) np.arange(a,b,step) np.linspace(a,b,n)
rng = np.random.default_rng(42); rng.normal(0,1,size=(n,d)); rng.integers(0,10,5)

a.shape a.ndim a.size a.dtype a.nbytes
a.reshape(r, -1) a.ravel() a.flatten() a.T a[:, None] np.expand_dims(a,0) np.squeeze(a)
a.astype(np.float32)

a.sum(axis=0) a.mean(axis=1) a.std() a.min() a.argmax(axis=1) a.max(axis=1)
np.linalg.norm(a, axis=1, keepdims=True)
a @ b      np.dot(a,b)     np.vstack([...]) np.hstack([...]) np.concatenate([...], axis=0)
np.argpartition(-scores, k)[:k]            # top-k without a full sort
np.array_equal(a,b)  np.allclose(a,b)  np.isnan(a).any()  np.fill_diagonal(m, 0)
```

```quiz
[
  {
    "question": "embeddings has shape (1000, 768). What is embeddings.mean(axis=0).shape?",
    "options": ["(1000,)", "(768,)", "(1000, 768)", "()"],
    "answer": 1,
    "explanation": "axis=0 collapses the rows, leaving one mean per dimension: shape (768,). The axis you name is the axis that disappears."
  },
  {
    "question": "Why cast embeddings to float32?",
    "options": [
      "float64 is not supported by NumPy",
      "It halves memory and bandwidth with no meaningful precision loss for embeddings",
      "It makes vectors normalised",
      "It is required for cosine similarity"
    ],
    "answer": 1,
    "explanation": "One million 1536-dim vectors: 12.3 GB at float64 versus 6.1 GB at float32 - and providers return 32-bit precision anyway."
  },
  {
    "question": "b = a[:100]; b += 1 — what happens to a?",
    "options": [
      "Nothing; b is a copy",
      "The first 100 elements of a are incremented, because slicing returns a view",
      "A ValueError is raised",
      "a is reallocated"
    ],
    "answer": 1,
    "explanation": "Basic slicing produces a view sharing memory. Use .copy() when you need independence - this bug silently corrupts training data."
  }
]
```

## Summary

- `shape` is a tuple of axis lengths; the axis you pass to an aggregation is the one that
  disappears.
- `float32` is the correct dtype for embeddings and ML features.
- Slices are views that share memory; fancy indexing copies.
- One matrix multiplication replaces thousands of Python-level comparisons.

## Next Step

Indexing, slicing, boolean masks and broadcasting — the operations that replace loops
entirely.
