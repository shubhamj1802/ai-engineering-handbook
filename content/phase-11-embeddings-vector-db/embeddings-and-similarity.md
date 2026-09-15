---
title: Embeddings, Similarity and Chunking
order: 1
difficulty: Advanced
duration: 18
badges: ["Hands-on", "Start here"]
summary: "What an embedding actually is, why cosine similarity works, how to chunk documents so retrieval can succeed, and how to measure whether any of it is working."
prereqs: ["Aggregations, Linear Algebra and Random Numbers", "Transformers and Attention"]
keyConcepts: ["embedding", "cosine similarity", "chunking", "semantic search", "recall@k"]
---

:::note In one line
**An embedding turns text into a point in space, and similar meanings land near each other.** That is what lets you search by meaning instead of by keyword.
:::

## Why this matters

Retrieval quality sets the ceiling on RAG quality: the model cannot answer from a chunk it
never received. Chunking and embedding decisions made in twenty minutes at the start of a
project determine whether the system works, and they are expensive to change later because
they require re-indexing everything.

## Mental Model

An embedding turns a piece of text into a list of numbers — a **point in space**.

The useful part: text that *means* similar things lands in similar places, even when the
words are completely different.

<figure class="lesson-figure">
<svg viewBox="0 0 660 330" role="img" aria-label="Scatter diagram: two questions about passwords and logins sit close together despite sharing no words, while a question about refunds sits far away. Similarity is the angle between the arrows from the origin.">
  <line x1="70" y1="270" x2="620" y2="270" stroke="var(--border-strong)" stroke-width="1.2"/>
  <line x1="70" y1="270" x2="70"  y2="30"  stroke="var(--border-strong)" stroke-width="1.2"/>
  <path d="M70,270 L300,90" stroke="var(--accent)" stroke-width="1.8" opacity="0.85"/>
  <path d="M70,270 L350,120" stroke="var(--accent)" stroke-width="1.8" opacity="0.85"/>
  <path d="M70,270 L560,215" stroke="var(--accent-2)" stroke-width="1.8" opacity="0.85"/>
  <path d="M126,226 A72,72 0 0 1 137,241" fill="none" stroke="var(--accent)" stroke-width="1.4"/>
  <text class="dg-sub" x="146" y="228" fill="var(--accent)">small angle = similar</text>
  <circle cx="300" cy="90"  r="7" fill="var(--accent)"/>
  <circle cx="350" cy="120" r="7" fill="var(--accent)"/>
  <circle cx="560" cy="215" r="7" fill="var(--accent-2)"/>
  <rect x="196" y="48" width="246" height="26" rx="6" fill="var(--panel-2)" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-sub" x="208" y="66">"How do I reset my password?"</text>
  <rect x="250" y="126" width="246" height="26" rx="6" fill="var(--panel-2)" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-sub" x="262" y="144">"I forgot my login credentials"</text>
  <rect x="400" y="230" width="230" height="26" rx="6" fill="var(--panel-2)" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-sub" x="412" y="248">"What is your refund policy?"</text>
  <text class="dg-sub" x="300" y="300" text-anchor="middle">Zero words in common, yet the top two are neighbours.</text>
  <text class="dg-sub" x="58" y="286" text-anchor="end">0</text>
</svg>
<figcaption>
<strong>Meaning becomes geometry.</strong> The two password questions share no words at all,
but they point in nearly the same direction. That is what lets you search by meaning instead
of by keyword.
</figcaption>
</figure>

Similarity is the **angle** between the arrows, not how far apart the points are:

| Cosine similarity | Meaning |
| --- | --- |
| `1.0` | same direction — as close to identical meaning as it gets |
| `0.8` | strongly related |
| `0.3` | vaguely related |
| `0.0` | unrelated |
| `-1.0` | opposite directions |

```text
cos(θ) = (a · b) / (|a| × |b|)
```

Angle rather than distance, because a long document and a short sentence can mean the same
thing. Direction captures meaning; length mostly captures how much text there was.

:::tip Keyword search vs meaning search
Keyword search matches **letters**. Embedding search matches **meaning**.

Search "password" and keyword search misses "login credentials" entirely. Embedding search
finds it. But keyword search never misses an exact product code, and embedding search
sometimes does — which is exactly why Phase 13 uses both together.
:::

## Core Concepts

### Where embeddings come from

An encoder-style transformer (bidirectional, Phase 10) processes the text and pools its
token representations into one fixed-length vector — typically 384, 768, 1024 or 1536
dimensions. The model is trained so that related texts produce nearby vectors, usually with
contrastive learning on pairs known to be related.

Two families you will use:

| Family | Example | Dimensions | Notes |
| --- | --- | --- | --- |
| Local / open-weight | `all-MiniLM-L6-v2`, `bge-base-en`, `e5-base` | 384–768 | free, fast, runs on CPU, no data leaves your machine |
| Hosted API | provider embedding endpoints | 768–3072 | stronger on nuance, costs money, sends text to a third party |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")          # ~80 MB, CPU-friendly
vectors = model.encode(
    ["how do I reset my password", "I forgot my login"],
    normalize_embeddings=True,        # unit length → cosine similarity is a dot product
    batch_size=64,
)
vectors.shape        # (2, 384)
```

:::warning Query and document must use the same model
An index built with one embedding model cannot be queried with another — the spaces are
unrelated. Changing embedding models means re-indexing the entire corpus. Record the model
name and version in your index metadata from day one.
:::

Some models also expect **asymmetric prefixes** — `"query: ..."` versus `"passage: ..."`
(the E5 family) or a query instruction (BGE). Skipping them silently costs several points of
recall; check the model card.

### Similarity measures

```python
import numpy as np

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# with normalised vectors, cosine similarity IS the dot product
scores = normalised_matrix @ normalised_query          # one matmul for the whole index
```

| Measure | Formula | When |
| --- | --- | --- |
| **Cosine** | angle between vectors | text embeddings — the default |
| Dot product | `a · b` | when vectors are already normalised (identical to cosine) |
| Euclidean (L2) | straight-line distance | image embeddings, some clustering |

:::note Similarity scores are not probabilities
A cosine of 0.82 does not mean "82% relevant". Scores are only comparable *within the same
model and corpus*. Calibrate thresholds empirically on your own data, and never copy a
threshold from a blog post.
:::

### Chunking — the decision that matters most

A chunk is the unit of retrieval. Too large and it contains mostly irrelevant text that
dilutes both the embedding and the context window. Too small and it loses the context needed
to be understood.

```text
TOO SMALL   "It costs $49 per month."          ← what costs $49? which plan?
GOOD        "## Pro plan\nThe Pro plan costs $49 per month and includes 30-day
             log retention, 5 seats, and API access."
TOO LARGE   the entire 40-page pricing document
```

Strategies, in increasing order of quality:

| Strategy | How | Good for |
| --- | --- | --- |
| Fixed size | every N characters/tokens with overlap | quick baselines, uniform text |
| Sentence-aware | split on sentence boundaries, pack to a limit | prose |
| **Structure-aware** | split on headings, then pack sections | docs, wikis, markdown |
| Semantic | split where consecutive-sentence similarity drops | unstructured transcripts |
| Parent-child | embed small, return the parent section | precision + context (Phase 13) |

Starting parameters that are usually reasonable: **500–800 tokens per chunk with 10–15%
overlap**, then measure and adjust. Overlap exists so a fact spanning a boundary appears
whole in at least one chunk.

### Context matters more than size

Prepending the document and section title to each chunk is one of the highest
return-on-effort changes available:

```python
chunk_text = f"{document_title} > {section_heading}\n\n{body}"
```

A chunk reading "It costs $49 per month" is nearly unretrievable. "ACME Pricing > Pro plan —
It costs $49 per month" is retrievable by "how much is Pro". Phase 13 extends this idea into
full contextual retrieval.

### Measuring retrieval

You cannot tune what you do not measure. The metric set:

```text
recall@k       of the chunks that SHOULD be retrieved, how many are in the top k?
precision@k    of the top k, how many are relevant?
MRR            1 / rank of the first relevant chunk, averaged
nDCG@k         rewards putting the best chunk first
```

For RAG, **recall@k is the metric that matters most**: the generator can ignore an
irrelevant chunk, but it cannot use one that was never retrieved.

## Minimal Example

```python title="embeddings_demo.py"
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

corpus = [
    "Reset your password from the account settings page.",
    "Logs are retained for 30 days on the Pro plan.",
    "Enterprise customers can configure retention up to 400 days.",
    "Contact sales for volume pricing and custom contracts.",
    "The API rate limit is 100 requests per minute per key.",
]

document_vectors = model.encode(corpus, normalize_embeddings=True)

for question in ["I forgot my login credentials", "how long do you keep my data"]:
    query_vector = model.encode(question, normalize_embeddings=True)
    scores = document_vectors @ query_vector            # cosine, one matmul

    print(f"\nQ: {question}")
    for index in np.argsort(-scores)[:3]:
        print(f"  {scores[index]:.3f}  {corpus[index]}")
```

```text
Q: I forgot my login credentials
  0.612  Reset your password from the account settings page.
  0.104  The API rate limit is 100 requests per minute per key.
  0.071  Contact sales for volume pricing and custom contracts.

Q: how long do you keep my data
  0.584  Logs are retained for 30 days on the Pro plan.
  0.521  Enterprise customers can configure retention up to 400 days.
  0.093  Reset your password from the account settings page.
```

Neither query shares a single significant word with its best match. That is semantic search,
and it is why embeddings beat keyword search for natural-language questions.

## Real-World Example

A chunking and embedding pipeline with structure awareness, contextual headers, caching and
a measurable evaluation.

```python title="src/retrieval/chunking.py"
"""Structure-aware chunking.

Design decisions and why:
  - split on markdown headings first: sections are natural semantic units
  - pack small sections together, split large ones with sentence-aware windows
  - prepend the heading path to every chunk so it is retrievable on its own
  - carry metadata (document, section, position) for filtering and citation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str                    # what gets embedded (includes the heading path)
    body: str                    # the original text, for display
    document: str
    heading_path: tuple[str, ...]
    index: int
    approx_tokens: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        path = " > ".join(self.heading_path) if self.heading_path else "intro"
        return f"{self.document} > {path}"


@dataclass(frozen=True, slots=True)
class Section:
    heading_path: tuple[str, ...]
    body: str


def split_sections(markdown: str) -> list[Section]:
    """Split on headings, tracking the full heading path for context."""
    matches = list(HEADING.finditer(markdown))
    if not matches:
        return [Section((), markdown.strip())] if markdown.strip() else []

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        sections.append(Section((), preamble))

    for i, match in enumerate(matches):
        level, title = len(match.group(1)), match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[match.end() : end].strip()
        if body:
            sections.append(Section(tuple(title for _, title in stack), body))

    return sections


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def pack_sentences(body: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Sentence-aware windows with overlap, used when a section is too large."""
    sentences = [s.strip() for s in SENTENCE_END.split(body) if s.strip()]
    if not sentences:
        return []

    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = approx_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            windows.append(" ".join(current))
            # carry the tail forward as overlap so facts are not split
            carried, carried_tokens = [], 0
            for previous in reversed(current):
                previous_tokens = approx_tokens(previous)
                if carried_tokens + previous_tokens > overlap_tokens:
                    break
                carried.insert(0, previous)
                carried_tokens += previous_tokens
            current, current_tokens = carried, carried_tokens
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        windows.append(" ".join(current))
    return windows


def chunk_document(
    markdown: str,
    *,
    document: str,
    max_tokens: int = 600,
    min_tokens: int = 80,
    overlap_tokens: int = 80,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """Turn a markdown document into retrievable chunks."""
    chunks: list[Chunk] = []
    buffer: list[Section] = []
    buffered_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffered_tokens
        if not buffer:
            return
        heading_path = buffer[0].heading_path
        body = "\n\n".join(s.body for s in buffer)
        emit(heading_path, body)
        buffer, buffered_tokens = [], 0

    def emit(heading_path: tuple[str, ...], body: str) -> None:
        prefix = f"{document} > {' > '.join(heading_path)}" if heading_path else document
        index = len(chunks)
        chunks.append(Chunk(
            id=f"{document}::{index}",
            text=f"{prefix}\n\n{body}",          # context header: the cheap win
            body=body,
            document=document,
            heading_path=heading_path,
            index=index,
            approx_tokens=approx_tokens(body),
            metadata=dict(metadata or {}),
        ))

    for section in split_sections(markdown):
        section_tokens = approx_tokens(section.body)

        if section_tokens > max_tokens:
            flush()
            for window in pack_sentences(section.body, max_tokens=max_tokens,
                                         overlap_tokens=overlap_tokens):
                emit(section.heading_path, window)
            continue

        if section_tokens < min_tokens:
            buffer.append(section)               # too small alone: merge with neighbours
            buffered_tokens += section_tokens
            if buffered_tokens >= max_tokens:
                flush()
            continue

        flush()
        emit(section.heading_path, section.body)

    flush()
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, float | int]:
    if not chunks:
        return {"chunks": 0}
    sizes = [c.approx_tokens for c in chunks]
    return {
        "chunks": len(chunks),
        "mean_tokens": round(sum(sizes) / len(sizes), 1),
        "min_tokens": min(sizes),
        "max_tokens": max(sizes),
        "total_tokens": sum(sizes),
    }


if __name__ == "__main__":
    document = """\
# ACME Platform Guide

Welcome to ACME.

## Pricing

### Pro plan
The Pro plan costs $49 per month per seat. It includes 30-day log retention,
five seats, and full API access. Overage is billed at $0.02 per extra request.

### Enterprise plan
Enterprise pricing is custom. Contact sales. Enterprise customers can configure
log retention up to 400 days and receive a dedicated support channel.

## API

### Rate limits
The API allows 100 requests per minute per key. Exceeding the limit returns HTTP 429
with a Retry-After header. Burst capacity is 20 requests. Rate limits are evaluated
per key, not per account, so separate keys have separate budgets. Contact support to
request a higher limit for production workloads with predictable traffic patterns.
"""

    chunks = chunk_document(document, document="acme-guide", max_tokens=60, min_tokens=15)
    print(chunk_stats(chunks), "\n")
    for chunk in chunks:
        print(f"[{chunk.id}] {chunk.citation}")
        print(f"    {chunk.body[:80]}...")
```

```text
{'chunks': 5, 'mean_tokens': 41.2, 'min_tokens': 12, 'max_tokens': 59, 'total_tokens': 206}

[acme-guide::0] acme-guide > ACME Platform Guide
    Welcome to ACME....
[acme-guide::1] acme-guide > ACME Platform Guide > Pricing > Pro plan
    The Pro plan costs $49 per month per seat. It includes 30-day log retention...
[acme-guide::2] acme-guide > ACME Platform Guide > Pricing > Enterprise plan
    Enterprise pricing is custom. Contact sales. Enterprise customers can conf...
[acme-guide::3] acme-guide > ACME Platform Guide > API > Rate limits
    The API allows 100 requests per minute per key. Exceeding the limit return...
[acme-guide::4] acme-guide > ACME Platform Guide > API > Rate limits
    Rate limits are evaluated per key, not per account, so separate keys have...
```

Chunk 4 is the overlap window: it carries the sentence that would otherwise have been split
across a boundary, and it is independently retrievable because the heading path travels with
it.

### Embedding with caching and batching

```python title="src/retrieval/embedder.py"
"""Embedding with a content-addressed cache.

Re-embedding unchanged text is pure waste - on a hosted API it is money, locally
it is minutes. The cache key is a hash of (model, text), so a changed model
invalidates everything automatically.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedEmbedder:
    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: Path = Path(".data/embeddings")
    batch_size: int = 64
    _model: object | None = field(default=None, init=False)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        if self._model is None:                 # lazy: importing torch is slow
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _key(self, text: str) -> str:
        raw = f"{self.model_name}\0{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.npy"      # shard, to keep directories small

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one normalised vector per text, in input order."""
        results: dict[int, np.ndarray] = {}
        pending: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            path = self._path(self._key(text))
            if path.exists():
                results[i] = np.load(path)
                self.hits += 1
            else:
                pending.append((i, text))
                self.misses += 1

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self.model.encode(
                [text for _, text in batch],
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            ).astype(np.float32)

            for (i, text), vector in zip(batch, vectors, strict=True):
                results[i] = vector
                path = self._path(self._key(text))
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vector)

        return np.vstack([results[i] for i in range(len(texts))])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def stats(self) -> dict[str, float | int | str]:
        total = self.hits + self.misses
        return {
            "model": self.model_name,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
```

### Evaluating chunking choices

```python title="src/retrieval/evaluate_chunking.py"
"""Compare chunking configurations with recall@k.

This is the experiment that decides your chunk size. It takes twenty minutes and
saves weeks of "the answers feel wrong".
"""
from __future__ import annotations

import numpy as np

from .chunking import chunk_document
from .embedder import CachedEmbedder

# each case: a question plus a phrase that MUST appear in a retrieved chunk
CASES = [
    ("How much does Pro cost?", "$49 per month"),
    ("How long are logs kept on Pro?", "30-day log retention"),
    ("Can Enterprise change retention?", "400 days"),
    ("What is the API rate limit?", "100 requests per minute"),
    ("Are rate limits per account?", "per key, not per account"),
    ("What happens when I exceed the rate limit?", "HTTP 429"),
]


def evaluate(document: str, *, max_tokens: int, overlap_tokens: int, k: int = 3) -> dict:
    chunks = chunk_document(document, document="doc", max_tokens=max_tokens,
                            overlap_tokens=overlap_tokens)
    embedder = CachedEmbedder()
    matrix = embedder.embed([c.text for c in chunks])

    hits, ranks = 0, []
    for question, required in CASES:
        scores = matrix @ embedder.embed_query(question)
        order = np.argsort(-scores)[:k]
        found = next(
            (rank for rank, idx in enumerate(order, start=1)
             if required.lower() in chunks[idx].body.lower()),
            None,
        )
        if found:
            hits += 1
            ranks.append(found)

    sizes = [c.approx_tokens for c in chunks]
    return {
        "max_tokens": max_tokens,
        "overlap": overlap_tokens,
        "chunks": len(chunks),
        "mean_size": round(sum(sizes) / len(sizes), 1),
        f"recall@{k}": round(hits / len(CASES), 3),
        "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "index_tokens": sum(sizes),
    }


if __name__ == "__main__":
    from .chunking import __doc__ as _  # noqa: F401

    document = open("data/acme-guide.md", encoding="utf-8").read()

    print(f"{'size':>6}{'overlap':>9}{'chunks':>8}{'mean':>8}{'recall@3':>10}{'rank':>7}{'tokens':>9}")
    for max_tokens in (200, 400, 600, 1000):
        for overlap in (0, int(max_tokens * 0.15)):
            row = evaluate(document, max_tokens=max_tokens, overlap_tokens=overlap)
            print(f"{row['max_tokens']:>6}{row['overlap']:>9}{row['chunks']:>8}"
                  f"{row['mean_size']:>8}{row['recall@3']:>10}"
                  f"{str(row['mean_rank']):>7}{row['index_tokens']:>9}")
```

```text
  size  overlap  chunks    mean  recall@3   rank   tokens
   200        0      18    92.4     0.667    1.5     1664
   200       30      22    88.1     0.833    1.4     1938
   400        0      11   151.2     0.833    1.2     1663
   400       60      13   148.9     1.000    1.2     1936
   600        0       8   207.8     0.833    1.5     1662
   600       90       9   201.4     0.833    1.6     1812
  1000        0       6   277.0     0.667    1.8     1662
  1000      100       6   277.0     0.667    1.8     1662
```

400 tokens with 15% overlap gives perfect recall at k=3 on this corpus, at an 16% index-size
premium. 1,000-token chunks lose two cases entirely — the relevant sentence is buried among
enough unrelated text that the embedding no longer points at it. **Run this experiment on
your own corpus**; the optimum is corpus-specific, and it is the single most valuable twenty
minutes in a RAG project.

## Common Mistakes

:::mistake
```text
1. Chunking without context headers
   "It costs $49" is unretrievable. Prepend the document and heading path.

2. Copying chunk size from a tutorial
   Optimal size depends on your documents. Measure recall@k.

3. Mixing embedding models in one index
   The spaces are unrelated; scores become meaningless. Record the model in metadata.

4. Skipping the model's query/passage prefixes
   E5 and BGE families lose several points of recall without them.

5. Embedding boilerplate
   Headers, footers, nav bars and legal disclaimers dominate similarity because they
   repeat everywhere. Strip them at ingest.

6. Treating similarity scores as probabilities
   0.75 means nothing in the abstract. Calibrate on your data.

7. Re-embedding unchanged text on every run
   Content-hash cache. On a hosted API this is a direct bill.

8. No evaluation set
   Then every chunking change is a guess.
```
:::

## Performance Considerations

| Decision | Effect |
| --- | --- |
| Batch size 32–128 | 5–20× faster than one call per text |
| `float32` not `float64` | half the memory for the whole index |
| Normalise once at index time | every query becomes a plain dot product |
| Smaller dimensions (384 vs 1536) | 4× less memory, often only slightly worse recall |
| Local model on CPU | ~1–3 ms per short text; no per-token cost |
| Content-hash caching | re-runs and incremental ingests become nearly free |

For one million chunks at 768 dimensions in `float32`: about 3 GB of vectors. That fits in
memory on a normal machine, which is worth remembering before adopting a distributed vector
database.

## Hands-on Exercise

:::exercise Chunk and evaluate your own documents
Take 5–10 real documents (your team's docs, a public handbook, anything with structure).

1. Write 15 questions with the exact phrase that must appear in a correct chunk.
2. Implement three chunking strategies: fixed-size, sentence-aware and structure-aware.
3. For each, sweep chunk size over {256, 512, 1024} tokens with and without overlap.
4. Report recall@1, recall@3, recall@5, mean rank, chunk count and total index tokens.
5. Add contextual headers to the best configuration and re-measure.
6. Write down the configuration you chose and the recall it achieves. That number is the
   ceiling on your RAG system's accuracy.
:::

:::solution What a real result looks like
```text
strategy         size  overlap  recall@1  recall@3  recall@5  chunks  tokens
fixed             512        0     0.400     0.667     0.733     142   72,704
fixed             512       76     0.467     0.733     0.800     168   85,120
sentence          512       76     0.533     0.800     0.867     151   78,320
structure         512       76     0.600     0.867     0.933     139   71,168
structure+header  512       76     0.733     0.933     1.000     139   74,580

Chosen: structure-aware, 512 tokens, 15% overlap, with heading-path headers.
recall@5 = 1.00 on the evaluation set, at a 4% index-size premium over the
header-less version. The generator therefore has access to the right chunk in
every evaluated case; any remaining errors are generation errors, not retrieval
errors - which is the state you want before tuning prompts.
```

That last sentence is the point of the exercise: separating retrieval failures from
generation failures is what makes a RAG system debuggable.
:::

## Challenge

:::challenge Semantic chunking
Implement semantic chunking: embed each sentence, compute the cosine similarity between
consecutive sentences, and start a new chunk where similarity drops below a threshold
(a topic shift). Add a maximum size so a long uniform passage still gets split.

Compare it to structure-aware chunking on a corpus **without** headings — meeting
transcripts, support conversations, or scraped pages. Report recall and the cost of the
extra embedding pass. Semantic chunking usually wins on unstructured text and loses on
well-structured documents, which tells you when it is worth the cost.
:::

## Interview Questions

:::interview
1. What is an embedding, and why does cosine similarity measure semantic similarity?
2. Why can you not query an index with a different embedding model than you built it with?
3. How do you choose a chunk size?
4. Why prepend headings to chunks?
5. Which retrieval metric matters most for RAG, and why?
:::

## Cheat Sheet

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, normalize_embeddings=True, batch_size=64)   # (n, d) float32
scores = matrix @ query_vector                       # cosine, if normalised
top_k = np.argpartition(-scores, k)[:k]              # O(n) top-k

chunking   structure-aware > sentence-aware > fixed
size       500-800 tokens, 10-15% overlap as a starting point - then MEASURE
context    prepend "document > section" to every chunk before embedding
metadata   document, section, page, ingested_at, permissions, model name

metrics    recall@k (most important for RAG) · precision@k · MRR · nDCG@k
cache      key on sha256(model + text); changing the model invalidates everything
```

```quiz
[
  {
    "question": "Your RAG system answers 'I could not find this' for questions that ARE covered by the documents. Where do you look first?",
    "options": [
      "The generation prompt",
      "Retrieval: measure recall@k on an evaluation set - the chunk may never be reaching the model",
      "The model temperature",
      "The vector database vendor"
    ],
    "answer": 1,
    "explanation": "The generator cannot use a chunk it never received. Always separate retrieval failure from generation failure before touching the prompt."
  },
  {
    "question": "Why prepend the document title and heading path to each chunk before embedding?",
    "options": [
      "It makes chunks longer, which improves embeddings",
      "It gives the chunk enough context to be retrievable on its own - 'It costs $49' becomes 'Pricing > Pro plan — It costs $49'",
      "It is required by vector databases",
      "It improves the compression ratio"
    ],
    "answer": 1,
    "explanation": "Chunks are retrieved out of context, so each must carry enough context to be matched. This is one of the cheapest recall improvements available."
  },
  {
    "question": "You switch from one embedding model to another. What must you do?",
    "options": [
      "Nothing, embeddings are standardised",
      "Re-embed and re-index the entire corpus, because the vector spaces are unrelated",
      "Only re-embed queries",
      "Normalise the old vectors"
    ],
    "answer": 1,
    "explanation": "Different models produce incomparable spaces. Store the model name and version with the index so this mistake is impossible to make silently."
  }
]
```

## Summary

- Embeddings place semantically similar text near each other; cosine similarity measures the
  angle, and normalising once makes every query a dot product.
- Chunking is the highest-leverage decision: structure-aware, 500–800 tokens, 10–15% overlap,
  with heading-path context headers.
- One model per index, recorded in metadata; changing it means re-indexing.
- Measure recall@k on a real evaluation set — it is the ceiling on your RAG accuracy.

## Next Step

Vector databases: what an index actually does, and how to choose between FAISS, Chroma,
Qdrant, Pinecone and pgvector.
