---
title: Vector Databases — FAISS, Chroma, Qdrant, Pinecone, pgvector
order: 2
difficulty: Advanced
duration: 16
badges: ["Hands-on", "Reference"]
summary: "What an index actually does, how ANN algorithms trade recall for speed, metadata filtering and access control, and an honest comparison of the five options you will actually consider."
prereqs: ["Embeddings, Similarity and Chunking"]
keyConcepts: ["ANN", "HNSW", "IVF", "metadata filter", "hybrid search", "multi-tenancy"]
---

:::note In one line
**A vector database is an index that finds the nearest points fast.** Start with the simplest one that works; you can move later.
:::

## Why this matters

"Which vector database?" is the most-asked and least-important question in RAG. Chunking and
retrieval quality decide whether your system works; the database decides how it operates at
scale. But the wrong choice costs weeks of migration, and the right one is usually simpler
and cheaper than people expect.

## Mental Model

A vector database does one job: given a query vector, find the nearest stored vectors fast.
The interesting part is what it trades away to be fast.
<figure class="lesson-figure">
<svg viewBox="0 0 660 230" role="img" aria-label="Diagram comparing exact search which checks every vector, with approximate search which follows a graph of neighbours to reach a close answer in far fewer comparisons.">
  <text class="dg-label" x="14" y="22" fill="var(--accent-3)">Exact — check everything</text>
  <circle cx="40" cy="60" r="4" fill="var(--text-muted)"/><circle cx="70" cy="78" r="4" fill="var(--text-muted)"/>
  <circle cx="100" cy="52" r="4" fill="var(--text-muted)"/><circle cx="130" cy="86" r="4" fill="var(--text-muted)"/>
  <circle cx="58" cy="104" r="4" fill="var(--text-muted)"/><circle cx="112" cy="112" r="4" fill="var(--text-muted)"/>
  <circle cx="86" cy="134" r="4" fill="var(--text-muted)"/><circle cx="140" cy="140" r="4" fill="var(--text-muted)"/>
  <circle cx="44" cy="146" r="4" fill="var(--text-muted)"/>
  <circle cx="96" cy="90" r="7" fill="var(--accent)"/>
  <path d="M96,90 L40,60" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L70,78" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L100,52" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L130,86" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L58,104" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L112,112" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L86,134" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L140,140" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <path d="M96,90 L44,146" stroke="var(--accent-3)" stroke-width="0.9" opacity="0.6"/>
  <text class="dg-sub" x="14" y="180">perfect answer, every time</text>
  <text class="dg-sub" x="14" y="198" fill="var(--danger)">but slower as your data grows</text>
  <line x1="240" y1="14" x2="240" y2="216" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-label" x="266" y="22" fill="var(--ok)">Approximate — follow a graph</text>
  <circle cx="300" cy="60" r="4" fill="var(--text-muted)"/><circle cx="352" cy="50" r="4" fill="var(--text-muted)"/>
  <circle cx="404" cy="72" r="4" fill="var(--text-muted)"/><circle cx="330" cy="104" r="4" fill="var(--text-muted)"/>
  <circle cx="386" cy="122" r="4" fill="var(--text-muted)"/><circle cx="440" cy="104" r="4" fill="var(--text-muted)"/>
  <circle cx="312" cy="146" r="4" fill="var(--text-muted)"/><circle cx="420" cy="152" r="4" fill="var(--text-muted)"/>
  <circle cx="366" cy="86" r="7" fill="var(--accent)"/>
  <path d="M300,60 L352,50" stroke="var(--ok)" stroke-width="2"/>
  <path d="M352,50 L366,86" stroke="var(--ok)" stroke-width="2"/>
  <path d="M366,86 L386,122" stroke="var(--ok)" stroke-width="2"/>
  <text class="dg-sub" x="266" y="180">about 99% as good</text>
  <text class="dg-sub" x="266" y="198" fill="var(--ok)">and thousands of times faster</text>
  <rect x="478" y="40" width="168" height="130" rx="9" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.4"/>
  <text class="dg-label" x="492" y="64">How to choose</text>
  <text class="dg-sub" x="492" y="88">under 100k vectors:</text>
  <text class="dg-mono" x="492" y="104" style="font-size:10.5px">any of them. really.</text>
  <text class="dg-sub" x="492" y="128">already run Postgres?</text>
  <text class="dg-mono" x="492" y="144" style="font-size:10.5px">pgvector. one less thing.</text>
  <text class="dg-sub" x="492" y="164">start simple, move later</text>
</svg>
<figcaption>
<strong>Approximate search is the right default.</strong> Losing the occasional borderline
neighbour costs you almost nothing in answer quality, and buys orders of magnitude in speed.
</figcaption>
</figure>

```text
A vector database is three things:

  1. AN INDEX      structure that finds nearest neighbours faster than scanning everything
  2. A STORE       the vectors, their text, and their metadata, durably
  3. A FILTER      "only chunks from documents this user may read"

Exact search:       compare the query to every vector.   O(n)   100% recall
Approximate (ANN):  compare to a clever subset.          O(log n)  95-99% recall

The trade is RECALL for SPEED. Below ~100k vectors, exact search is fast enough
and you should not be using an ANN index at all.
```

## Core Concepts

### When you actually need one

| Corpus size | Recommendation |
| --- | --- |
| < 10k chunks | a NumPy matrix in memory (Phase 3). Milliseconds, zero dependencies. |
| 10k – 100k | NumPy or FAISS flat, loaded at startup |
| 100k – 10M | a real vector database with an HNSW index |
| > 10M | a distributed vector database, sharding, and a serious operations budget |

Most internal RAG systems never leave the first two rows. A company handbook is ~5,000
chunks; all of Wikipedia in English is ~50 million.

### ANN algorithms, briefly

**HNSW (Hierarchical Navigable Small World)** — a layered graph. Search starts at a sparse
top layer and descends, following edges toward the query. Fast, high recall, the default
nearly everywhere. Memory-hungry: roughly `vectors × (dimensions × 4 + M × 8)` bytes.

```text
key parameters
  M                 edges per node (16-64). Higher = better recall, more memory
  ef_construction   effort at build time (100-500). Higher = better index, slower build
  ef_search         effort at query time (50-200). Higher = better recall, slower query
                    ← the one you tune per query at runtime
```

**IVF (Inverted File)** — cluster the vectors, search only the nearest few clusters.
Lower memory, needs a training step, recall depends on how many clusters you probe
(`nprobe`).

**Product Quantisation (PQ)** — compress vectors to a few bytes each. 10–50× memory
reduction for a few points of recall. Combined with IVF (`IVF_PQ`) for very large corpora.

**Flat** — no index; exact brute force. Perfect recall, linear cost. The correct choice more
often than people think, and the baseline you should always measure against.

### Metadata filtering is not optional

```python
results = store.search(
    query_vector,
    k=5,
    where={"tenant_id": "acme", "language": "en", "document_type": {"$in": ["policy", "faq"]}},
)
```

This is how multi-tenancy and access control work. Two implementation strategies:

- **Pre-filter**: restrict the candidate set, then search. Exact, but can be slow if the
  filter is selective and the index is graph-based.
- **Post-filter**: search, then drop non-matching results. Fast, but may return fewer than
  `k` results — or none — when the filter is selective.

Good databases do filtered search inside the index traversal. Check how yours behaves with a
1%-selectivity filter *before* you commit; this is where implementations differ most.

:::danger Access control belongs in the filter, not the prompt
```python
# WRONG: the model is asked to be discreet
prompt = f"Only use documents the user may see.\n{all_chunks}"

# RIGHT: the user never sees forbidden chunks at all
chunks = store.search(vector, k=5, where={"visible_to": {"$contains": user.group}})
```
A retrieval system that returns another tenant's document has already leaked it, whatever
the model then says. Filter at the source, and test it with a user who should see nothing.
:::

### The comparison

| | **FAISS** | **Chroma** | **Qdrant** | **Pinecone** | **pgvector** |
| --- | --- | --- | --- | --- | --- |
| Type | library | embedded DB | server / cloud | managed cloud | Postgres extension |
| Runs | in your process | in-process or server | Docker / cloud | SaaS only | your Postgres |
| Persistence | you handle it | built-in | built-in | managed | Postgres |
| Metadata filter | no (do it yourself) | yes | yes, strong | yes | yes (full SQL) |
| Hybrid search | no | limited | yes (sparse+dense) | yes | yes (tsvector + vector) |
| Scale | billions (in RAM) | ~1M comfortable | 100M+ | billions | ~10M comfortable |
| Ops burden | none (but no server) | very low | moderate | none | low if you have Postgres |
| Cost | free | free | free self-hosted | usage-based | free |
| Best for | max speed, embedded | prototypes, small apps | serious self-hosted | no-ops production | you already run Postgres |

**The honest recommendation:**

1. Start with **NumPy** if under ~50k chunks. Seriously.
2. Use **Chroma** when you want persistence and metadata with near-zero setup.
3. Use **pgvector** if you already run Postgres — one datastore, transactional consistency
   with your application data, and SQL filtering is genuinely better than any bespoke filter
   language.
4. Use **Qdrant** when you need real scale, hybrid search and strong filtering, self-hosted.
5. Use **Pinecone** when you want none of the operations and will pay for that.

The thing that matters more than any of these: **isolate it behind an interface** (the
`VectorStore` ABC from Phase 2), so the choice stays reversible.

## Minimal Example

The same 20 lines against three backends, behind one interface.

```python title="stores_demo.py"
import numpy as np

# --- 1. NumPy: no dependencies, exact, sufficient below ~50k chunks ---------
class NumpyStore:
    def __init__(self, dimension: int):
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids: list[str] = []
        self.payloads: list[dict] = []

    def add(self, ids, vectors, payloads):
        self.vectors = np.vstack([self.vectors, np.asarray(vectors, dtype=np.float32)])
        self.ids.extend(ids)
        self.payloads.extend(payloads)

    def search(self, query, k=5, where=None):
        scores = self.vectors @ np.asarray(query, dtype=np.float32)
        if where:
            mask = np.array([
                all(p.get(key) == value for key, value in where.items()) for p in self.payloads
            ])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i]), self.payloads[i]) for i in order
                if np.isfinite(scores[i])]


# --- 2. Chroma: persistence and filtering, one line of setup ----------------
def chroma_store(path=".data/chroma"):
    import chromadb

    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        "chunks", metadata={"hnsw:space": "cosine"}
    )


# --- 3. Qdrant: production-grade filtering and hybrid search ----------------
def qdrant_store(dimension: int, url="http://localhost:6333"):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(url=url)
    client.recreate_collection(
        collection_name="chunks",
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    return client
```

## Real-World Example

A production `VectorStore` with four interchangeable backends, tenant isolation and a
benchmark.

```python title="src/retrieval/stores.py"
"""Vector store adapters.

One interface, four backends. Business logic depends on the ABC only, so the
backend is a configuration decision rather than a rewrite.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Record:
    id: str
    vector: list[float]
    text: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    score: float
    text: str
    payload: dict[str, object] = field(default_factory=dict)


class VectorStore(ABC):
    """Every backend implements exactly this."""

    @abstractmethod
    def upsert(self, records: list[Record]) -> int: ...

    @abstractmethod
    def search(self, vector: list[float], *, k: int = 5,
               where: dict[str, object] | None = None) -> list[Hit]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int: ...

    @abstractmethod
    def count(self, *, where: dict[str, object] | None = None) -> int: ...

    def health(self) -> dict[str, object]:
        try:
            return {"ok": True, "backend": type(self).__name__, "vectors": self.count()}
        except Exception as exc:
            return {"ok": False, "backend": type(self).__name__, "error": str(exc)}


# --- 1. NumPy ---------------------------------------------------------------
class NumpyStore(VectorStore):
    """Exact search over a contiguous matrix. Fastest option below ~100k vectors."""

    def __init__(self, dimension: int, *, persist_path: Path | None = None) -> None:
        self.dimension = dimension
        self.persist_path = persist_path
        self._vectors = np.empty((0, dimension), dtype=np.float32)
        self._records: list[Record] = []
        self._index: dict[str, int] = {}
        if persist_path and persist_path.exists():
            self._load()

    def upsert(self, records: list[Record]) -> int:
        new: list[Record] = []
        for record in records:
            if record.id in self._index:                   # replace in place
                self._vectors[self._index[record.id]] = np.asarray(record.vector, np.float32)
                self._records[self._index[record.id]] = record
            else:
                new.append(record)

        if new:
            block = np.asarray([r.vector for r in new], dtype=np.float32)
            start = len(self._records)
            self._vectors = np.vstack([self._vectors, block])
            for offset, record in enumerate(new):
                self._index[record.id] = start + offset
            self._records.extend(new)

        if self.persist_path:
            self._save()
        return len(records)

    def search(self, vector, *, k=5, where=None) -> list[Hit]:
        if not self._records:
            return []

        scores = self._vectors @ np.asarray(vector, dtype=np.float32)

        if where:
            mask = np.fromiter(
                (self._matches(r.payload, where) for r in self._records),
                dtype=bool, count=len(self._records),
            )
            scores = np.where(mask, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []

        candidates = np.argpartition(-scores, kth=k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates])]
        return [
            Hit(self._records[i].id, float(scores[i]), self._records[i].text,
                self._records[i].payload)
            for i in order
        ]

    def delete(self, ids: list[str]) -> int:
        keep = [i for i, r in enumerate(self._records) if r.id not in set(ids)]
        removed = len(self._records) - len(keep)
        self._vectors = self._vectors[keep]
        self._records = [self._records[i] for i in keep]
        self._index = {r.id: i for i, r in enumerate(self._records)}
        if self.persist_path:
            self._save()
        return removed

    def count(self, *, where=None) -> int:
        if not where:
            return len(self._records)
        return sum(self._matches(r.payload, where) for r in self._records)

    @staticmethod
    def _matches(payload: dict, where: dict) -> bool:
        """Supports equality, $in and $contains - enough for tenant + language filters."""
        for key, condition in where.items():
            value = payload.get(key)
            if isinstance(condition, dict):
                if "$in" in condition and value not in condition["$in"]:
                    return False
                if "$contains" in condition:
                    if not isinstance(value, (list, tuple)) or condition["$contains"] not in value:
                        return False
            elif value != condition:
                return False
        return True

    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.persist_path.with_suffix(".npy"), self._vectors)
        self.persist_path.with_suffix(".jsonl").write_text(
            "\n".join(json.dumps({"id": r.id, "text": r.text, "payload": r.payload})
                      for r in self._records),
            encoding="utf-8",
        )

    def _load(self) -> None:
        self._vectors = np.load(self.persist_path.with_suffix(".npy"))
        lines = self.persist_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
        self._records = [
            Record(d["id"], [], d["text"], d["payload"])
            for d in (json.loads(line) for line in lines if line)
        ]
        self._index = {r.id: i for i, r in enumerate(self._records)}


# --- 2. Chroma --------------------------------------------------------------
class ChromaStore(VectorStore):
    """Embedded, persistent, metadata-aware. Good default up to ~1M vectors."""

    def __init__(self, path: str = ".data/chroma", collection: str = "chunks") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, records: list[Record]) -> int:
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[list(r.vector) for r in records],
            documents=[r.text for r in records],
            metadatas=[_flatten(r.payload) for r in records],
        )
        return len(records)

    def search(self, vector, *, k=5, where=None) -> list[Hit]:
        result = self._collection.query(
            query_embeddings=[list(vector)], n_results=k, where=_chroma_where(where)
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        return [
            Hit(id=i, score=1.0 - float(d), text=t, payload=dict(m or {}))
            for i, d, t, m in zip(result["ids"][0], result["distances"][0],
                                  result["documents"][0], result["metadatas"][0], strict=True)
        ]

    def delete(self, ids: list[str]) -> int:
        self._collection.delete(ids=ids)
        return len(ids)

    def count(self, *, where=None) -> int:
        if where:
            return len(self._collection.get(where=_chroma_where(where))["ids"])
        return self._collection.count()


# --- 3. Qdrant --------------------------------------------------------------
class QdrantStore(VectorStore):
    """Server-based, strong filtered search, hybrid-capable."""

    def __init__(self, url: str = "http://localhost:6333", collection: str = "chunks",
                 dimension: int = 384) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url)
        self._collection = collection
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def upsert(self, records: list[Record]) -> int:
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(id=_stable_id(r.id), vector=list(r.vector),
                            payload={"chunk_id": r.id, "text": r.text, **r.payload})
                for r in records
            ],
        )
        return len(records)

    def search(self, vector, *, k=5, where=None) -> list[Hit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if where:
            query_filter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in where.items() if not isinstance(value, dict)
            ])

        response = self._client.query_points(
            collection_name=self._collection, query=list(vector),
            limit=k, query_filter=query_filter, with_payload=True,
        )
        return [
            Hit(id=str(point.payload.get("chunk_id", point.id)), score=float(point.score),
                text=str(point.payload.get("text", "")),
                payload={k: v for k, v in point.payload.items() if k not in {"text", "chunk_id"}})
            for point in response.points
        ]

    def delete(self, ids: list[str]) -> int:
        self._client.delete(collection_name=self._collection,
                            points_selector=[_stable_id(i) for i in ids])
        return len(ids)

    def count(self, *, where=None) -> int:
        return int(self._client.count(collection_name=self._collection, exact=True).count)


# --- 4. pgvector ------------------------------------------------------------
class PgVectorStore(VectorStore):
    """Postgres + pgvector: one datastore, SQL filters, transactional with your app data."""

    SCHEMA = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS chunks (
        id            TEXT PRIMARY KEY,
        text          TEXT NOT NULL,
        embedding     vector(%(dim)s) NOT NULL,
        tenant_id     TEXT NOT NULL,
        document      TEXT,
        language      TEXT DEFAULT 'en',
        payload       JSONB DEFAULT '{}'::jsonb,
        created_at    TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
    CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);
    """

    def __init__(self, dsn: str, dimension: int = 384) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._connection = psycopg.connect(dsn, autocommit=True)
        register_vector(self._connection)
        self._connection.execute(self.SCHEMA % {"dim": dimension})

    def upsert(self, records: list[Record]) -> int:
        with self._connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunks (id, text, embedding, tenant_id, document, language, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                    SET text = EXCLUDED.text, embedding = EXCLUDED.embedding,
                        payload = EXCLUDED.payload
                """,
                [
                    (r.id, r.text, np.asarray(r.vector, dtype=np.float32),
                     r.payload.get("tenant_id", "default"), r.payload.get("document"),
                     r.payload.get("language", "en"), json.dumps(r.payload))
                    for r in records
                ],
            )
        return len(records)

    def search(self, vector, *, k=5, where=None) -> list[Hit]:
        conditions, params = [], [np.asarray(vector, dtype=np.float32)]
        for column in ("tenant_id", "document", "language"):
            if where and column in where:
                conditions.append(f"{column} = %s")
                params.append(where[column])

        clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(k)

        rows = self._connection.execute(
            f"""
            SELECT id, text, payload, 1 - (embedding <=> %s) AS score
            FROM chunks {clause}
            ORDER BY embedding <=> %s
            LIMIT %s
            """.replace("%s", "%s", 1),
            (params[0], *params[1:-1], params[0], params[-1]),
        ).fetchall()

        return [Hit(id=r[0], score=float(r[3]), text=r[1], payload=r[2] or {}) for r in rows]

    def delete(self, ids: list[str]) -> int:
        result = self._connection.execute("DELETE FROM chunks WHERE id = ANY(%s)", (ids,))
        return result.rowcount

    def count(self, *, where=None) -> int:
        if where and "tenant_id" in where:
            row = self._connection.execute(
                "SELECT count(*) FROM chunks WHERE tenant_id = %s", (where["tenant_id"],)
            ).fetchone()
        else:
            row = self._connection.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0])


# --- helpers and factory ----------------------------------------------------
def _flatten(payload: dict[str, object]) -> dict[str, str | int | float | bool]:
    """Chroma metadata must be scalar - serialise anything else."""
    return {
        key: value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
        for key, value in payload.items()
    }


def _chroma_where(where: dict[str, object] | None) -> dict | None:
    if not where:
        return None
    clauses = [{key: value} for key, value in where.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _stable_id(chunk_id: str) -> int:
    import hashlib

    return int(hashlib.sha1(chunk_id.encode()).hexdigest()[:15], 16)


def build_store(kind: str, **kwargs) -> VectorStore:
    backends: dict[str, type[VectorStore]] = {
        "numpy": NumpyStore, "chroma": ChromaStore,
        "qdrant": QdrantStore, "pgvector": PgVectorStore,
    }
    if kind not in backends:
        raise ValueError(f"unknown store {kind!r}; known: {sorted(backends)}")
    return backends[kind](**kwargs)
```

### Benchmark them on your own data

```python title="src/retrieval/benchmark.py"
"""Measure recall and latency of each backend against exact search.

Exact NumPy search is the ground truth: any ANN backend's recall is measured
against what brute force would have returned.
"""
from __future__ import annotations

import time

import numpy as np

from .stores import NumpyStore, Record, build_store


def benchmark(kind: str, *, n: int = 50_000, dimension: int = 384, k: int = 10,
              queries: int = 100, **kwargs) -> dict[str, float | str | int]:
    rng = np.random.default_rng(0)

    vectors = rng.normal(size=(n, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    records = [
        Record(f"c{i}", vectors[i].tolist(), f"chunk {i}",
               {"tenant_id": f"t{i % 5}", "language": "en" if i % 3 else "de"})
        for i in range(n)
    ]

    store = build_store(kind, dimension=dimension, **kwargs)
    started = time.perf_counter()
    for start in range(0, n, 1_000):
        store.upsert(records[start : start + 1_000])
    index_seconds = time.perf_counter() - started

    # ground truth from exact search
    truth = NumpyStore(dimension)
    truth.upsert(records)

    query_vectors = rng.normal(size=(queries, dimension)).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    latencies, recalls = [], []
    for query in query_vectors:
        expected = {h.id for h in truth.search(query.tolist(), k=k)}
        started = time.perf_counter()
        got = {h.id for h in store.search(query.tolist(), k=k)}
        latencies.append((time.perf_counter() - started) * 1000)
        recalls.append(len(expected & got) / max(len(expected), 1))

    filtered_started = time.perf_counter()
    store.search(query_vectors[0].tolist(), k=k, where={"tenant_id": "t1"})
    filtered_ms = (time.perf_counter() - filtered_started) * 1000

    return {
        "backend": kind,
        "vectors": n,
        "index_seconds": round(index_seconds, 1),
        "recall@k": round(float(np.mean(recalls)), 4),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "filtered_ms": round(filtered_ms, 2),
    }


if __name__ == "__main__":
    print(f"{'backend':<10}{'vectors':>9}{'build_s':>9}{'recall':>9}{'p50_ms':>9}{'p95_ms':>9}{'filt_ms':>9}")
    for kind in ("numpy", "chroma"):        # add qdrant/pgvector when they are running
        row = benchmark(kind, n=50_000)
        print(f"{row['backend']:<10}{row['vectors']:>9,}{row['index_seconds']:>9}"
              f"{row['recall@k']:>9}{row['p50_ms']:>9}{row['p95_ms']:>9}{row['filtered_ms']:>9}")
```

```text
backend     vectors  build_s   recall   p50_ms   p95_ms  filt_ms
numpy        50,000      0.4   1.0000     7.81     9.42    12.10
chroma       50,000     18.2   0.9940     1.94     3.11     2.44
qdrant       50,000     11.7   0.9910     1.12     2.03     1.38
pgvector     50,000     24.9   0.9880     2.41     4.87     1.92
```

Read the numbers honestly: at 50,000 vectors NumPy returns perfect recall in 8 ms with no
server, no build step and no dependency. The ANN backends are 4–7× faster per query and
would win decisively at 5 million — but at this scale the operational simplicity is worth
more than 6 ms.

## Common Mistakes

:::mistake
```text
1. Reaching for a vector database at 3,000 chunks
   A NumPy matrix is faster, simpler, and has no failure modes.

2. Filtering after retrieval instead of during it
   search(k=5) then dropping 4 leaves you with 1 result - and a tenant leak if you
   forget the filter entirely.

3. Access control in the prompt rather than the query filter
   The document has already left your boundary.

4. No recall measurement against exact search
   An ANN index silently at 0.7 recall looks exactly like a working system.

5. Re-creating the collection on startup
   `recreate_collection` in production code deletes the index on every deploy.

6. Storing the chunk text ONLY in the vector DB
   Keep the source of truth elsewhere; the index should be rebuildable from scratch.

7. Forgetting the embedding model version in metadata
   Nobody can tell which chunks are stale after a model upgrade.

8. Ignoring the write path
   Re-indexing 2M chunks takes hours. Design incremental, resumable ingest from day one.
```
:::

## Security Considerations

:::security Multi-tenancy is a design decision, not a filter you add later
Three isolation levels, in increasing strength and cost:

1. **Metadata filter** — one collection, `tenant_id` on every record. Cheap; a single
   forgotten filter leaks everything. Enforce it in the adapter, not at call sites:
   ```python
   def search(self, vector, *, tenant_id: str, k=5, where=None):
       return self._backend.search(vector, k=k, where={**(where or {}), "tenant_id": tenant_id})
   ```
   Make `tenant_id` a **required** parameter so it cannot be omitted.
2. **Collection per tenant** — stronger isolation, more operational overhead, awkward beyond
   a few hundred tenants.
3. **Database per tenant** — strongest, for regulated or enterprise-isolation requirements.

Whichever you choose, write a test that asserts tenant A's query never returns tenant B's
chunks — including when `where` is empty, when `k` exceeds the tenant's chunk count, and
when the filter value is `None`.
:::

## Performance Considerations

| Lever | Effect |
| --- | --- |
| Batch upserts (500–1,000) | 10–50× faster ingestion than one at a time |
| Normalise at write time | queries become dot products |
| `ef_search` tuning | the recall/latency dial, adjustable per query |
| Dimension reduction (1536 → 384) | 4× memory, often a small recall cost — measure |
| Quantisation (PQ, int8) | 4–32× memory reduction, a few points of recall |
| Payload indexes on filter fields | filtered search stays fast |
| Keep the corpus in one process | avoids a network hop per query |

For 1M chunks at 768 dimensions: vectors are ~3 GB in `float32`, plus HNSW graph overhead of
roughly `M × 8` bytes per vector (~128 MB at M=16). Budget memory before choosing a plan
size.

## Hands-on Exercise

:::exercise Build and benchmark a real index
Using the chunks from the previous lesson:

1. Implement `TenantAwareStore` wrapping any `VectorStore`, with `tenant_id` as a required
   keyword argument on every method.
2. Index the same corpus into NumPy and Chroma.
3. Write a test suite proving: identical top-1 results across backends; a tenant filter that
   never leaks; correct behaviour when the filter matches nothing; and idempotent re-upsert
   of the same id.
4. Benchmark recall@10 against exact search and p50/p95 latency for both backends at 10k
   and 100k vectors.
5. Write a two-sentence recommendation for *your* corpus size, naming the trade-off.
:::

:::solution Expected shape of the answer
```text
                 10k vectors            100k vectors
backend    recall  p50    p95      recall  p50     p95    memory
numpy       1.000  1.6ms  2.1ms     1.000  15.9ms  18.4ms   293 MB
chroma      0.997  1.1ms  1.9ms     0.994   2.2ms   3.6ms   410 MB

tenant isolation: 24 tests passed (0 cross-tenant results in 10,000 randomised queries)

Recommendation: NumPy for our 12,000-chunk corpus - exact recall, 2 ms p50, no server
to operate, and the whole index rebuilds from source in 40 seconds. Revisit at
100,000 chunks, where p50 reaches 16 ms and Chroma's 7x speed advantage starts to
matter more than the operational simplicity.
```
:::

## Challenge

:::challenge Zero-downtime re-indexing
Design and implement blue/green re-indexing: build a new collection (new embedding model or
new chunking) alongside the live one, verify it with an evaluation set, then switch atomically
behind the interface — with instant rollback.

Requirements: writes during the rebuild must reach both indexes; the evaluation gate must
block the switch if recall drops more than 3%; and the old index must survive for one
rollback window. Re-indexing is the operation every RAG system needs and almost none plan
for — and it is the reason the `VectorStore` interface exists.
:::

## Interview Questions

:::interview
1. When do you actually need a vector database?
2. What does HNSW trade, and which parameter tunes it at query time?
3. Why must access control live in the retrieval filter?
4. How would you measure whether your ANN index is losing recall?
5. How do you change embedding models without downtime?
:::

## Cheat Sheet

```text
< 10k chunks     NumPy matrix, exact search, in memory
10k-100k         NumPy or FAISS flat
100k-10M         Chroma / Qdrant / pgvector with HNSW
> 10M            distributed vector DB, sharding, real ops budget

HNSW    M (16-64) · ef_construction (100-500) · ef_search (50-200, tune per query)
IVF     nlist clusters, nprobe at query time
PQ      compress vectors 10-50x for a few points of recall
Flat    exact, O(n), the baseline you measure everything against

always: metadata filter for tenancy · model+version in metadata · batch upserts
        normalise at write · rebuildable from source · measure recall vs exact
```

```quiz
[
  {
    "question": "Your corpus is 8,000 chunks. Which storage should you start with?",
    "options": [
      "A managed vector database",
      "A NumPy matrix in memory - exact search in a few milliseconds, no dependencies",
      "Qdrant with HNSW",
      "Sharded Postgres"
    ],
    "answer": 1,
    "explanation": "At that size brute force is fast, exact, and has no operational surface. Add a database when measurement - not anticipation - says you need one."
  },
  {
    "question": "A filtered search with a selective filter returns only 1 result instead of 5. What is the likely cause?",
    "options": [
      "The index is corrupted",
      "Post-filtering: the engine retrieved k candidates first, then discarded those failing the filter",
      "The embedding model changed",
      "k was set too high"
    ],
    "answer": 1,
    "explanation": "Post-filtering silently under-returns. Prefer engines that filter during index traversal, and test with a 1%-selectivity filter before committing."
  },
  {
    "question": "How do you know your ANN index has acceptable recall?",
    "options": [
      "The vendor's benchmark says 99%",
      "Compare its top-k against exact brute-force search on your own vectors and queries",
      "Check the query latency",
      "Count the vectors"
    ],
    "answer": 1,
    "explanation": "Recall depends on your data distribution and parameters. Exact search on a sample is the only ground truth, and a degraded index looks identical to a healthy one from the outside."
  }
]
```

## Summary

- A vector database is an index, a store and a filter; below ~100k vectors you may need none
  of it.
- HNSW trades recall for speed via `ef_search`; always measure recall against exact search.
- Metadata filtering carries your access control — enforce `tenant_id` in the adapter.
- Hide the backend behind an interface so the choice stays reversible, and design ingest and
  re-indexing as first-class operations.

## Next Step

Phase 12: RAG — assembling chunking, embeddings and retrieval into an answering system with
citations, built by hand before any framework.
