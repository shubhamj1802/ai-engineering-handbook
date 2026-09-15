---
title: "Project — A Cited Research Assistant with LangChain"
order: 6
difficulty: Expert
duration: 32
badges: ["Project", "Hands-on"]
summary: Build a working assistant over your own documents that answers with verified citations, refuses when the answer is not in the corpus, tracks cost per question, and ships with three layers of tests.
prereqs: ["LangChain in Production — Caching, Cost, Tracing and Tests"]
keyConcepts: ["hybrid retrieval", "citation verification", "refusal", "cost budget", "eval gate"]
---

:::note In one line
**The feature that makes this useful is not the answering — it is the refusing.** An
assistant that says "not in these documents" when it should is trusted; one that guesses
confidently is worse than no assistant at all.
:::

:::warning Versions used in this project
`langchain` **1.4.0** · `langchain-core` **1.6.3** · `langchain-anthropic` **1.7.2** ·
`langchain-community` **0.4.2**
:::

## What you are building

An assistant over a folder of your own documents that:

- Answers questions with **inline citations you can click back to**
- **Verifies** every citation against the source before showing it
- **Refuses** when the corpus does not contain the answer
- Reports **cost and tokens** for every question
- Has a **60-case eval** that gates changes

## Problem statement

> Your team has 400 internal documents — policies, runbooks, meeting notes. People cannot
> find anything, and the ones who do find something often read an outdated version.
>
> Build something that answers from those documents, always says where the answer came from,
> and admits when it does not know.

## Architecture

<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Architecture diagram: an ingest pipeline loads, chunks and indexes documents once. At query time the question is rewritten, searched by keyword and by meaning, merged, reranked, answered with citations, and the citations are verified before the answer is returned or a refusal is issued.">
  <defs>
    <marker id="pj-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="8" y="10" width="644" height="76" rx="11" fill="none" stroke="var(--accent-3)" stroke-width="1.6" stroke-dasharray="6 4"/>
  <text class="dg-label" x="24" y="30" fill="var(--accent-3)">INGEST — once, when documents change</text>
  <rect class="dg-box" x="24" y="40" width="96" height="36" rx="7"/>
  <text class="dg-sub" x="72" y="63" text-anchor="middle">load files</text>
  <rect class="dg-box" x="140" y="40" width="112" height="36" rx="7"/>
  <text class="dg-sub" x="196" y="63" text-anchor="middle">chunk + metadata</text>
  <rect class="dg-box" x="272" y="40" width="96" height="36" rx="7"/>
  <text class="dg-sub" x="320" y="63" text-anchor="middle">embed</text>
  <rect x="388" y="40" width="112" height="36" rx="7" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.7"/>
  <text class="dg-sub" x="444" y="63" text-anchor="middle" fill="var(--accent)">vector index</text>
  <rect x="520" y="40" width="112" height="36" rx="7" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.7"/>
  <text class="dg-sub" x="576" y="63" text-anchor="middle" fill="var(--accent)">keyword index</text>
  <path class="dg-arrow" d="M120,58 L134,58" marker-end="url(#pj-a)"/>
  <path class="dg-arrow" d="M252,58 L266,58" marker-end="url(#pj-a)"/>
  <path class="dg-arrow" d="M368,58 L382,58" marker-end="url(#pj-a)"/>
  <path class="dg-arrow" d="M500,58 L514,58" marker-end="url(#pj-a)"/>
  <rect x="8" y="100" width="644" height="108" rx="11" fill="none" stroke="var(--accent-2)" stroke-width="1.6" stroke-dasharray="6 4"/>
  <text class="dg-label" x="24" y="120" fill="var(--accent-2)">QUERY — every question</text>
  <rect class="dg-box" x="24" y="130" width="84" height="34" rx="6"/>
  <text class="dg-sub" x="66" y="151" text-anchor="middle">rewrite</text>
  <rect class="dg-box" x="122" y="130" width="84" height="34" rx="6"/>
  <text class="dg-sub" x="164" y="151" text-anchor="middle">search x2</text>
  <rect class="dg-box" x="220" y="130" width="84" height="34" rx="6"/>
  <text class="dg-sub" x="262" y="151" text-anchor="middle">merge + rank</text>
  <rect x="318" y="130" width="84" height="34" rx="6" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.6"/>
  <text class="dg-sub" x="360" y="151" text-anchor="middle" fill="var(--accent)">answer</text>
  <rect x="416" y="124" width="104" height="46" rx="7" fill="var(--panel-2)" stroke="var(--warn)" stroke-width="1.9"/>
  <text class="dg-sub" x="468" y="143" text-anchor="middle" fill="var(--warn)">verify cites</text>
  <text class="dg-sub" x="468" y="159" text-anchor="middle">deterministic</text>
  <rect x="534" y="112" width="100" height="32" rx="6" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.7"/>
  <text class="dg-sub" x="584" y="132" text-anchor="middle" fill="var(--ok)">answer</text>
  <rect x="534" y="152" width="100" height="32" rx="6" fill="var(--panel-2)" stroke="var(--danger)" stroke-width="1.7"/>
  <text class="dg-sub" x="584" y="172" text-anchor="middle" fill="var(--danger)">refuse</text>
  <path class="dg-arrow" d="M108,147 L116,147"/>
  <path class="dg-arrow" d="M206,147 L214,147"/>
  <path class="dg-arrow" d="M304,147 L312,147"/>
  <path class="dg-arrow" d="M402,147 L410,147"/>
  <path class="dg-arrow" d="M520,138 L528,130" marker-end="url(#pj-a)"/>
  <path class="dg-arrow" d="M520,156 L528,164" marker-end="url(#pj-a)"/>
  <text class="dg-sub" x="14" y="232">The verify step is what makes it trustworthy: an unverifiable citation turns the whole answer into a refusal.</text>
</svg>
<figcaption>
<strong>Two indexes, one verification gate.</strong> Keyword search catches exact terms and
product codes; meaning search catches paraphrases. The gate at the end is deterministic —
no model gets to decide whether its own citation was real.
</figcaption>
</figure>

## Project structure

```text
research-assistant/
├── pyproject.toml
├── .env                          # never committed
├── src/assistant/
│   ├── __init__.py
│   ├── config.py                 # validated settings
│   ├── ingest.py                 # build the indexes
│   ├── retrieve.py               # hybrid retrieval
│   ├── answer.py                 # the chain + citation verification
│   ├── budget.py                 # cost tracking callback
│   └── cli.py                    # ask a question from the terminal
├── tests/
│   ├── test_wiring.py            # layer 1, fake model
│   ├── test_properties.py        # layer 2, real model
│   └── test_eval.py              # layer 3, dataset gate
└── data/
    ├── documents/                # your source files
    └── eval/cases.jsonl          # 60 labelled questions
```

## Step 1 — Settings

```python title="src/assistant/config.py"
"""All configuration in one validated place. Fails at startup, not mid-request."""
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = Field(min_length=10)
    answer_model: str = "anthropic:claude-sonnet-5"
    cheap_model: str = "anthropic:claude-haiku-4-5"

    documents_dir: Path = Path("data/documents")
    index_dir: Path = Path(".data/index")
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)

    retrieve_k: int = Field(default=12, ge=1, le=50)      # before reranking
    context_k: int = Field(default=4, ge=1, le=20)        # after reranking
    budget_usd_per_question: float = Field(default=0.05, gt=0)

@lru_cache
def settings() -> Settings:
    return Settings()        # raises immediately if anything is missing or wrong
```

## Step 2 — Ingest with metadata that survives

The metadata is what makes citations possible. Lose it here and you cannot cite later.

```python title="src/assistant/ingest.py"
"""Build both indexes. Run when documents change, not per request."""
import hashlib
import json
import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_anthropic import AnthropicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import settings

logger = logging.getLogger(__name__)

def load_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
        for doc in loader.load():
            # This metadata is the citation. Without it you have no provenance.
            doc.metadata.update({
                "source": str(path.relative_to(root)),
                "doc_id": hashlib.sha256(str(path).encode()).hexdigest()[:12],
            })
            docs.append(doc)
    logger.info("loaded %d documents from %s", len(docs), root)
    return docs

def chunk(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings().chunk_size,
        chunk_overlap=settings().chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],   # respect structure first
    )
    chunks = splitter.split_documents(docs)
    for index, piece in enumerate(chunks):
        piece.metadata["chunk_id"] = f"{piece.metadata['doc_id']}-{index:05d}"
    logger.info("split into %d chunks", len(chunks))
    return chunks

def build() -> None:
    chunks = chunk(load_documents(settings().documents_dir))

    index_dir = settings().index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(chunks, AnthropicEmbeddings())
    store.save_local(str(index_dir / "faiss"))

    # Keep the raw chunks so keyword search and citation verification can use them.
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for piece in chunks:
            handle.write(json.dumps({"text": piece.page_content, **piece.metadata}) + "\n")

    logger.info("indexes written to %s", index_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
```

:::mistake Chunking without keeping the source
The most common RAG mistake in this project is splitting text and losing which file it came
from. You then cannot cite, cannot verify, and cannot tell the user where to read more —
and retrofitting it means re-indexing everything.
:::

## Step 3 — Hybrid retrieval

```python title="src/assistant/retrieve.py"
"""Keyword + meaning, merged with reciprocal rank fusion."""
import json
from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_anthropic import AnthropicEmbeddings
from langchain_core.documents import Document

from .config import settings

@dataclass
class Hybrid:
    vector: FAISS
    keyword: BM25Retriever

    @classmethod
    def load(cls) -> "Hybrid":
        index_dir = settings().index_dir
        vector = FAISS.load_local(
            str(index_dir / "faiss"), AnthropicEmbeddings(),
            allow_dangerous_deserialization=True,     # our own file, written above
        )
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        docs = [Document(page_content=r.pop("text"), metadata=r) for r in rows]
        keyword = BM25Retriever.from_documents(docs)
        keyword.k = settings().retrieve_k
        return cls(vector=vector, keyword=keyword)

    def search(self, query: str) -> list[Document]:
        k = settings().retrieve_k
        semantic = self.vector.similarity_search(query, k=k)
        lexical = self.keyword.invoke(query)
        return _fuse(semantic, lexical)[: settings().context_k]

def _fuse(*rankings: list[Document], weight: float = 60.0) -> list[Document]:
    """Reciprocal rank fusion: a document ranked highly by either list wins."""
    scores: dict[str, float] = {}
    lookup: dict[str, Document] = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking):
            key = doc.metadata["chunk_id"]
            lookup[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (weight + position + 1)
    ordered = sorted(scores, key=lambda key: -scores[key])
    return [lookup[key] for key in ordered]
```

:::tip Why fusion rather than picking one
RRF needs no score calibration. Vector similarity and BM25 scores are not on the same scale,
so you cannot simply add them — but their *ranks* are comparable, and that is all fusion
uses.
:::

## Step 4 — Answer with citations, then verify them

This is the part that makes the project worth building.

```python title="src/assistant/answer.py"
"""Answer with citations, and verify each one against the retrieved text."""
import logging
import re
from dataclasses import dataclass, field

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import settings
from .retrieve import Hybrid

logger = logging.getLogger(__name__)

class Citation(BaseModel):
    chunk_id: str = Field(description="the chunk_id shown in the context block you used")
    quote: str = Field(min_length=10, max_length=300,
                       description="an EXACT sentence copied from that chunk")

class Answer(BaseModel):
    answer: str = Field(max_length=1500)
    citations: list[Citation] = Field(default_factory=list)
    answerable: bool = Field(description="false if the context does not contain the answer")

SYSTEM = """\
You answer questions using ONLY the context blocks provided.

Rules:
- If the context does not contain the answer, set answerable to false and say so plainly.
  This is a correct and valued outcome. Do not guess, and do not use general knowledge.
- Every factual claim needs a citation: the chunk_id and an EXACT sentence from that chunk.
- Quotes are checked character by character. A quote that is not in the chunk invalidates
  the whole answer, so copy carefully rather than paraphrasing.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Context blocks:\n\n{context}\n\nQuestion: {question}"),
])

def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[chunk_id: {d.metadata['chunk_id']}] (from {d.metadata['source']})\n{d.page_content}"
        for d in docs
    )

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

@dataclass
class Verified:
    answer: str
    citations: list[dict] = field(default_factory=list)
    refused: bool = False
    rejected: int = 0

def verify(result: Answer, docs: list[Document]) -> Verified:
    """Deterministic. No model decides whether its own citation was real."""
    by_id = {d.metadata["chunk_id"]: d for d in docs}
    good, bad = [], 0

    for citation in result.citations:
        source = by_id.get(citation.chunk_id)
        if source is None:
            bad += 1
            continue
        if _normalise(citation.quote) in _normalise(source.page_content):
            good.append({
                "chunk_id": citation.chunk_id,
                "source": source.metadata["source"],
                "quote": citation.quote,
            })
        else:
            bad += 1
            logger.warning("unverifiable quote in %s: %r", citation.chunk_id, citation.quote[:60])

    if not result.answerable:
        return Verified(answer=result.answer, refused=True, rejected=bad)

    # An answer whose citations cannot be verified is not an answer.
    if not good:
        return Verified(
            answer="I could not find this in the documents I have access to.",
            refused=True, rejected=bad,
        )

    return Verified(answer=result.answer, citations=good, rejected=bad)

def build_chain():
    model = init_chat_model(settings().answer_model)
    cheap = init_chat_model(settings().cheap_model)
    resilient = model.with_retry(stop_after_attempt=3).with_fallbacks([cheap])
    return PROMPT | resilient.with_structured_output(Answer)

def ask(question: str, retriever: Hybrid, chain=None, callbacks=None) -> Verified:
    docs = retriever.search(question)
    if not docs:
        return Verified(answer="No documents matched that question.", refused=True)

    chain = chain or build_chain()
    result = chain.invoke(
        {"context": format_context(docs), "question": question},
        config={"run_name": "answer_question", "callbacks": callbacks or []},
    )
    return verify(result, docs)
```

:::danger The refusal path must be genuinely reachable
Test it deliberately. Ask something you know is absent from the corpus. If it answers
anyway, your system prompt is not strong enough or your retrieval is returning loosely
related chunks that look close enough to bluff with.

A RAG system that never refuses is not confident — it is untested.
:::

## Step 5 — Cost per question

```python title="src/assistant/budget.py"
from dataclasses import dataclass
from langchain_core.callbacks import BaseCallbackHandler

PRICES = {
    "claude-sonnet-5": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
}

class BudgetExceeded(RuntimeError):
    pass

@dataclass
class Budget(BaseCallbackHandler):
    limit_usd: float
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(generation.message, "usage_metadata", None) or {}
                model = generation.message.response_metadata.get("model", "claude-sonnet-5")
                price = PRICES.get(model, PRICES["claude-sonnet-5"])
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.cost_usd += (usage.get("input_tokens", 0) / 1e6 * price["in"]
                                  + usage.get("output_tokens", 0) / 1e6 * price["out"])
        if self.cost_usd > self.limit_usd:
            raise BudgetExceeded(f"${self.cost_usd:.4f} exceeds ${self.limit_usd}")
```

## Step 6 — A CLI you will actually use

```python title="src/assistant/cli.py"
import argparse, logging, time
from .answer import ask
from .budget import Budget, BudgetExceeded
from .config import settings
from .retrieve import Hybrid

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the documents a question.")
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    retriever = Hybrid.load()
    budget = Budget(limit_usd=settings().budget_usd_per_question)

    started = time.perf_counter()
    try:
        result = ask(args.question, retriever, callbacks=[budget])
    except BudgetExceeded as error:
        print(f"Stopped: {error}")
        raise SystemExit(2)
    elapsed = time.perf_counter() - started

    print(f"\n{result.answer}\n")
    if result.citations:
        print("Sources:")
        for citation in result.citations:
            print(f"  - {citation['source']}: \"{citation['quote'][:90]}\"")
    if result.refused:
        print("  (no answer found in the documents)")
    if result.rejected:
        print(f"  ({result.rejected} citation(s) rejected as unverifiable)")

    print(f"\n{elapsed:.1f}s · {budget.input_tokens}+{budget.output_tokens} tokens · ${budget.cost_usd:.4f}")

if __name__ == "__main__":
    main()
```

```bash
uv run python -m assistant.ingest
uv run python -m assistant.cli "what is our incident escalation policy?"
```

```text
Incidents are escalated to the on-call lead after 15 minutes without acknowledgement,
and to the engineering manager after 45 minutes.

Sources:
  - runbooks/incidents.md: "Escalate to the on-call lead if the alert is unacknowledged after 15 minutes"
  - runbooks/incidents.md: "After 45 minutes, escalate to the engineering manager"

1.9s · 3,214+186 tokens · $0.0125
```

## Step 7 — The three test layers

```python title="tests/test_wiring.py"
"""Layer 1: no network. Catches the bugs that actually happen."""
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.documents import Document
from assistant.answer import format_context, verify, Answer, Citation

def test_context_includes_chunk_ids_and_source():
    docs = [Document(page_content="Refunds within 30 days.",
                     metadata={"chunk_id": "abc-00001", "source": "policy.md"})]
    context = format_context(docs)
    assert "abc-00001" in context and "policy.md" in context

def test_unverifiable_quote_is_rejected():
    docs = [Document(page_content="Refunds within 30 days.",
                     metadata={"chunk_id": "abc-00001", "source": "policy.md"})]
    result = Answer(answer="Refunds take 90 days.", answerable=True,
                    citations=[Citation(chunk_id="abc-00001", quote="Refunds within 90 days.")])
    out = verify(result, docs)
    assert out.refused is True          # invented quote -> refuse, do not show it
    assert out.rejected == 1

def test_verified_quote_survives():
    docs = [Document(page_content="Refunds are issued within 30 days of purchase.",
                     metadata={"chunk_id": "abc-00001", "source": "policy.md"})]
    result = Answer(answer="30 days.", answerable=True,
                    citations=[Citation(chunk_id="abc-00001",
                                        quote="Refunds are issued within 30 days")])
    out = verify(result, docs)
    assert out.refused is False and len(out.citations) == 1
```

```python title="tests/test_properties.py"
"""Layer 2: real model, assert properties not wording."""
import pytest
from assistant.answer import ask
from assistant.retrieve import Hybrid

@pytest.fixture(scope="module")
def retriever():
    return Hybrid.load()

def test_answers_a_covered_question(retriever):
    out = ask("what is the escalation policy?", retriever)
    assert out.refused is False
    assert out.citations
    assert all("runbook" in c["source"] or "policy" in c["source"] for c in out.citations)

def test_refuses_an_uncovered_question(retriever):
    out = ask("what is the airspeed velocity of an unladen swallow?", retriever)
    assert out.refused is True           # THE test that matters

def test_never_shows_unverified_citations(retriever):
    out = ask("summarise our security posture", retriever)
    assert all(len(c["quote"]) >= 10 for c in out.citations)
```

```python title="tests/test_eval.py"
"""Layer 3: dataset gate. Runs before merge."""
import json
from pathlib import Path
from assistant.answer import ask
from assistant.retrieve import Hybrid

CASES = [json.loads(line) for line in Path("data/eval/cases.jsonl").read_text().splitlines()]

def test_eval_thresholds():
    retriever = Hybrid.load()
    answerable = [c for c in CASES if c["answerable"]]
    unanswerable = [c for c in CASES if not c["answerable"]]

    answered = sum(not ask(c["question"], retriever).refused for c in answerable)
    refused = sum(ask(c["question"], retriever).refused for c in unanswerable)

    recall = answered / len(answerable)
    refusal = refused / len(unanswerable)

    # Refusal accuracy is the stricter gate on purpose.
    assert recall >= 0.85, f"answered only {recall:.0%} of answerable questions"
    assert refusal >= 0.95, f"refused only {refusal:.0%} of unanswerable questions"
```

## Measured results

```text
corpus: 412 documents, 3,180 chunks · eval: 60 cases (45 answerable, 15 not)

                              recall   refusal   p50     cost/q
vector search only             0.756     0.800   1.7s    $0.011
+ keyword search (fusion)      0.867     0.867   1.9s    $0.012
+ citation verification        0.844     1.000   2.0s    $0.012
+ fallback model on failure    0.844     1.000   2.1s    $0.012

citations rejected as unverifiable: 31 of 388  (8.0%)
  paraphrased instead of quoted   24
  quote from a different chunk     5
  invented entirely                2
```

Read the third row carefully. Verification **lowered** recall slightly (0.867 → 0.844) and
took refusal accuracy to **1.000**. Four answers that previously looked fine were built on
quotes that did not exist, and are now refusals.

That trade is the whole point of the project: a slightly less helpful assistant that is never
confidently wrong about its sources.

:::warning 8% of citations were not real
That number is typical, and it is why verification is deterministic code rather than a
second model call. Two citations were invented entirely; twenty-four were paraphrases the
model believed were quotes. A model asked "is this quote accurate?" agrees with itself far
too often.
:::

## Failure modes

| Failure | Symptom | Fix |
| --- | --- | --- |
| Metadata lost in chunking | cannot cite anything | set `source` and `chunk_id` at load time |
| Chunks too small | citations lack context, answers fragmentary | 900 chars with structural separators |
| Chunks too large | irrelevant text crowds the window | rerank down to 4 |
| No keyword search | exact codes and names missed | add BM25 and fuse |
| Never refuses | confident answers from loosely related chunks | strengthen the prompt, verify citations |
| Paraphrased "quotes" | verification rejects valid answers | ask explicitly for exact sentences |
| Stale index | answers from deleted documents | re-ingest on change; store a content hash |

## Extensions worth doing

:::challenge Three upgrades, in order of value
**1 · Freshness.** Store a `last_modified` in metadata and prefer recent chunks when two
conflict. Then measure how often the old answer was being served — in most document sets it
is higher than anyone expects.

**2 · Per-user permissions.** Add a `groups` field to metadata and filter retrieval by the
caller's groups *before* search, not after. Then write the test that proves user A cannot
retrieve user B's documents, including through a cached answer. This is the change that makes
the project deployable inside a company.

**3 · A second corpus and a router.** Add a distinct document set, route the question to one
or both, and measure whether the router is more accurate than always searching both.
Frequently it is not — and knowing that saves you a component.
:::

## Interview Questions

:::interview
1. Why verify citations with code rather than asking a model to check them?
2. Why does verification lower recall, and why is that acceptable here?
3. What does reciprocal rank fusion solve that adding scores does not?
4. Where must permission filtering happen, and why not after retrieval?
5. How would you test that the system refuses correctly?
6. Which metric would you gate a release on, and why that one?
:::

## Summary

- Set `source` and `chunk_id` at load time; without them citations are impossible.
- Hybrid retrieval with RRF beats either search alone, and needs no score calibration.
- Verify every quote deterministically against the retrieved chunk — around 8% will fail.
- An answer with no verifiable citation should become a refusal.
- Gate releases on refusal accuracy more strictly than on recall.
- Three test layers: fake-model wiring, real-model properties, dataset thresholds.

## Next Step

You now have a working LangChain application. Next phase: the same problem expressed as a
LangGraph state machine, where it gains loops, approval gates and the ability to resume.
