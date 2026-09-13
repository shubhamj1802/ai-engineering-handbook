---
title: "AI Engineering Cheat Sheet and What Comes Next"
order: 4
difficulty: Production
duration: 12
badges: ["Reference", "Read once, refer often"]
summary: "One page of the commands, patterns, numbers and rules that recur across the whole handbook — plus where to go after finishing it."
prereqs: ["The Production Checklist"]
keyConcepts: ["reference", "defaults", "heuristics", "next steps"]
---

## Environment and tooling

```bash
uv init project && cd project && uv python pin 3.12
uv add pkg / uv add --dev pytest ruff mypy
uv sync --frozen          # CI: reproduce exactly
uv run python -m pkg.cli  # always run inside the project environment

uv run ruff check . --fix && uv run ruff format .
uv run mypy src --strict
uv run pytest -m "not slow" --cov=src

.env → .gitignore      .env.example → committed, blank values
```

## Python patterns that recur

```python
# never
def f(items=[]): ...                    # mutable default
except Exception: pass                   # silent failure
if not value:                            # when 0 or "" are valid
open(path)                               # no encoding
requests.get(url)                        # no timeout

# always
def f(items=None): items = items or []
except (SpecificError, OtherError) as exc: ... raise Wrapped() from exc
if value is None: ...
open(path, encoding="utf-8")
httpx.get(url, timeout=10)

@dataclass(frozen=True, slots=True)      # values you hand out
Protocol                                  # interfaces without inheritance
asyncio.gather(*coros)                    # concurrency, with a Semaphore
```

## LLM API defaults

```python
response.content        # a LIST of blocks - check block.type
response.stop_reason    # end_turn | max_tokens | tool_use | refusal
response.usage          # input, output, cache_read tokens

messages.parse(..., output_format=PydanticModel)   # structured output, validated
messages.stream(...)                                # whenever a human waits
messages.count_tokens(...)                          # before spending

retry:  RateLimit · Connection · Timeout · 5xx
never:  BadRequest · Authentication · NotFound

prompt order:  [system][tools][examples]  ← stable, cacheable
               [retrieved][history][question]  ← volatile
```

## RAG defaults

```text
chunking      structure-aware · 500-800 tokens · 10-15% overlap
              prepend "document > section" to every chunk before embedding
embeddings    one model per index, recorded in metadata · normalise at write
retrieval     k=10-20 candidates → threshold → dedupe → cap per document → top 3-6
hybrid        vector + BM25 → reciprocal rank fusion (k=60)
rerank        cross-encoder over 20-50 candidates → the biggest quality gain per ms
generation    context only · cite every claim · refusal permitted · ≤200 words
verify        every cited id ∈ retrieved set, in CODE, blocking

metrics       recall@k (the ceiling) · faithfulness · correct refusal · citation validity
debug         print the retrieved chunks and scores FIRST, always
```

## Agent defaults

```text
agent = goal + tools + bounded loop + stop conditions

caps          max_iterations 6-10 · max_cost · max_seconds · consecutive errors ·
              repeated identical calls
tools         5-12 per agent · read-only by default · errors RETURNED not raised ·
              results capped at 2-4k chars · idempotent writes
patterns      routing (cheapest win) → planning → reflection → external verification
              multi-agent only for tool separation, distinct models, or parallelism
measure       completion RATE over N runs · steps · cost · safety violations (zero)
```

## Numbers worth remembering

```text
TOKENS       ~4 chars / ~0.75 words per token (English); code and CJK are denser
             output tokens cost ~5x input tokens and dominate latency

ATTENTION    O(n²) in sequence length: 10x context ≈ 100x attention compute
PROMPT CACHE 50-90% of input cost in agent loops; requires a byte-stable prefix

RETRIEVAL    < 10k chunks: a NumPy matrix beats a vector database
             1M chunks at 768 dims float32 ≈ 3 GB
             hybrid adds ~10 points of recall on identifier queries
             reranking adds ~10 points of recall and ~0.15 MRR for 40-80 ms

AGENTS       10-30x a workflow in cost and latency
             3 stages at 90% reliability = 73% end to end
             supervisor coordination overhead: often 30% of spend

ML           always compute the trivial baseline first
             3 chunk sizes x 2 overlaps = the 20 minutes that sets your quality ceiling
```

## The rules that do not change

```text
1.  Rules that must hold, hold in CODE. Prompts are guidance.
2.  Retrieval quality bounds answer quality. Measure it separately.
3.  An agent's loop must terminate for reasons your code controls.
4.  Tool errors are observations; exceptions end runs.
5.  Least privilege beats detection. An absent tool cannot be misused.
6.  Access control belongs in the query filter, never in the prompt.
7.  Verify citations deterministically. Block what you cannot verify.
8.  Log metadata, not content.
9.  Measure p95, not the mean; measure cost per request, not per token.
10. Make the expensive path rarer before making it better.
11. Build it by hand once before adopting the framework.
12. An evaluation set that never grows is measuring the past.
```

## Choosing, quickly

| Question | Answer |
| --- | --- |
| Rules, classifier or LLM? | cheapest that works; usually a layered hybrid |
| Workflow or agent? | can you draw the flowchart in advance? then workflow |
| Which vector store? | < 10k chunks: NumPy. Have Postgres? pgvector. Scale: Qdrant |
| Which framework? | chain → LangChain · control flow + HITL → LangGraph · role pipeline → CrewAI · simple + fast → none |
| Fine-tune or RAG? | facts → RAG. Form and behaviour → fine-tune. Usually RAG |
| Single or multi-agent? | single, until measurement says otherwise |
| Bigger model or better retrieval? | retrieval, nearly always, at a fraction of the cost |
| Where do I start optimising cost? | cache → route → retrieval size → model tier |

## What to do next

### 1. Ship something real (the only essential step)

Pick a problem you personally have, build it, and put it in front of five users. Everything
in this handbook was written against the failures that appear when real people use a system:
the questions you did not anticipate, the documents that extract badly, the cost that
surprises you.

### 2. Deepen in one direction

| Direction | What to study |
| --- | --- |
| **Retrieval** | learned sparse retrieval, ColBERT-style late interaction, query understanding, evaluation methodology |
| **Agents** | planning algorithms, environment design, RL for tool use, formal verification of action safety |
| **Model layer** | fine-tuning (LoRA/QLoRA), quantisation, serving (vLLM), distillation, inference economics |
| **Evaluation** | judge calibration, statistical significance in small evaluation sets, human annotation design |
| **Platform** | multi-region deployment, cost engineering at scale, model governance, compliance |

Depth in one of these plus the breadth from this handbook is a strong, uncommon profile.

### 3. Read primary sources

Skip summaries of summaries. Read the paper, the release notes, the SDK source:

- "Attention Is All You Need" (2017) — the architecture
- The RAG paper (2020) and the "lost in the middle" study — why retrieval design matters
- Scaling-law papers — why models got better
- Your framework's source code — the fastest way to stop treating it as magic
- Provider changelogs — the field moves faster than any book

### 4. Build the habits that compound

```text
weekly    read one paper or one framework changelog properly
          add every production failure to an evaluation set
monthly   run a dependency upgrade with the tests as your gate
          review cost per request and act on the biggest line
quarterly re-run your evaluations against a newer model; the best default changes
          delete something: a tool nobody uses, a prompt rule doing nothing
```

### 5. Teach it

Write up one thing you got wrong and how you found out. The measurement you had to build to
discover it is almost always more useful to other people than the fix.

## A closing thought

The technology on these pages will change — model names, framework APIs, which vector store
is fashionable. What will not change is the engineering underneath:

> Understand the problem. Choose the simplest thing that could work. Measure whether it does.
> Bound what it can do wrong. Make the failure modes visible. Ship it, watch it, and fix what
> the measurements tell you to fix.

That is what AI engineering is, and it is what you now know how to do.

## Summary

- One page of defaults: environment, Python, LLM calls, RAG, agents, and the numbers.
- Twelve rules that survive every framework change.
- Next: ship something real, then go deep in one direction.
- Read primary sources, keep an evaluation set that grows, and re-check your defaults
  quarterly.

## Next Step

Back to [the roadmap](/roadmap) — or start your own project. That is the actual next step.
