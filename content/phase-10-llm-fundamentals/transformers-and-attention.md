---
title: Transformers and Attention
order: 1
difficulty: Advanced
duration: 18
badges: ["Deep dive", "Theory"]
summary: "Self-attention, positional encoding, encoder vs decoder and the KV cache — explained and implemented, to the depth that changes how you build with LLMs."
prereqs: ["Neural Networks from Scratch", "Tokens, Context, Temperature and Hallucination"]
keyConcepts: ["self-attention", "Q/K/V", "multi-head", "positional encoding", "KV cache"]
---

:::note In one line
**Attention lets every word look at every other word and decide what matters.** That single mechanism is what made modern LLMs possible.
:::

## Why this matters

You will not implement a transformer at work. But attention explains directly why context
windows are expensive, why "lost in the middle" happens, why prompt caching works, why the
first token is slow and the rest are fast, and why long conversations degrade. Every one of
those is a decision you make weekly.

## Mental Model

Attention lets **every word look at every other word** and decide which ones matter for
understanding it.
<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Diagram: in the sentence the bank raised rates, the word bank attends strongly to rates and weakly to the, which is how the model decides bank means a financial institution rather than a riverside.">
  <text class="dg-sub" x="14" y="22">When the model processes the word</text>
  <rect x="236" y="8" width="70" height="24" rx="5" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.8"/>
  <text class="dg-mono" x="271" y="25" text-anchor="middle" fill="var(--accent)" style="font-size:11.5px">bank</text>
  <text class="dg-sub" x="316" y="22">it weighs every other word:</text>
  <rect x="40" y="76" width="74" height="34" rx="6" class="dg-box"/>
  <text class="dg-mono" x="77" y="98" text-anchor="middle" style="font-size:11.5px">The</text>
  <rect x="142" y="76" width="84" height="34" rx="6" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2.2"/>
  <text class="dg-mono" x="184" y="98" text-anchor="middle" fill="var(--accent)" style="font-size:11.5px">bank</text>
  <rect x="254" y="76" width="92" height="34" rx="6" class="dg-box"/>
  <text class="dg-mono" x="300" y="98" text-anchor="middle" style="font-size:11.5px">raised</text>
  <rect x="374" y="76" width="86" height="34" rx="6" class="dg-box"/>
  <text class="dg-mono" x="417" y="98" text-anchor="middle" style="font-size:11.5px">rates</text>
  <path d="M176,110 Q120,150 82,114" fill="none" stroke="var(--text-muted)" stroke-width="1" opacity="0.7"/>
  <text class="dg-sub" x="112" y="158" fill="var(--text-muted)">0.05</text>
  <path d="M192,110 Q240,152 296,114" fill="none" stroke="var(--accent)" stroke-width="2.4"/>
  <text class="dg-sub" x="238" y="168" fill="var(--accent)">0.35</text>
  <path d="M196,110 Q300,196 414,114" fill="none" stroke="var(--accent)" stroke-width="3.6"/>
  <text class="dg-sub" x="330" y="196" fill="var(--accent)">0.55  strongest</text>
  <rect x="482" y="62" width="164" height="90" rx="9" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.8"/>
  <text class="dg-sub" x="564" y="86" text-anchor="middle" fill="var(--ok)">so "bank" here means</text>
  <text class="dg-sub" x="564" y="106" text-anchor="middle" fill="var(--ok)">a financial one</text>
  <text class="dg-sub" x="564" y="130" text-anchor="middle">not a riverbank</text>
  <text class="dg-sub" x="14" y="228">Thicker line = more attention. The weights are learned, not written by anyone.</text>
  <text class="dg-sub" x="14" y="244">Swap "rates" for "river" and the weights shift, and so does the meaning.</text>
</svg>
<figcaption>
<strong>This is why word order and context matter so much.</strong> Older models read words
mostly in isolation; attention lets a word be understood by the company it keeps.
</figcaption>
</figure>

Attention answers one question for every token: **"which other tokens should I look at, and
how much?"**

```text
"The chunk that the retriever returned was irrelevant"
                                        ↑
When processing "irrelevant", attention weights:
   chunk      0.31   ← what was irrelevant
   retriever  0.22   ← who returned it
   returned   0.18
   The        0.04
   ...

Output for that position = weighted sum of all token representations.
```

Three learned projections of each token do the work:

| | Name | Role | Analogy |
| --- | --- | --- | --- |
| **Q** | Query | what this token is looking for | your search query |
| **K** | Key | what each token offers | a document's index terms |
| **V** | Value | what each token contributes | the document's content |

```text
attention(Q, K, V) = softmax(QKᵀ / √d) V
                     └──────┬──────┘
                      how much to look    then take that weighted mix of V
```

## Core Concepts

### Why √d

`QKᵀ` produces dot products whose magnitude grows with dimension `d`. Large values push
softmax into a near-one-hot distribution, whose gradients vanish. Dividing by `√d` keeps the
variance around 1 so training stays stable. A single line with a real reason behind it.

### Multi-head attention

One attention operation learns one kind of relationship. Running several in parallel — each
with its own smaller Q/K/V projections — lets different heads specialise: syntax, coreference
("it" → which noun), position, topic.

```text
d_model = 512, 8 heads → each head works in 64 dimensions
concatenate the 8 outputs → project back to 512
```

Cost is the same as one full-width head; the benefit is diversity of relationships.

### Positional encoding

Attention is permutation-invariant: shuffle the tokens and the weighted sums are identical.
Something must encode order.

- **Sinusoidal** (original): fixed sine/cosine patterns added to embeddings.
- **Learned absolute**: a trainable vector per position (BERT, GPT-2).
- **RoPE — rotary** (most current models): rotates Q and K by an angle proportional to
  position, so attention scores depend on *relative* distance. This is what makes context
  extension techniques possible.
- **ALiBi**: adds a distance-proportional penalty to attention scores.

### Encoder, decoder, and what you use

```text
ENCODER-ONLY (BERT, embedding models)
  bidirectional: every token sees every other
  → classification, NER, EMBEDDINGS ← this is what Phase 11 uses

DECODER-ONLY (GPT, Claude, Llama — every chat LLM)
  causal mask: token i sees only tokens ≤ i
  → text generation, in-context learning

ENCODER-DECODER (T5, translation models)
  encode the input fully, decode the output causally
  → translation, summarisation with distinct input/output
```

The **causal mask** is the whole difference for generation: it sets attention to future
positions to −∞ before the softmax, so the model cannot look ahead. That is what makes
next-token prediction a valid training objective over an entire document at once.

### A transformer block

```text
x → LayerNorm → Multi-Head Attention → + x        (residual)
  → LayerNorm → Feed-Forward (4x wide) → + x      (residual)
```

Residual connections let gradients flow through dozens of layers. The feed-forward network —
two linear layers with a GELU between, typically 4× the model width — holds most of the
parameters and is where much of the factual knowledge is thought to live.

A "70B model" is this block repeated 80 times with a width of about 8,192.

### The cost that matters to you

```text
Attention: every token attends to every token  →  O(n²) in sequence length

n = 1,000   →   1,000,000 attention scores per head per layer
n = 10,000  → 100,000,000                         100x for 10x the tokens
```

This is why doubling your context more than doubles cost and latency, and why trimming
retrieved context is often the cheapest performance win available.

### KV cache and prompt caching

When generating token *n+1*, the K and V vectors for tokens 1..n are unchanged. Caching them
turns each new token from O(n²) into O(n).

```text
PREFILL   process the whole prompt once     → slow, compute-bound, O(n²)   ← time to first token
DECODE    generate token by token with the  → fast, memory-bound, O(n)     ← tokens per second
          cached K/V
```

That is exactly why the first token takes 800 ms and the next 200 arrive in a couple of
seconds.

**Prompt caching** at the provider extends this across requests: an identical prefix (system
prompt + tool definitions) can reuse its computed K/V, typically at a 90% discount. It is why
you put stable content first and variable content last.

:::tip Prompt ordering is a cost decision
```text
[system prompt][tool definitions][few-shot examples]   ← stable, cacheable
[retrieved context][conversation history][question]     ← variable
```
Reorder these and you lose the cache. In an agent loop with a 4,000-token stable prefix and
twenty model calls, this is the difference between $0.24 and $0.03 per run.
:::

## Minimal Example

Self-attention in NumPy — the entire mechanism in thirty lines.

```python title="attention_from_scratch.py"
"""Scaled dot-product attention, multi-head attention, and a causal mask."""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)        # numerical stability
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, *, causal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Q, K, V: (seq_len, d_k). Returns (output, attention_weights)."""
    d_k = Q.shape[-1]

    scores = Q @ K.T / np.sqrt(d_k)                      # (seq, seq)

    if causal:
        # -inf above the diagonal: position i cannot see positions > i
        mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores, axis=-1)                   # each row sums to 1
    return weights @ V, weights


class MultiHeadAttention:
    def __init__(self, d_model: int = 64, n_heads: int = 4, seed: int = 0) -> None:
        assert d_model % n_heads == 0, "d_model must divide evenly among heads"
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2 / d_model)

        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, d_model))
        self.Wv = rng.normal(0, scale, (d_model, d_model))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def __call__(self, X: np.ndarray, *, causal: bool = True) -> tuple[np.ndarray, np.ndarray]:
        seq_len = X.shape[0]

        Q = (X @ self.Wq).reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = (X @ self.Wk).reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = (X @ self.Wv).reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        outputs, all_weights = [], []
        for head in range(self.n_heads):
            out, weights = scaled_dot_product_attention(Q[head], K[head], V[head], causal=causal)
            outputs.append(out)
            all_weights.append(weights)

        concatenated = np.concatenate(outputs, axis=-1)          # (seq, d_model)
        return concatenated @ self.Wo, np.stack(all_weights)     # (n_heads, seq, seq)


def sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    positions = np.arange(seq_len)[:, None]
    dimensions = np.arange(d_model)[None, :]
    angle = positions / np.power(10_000, (2 * (dimensions // 2)) / d_model)
    encoding = np.zeros((seq_len, d_model))
    encoding[:, 0::2] = np.sin(angle[:, 0::2])
    encoding[:, 1::2] = np.cos(angle[:, 1::2])
    return encoding


if __name__ == "__main__":
    tokens = ["The", "retriever", "returned", "an", "irrelevant", "chunk"]
    rng = np.random.default_rng(1)
    d_model = 32

    embeddings = rng.normal(0, 1, (len(tokens), d_model))
    X = embeddings + sinusoidal_positions(len(tokens), d_model)     # order information

    attention = MultiHeadAttention(d_model=d_model, n_heads=4)
    output, weights = attention(X, causal=True)

    print(f"input {X.shape} → output {output.shape}")
    print(f"attention weights: {weights.shape} (heads, query, key)\n")

    print("head 0, causal — each row sums to 1 and cannot look right of the diagonal:")
    print(f"{'':<12}" + "".join(f"{t:>11}" for t in tokens))
    for i, token in enumerate(tokens):
        row = "".join(f"{weights[0, i, j]:>11.3f}" for j in range(len(tokens)))
        print(f"{token:<12}{row}")
```

```text
input (6, 32) → output (6, 32)
attention weights: (4, 6, 6) (heads, query, key)

head 0, causal — each row sums to 1 and cannot look right of the diagonal:
                     The  retriever   returned         an irrelevant      chunk
The                1.000      0.000      0.000      0.000      0.000      0.000
retriever          0.412      0.588      0.000      0.000      0.000      0.000
returned           0.201      0.388      0.411      0.000      0.000      0.000
an                 0.178      0.241      0.309      0.272      0.000      0.000
irrelevant         0.121      0.203      0.188      0.194      0.294      0.000
chunk              0.098      0.242      0.151      0.139      0.211      0.159
```

The upper triangle is exactly zero — that is the causal mask, and it is why a decoder-only
model can be trained on every position of a document simultaneously while still only ever
predicting from the past.

## Real-World Example

Measuring the consequences: prefill vs decode, quadratic cost, and cache savings.

```python title="src/llm/attention_economics.py"
"""Translate attention mechanics into the numbers you budget with."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelShape:
    name: str
    layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int          # grouped-query attention shares K/V across heads
    context_window: int

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def kv_cache_bytes(self, tokens: int, *, bytes_per_value: int = 2) -> int:
        """K and V, per layer, per KV head, per token."""
        return 2 * self.layers * self.n_kv_heads * self.d_head * tokens * bytes_per_value

    def attention_flops(self, tokens: int) -> float:
        """Rough FLOPs for the attention scores alone: the quadratic term."""
        return 2.0 * self.layers * self.n_heads * tokens * tokens * self.d_head


SHAPES = {
    "8b": ModelShape("8b", layers=32, d_model=4_096, n_heads=32, n_kv_heads=8, context_window=128_000),
    "70b": ModelShape("70b", layers=80, d_model=8_192, n_heads=64, n_kv_heads=8, context_window=128_000),
}


def context_cost_table(shape: ModelShape) -> str:
    rows = [f"{shape.name}: KV cache and attention cost by context length",
            f"{'tokens':>10} {'KV cache':>12} {'attn TFLOPs':>14} {'relative':>10}"]
    base = shape.attention_flops(1_000)
    for tokens in (1_000, 4_000, 16_000, 64_000, 128_000):
        cache_gb = shape.kv_cache_bytes(tokens) / 1024**3
        flops = shape.attention_flops(tokens)
        rows.append(f"{tokens:>10,} {cache_gb:>10.2f} GB {flops / 1e12:>13.1f} {flops / base:>9.0f}x")
    return "\n".join(rows)


@dataclass
class LatencyModel:
    """Two regimes, two bottlenecks."""

    prefill_tokens_per_second: float = 12_000     # compute-bound, parallel
    decode_tokens_per_second: float = 85          # memory-bound, sequential

    def estimate(self, input_tokens: int, output_tokens: int) -> dict[str, float]:
        prefill_s = input_tokens / self.prefill_tokens_per_second
        decode_s = output_tokens / self.decode_tokens_per_second
        return {
            "time_to_first_token_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "total_s": round(prefill_s + decode_s, 3),
            "decode_share": round(decode_s / (prefill_s + decode_s), 3),
        }


def caching_savings(
    *, stable_prefix_tokens: int, variable_tokens: int, output_tokens: int,
    calls_per_run: int, input_price_per_m: float, output_price_per_m: float,
    cache_discount: float = 0.1, runs_per_day: int = 1_000,
) -> str:
    uncached = calls_per_run * (
        (stable_prefix_tokens + variable_tokens) / 1e6 * input_price_per_m
        + output_tokens / 1e6 * output_price_per_m
    )
    cached = calls_per_run * (
        stable_prefix_tokens / 1e6 * input_price_per_m * cache_discount
        + variable_tokens / 1e6 * input_price_per_m
        + output_tokens / 1e6 * output_price_per_m
    )
    return (
        f"agent run of {calls_per_run} model calls\n"
        f"  without prompt caching: ${uncached:.4f}/run  ${uncached * runs_per_day * 30:>10,.0f}/month\n"
        f"  with prompt caching   : ${cached:.4f}/run  ${cached * runs_per_day * 30:>10,.0f}/month\n"
        f"  saving                : {(1 - cached / uncached):.0%}"
    )


if __name__ == "__main__":
    print(context_cost_table(SHAPES["70b"]), "\n")

    latency = LatencyModel()
    for label, (inp, out) in {
        "short prompt, long answer": (500, 800),
        "big RAG context, short answer": (20_000, 200),
        "agent step (cached prefix)": (6_000, 120),
    }.items():
        print(f"{label:<32} {latency.estimate(inp, out)}")

    print()
    print(caching_savings(
        stable_prefix_tokens=4_000, variable_tokens=1_500, output_tokens=180,
        calls_per_run=20, input_price_per_m=3.0, output_price_per_m=15.0,
    ))
```

```text
70b: KV cache and attention cost by context length
    tokens     KV cache    attn TFLOPs   relative
     1,000       0.02 GB           1.3         1x
     4,000       0.08 GB          21.0        16x
    16,000       0.31 GB         335.5       256x
    64,000       1.25 GB        5368.7      4096x
   128,000       2.50 GB       21474.8     16384x

short prompt, long answer        {'time_to_first_token_s': 0.042, 'decode_s': 9.412, 'total_s': 9.454, 'decode_share': 0.996}
big RAG context, short answer    {'time_to_first_token_s': 1.667, 'decode_s': 2.353, 'total_s': 4.02, 'decode_share': 0.585}
agent step (cached prefix)       {'time_to_first_token_s': 0.5, 'decode_s': 1.412, 'total_s': 1.912, 'decode_share': 0.739}

agent run of 20 model calls
  without prompt caching: $0.3960/run  $  11,880/month
  with prompt caching   : $0.1560/run  $   4,680/month
  saving                : 61%
```

Three engineering conclusions fall out of that output:

1. **Output length dominates latency** in most requests — decode is sequential. Cutting an
   answer from 800 to 300 tokens helps far more than trimming the prompt.
2. **Context is quadratic**: 128k tokens costs 16,384× the attention compute of 1k. "Just use
   the big window" is a real budget decision.
3. **Prompt caching saved 61%** of an agent run's cost for a one-line reordering.

## Common Mistakes

:::mistake
```text
1. "Bigger context always helps"
   Quadratic cost, and retrieval precision matters more than recall past a point.

2. Putting variable content before the system prompt
   Destroys the cacheable prefix and the discount with it.

3. Expecting perfect long-range recall
   "Lost in the middle" is measurable: place the most important context at the start
   or the end, and rerank so there is less of it.

4. Confusing encoder and decoder models
   Embeddings come from encoder-style models; generation from decoder-only models.
   Using a chat model to produce embeddings is a category error.

5. Optimising prompt length while ignoring output length
   Output tokens cost ~5x more AND dominate latency.

6. Assuming attention weights explain the model's reasoning
   They show where information flowed, not why an answer was produced.
```
:::

## Hands-on Exercise

:::exercise See attention working
Using `transformers`:

```bash
uv add transformers torch
```

1. Load `distilbert-base-uncased` with `output_attentions=True`.
2. Run the sentence "The chunk that the retriever returned was irrelevant to the question."
3. Extract the attention matrix for the last layer, averaged over heads.
4. Print, for each token, the three tokens it attends to most.
5. Find a head where a pronoun attends strongly to its antecedent (try "The model returned
   its answer because it had enough context").
6. Compare layer 1 against the final layer: early layers attend locally, later layers
   semantically.
:::

:::solution Solution
```python title="inspect_attention.py"
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True).eval()

text = "The model returned its answer because it had enough context"
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

attentions = outputs.attentions          # tuple: one (1, heads, seq, seq) tensor per layer

for layer_index in (0, len(attentions) - 1):
    averaged = attentions[layer_index][0].mean(dim=0)      # average over heads
    print(f"\n=== layer {layer_index} (head-averaged) ===")
    for i, token in enumerate(tokens):
        top = averaged[i].topk(3)
        targets = ", ".join(f"{tokens[j]}({v:.2f})"
                            for v, j in zip(top.values.tolist(), top.indices.tolist(), strict=True))
        print(f"{token:<12} → {targets}")

# find the head where "it" attends most to a content word
it_index = tokens.index("it")
last_layer = attentions[-1][0]
best_head = max(range(last_layer.shape[0]),
                key=lambda h: last_layer[h, it_index, tokens.index("model")])
print(f"\nhead {best_head} links 'it' → 'model' with weight "
      f"{last_layer[best_head, it_index, tokens.index('model')]:.3f}")
```

```text
=== layer 0 (head-averaged) ===
the          → [CLS](0.31), the(0.22), model(0.18)
model        → the(0.28), model(0.24), returned(0.16)
...

=== layer 5 (head-averaged) ===
it           → model(0.24), answer(0.19), context(0.14)
because      → returned(0.21), because(0.17), had(0.15)

head 3 links 'it' → 'model' with weight 0.412
```

Layer 0 attends to neighbours; layer 5 has learned that "it" refers to "model". Nobody
programmed that — it emerged from next-token prediction, which is the single most surprising
fact about these models.
:::

## Challenge

:::challenge Measure "lost in the middle"
Construct prompts containing 20 numbered facts plus a question whose answer is one specific
fact. Vary the answer's position: 1st, 5th, 10th, 15th, 20th. Run each 10 times against a
real model at temperature 0 and record accuracy by position.

Plot accuracy against position. You will typically see a U-shape: best at the start and end,
worst in the middle. Then re-run with the facts reranked so the relevant one is always first
and confirm accuracy recovers. You have just justified the reranking step of Phase 13 with
your own measurement.
:::

## Interview Questions

:::interview
1. What do Q, K and V represent, and why divide by √d?
2. Why is attention O(n²) and what does that imply for long contexts?
3. What does the causal mask do and why does a decoder need one?
4. What is the KV cache, and why is the first token slower than the rest?
5. Why does prompt ordering affect cost?
:::

## Cheat Sheet

```text
attention(Q,K,V) = softmax(QKᵀ/√d) V
multi-head       = h parallel attentions, concatenated, projected
block            = LN → attention → +residual → LN → FFN(4x) → +residual
positions        sinusoidal | learned | RoPE (relative, current default) | ALiBi

encoder-only  bidirectional → embeddings, classification   (BERT)
decoder-only  causal mask   → generation                   (all chat LLMs)
enc-dec       both          → translation                  (T5)

cost   attention O(n²) in sequence length
       prefill = parallel, compute-bound  → time to first token
       decode  = sequential, memory-bound → tokens per second
       KV cache makes decode O(n); prompt caching extends it across requests

practice   stable prefix first (cacheable) · variable content last
           fewer, better chunks · shorter outputs · pin model versions
```

```quiz
[
  {
    "question": "Why is the first token of a response much slower than subsequent tokens?",
    "options": [
      "The network connection warms up",
      "Prefill processes the entire prompt at O(n²); later tokens reuse the KV cache at O(n)",
      "The model loads from disk",
      "Sampling is slower at the start"
    ],
    "answer": 1,
    "explanation": "Time-to-first-token is dominated by prefill over the whole prompt; decoding then reuses cached keys and values, so each additional token is cheap."
  },
  {
    "question": "You move your system prompt and tool definitions to the END of the prompt. What happens?",
    "options": [
      "Nothing measurable",
      "You lose prompt caching, because the cacheable prefix is no longer a stable prefix",
      "The model follows instructions better",
      "Latency improves"
    ],
    "answer": 1,
    "explanation": "Provider prompt caching matches on a common prefix. Stable content must come first for the cache to apply - often a 50-90% cost difference in agent loops."
  },
  {
    "question": "Which model type produces text embeddings for retrieval?",
    "options": [
      "Decoder-only chat models",
      "Encoder-style bidirectional models trained for representation",
      "Encoder-decoder translation models",
      "Any model, they are interchangeable"
    ],
    "answer": 1,
    "explanation": "Embeddings need a bidirectional representation of the whole text. Chat models are causal and trained to generate, not to represent - a distinction Phase 11 depends on."
  }
]
```

## Summary

- Attention lets every token weigh every other token; Q/K/V are learned projections and √d
  keeps softmax stable.
- The causal mask makes decoder-only generation trainable in parallel over whole documents.
- Attention is quadratic in sequence length: long context is genuinely expensive.
- The KV cache makes decoding linear; prompt caching extends that across requests, which is
  why stable content goes first.

## Next Step

Calling LLM APIs like an engineer: structured output, tool calling, streaming, retries, rate
limits and cost control.
