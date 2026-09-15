---
title: AI vs ML vs Deep Learning vs Generative AI
order: 2
difficulty: Beginner
duration: 10
badges: ["Read once, refer often"]
summary: The words everyone mixes up, sorted out once — AI, ML, deep learning, generative AI, LLMs, RAG and agents — with a picture that makes the nesting obvious.
prereqs: ["What is AI Engineering?"]
keyConcepts: ["AI", "ML", "deep learning", "generative AI", "LLM", "RAG", "agent"]
---

:::note In one line
**These words are boxes inside boxes, not synonyms.** All deep learning is machine learning.
All machine learning is AI. But plenty of AI is neither.
:::

## Why this matters

Mixing these up is not just sloppy talk. It causes real mistakes:

- Reaching for a **language model** when a simple formula would be better, faster and free
- Calling a two-step script an **"agent"**, then wondering why it needs a supervisor
- Promising someone **"AI"** when you are going to deliver a list of `if` statements

Ten minutes here saves a lot of confusion later.

## The picture

<figure class="lesson-figure">
<svg viewBox="0 0 660 360" role="img" aria-label="Diagram: nested boxes. Artificial intelligence is the outermost box and contains rule-based AI and machine learning. Machine learning contains classical machine learning and deep learning. Deep learning contains generative AI, which contains large language models.">
  <rect x="10" y="10" width="640" height="300" rx="16" fill="none" stroke="var(--border-strong)" stroke-width="2"/>
  <text class="dg-label" x="28" y="36">ARTIFICIAL INTELLIGENCE</text>
  <text class="dg-sub"   x="28" y="52">any machine doing something we'd call smart</text>
  <rect x="28" y="228" width="240" height="62" rx="10" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.5"/>
  <text class="dg-label" x="44" y="252">Rule-based AI</text>
  <text class="dg-sub"   x="44" y="270">chess engines, expert systems</text>
  <text class="dg-sub"   x="44" y="284">no learning at all</text>
  <rect x="286" y="64" width="350" height="226" rx="14" fill="none" stroke="var(--accent-3)" stroke-width="2"/>
  <text class="dg-label" x="302" y="88" fill="var(--accent-3)">MACHINE LEARNING</text>
  <text class="dg-sub"   x="302" y="104">learned from examples, not hand-written</text>
  <rect x="302" y="216" width="150" height="60" rx="10" fill="var(--panel-2)" stroke="var(--border-strong)" stroke-width="1.5"/>
  <text class="dg-label" x="316" y="240">Classical ML</text>
  <text class="dg-sub"   x="316" y="257">trees, regression</text>
  <text class="dg-sub"   x="316" y="270">still the right tool often</text>
  <rect x="470" y="118" width="152" height="158" rx="12" fill="none" stroke="var(--accent-2)" stroke-width="2"/>
  <text class="dg-label" x="484" y="140" fill="var(--accent-2)">DEEP LEARNING</text>
  <text class="dg-sub"   x="484" y="155">neural networks</text>
  <rect x="484" y="168" width="124" height="94" rx="10" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-label" x="496" y="190" fill="var(--accent)">Generative AI</text>
  <text class="dg-sub"   x="496" y="205">makes new content</text>
  <rect x="496" y="216" width="100" height="34" rx="8" fill="var(--panel)" stroke="var(--accent)" stroke-width="1.5"/>
  <text class="dg-label" x="546" y="237" text-anchor="middle" fill="var(--accent)">LLMs</text>
  <text class="dg-sub" x="330" y="336">Read it inwards: every box is a special case of the box around it.</text>
</svg>
<figcaption>
<strong>LLMs sit four layers deep.</strong> When someone says "we should use AI", ask which
box they mean — the answer changes the cost, the speed and the accuracy by orders of magnitude.
</figcaption>
</figure>

## The four boxes, plainly

### Artificial Intelligence — the outer box

Any machine doing something that looks intelligent. **Learning is not required.**

A chess engine that searches ahead is AI. It never learned anything; someone wrote the rules.

### Machine Learning — learned from examples

Instead of writing the rules, you show the computer examples and it works the rules out.

You do not write "if the email says FREE MONEY, it is spam". You show it 10,000 emails
already marked spam or not, and it finds the pattern.

### Deep Learning — neural networks with many layers

A particular way of doing machine learning, using layered networks loosely inspired by
brains. It wins when data is messy and huge: images, audio, language.

It needs a lot of data and a lot of computing power. That is the trade.

### Generative AI — it produces new things

Older models mostly **judged** things: spam or not, £340,000 or £360,000.

Generative models **produce** things: a sentence, an image, a block of code. That is the
whole difference, and it is why the last few years felt sudden.

:::tip A one-question test
Ask: *does it pick from options, or does it make something new?*

Picking is classification. Making is generative. Spam filters pick. ChatGPT makes.
:::

## The words that come after

These are not in the nesting diagram, because they are things you **build**, not types of
model.

| Word | What it actually is | Built in |
| --- | --- | --- |
| **LLM** | A generative model trained on text. Takes text, predicts what comes next. | Phase 10 |
| **RAG** | Search your documents first, then paste the results into the prompt. | Phase 12 |
| **Agent** | A model in a loop that can use tools and decides when it is finished. | Phase 14 |
| **Agentic AI** | Agents that plan, work over many steps, and hand off to each other. | Phase 15 |

:::warning "Agent" is the most abused word in this list
If it runs a fixed sequence of steps, it is a **workflow** — and that is usually better:
cheaper, faster, easier to debug. It is only an agent if the *model* decides what happens
next. Phase 14 makes this distinction carefully, because choosing wrong is expensive.
:::

## When the boring option wins

This is the part most AI courses skip.

| Your problem | Best tool | Why |
| --- | --- | --- |
| Is this transaction fraud? | Classical ML | Faster, cheaper, and you can explain the decision |
| Predict next month's sales | Classical ML | Numbers in, number out — no language involved |
| Sort support emails into 5 buckets | Start classical | An LLM works, but costs 100× more per email |
| Answer questions about our handbook | LLM + RAG | Needs language understanding |
| Write a first-draft reply | LLM | Generation is the whole point |
| Convert dates to a standard format | **Plain code** | Seriously. No model. A regex. |

:::mistake The expensive habit
Using an LLM for something a `for` loop could do. It is slower, costs money per call, and
occasionally gets it wrong — whereas the loop is instant, free and correct every time.

Always ask: *could ordinary code do this?* Surprisingly often, yes.
:::

## Quick check

```quiz
[
  {
    "question": "Every deep learning system is also a machine learning system.",
    "options": ["True", "False"],
    "answer": 0,
    "explanation": "Deep learning sits inside machine learning, which sits inside AI. The boxes nest inwards, so anything deep is also ML and also AI."
  },
  {
    "question": "You need to flag transactions as fraud or not, with an explanation for auditors. What should you reach for first?",
    "options": ["A large language model", "Classical machine learning", "A multi-agent system", "Rule-based AI only"],
    "answer": 1,
    "explanation": "It is a classification problem on numeric data, and you need explainability. Classical ML is faster, far cheaper, and its decisions can be explained — all three matter to an auditor."
  },
  {
    "question": "A script always runs: fetch data, summarise it, email it. Is it an agent?",
    "options": ["Yes, it uses an LLM", "No, it is a workflow"],
    "answer": 1,
    "explanation": "The steps are fixed and decided by you, not the model. That is a workflow. It becomes an agent only when the model chooses what to do next."
  }
]
```

## Summary

- The terms nest: AI ⊃ ML ⊃ deep learning ⊃ generative AI ⊃ LLMs.
- Learning is not required for something to count as AI.
- Generative means it **makes** things rather than **picks** between them.
- RAG and agents are things you build, not kinds of model.
- Ordinary code and classical ML are still the right answer more often than people admit.

## Next Step

Now that the words are clear, let's see how the pieces fit together into a working system —
and which parts you will build in which order.
