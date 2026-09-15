---
title: What is AI Engineering?
order: 1
difficulty: Beginner
duration: 12
badges: ["Start here", "Read once, refer often"]
summary: What the job actually is, in plain language — building reliable software around a model that is sometimes wrong, and how that differs from data science and ML.
prereqs: []
keyConcepts: ["AI engineering", "foundation model", "non-determinism", "eval", "agent"]
---

:::note In one line
**AI engineering is normal software engineering, with one unusual part: a component that is sometimes wrong.** Your job is to build everything around that component so the whole system stays useful anyway.
:::

## Why this matters

Most people think the hard part of AI is the model.

It isn't. The model is a service you call, like a payment API. Someone else trained it. You
send text, you get text back.

The hard part is everything around it:

- Getting the **right information** into the model
- Limiting **what it is allowed to do**
- **Checking** whether the answer was any good
- Making it **fast and cheap** enough
- Making it **fail safely** when it is wrong — because it will be wrong

That surrounding work is AI engineering. It is a software job, not a maths job.

**If you can write clear Python, test your code, and think about cost and failure, you can
do this.** That is the whole entry requirement.

## The big picture

Here is the shape of almost every serious AI system. Look at how small the model's box is:

<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Diagram: a user request flows through your application, context assembly, the model, and a validation step before becoming an answer. Traces and evaluations observe the result, and invalid answers loop back to context assembly.">
  <defs>
    <marker id="ae-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
    <marker id="ae-b" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--warn)"/>
    </marker>
  </defs>

  <rect class="dg-box" x="8"   y="52" width="112" height="52" rx="9"/>
  <text class="dg-label" x="64"  y="76" text-anchor="middle">User asks</text>
  <text class="dg-sub"   x="64"  y="92" text-anchor="middle">a question</text>

  <rect class="dg-box" x="148" y="52" width="122" height="52" rx="9"/>
  <text class="dg-label" x="209" y="72" text-anchor="middle">Your app</text>
  <text class="dg-sub"   x="209" y="88" text-anchor="middle">login · rules · budget</text>

  <rect class="dg-box" x="298" y="52" width="122" height="52" rx="9"/>
  <text class="dg-label" x="359" y="72" text-anchor="middle">Find context</text>
  <text class="dg-sub"   x="359" y="88" text-anchor="middle">docs · memory · tools</text>

  <rect x="448" y="46" width="104" height="64" rx="10" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-label" x="500" y="72" text-anchor="middle" fill="var(--accent)">The model</text>
  <text class="dg-sub"   x="500" y="90" text-anchor="middle">you just call it</text>

  <rect class="dg-box" x="448" y="152" width="104" height="52" rx="9"/>
  <text class="dg-label" x="500" y="172" text-anchor="middle">Check it</text>
  <text class="dg-sub"   x="500" y="188" text-anchor="middle">is it valid?</text>

  <rect class="dg-box" x="580" y="52" width="72" height="52" rx="9"/>
  <text class="dg-label" x="616" y="76" text-anchor="middle">Answer</text>
  <text class="dg-sub"   x="616" y="92" text-anchor="middle">to user</text>

  <path class="dg-arrow" d="M120,78 L142,78" marker-end="url(#ae-a)"/>
  <path class="dg-arrow" d="M270,78 L292,78" marker-end="url(#ae-a)"/>
  <path class="dg-arrow" d="M420,78 L442,78" marker-end="url(#ae-a)"/>
  <path class="dg-arrow" d="M500,110 L500,146" marker-end="url(#ae-a)"/>
  <path class="dg-arrow" d="M552,170 Q616,170 616,110" marker-end="url(#ae-a)"/>
  <path d="M448,178 Q359,178 359,110" stroke="var(--warn)" stroke-width="1.6" fill="none" stroke-dasharray="4 3" marker-end="url(#ae-b)"/>
  <text class="dg-sub" x="392" y="198" fill="var(--warn)">if the answer is bad, try again</text>

  <text class="dg-sub" x="330" y="228" text-anchor="middle">You build every box except the green one.</text>
</svg>
<figcaption>
<strong>The model is one box out of six.</strong> Everything else — the context you feed it,
the checks on its output, the retry when it fails — is ordinary code that you write.
</figcaption>
</figure>

Everything in this handbook fits into one of those boxes:

| Box | What you build | Where you learn it |
| --- | --- | --- |
| Your app | Python services, APIs, logins | Phases 1–2, 25 |
| Find context | search, embeddings, RAG, memory | Phases 11–13, 21–22 |
| The model | prompting, structured output, model choice | Phases 8–10 |
| Check it | guardrails, schemas, human approval | Phases 19–20 |
| Answer / act | agents and workflows | Phases 14–18, 26 |
| Watching it | tracing, evaluation, cost control | Phases 23–24 |

## Three ideas that change how you build

### 1. The model gives different answers to the same question

A function you write is predictable. `add(2, 2)` returns `4` every single time.

A model is not. The same prompt can give you different words, a different JSON shape, or a
confident lie. This one fact changes three habits:

| Normal software | AI software |
| --- | --- |
| `assert result == expected` | "Is this good enough, often enough?" |
| Type hints catch mistakes | You check the output at runtime, every time |
| A retry means something broke | A retry is a normal Tuesday |

:::tip Think of it like a weather forecast
You would not write `assert forecast == "sunny"`. You would ask whether the forecast is
right often enough to be useful. Model output works the same way.
:::

### 2. Context is the product

The model does not know your documents, your database, your customer, or today's date.

It only knows what you put in the message. So almost all of the quality in a real AI
product comes from **what you put in** and **what you leave out**.

This is why retrieval (Phase 12) is such a big topic. Most "the AI is dumb" problems are
really "the AI never saw the right information" problems.

### 3. What it can do is what you let it do

A model cannot delete your database. It can only produce text.

It becomes dangerous when *you* connect that text to something real — a shell, an email
send, a refund button. Every capability you add is a capability an attacker can try to
reach through the model.

:::warning The rule that keeps you out of trouble
Give the model the **smallest** set of tools that lets it do the job. Not the most
convenient set. Phase 19 covers this properly.
:::

## How this job differs from the neighbours

People mix these three up constantly. They are different jobs:

| | Data scientist | ML engineer | **AI engineer** |
| --- | --- | --- | --- |
| Main question | What does the data say? | How do we train and serve a model? | How do we build a reliable product on a model someone else trained? |
| Builds | analyses, dashboards | training pipelines | applications, agents, RAG systems |
| Maths needed | statistics | a lot | **surprisingly little** |
| Core skill | analysis | modelling | **software engineering** |

You do not need to train a model to be an AI engineer. You need to be good at building
systems around one.

## Words you will keep hearing

Learn these five now and most articles stop being confusing:

| Word | Plain meaning |
| --- | --- |
| **Foundation model** | A big general-purpose model someone else trained. You rent it. |
| **Token** | A chunk of text, roughly ¾ of a word. You are billed per token. |
| **Context window** | How much text the model can read at once. |
| **Eval** | A test for something that has no single right answer. |
| **Agent** | A model in a loop that can call tools and decide when to stop. |

Phase 8 covers the full vocabulary. These five carry you a long way.

## What you will be able to do

By the end of this handbook you will have built:

- A search system over your own documents that answers with citations
- An agent that uses tools and knows when to stop
- A multi-agent system with real safety limits
- A service you can deploy, monitor, and control the cost of

You will start from `print("hello")`. That is genuinely fine.

## Hands-on Exercise

:::exercise Ten minutes, no code
Pick any AI product you have used — a chatbot, an email assistant, a code helper.

1. Draw the six boxes from the diagram above on paper.
2. For each box, guess what that product does. What context does it fetch? What does it
   check before showing you the answer?
3. Find one place it could go wrong, and write down what the user would see.

This is the exact thinking the rest of the handbook trains. Doing it badly now is useful —
you will redo it properly in Phase 26.
:::

:::solution What a good answer looks like (a code assistant)
```text
User asks       "why is this test failing?"
Your app         checks I am logged in and have quota left
Find context     pulls the failing test, the source file, the error output
The model        reads it, suggests a fix
Check it         does the suggested code parse? does the test pass now?
Answer           show the diff
Watching it      log tokens, latency, whether I accepted the fix

Where it goes wrong:
  the retriever grabs the wrong file, so the model explains code I never ran.
  The user sees a confident, detailed, completely irrelevant answer.
```
That last line is the most common failure in real AI products, and it is a **retrieval**
bug, not a model bug. Phase 13 is largely about preventing it.
:::

## Summary

- The model is one small part. You build everything around it.
- The model is sometimes wrong. Plan for it instead of hoping.
- Quality comes from context — what you feed in and leave out.
- Limit what the model can reach. Small permissions, always.
- This is a software job. Your Python skills matter more than maths.

## Next Step

Next we untangle the words people use interchangeably: AI, machine learning, deep learning
and generative AI. It takes five minutes and removes a lot of confusion.
