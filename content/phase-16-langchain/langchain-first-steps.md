---
title: "LangChain from Zero — Your First Calls"
order: 1
difficulty: Intermediate
duration: 16
badges: ["Start here", "Hands-on"]
summary: Install it, make one call, understand messages, and see exactly what LangChain adds over calling a provider SDK yourself — written against LangChain 1.4.
prereqs: ["LLM API Engineering", "Typing, Protocols and Static Analysis"]
keyConcepts: ["init_chat_model", "messages", "invoke", "stream", "provider string"]
---

:::note In one line
**LangChain is a standard plug socket for language models.** You write your code once against
its interface, and swapping the model underneath becomes a one-line change.
:::

:::warning Versions used on this page
`langchain` **1.4.0** · `langchain-core` **1.6.3** · `langchain-anthropic` **1.7.2**

LangChain changed its import paths significantly at 1.0. If you find a tutorial using
`from langchain.chat_models import ChatAnthropic` or `LLMChain`, it is written for 0.x and
will not work here. Everything below is verified against 1.4.
:::

## Why this matters

You already know how to call a model with a provider SDK. So why add a library?

One honest answer: **often you should not**. If you call one provider, one way, the raw SDK
is simpler and has fewer moving parts.

LangChain starts paying for itself when you need any of these:

- Swap or compare providers without rewriting your code
- Get **validated objects** back instead of a string you have to parse
- Reuse the same retrieval and tool plumbing across several features
- Hand work to LangGraph later without rebuilding everything

This lesson gets you to a working call and shows you exactly what the library is doing, so
nothing later feels like magic.

## Install

```bash
uv add langchain langchain-anthropic
```

`langchain` is the framework. `langchain-anthropic` is the adapter for Claude. Each provider
lives in its own package, so you install only what you use.

```bash title="set your key - never hardcode it"
export ANTHROPIC_API_KEY="sk-ant-..."
```

:::security The key lives in the environment, always
LangChain reads `ANTHROPIC_API_KEY` from the environment by itself. You never pass it in
code, which means it never lands in a source file or a git history.
:::

## Your first call

```python title="src/lc/first_call.py"
"""The smallest useful LangChain program."""
from langchain.chat_models import init_chat_model

# "provider:model" - one string picks both the adapter and the model
model = init_chat_model("anthropic:claude-sonnet-5")

response = model.invoke("Explain what a vector database does, in two sentences.")

print(response.text)
```

```bash
uv run python src/lc/first_call.py
```

```text
A vector database stores text as numeric vectors that capture meaning, so it can find
passages that are semantically similar to a query rather than just keyword matches. It is
the storage layer behind retrieval-augmented generation.
```

Three things just happened, and all three are worth naming:

1. `init_chat_model` looked at the `anthropic:` prefix and loaded `langchain-anthropic`
2. Your plain string was wrapped into a **HumanMessage** for you
3. You got back an **AIMessage** object, not a string — `.text` is the text part of it

## What the library actually added

<figure class="lesson-figure">
<svg viewBox="0 0 660 230" role="img" aria-label="Diagram: with a provider SDK your code speaks that provider's shapes directly. With LangChain your code speaks one standard shape, and adapters translate it to each provider, so changing provider is a one-line change.">
  <defs>
    <marker id="l1-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <text class="dg-label" x="14" y="22" fill="var(--danger)">Provider SDK directly</text>
  <rect x="14" y="34" width="98" height="34" rx="6" class="dg-box"/>
  <text class="dg-sub" x="63" y="55" text-anchor="middle">your code</text>
  <rect x="140" y="34" width="150" height="34" rx="6" fill="var(--panel-2)" stroke="var(--danger)" stroke-width="1.5"/>
  <text class="dg-sub" x="215" y="55" text-anchor="middle" fill="var(--danger)">that provider&apos;s shapes</text>
  <path class="dg-arrow" d="M112,51 L134,51" marker-end="url(#l1-a)"/>
  <text class="dg-sub" x="306" y="55" fill="var(--danger)">switching provider = rewrite</text>
  <line x1="14" y1="88" x2="646" y2="88" stroke="var(--border)" stroke-width="1"/>
  <text class="dg-label" x="14" y="114" fill="var(--ok)">Through LangChain</text>
  <rect x="14" y="126" width="98" height="40" rx="6" class="dg-box"/>
  <text class="dg-sub" x="63" y="151" text-anchor="middle">your code</text>
  <rect x="140" y="120" width="160" height="52" rx="8" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="2"/>
  <text class="dg-label" x="220" y="141" text-anchor="middle" fill="var(--accent)">one standard shape</text>
  <text class="dg-sub"   x="220" y="159" text-anchor="middle">messages in, AIMessage out</text>
  <rect x="332" y="112" width="120" height="28" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.3"/>
  <text class="dg-sub" x="392" y="131" text-anchor="middle">anthropic adapter</text>
  <rect x="332" y="146" width="120" height="28" rx="5" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.3"/>
  <text class="dg-sub" x="392" y="165" text-anchor="middle">another adapter</text>
  <path class="dg-arrow" d="M112,146 L134,146" marker-end="url(#l1-a)"/>
  <path class="dg-arrow" d="M300,138 L326,126"/>
  <path class="dg-arrow" d="M300,154 L326,160"/>
  <rect x="478" y="118" width="168" height="56" rx="8" fill="var(--panel-2)" stroke="var(--accent)" stroke-width="1.5"/>
  <text class="dg-sub" x="562" y="140" text-anchor="middle">change one string:</text>
  <text class="dg-mono" x="562" y="160" text-anchor="middle" style="font-size:10.5px">"anthropic:..." -&gt; other</text>
  <path class="dg-arrow" d="M452,146 L472,146" marker-end="url(#l1-a)"/>
  <text class="dg-sub" x="14" y="206">The cost of this: one more dependency, and its abstractions to learn when something breaks.</text>
  <text class="dg-sub" x="14" y="222">The benefit only arrives if you actually use the portability. Be honest about which you need.</text>
</svg>
<figcaption>
<strong>One shape in front of many providers.</strong> That is the whole pitch. Everything
else LangChain offers — tools, retrieval, agents — is built on this one standard shape.
</figcaption>
</figure>

## Messages: the real input type

That string you passed was a convenience. The actual input is a **list of messages**, and
each message has a role.

```python title="src/lc/messages.py"
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

model = init_chat_model("anthropic:claude-sonnet-5")

messages = [
    SystemMessage("You are a terse assistant. Answer in one sentence, no preamble."),
    HumanMessage("What is a token?"),
]

response = model.invoke(messages)
print(response.text)
```

```text
A token is a chunk of text, roughly three quarters of a word, and the unit you are billed in.
```

The four roles you will meet:

| Class | Role | What it is for |
| --- | --- | --- |
| `SystemMessage` | system | standing instructions — the job, the rules |
| `HumanMessage` | user | what the user said |
| `AIMessage` | assistant | what the model said, including tool requests |
| `ToolMessage` | tool | the result of running a tool |

```python
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
```

:::tip Plain dicts work too
`model.invoke([{"role": "user", "content": "hi"}])` is accepted everywhere the classes are.
Use the classes when you want your editor to help you; use dicts when the data arrived as
JSON from somewhere else and you are just passing it through.
:::

## Holding a conversation

There is no hidden session. **A conversation is just a list you keep appending to.**

```python title="src/lc/conversation.py"
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

model = init_chat_model("anthropic:claude-sonnet-5")

history = [SystemMessage("You are a helpful Python tutor. Keep answers short.")]

for question in ["What is a list comprehension?", "Show me one that filters.", "Why is that better than a loop?"]:
    history.append(HumanMessage(question))
    reply = model.invoke(history)          # the WHOLE list goes every time
    history.append(reply)                  # keep the reply, or it forgets
    print(f"\nQ: {question}\nA: {reply.text}")
```

Two consequences worth internalising now:

- **If you forget to append the reply, the model has no memory.** There is nowhere else it
  is stored.
- **Every turn re-sends the whole history**, so cost grows with the square of the
  conversation length. This is the problem LangGraph's checkpointing and Phase 21's memory
  layers exist to manage.

## Streaming

`invoke` waits for the whole answer. `stream` gives you pieces as they arrive, which is the
difference between an app that feels fast and one that feels broken.

```python title="src/lc/streaming.py"
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-5")

for chunk in model.stream("List three uses for embeddings."):
    print(chunk.text, end="", flush=True)
print()
```

`flush=True` matters — without it Python buffers the output and you lose the effect you were
trying to create.

## Reading the response object

`invoke` returns an `AIMessage`. The useful parts:

```python title="src/lc/inspect_response.py"
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-5")
response = model.invoke("Say hello.")

print(response.text)                       # the text
print(response.usage_metadata)             # tokens in, out, total
print(response.response_metadata)          # model id, stop reason, provider details
print(type(response).__name__)             # AIMessage
```

```text
Hello!
{'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}
{'model': 'claude-sonnet-5', 'stop_reason': 'end_turn', ...}
AIMessage
```

:::production Log usage_metadata from day one
This is where your cost data lives. Log `input_tokens` and `output_tokens` per request
alongside a request id, and cost analysis later is a database query rather than an
archaeology project. Retrofitting it is much harder than adding it now.
:::

## Common mistakes

:::mistake Using a 0.x tutorial
```python
# BROKEN on 1.x - these are all gone
from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic
chain = LLMChain(llm=llm, prompt=prompt)
chain.run("hello")
```
`LLMChain` and `.run()` were removed. The 1.x equivalents are `init_chat_model`, the pipe
operator, and `.invoke()`. If a snippet has `LLMChain` in it, close the tab.
:::

:::mistake Treating the response as a string
```python
response = model.invoke("hi")
print("Reply: " + response)          # TypeError
print("Reply: " + response.text)     # correct
```
:::

:::mistake Forgetting to append the reply to history
The single most common cause of "why does it keep forgetting?" There is no session; the list
is the memory.
:::

## Hands-on Exercise

:::exercise A three-turn assistant with cost tracking
Write `src/lc/exercise.py` that:

1. Creates a model with `init_chat_model("anthropic:claude-sonnet-5")`
2. Holds a **three-turn** conversation where turn 2 and 3 refer back to turn 1
   (for example: "Explain RAG" → "What breaks most often in it?" → "Which of those would
   you fix first?")
3. Streams each answer so it appears as it is generated
4. Accumulates `usage_metadata` across all three calls and prints the total tokens
5. Prints the total a second time **with the history-appending line removed**, so you can
   see the difference

Report both totals and explain the gap.
:::

:::solution What you should see, and why
```python title="src/lc/exercise.py"
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("anthropic:claude-sonnet-5")

QUESTIONS = [
    "Explain RAG in two sentences.",
    "What breaks most often in it?",
    "Which of those would you fix first?",
]

def run(keep_history: bool) -> int:
    history = [SystemMessage("You are concise. Two sentences maximum.")]
    total = 0
    for question in QUESTIONS:
        history.append(HumanMessage(question))
        chunks = []
        for chunk in model.stream(history):
            chunks.append(chunk.text)
            print(chunk.text, end="", flush=True)
        print()
        reply = AIMessage("".join(chunks))
        if keep_history:
            history.append(reply)
        else:
            history.pop()            # drop the question too: no memory at all
    return total

# Note: model.stream() chunks carry usage on the final chunk; for a simple
# total, invoke() is easier to account for:
def totals() -> tuple[int, int]:
    with_memory = without_memory = 0
    for keep in (True, False):
        history = [SystemMessage("You are concise. Two sentences maximum.")]
        running = 0
        for question in QUESTIONS:
            history.append(HumanMessage(question))
            reply = model.invoke(history)
            running += reply.usage_metadata["total_tokens"]
            if keep:
                history.append(reply)
        if keep:
            with_memory = running
        else:
            without_memory = running
    return with_memory, without_memory

print(totals())
```

```text
(1184, 612)
```

**Roughly double, for three turns.** Keeping the history costs tokens because every turn
re-sends everything before it. Without it the cost is lower and the answers to turns 2 and 3
are useless, because "what breaks most often in it?" has no *it*.

That trade — memory costs tokens — is the whole reason Phase 21 exists. You do not simply
keep everything; you decide what deserves the space.
:::

## Challenge

:::challenge Prove the portability claim
The pitch is that swapping providers is a one-line change. Test it honestly.

Write a script that runs the same three prompts through **two different models** via
`init_chat_model` (two Claude models is fine — `claude-sonnet-5` and `claude-haiku-4-5`),
changing only the model string, and prints a table of answer, tokens and rough cost side by
side.

Then find where the abstraction leaks. Try passing a provider-specific parameter through
`init_chat_model(..., model_kwargs={...})` and see what happens on each. Note in comments
exactly which parts were genuinely portable and which were not — that honest list is what
tells you whether the dependency is earning its place in your project.
:::

## Interview Questions

:::interview
1. What does `init_chat_model("anthropic:claude-sonnet-5")` actually do?
2. Where is conversation history stored in LangChain?
3. Why does a long conversation get more expensive per turn?
4. What is the difference between `invoke` and `stream`?
5. Name a case where you would skip LangChain and use the provider SDK directly.
:::

## Cheat Sheet

```python
# setup
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

model = init_chat_model("anthropic:claude-sonnet-5")

# calling
model.invoke("a string")                       # convenience
model.invoke([SystemMessage(...), HumanMessage(...)])
for chunk in model.stream(messages):
    chunk.text

# reading the reply
reply.text                                     # the text
reply.usage_metadata["total_tokens"]           # cost data - log this
reply.response_metadata                        # model id, stop reason

# conversation = a list you append to
history.append(HumanMessage(q))
history.append(model.invoke(history))
```

## Summary

- `init_chat_model("provider:model")` is the single entry point in LangChain 1.x.
- The real input is a list of messages with roles; a bare string is a shortcut.
- The response is an `AIMessage`. Use `.text`, and log `.usage_metadata`.
- Conversation memory is a list you maintain. Nothing is stored for you.
- Every turn re-sends the history, so cost grows faster than turn count.
- If you will never swap providers or reuse the plumbing, the raw SDK is a fair choice.

## Next Step

Next: structured output and tools — how to get a validated Python object back instead of a
string you have to parse, which is what makes model output safe to put into a database.
