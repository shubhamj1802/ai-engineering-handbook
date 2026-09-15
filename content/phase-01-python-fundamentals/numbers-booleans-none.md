---
title: Numbers, Booleans, None and Operators
order: 3
difficulty: Beginner
duration: 12
badges: ["Hands-on"]
summary: Integer and float arithmetic, truthiness, the None sentinel, and the operator precedence rules that decide what your conditions really mean.
prereqs: ["Variables and Data Types"]
keyConcepts: ["int", "float", "truthiness", "None", "short-circuit", "walrus"]
---

:::note In one line
**`0.1 + 0.2` is not `0.3`.** Floats are approximations, `None` means "no value" rather than zero, and `and`/`or` hand back one of your values instead of `True`/`False`.
:::

## Why this matters

Cost calculations, similarity thresholds, retry counters, confidence scores, token budgets
— numeric logic sits under every AI system decision. And truthiness (`if value:`) is the
most common source of silent bugs in Python, because `0`, `""`, `[]` and `None` are all
falsy but mean completely different things.

## Mental Model

```text
Truthy / falsy — what `if x:` really tests

FALSY:  False   None   0   0.0   ""   []   {}   ()   set()
TRUTHY: everything else, including "0", " ", [0], {"a": None}, -1
```

`if scores:` asks "is this list non-empty?"
`if scores is not None:` asks "does this list exist?"
These are different questions, and confusing them is a real bug: an empty retrieval result
is not the same as a retrieval that never ran.

## Core Concepts

### Integers and floats

```python
tokens = 1_847          # int: arbitrary precision, no overflow
temperature = 0.7       # float: 64-bit IEEE 754, approximate
big = 2 ** 200          # perfectly fine in Python
int("42")               # 42     parse
float("0.7")            # 0.7
int(3.9)                # 3      truncates toward zero
round(3.5)              # 4      banker's rounding: round(2.5) is 2!
```

### Arithmetic operators

| Operator | Meaning | Example |
| --- | --- | --- |
| `+ - *` | add, subtract, multiply | `3 * 4 == 12` |
| `/` | true division (always float) | `7 / 2 == 3.5` |
| `//` | floor division | `7 // 2 == 3`, `-7 // 2 == -4` |
| `%` | remainder | `7 % 2 == 1` |
| `**` | power | `2 ** 10 == 1024` |
| `+= -= *= /=` | augmented assignment | `count += 1` |

### Comparison and chaining

```python
score = 0.83

0.0 <= score <= 1.0       # True - Python allows chained comparison
score > 0.8 and score < 0.9   # same thing, more verbose
0.8 < score < 0.9         # idiomatic
```

### Boolean operators and short-circuiting

```python
a and b     # if a is falsy, return a (b never evaluated); else return b
a or b      # if a is truthy, return a; else return b
not a       # True/False
```

`and`/`or` return **one of the operands**, not necessarily `True`/`False`. That is what
makes this idiom work:

```python
name = user_name or "anonymous"      # default when user_name is falsy
timeout = config_timeout or 30
```

:::warning The `or`-default trap
`value = given or 10` also replaces a legitimate `0` with `10`, because `0` is falsy. When
zero is a valid value, test for `None` explicitly:
```python
value = 10 if given is None else given
```
:::

Short-circuiting also guards expensive or unsafe operations:

```python
if results and results[0].score > 0.7:   # safe: results[0] only runs if results is non-empty
    ...
```

### None

`None` is the single object meaning "no value". Use it for optional parameters, missing
data and "not computed yet".

```python
result = None
if result is None:          # always `is`, never ==
    result = compute()
```

### The walrus operator

`:=` assigns inside an expression — useful when you need a value both in the condition and
the body.

```python
if (match := PATTERN.search(text)) is not None:
    print(match.group(1))

while (line := file.readline()):
    process(line)
```

## Syntax

```python
# precedence, highest to lowest (the part you actually need)
**                     # power
-x                     # unary minus
* / // %
+ -
< <= > >= == != is in  # comparisons
not
and
or
```

When in doubt, add parentheses. `not a == b` means `not (a == b)`, which surprises people.

## Minimal Example

```python title="scores.py"
similarity = 0.8123
threshold = 0.75
retries_used = 2
max_retries = 3

passed = similarity >= threshold
can_retry = retries_used < max_retries

print(f"similarity {similarity:.3f} >= {threshold} -> {passed}")
print(f"retries left: {max_retries - retries_used} (can retry: {can_retry})")
print(f"confidence band: {'high' if similarity > 0.85 else 'medium' if similarity > 0.7 else 'low'}")
```

```text
similarity 0.812 >= 0.75 -> True
retries left: 1 (can retry: True)
confidence band: medium
```

## Real-World Example

Token cost accounting with correct handling of optional values.

```python title="src/costs.py"
"""Cost accounting for LLM calls.

Prices are per million tokens and change often, so they are configuration,
never constants buried in logic.
"""
from __future__ import annotations

from dataclasses import dataclass

# USD per 1,000,000 tokens. Replace with live values from your provider.
PRICES: dict[str, tuple[float, float]] = {
    "small": (0.25, 1.25),      # (input, output)
    "medium": (3.00, 15.00),
    "large": (15.00, 75.00),
}

MICRO = 1_000_000


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0     # often billed at ~10% of input price

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cached_input_tokens


def cost_usd(usage: Usage, tier: str, *, cache_discount: float = 0.1) -> float:
    """Cost of one call in dollars.

    Raises:
        KeyError: if `tier` is unknown - fail loudly rather than bill silently wrong.
    """
    input_price, output_price = PRICES[tier]
    return (
        usage.input_tokens / MICRO * input_price
        + usage.cached_input_tokens / MICRO * input_price * cache_discount
        + usage.output_tokens / MICRO * output_price
    )


def within_budget(spent: float, budget: float | None) -> bool:
    """A budget of None means unlimited. Note the explicit None check:
    `if not budget` would treat a 0.0 budget as unlimited - the opposite of correct."""
    if budget is None:
        return True
    return spent < budget


if __name__ == "__main__":
    usage = Usage(input_tokens=12_400, output_tokens=830, cached_input_tokens=8_000)
    for tier in PRICES:
        c = cost_usd(usage, tier)
        print(f"{tier:>6}: ${c:.5f}  ({usage.total:,} tokens)")

    print("budget 0.01 ok? ", within_budget(0.004, 0.01))
    print("budget 0.0 ok?  ", within_budget(0.004, 0.0))     # False - correct
    print("budget None ok? ", within_budget(999.0, None))    # True  - unlimited
```

```text
 small: $0.00136  (21,230 tokens)
medium: $0.01293  (21,230 tokens)
 large: $0.06457  (21,230 tokens)
budget 0.01 ok?  True
budget 0.0 ok?   False
budget None ok?  True
```

## Common Mistakes

:::mistake
```python
# 1. float equality
if score == 0.3: ...                 # fragile
if math.isclose(score, 0.3): ...     # correct

# 2. `if not x` when 0 is meaningful
def retry(delay=None):
    delay = delay or 1.0             # a caller passing 0 gets 1.0 - wrong
    delay = 1.0 if delay is None else delay   # correct

# 3. `is` with numbers or strings
if count is 0: ...                   # works by accident for small ints, breaks for large
if count == 0: ...                   # correct

# 4. Integer division surprise with negatives
-7 // 2                              # -4, not -3

# 5. Mixing int and float without noticing
total = 0
total += 0.1                         # total is now a float
```
:::

## Debugging

```python
import math

math.isclose(a, b, rel_tol=1e-9)     # float comparison
math.isnan(x)                        # NaN never equals itself: x == x is False for NaN
math.isinf(x)
round(x, 3)                          # display only - never for money
```

For money, use `decimal.Decimal` — floats cannot represent 0.10 exactly, and errors
accumulate over millions of API calls.

```python
from decimal import Decimal
Decimal("0.1") + Decimal("0.2") == Decimal("0.3")     # True
```

## Best Practices

1. `is None` for existence, `== value` for equality, `math.isclose` for floats.
2. Name your thresholds: `SIMILARITY_THRESHOLD = 0.75`, never a bare `0.75` in a condition.
3. Keep prices, limits and thresholds in configuration, not in the code that uses them.
4. Use `Decimal` for billing, `float` for scores.
5. Prefer explicit `if x is not None` over truthiness whenever `0`, `""` or `[]` are valid.

## Hands-on Exercise

:::exercise Retry policy calculator
Write `backoff_schedule(attempts, base=1.0, factor=2.0, cap=30.0, jitter=0.1)` that returns
the list of sleep durations for a retry policy:

- attempt *n* waits `base * factor ** (n-1)` seconds,
- never more than `cap`,
- plus up to `jitter` fraction of random extra time,
- rounded to 2 decimals.

Print the schedule for 6 attempts and the total worst-case wait. Then answer in a comment:
why is jitter necessary when a thousand clients retry after the same outage?
:::

:::solution Solution
```python title="backoff.py"
import random


def backoff_schedule(
    attempts: int,
    *,
    base: float = 1.0,
    factor: float = 2.0,
    cap: float = 30.0,
    jitter: float = 0.1,
) -> list[float]:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    schedule: list[float] = []
    for n in range(1, attempts + 1):
        delay = min(base * factor ** (n - 1), cap)
        delay += random.uniform(0, delay * jitter)
        schedule.append(round(delay, 2))
    return schedule


if __name__ == "__main__":
    random.seed(7)                      # deterministic for the example
    schedule = backoff_schedule(6)
    print(schedule)
    print(f"worst case total: {sum(schedule):.2f}s")

# Without jitter every client that failed during an outage retries at exactly
# t+1, t+3, t+7... recreating the traffic spike that caused the outage - the
# "thundering herd". Jitter spreads them out.
```

```text
[1.09, 2.1, 4.28, 8.32, 16.63, 31.6]
worst case total: 64.02s
```
:::

## Challenge

:::challenge Confidence routing
Write `route(similarity, has_citation, user_is_internal)` returning one of
`"answer"`, `"answer_with_caveat"`, `"escalate"` using these rules, expressed with the
smallest number of clear conditions you can manage:

- similarity ≥ 0.82 and has_citation → `answer`
- 0.6 ≤ similarity < 0.82 and has_citation → `answer_with_caveat`
- anything else → `escalate`, except internal users, who get `answer_with_caveat` when
  similarity ≥ 0.5.

Write five assertions covering the boundaries (exactly 0.82, exactly 0.6, missing citation,
internal at 0.5, internal at 0.49). Boundary conditions are where routing bugs live.
:::

## Interview Questions

:::interview
1. Which values are falsy in Python?
2. What does `a or b` return when both are truthy?
3. Why is `if not timeout:` dangerous when `timeout=0` is valid?
4. What is the difference between `/`, `//` and `%`?
5. When would you use `Decimal` instead of `float`?
:::

## Cheat Sheet

```python
+ - * /  //  %  **          # / is float division, // floors
0.8 < x < 0.9               # chained comparison
a and b / a or b            # return an operand, short-circuit
not a
x is None / x is not None   # existence
math.isclose(a, b)          # float equality
Decimal("0.1")              # money
(n := compute())            # walrus: assign inside an expression

falsy: False None 0 0.0 "" [] {} () set()
```

```quiz
[
  {
    "question": "What does `[] or 'default'` evaluate to?",
    "options": ["[]", "'default'", "True", "TypeError"],
    "answer": 1,
    "explanation": "An empty list is falsy, so `or` returns the right operand. This is the idiom behind `value = maybe or fallback` - and the reason it misfires when 0 or '' are legitimate values."
  },
  {
    "question": "Retrieval returned zero chunks. Which check distinguishes 'ran and found nothing' from 'never ran'?",
    "options": [
      "if not chunks:",
      "if chunks is None:",
      "if len(chunks) == 0:",
      "if chunks == False:"
    ],
    "answer": 1,
    "explanation": "None means the step did not produce a result; an empty list means it ran and found nothing. Only `is None` separates them - the others treat both as the same."
  }
]
```

## Summary

- `/` is float division, `//` floors (toward negative infinity), `%` is the remainder.
- Truthiness collapses `0`, `""`, `[]`, `{}` and `None` into "falsy" — be explicit when the
  difference matters.
- `and`/`or` short-circuit and return operands, enabling safe guards and defaults.
- `is None` for existence, `math.isclose` for floats, `Decimal` for money.

## Next Step

Lists and tuples: the ordered collections that hold your messages, chunks and results.
