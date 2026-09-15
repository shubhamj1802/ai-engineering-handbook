---
title: Files, JSON and CSV
order: 10
difficulty: Beginner
duration: 14
badges: ["Hands-on"]
summary: Reading and writing text, JSON, JSONL and CSV with pathlib and context managers — plus encodings, atomic writes and streaming large files.
prereqs: ["Errors and Exceptions", "Dictionaries and Sets"]
keyConcepts: ["pathlib", "with", "json", "jsonl", "csv", "encoding"]
---

:::note In one line
**Always open files with `with`, and always say `encoding="utf-8"`.** That single habit prevents corrupted text and files left half-written when something goes wrong.
:::

## Why this matters

Your documents arrive as files. Your evaluation datasets are JSONL. Your chunk metadata is
JSON. Your experiment results go to CSV. Every ingestion pipeline in Phases 12–13 starts
with reading files correctly — including the encoding detail that breaks on Windows and
the streaming pattern that lets you process a file larger than RAM.

## Mental Model

```text
open(path)  →  file object  →  read/write  →  close   ← forgetting this leaks handles
with open(path) as f:  ...                            ← closes automatically, even on error

FORMAT      SHAPE                       USE
.txt        free text                   documents to chunk
.json       one object/array            config, metadata, API payloads
.jsonl      one JSON object per line    datasets, logs, evals ← stream-friendly
.csv        rows and columns            tabular data, results, spreadsheets
```

## Core Concepts

### pathlib, not string paths

```python
from pathlib import Path

data_dir = Path("data")
path = data_dir / "raw" / "handbook.txt"     # / works on every OS

path.exists()      path.is_file()      path.suffix      # '.txt'
path.stem          # 'handbook'
path.name          # 'handbook.txt'
path.parent        # Path('data/raw')
path.stat().st_size
path.parent.mkdir(parents=True, exist_ok=True)

list(data_dir.glob("*.pdf"))
list(data_dir.rglob("*.md"))                 # recursive
```

Never build paths with string concatenation — `"data/" + name` breaks on Windows and
silently mishandles trailing slashes.

### Reading and writing text

```python
# whole file, small inputs
text = path.read_text(encoding="utf-8")
path.write_text(text, encoding="utf-8")

# explicit handle, when you need control
with open(path, "r", encoding="utf-8") as handle:
    for line in handle:              # streams: one line at a time, constant memory
        process(line.rstrip("\n"))

with open(path, "w", encoding="utf-8") as handle:
    handle.write("line\n")
    handle.writelines(f"{x}\n" for x in items)
```

Modes: `"r"` read, `"w"` write (**truncates**), `"a"` append, `"x"` create-or-fail,
`"rb"`/`"wb"` binary.

:::danger Always pass `encoding="utf-8"`
Without it, Python uses the platform default — often UTF-8 on Linux/macOS and cp1252 on
Windows. The same code then works on your laptop and corrupts text in CI, or crashes with
`UnicodeDecodeError` on a document containing an em dash.
:::

### JSON

```python
import json

# file
data = json.loads(path.read_text(encoding="utf-8"))
path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# strings
obj = json.loads('{"k": 1}')
text = json.dumps(obj, indent=2, sort_keys=True, default=str)
```

`ensure_ascii=False` keeps non-English text readable instead of `é` escapes.
`default=str` stops `datetime` and `Path` objects raising `TypeError`.

JSON types map to Python as: object→dict, array→list, string→str, number→int/float,
true/false→bool, null→None. Tuples become arrays; sets and datetimes are **not**
serialisable without help.

### JSONL — the format for datasets

One JSON object per line. Appendable, streamable, diff-friendly, and what every evaluation
harness expects.

```python
# write
with open(path, "w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

# read, one line at a time - works on a 10 GB file
with open(path, "r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            record = json.loads(line)
```

### CSV

```python
import csv

with open(path, newline="", encoding="utf-8") as handle:      # newline="" is required
    for row in csv.DictReader(handle):
        print(row["question"], row["expected"])

with open(out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["id", "score", "latency_ms"])
    writer.writeheader()
    writer.writerows(rows)
```

:::warning Two CSV gotchas
1. `newline=""` prevents blank lines between rows on Windows.
2. Every CSV value is a **string** — `row["score"]` is `"0.83"`, not `0.83`. Convert
   explicitly, and handle the empty-string case.
:::

## Minimal Example

```python title="io_demo.py"
import json
from pathlib import Path

out = Path("data/eval")
out.mkdir(parents=True, exist_ok=True)

records = [
    {"id": "q1", "question": "What is RAG?", "expected": "retrieval augmented generation"},
    {"id": "q2", "question": "What is HNSW?", "expected": "a graph-based vector index"},
]

jsonl = out / "dataset.jsonl"
with jsonl.open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

loaded = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
print(f"wrote and read back {len(loaded)} records")
print(loaded[0]["question"])
```

```text
wrote and read back 2 records
What is RAG?
```

## Real-World Example

A dataset module: stream JSONL, validate records, append results atomically, export CSV for
a spreadsheet review.

```python title="src/datasets/store.py"
"""Evaluation dataset I/O.

Three habits that matter in production:
  - stream JSONL so file size never matters
  - validate on read, with the line number in the error
  - write atomically (temp file + rename) so a crash cannot corrupt the dataset
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    expected: str
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict, *, line: int) -> "EvalCase":
        missing = [k for k in ("id", "question", "expected") if not raw.get(k)]
        if missing:
            raise ValueError(f"line {line}: missing or empty fields {missing}")
        return cls(
            id=str(raw["id"]),
            question=str(raw["question"]),
            expected=str(raw["expected"]),
            tags=tuple(raw.get("tags", ())),
        )


def read_cases(path: Path) -> Iterator[EvalCase]:
    """Yield validated cases, one at a time."""
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON ({exc.msg})") from exc
            yield EvalCase.from_dict(raw, line=line_number)


def write_cases(path: Path, cases: list[EvalCase]) -> None:
    """Atomic write: a crash mid-write leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)                     # atomic on POSIX and Windows
    finally:
        tmp.unlink(missing_ok=True)


def append_result(path: Path, result: dict) -> None:
    """Append one result line. Safe to call from a long-running eval loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def export_csv(results_path: Path, csv_path: Path) -> int:
    """Flatten results to CSV for review in a spreadsheet."""
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return 0

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return len(rows)


if __name__ == "__main__":
    base = Path("data/eval")
    dataset = base / "cases.jsonl"

    write_cases(dataset, [
        EvalCase("q1", "What is RAG?", "retrieval augmented generation", ("rag",)),
        EvalCase("q2", "What is a reranker?", "a model that reorders candidates", ("rag", "advanced")),
    ])

    for case in read_cases(dataset):
        append_result(base / "results.jsonl", {
            "id": case.id, "score": 0.91, "latency_ms": 812, "passed": True,
        })

    n = export_csv(base / "results.jsonl", base / "results.csv")
    print(f"exported {n} rows to {base / 'results.csv'}")
    print((base / "results.csv").read_text(encoding="utf-8"))
```

```text
exported 2 rows to data/eval/results.csv
id,latency_ms,passed,score
q1,812,True,0.91
q2,812,True,0.91
```

## Common Mistakes

:::mistake
```python
# 1. Forgetting to close (or not using with)
f = open(path); data = f.read()        # handle leaks on exception

# 2. Missing encoding
open(path)                             # platform-dependent: works locally, breaks in CI
open(path, encoding="utf-8")

# 3. "w" when you meant "a"
open(path, "w")                        # truncates the file you wanted to append to

# 4. Reading a huge file into memory
text = path.read_text()                # 4 GB file, 4 GB of RAM
for line in path.open():               # streams

# 5. Assuming CSV values are typed
score = row["score"] + 0.1             # TypeError: it is a string
score = float(row["score"] or 0)

# 6. Writing JSON with non-serialisable objects
json.dumps({"when": datetime.now()})   # TypeError
json.dumps({"when": datetime.now()}, default=str)
```
:::

## Debugging

```python
print(path.resolve())            # absolute path - "file not found" is usually a wrong cwd
print(path.exists(), path.stat().st_size if path.exists() else "-")
print(repr(path.read_text(encoding="utf-8")[:200]))   # reveals BOM, \r\n, odd whitespace
```

A leading `'﻿'` in the repr is a UTF-8 BOM; read with `encoding="utf-8-sig"` to strip
it. It is the classic cause of a first column named `"﻿id"` in a CSV exported from
Excel.

## Security Considerations

:::security Never trust a path that came from user input
```python
# Path traversal: "../../etc/passwd" or "..\\..\\secrets.env"
requested = Path(user_input)
full = (UPLOAD_DIR / requested).resolve()
if not full.is_relative_to(UPLOAD_DIR.resolve()):
    raise ValueError("path escapes the upload directory")
```
Also: cap the file size before reading, validate the extension against an allowlist, and
never `eval`/`pickle.load` a file you did not write — `pickle` executes arbitrary code on
load. Use JSON for anything that crosses a trust boundary.
:::

## Performance Considerations

| Task | Approach |
| --- | --- |
| 10 GB JSONL | stream line by line; never `read_text()` |
| Many small writes | open once outside the loop, or buffer and write in batches |
| Tabular analysis | Pandas `read_csv` (Phase 4) is far faster than the `csv` module |
| Repeated reads of the same file | cache the parsed result in memory |
| Very large JSON (single object) | `ijson` for incremental parsing |

## Hands-on Exercise

:::exercise Corpus ingest report
Write `ingest_report(directory)` that walks a directory of `.txt` and `.md` files and
produces a JSONL manifest plus a printed summary:

1. For each file: path, size in bytes, character count, line count, a sha256 of the
   contents (first 16 hex chars), and `approx_tokens = chars // 4`.
2. Skip files over 5 MB and record them as skipped, with the reason.
3. Detect duplicate content by hash and report which files are duplicates of which.
4. Write `manifest.jsonl` atomically and print totals: files, skipped, duplicates, total
   approximate tokens.
:::

:::solution Solution
```python title="ingest_report.py"
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
EXTENSIONS = {".txt", ".md"}


def ingest_report(directory: str | Path) -> dict[str, object]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    records: list[dict] = []
    skipped: list[dict] = []
    by_hash: dict[str, str] = {}
    duplicates: list[dict] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        size = path.stat().st_size
        if size > MAX_BYTES:
            skipped.append({"path": str(path), "reason": f"too large ({size:,} bytes)"})
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped.append({"path": str(path), "reason": "not valid utf-8"})
            continue

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if digest in by_hash:
            duplicates.append({"path": str(path), "duplicate_of": by_hash[digest]})
            continue
        by_hash[digest] = str(path)

        records.append({
            "path": str(path),
            "bytes": size,
            "chars": len(text),
            "lines": text.count("\n") + 1,
            "sha256_16": digest,
            "approx_tokens": len(text) // 4,
        })

    manifest = directory / "manifest.jsonl"
    tmp = manifest.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(manifest)

    summary = {
        "files": len(records),
        "skipped": len(skipped),
        "duplicates": len(duplicates),
        "approx_tokens": sum(r["approx_tokens"] for r in records),
    }
    print(json.dumps(summary, indent=2))
    for dup in duplicates:
        print(f"  duplicate: {dup['path']} == {dup['duplicate_of']}")
    for skip in skipped:
        print(f"  skipped:   {skip['path']} ({skip['reason']})")
    return summary


if __name__ == "__main__":
    demo = Path("data/corpus")
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "a.md").write_text("# Intro\nRAG grounds answers.\n", encoding="utf-8")
    (demo / "b.md").write_text("# Intro\nRAG grounds answers.\n", encoding="utf-8")
    (demo / "c.txt").write_text("Rerankers reorder candidates.\n", encoding="utf-8")
    ingest_report(demo)
```

```text
{
  "files": 2,
  "skipped": 0,
  "duplicates": 1,
  "approx_tokens": 13
}
  duplicate: data/corpus/b.md == data/corpus/a.md
```

Deduplicating by content hash before embedding is a real cost saving: duplicate documents
are extremely common in shared drives and wikis.
:::

## Challenge

:::challenge Resumable ingest
Extend the report so a second run skips files already present in `manifest.jsonl` (matched
by path *and* hash), appends only new ones, and prints `new=N unchanged=M changed=K`. Then
make it safe to interrupt: kill the process halfway and confirm the manifest is still valid
JSONL. This "resumable, idempotent ingest" property is what makes Phase 12's pipeline
usable on real document sets.
:::

## Interview Questions

:::interview
1. Why use `with open(...)` instead of `open(...)`?
2. What does `newline=""` do in `csv.reader`?
3. When would you choose JSONL over JSON?
4. How do you write a file atomically, and why does it matter?
5. What is a path traversal attack and how do you prevent it?
:::

## Cheat Sheet

```python
from pathlib import Path
p = Path("data") / "file.jsonl"
p.exists() p.suffix p.stem p.parent p.stat().st_size
p.parent.mkdir(parents=True, exist_ok=True)
p.read_text(encoding="utf-8") / p.write_text(s, encoding="utf-8")
p.glob("*.md") / p.rglob("*.pdf")

with open(p, "r", encoding="utf-8") as f:      # r w a x, + b for binary
    for line in f: ...                          # streaming

json.loads(s) json.dumps(obj, indent=2, ensure_ascii=False, default=str)
jsonl: one json.dumps(...) + "\n" per record

csv.DictReader(f) / csv.DictWriter(f, fieldnames=[...])   # open with newline=""
atomic write: tempfile -> flush -> os.fsync -> Path.replace(target)
```

```quiz
[
  {
    "question": "Which opens a file safely for reading text on any platform?",
    "options": [
      "open(path)",
      "open(path, 'r', encoding='utf-8')",
      "open(path, 'rb')",
      "open(path, 'w', encoding='utf-8')"
    ],
    "answer": 1,
    "explanation": "Always specify the encoding; the platform default differs between your laptop and CI. Mode 'w' would destroy the file."
  },
  {
    "question": "Why is JSONL preferred for evaluation datasets?",
    "options": [
      "It is smaller",
      "Each line is an independent record, so files stream and append without rewriting",
      "It supports comments",
      "It is required by pytest"
    ],
    "answer": 1,
    "explanation": "Streaming, appending and line-level diffs are exactly what a growing dataset and an incremental eval run need."
  },
  {
    "question": "Your process crashes while writing a dataset file. How do you ensure the previous version survives?",
    "options": [
      "Write with mode 'a'",
      "Write to a temp file in the same directory, then Path.replace() it over the target",
      "Write twice",
      "Catch KeyboardInterrupt"
    ],
    "answer": 1,
    "explanation": "Rename within a filesystem is atomic, so readers see either the old file or the complete new one, never a half-written file."
  }
]
```

## Summary

- Use `pathlib` for paths and `with` for handles; always pass `encoding="utf-8"`.
- JSON for single objects, JSONL for datasets and logs, CSV for spreadsheet hand-off.
- Stream line-by-line for large files; never read a multi-GB file into memory.
- Write atomically via a temp file plus rename; validate on read with line numbers.
- Validate user-supplied paths and never unpickle untrusted data.

## Next Step

Modules and packages — how to split your growing code across files and import it reliably.
