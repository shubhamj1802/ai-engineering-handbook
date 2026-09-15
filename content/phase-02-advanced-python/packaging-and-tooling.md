---
title: Packaging, Dependencies and Project Tooling
order: 10
difficulty: Intermediate
duration: 12
badges: ["Hands-on", "Reference"]
summary: "Turn a folder of modules into an installable, reproducible, linted project — pyproject, lock files, ruff, pre-commit and a CI pipeline you can trust."
prereqs: ["Modules, Packages and Imports", "Testing with pytest"]
keyConcepts: ["pyproject.toml", "lock file", "ruff", "pre-commit", "CI"]
---

:::note In one line
**A project that others can install and run is a different artefact from a folder of scripts.** This is the difference.
:::

## Why this matters

Every project in Phases 7, 12, 17 and 26 ships as a package with a lock file, a test suite
and a lint gate. That is not ceremony: it is what makes an AI project reproducible when a
dependency releases a breaking change — which, in this ecosystem, happens monthly.

## Mental Model

A folder of scripts and an installable project are different things. The difference is a
single file that says what this project is and what it needs.

<figure class="lesson-figure">
<svg viewBox="0 0 660 250" role="img" aria-label="Diagram: a loose folder of scripts with imports that only work from one directory, compared with a packaged project whose pyproject.toml declares its dependencies and lets it be installed and run from anywhere.">
  <defs>
    <marker id="pk-a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <text class="dg-label" x="14" y="22" fill="var(--warn)">A folder of scripts</text>
  <rect x="14" y="34" width="250" height="118" rx="10" fill="var(--panel-2)" stroke="var(--warn)" stroke-width="1.6"/>
  <text class="dg-mono" x="30" y="58" style="font-size:11.5px">my_stuff/</text>
  <text class="dg-mono" x="30" y="78" style="font-size:11.5px">  script.py</text>
  <text class="dg-mono" x="30" y="96" style="font-size:11.5px">  helpers.py</text>
  <text class="dg-mono" x="30" y="114" style="font-size:11.5px">  notes.txt</text>
  <text class="dg-sub"  x="30" y="138" fill="var(--warn)">works only from this folder</text>
  <path d="M272,92 L322,92" stroke="var(--accent)" stroke-width="2" fill="none" marker-end="url(#pk-a)"/>
  <text class="dg-label" x="344" y="22" fill="var(--ok)">A real project</text>
  <rect x="344" y="34" width="302" height="118" rx="10" fill="var(--panel-2)" stroke="var(--ok)" stroke-width="1.9"/>
  <text class="dg-mono" x="360" y="58" style="font-size:11.5px">my_project/</text>
  <text class="dg-mono" x="360" y="76" style="font-size:11.5px">  pyproject.toml</text>
  <text class="dg-sub"  x="486" y="76" style="font-size:10.5px" fill="var(--ok)">the declaration</text>
  <text class="dg-mono" x="360" y="94" style="font-size:11.5px">  src/my_project/</text>
  <text class="dg-mono" x="360" y="112" style="font-size:11.5px">  tests/</text>
  <text class="dg-sub"  x="360" y="136" fill="var(--ok)">installable, importable, runnable anywhere</text>
  <rect x="14" y="176" width="632" height="58" rx="9" fill="var(--panel)" stroke="var(--border-strong)" stroke-width="1.3"/>
  <text class="dg-sub" x="330" y="198" text-anchor="middle">pyproject.toml is the source of truth: the name, the Python version, the dependencies,</text>
  <text class="dg-sub" x="330" y="218" text-anchor="middle">and the settings for your formatter, linter and type checker, all in one place.</text>
</svg>
<figcaption>
<strong>The <code>src/</code> layout is worth the extra folder.</strong> It forces you to
install your own package to import it, which means your tests exercise the same code path a
real user would — and import bugs surface on your machine instead of theirs.
</figcaption>
</figure>

```toml
# pyproject.toml - the whole project in one file
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["httpx>=0.27", "pydantic>=2.7"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```bash
uv sync                  # build the environment from this file
uv run pytest            # run tests inside it
```

:::tip Pin what you install, declare what you need
`pyproject.toml` says "I need httpx 0.27 or newer". The lock file records "we actually
installed 0.27.2" so a colleague gets byte-identical packages.

Commit both. The declaration is your intent; the lock file is what reproducibly works.
:::

## Core Concepts

### A complete pyproject.toml

```toml title="pyproject.toml"
[project]
name = "rag-service"
version = "0.1.0"
description = "Retrieval-augmented question answering over internal documents"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["rag", "llm", "retrieval"]

dependencies = [
    "fastapi>=0.115",
    "httpx>=0.27",
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "uvicorn[standard]>=0.30",
]

[project.optional-dependencies]
qdrant = ["qdrant-client>=1.11"]
postgres = ["psycopg[binary]>=3.2", "pgvector>=0.3"]

[dependency-groups]                 # dev tools: installed locally, never shipped
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "mypy>=1.11",
    "ruff>=0.6",
    "pre-commit>=3.8",
]

[project.scripts]
rag = "rag_service.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_service"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q --strict-markers"
asyncio_mode = "auto"
markers = ["slow: touches the network", "eval: costs money"]

[tool.coverage.run]
source = ["src"]
omit = ["*/cli.py"]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "W",    # pycodestyle
    "F",         # pyflakes
    "I",         # import sorting
    "N",         # naming
    "UP",        # pyupgrade: modern syntax
    "B",         # bugbear: real bug patterns
    "A",         # shadowing builtins
    "C4",        # comprehensions
    "SIM",       # simplifications
    "RUF",       # ruff-specific
]
ignore = ["E501"]                   # the formatter handles line length

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]                # asserts are the point in tests

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

### Dependency version ranges

| Spec | Meaning | Use for |
| --- | --- | --- |
| `httpx>=0.27` | any newer version | libraries you trust to be stable |
| `httpx>=0.27,<0.29` | bounded | fast-moving libraries |
| `langchain==1.4.0` | exact | frameworks with frequent breaking changes |
| `httpx~=0.27.0` | `>=0.27.0,<0.28.0` | compatible-release shorthand |

:::tip Pin frameworks, range libraries
In AI work, pin `langchain`, `langgraph`, `crewai` and provider SDKs to exact or tightly
bounded versions, and upgrade deliberately with the tests running. A minor version can move
a class between packages.
:::

### Optional dependency groups

```bash
uv sync --extra qdrant                 # install the qdrant extra
uv sync --all-extras                   # everything
uv sync --no-dev                       # production image: no test/lint tooling
```

This keeps your container small: a production image installs `--no-dev --frozen` and nothing
else.

### Ruff — lint and format

```bash
uv run ruff check .            # lint
uv run ruff check . --fix      # fix what can be fixed automatically
uv run ruff format .           # format (a drop-in for black)
```

Ruff replaces flake8, isort, pyupgrade, black and a dozen plugins, and runs in milliseconds
on a large codebase.

### pre-commit — catch it before the commit

```yaml title=".pre-commit-config.yaml"
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=2000]
      - id: detect-private-key

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.19.2
    hooks:
      - id: gitleaks          # blocks commits containing API keys
```

```bash
uv run pre-commit install       # once per clone
uv run pre-commit run --all-files
```

The `gitleaks` hook is the one that matters most in AI projects: it stops the API key before
it enters history.

### Continuous integration

```yaml title=".github/workflows/ci.yml"
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --frozen --all-extras

      - name: Lint
        run: uv run ruff check .

      - name: Format check
        run: uv run ruff format --check .

      - name: Type check
        run: uv run mypy src

      - name: Test
        run: uv run pytest -m "not slow and not eval" --cov=src --cov-report=term-missing --cov-fail-under=80
```

`uv sync --frozen` fails if `uv.lock` does not match `pyproject.toml` — which catches the
"I added a dependency but forgot to commit the lock" class of breakage.

### Building and distributing

```bash
uv build                     # produces dist/*.whl and dist/*.tar.gz
uv publish                   # to PyPI (needs a token)
uv pip install dist/rag_service-0.1.0-py3-none-any.whl     # verify the wheel
```

:::warning Verify the wheel contains what you think
```bash
unzip -l dist/rag_service-0.1.0-py3-none-any.whl
```
Missing data files (prompt templates, JSON schemas) are the classic packaging bug: the code
works from the repo and fails when installed. Include them explicitly:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/rag_service/prompts" = "rag_service/prompts"
```
:::

## Real-World Example

The structure every project in this handbook uses:

```text
rag-service/
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── .gitignore
├── .env.example
├── README.md
├── pyproject.toml
├── uv.lock                      ← committed
├── Dockerfile
├── src/rag_service/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── errors.py
│   ├── models.py
│   ├── adapters/
│   ├── retrieval/
│   ├── generation/
│   └── prompts/                 ← data files, must be included in the wheel
│       └── answer.md
├── tests/
│   ├── conftest.py
│   ├── fakes.py
│   └── test_*.py
└── evals/
    ├── dataset.jsonl
    └── run_eval.py
```

```dockerfile title="Dockerfile"
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first: this layer is cached unless the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app

RUN useradd --create-home --uid 1000 app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import httpx;httpx.get('http://localhost:8000/health',timeout=2)"
CMD ["uvicorn", "rag_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Three details that matter: dependencies are installed **before** the source is copied (layer
caching), `--no-dev` keeps test tooling out of the image, and the process runs as a
**non-root** user.

## Common Mistakes

:::mistake
```text
1. Not committing uv.lock            → "works on my machine", unreproducible builds
2. Committing .venv                  → a huge repo and broken paths on other machines
3. Dev tools in [project.dependencies] → pytest and mypy shipped to production
4. No upper bound on a volatile framework → a minor release breaks the build silently
5. Data files missing from the wheel  → works from source, fails when installed
6. Running as root in the container   → an avoidable security finding
7. requirements.txt AND pyproject.toml → two sources of truth that drift apart
```
:::

## Security Considerations

:::security Supply chain
- Commit the lock file: it pins hashes, so a compromised republished version is rejected.
- Run `uv pip audit` (or `pip-audit`) in CI to catch known vulnerabilities.
- Check the name before adding a dependency — typosquatting on popular AI packages is
  common (`langchian`, `openal`, `chroma-db`).
- Never add a dependency you have not looked at, and prefer fewer, well-maintained ones.
- `gitleaks` in pre-commit and in CI; assume any key that reaches history is compromised.
:::

## Hands-on Exercise

:::exercise Make your `ragkit` package shippable
Take the `ragkit` package from Phase 1 and bring it up to production standard:

1. Full `pyproject.toml`: metadata, `requires-python`, a `[project.scripts]` entry, hatchling
   build config, and ruff/mypy/pytest configuration.
2. `uv lock`, commit `uv.lock`.
3. Add `.pre-commit-config.yaml` with ruff, gitleaks and the standard hooks; install and run
   it on all files, fixing what it reports.
4. Add the CI workflow and confirm it passes locally with the same commands.
5. `uv build`, inspect the wheel contents, install it into a *fresh* environment and run the
   CLI from a different directory.
:::

:::solution What "done" looks like
```bash
$ uv run ruff check . && uv run mypy src && uv run pytest -q
All checks passed!
Success: no issues found in 6 source files
14 passed in 0.09s

$ uv build
Successfully built dist/ragkit-0.1.0.tar.gz and dist/ragkit-0.1.0-py3-none-any.whl

$ unzip -l dist/ragkit-0.1.0-py3-none-any.whl | head
    1204  ragkit/__init__.py
    3821  ragkit/text.py
    2410  ragkit/scoring.py
    1902  ragkit/select.py
     612  ragkit/prompts/answer.md         ← data file present

$ cd /tmp && uv venv fresh && uv pip install --python fresh dist/ragkit-0.1.0-py3-none-any.whl
$ fresh/bin/ragkit --help
usage: ragkit [-h] {chunk,select} ...
```

Installing into a fresh environment and running from an unrelated directory is the test that
matters — it is what your Docker build and your colleague's laptop will do.
:::

## Challenge

:::challenge Dependency upgrade drill
Run `uv lock --upgrade` on a project with a real dependency tree, then:

1. Inspect the diff of `uv.lock` and identify which packages changed major or minor versions.
2. Run the full test suite and the type checker.
3. For anything that broke, write the smallest failing test that captures the behaviour
   change, then fix the code.
4. Record the upgrade in a `CHANGELOG.md` entry.

Doing this deliberately once a month is the difference between routine maintenance and a
panicked emergency upgrade when a security advisory lands.
:::

## Interview Questions

:::interview
1. What is the difference between `pyproject.toml` and a lock file?
2. Why commit the lock file?
3. How do you keep test tooling out of a production image?
4. What does `uv sync --frozen` protect you from in CI?
5. How do you reduce supply-chain risk in a Python project?
:::

## Cheat Sheet

```bash
uv init / uv add X / uv add --dev X / uv remove X
uv sync            # match the lock
uv sync --frozen   # CI: fail if the lock is stale
uv sync --no-dev   # production install
uv lock --upgrade  # deliberate refresh
uv tree            # dependency tree
uv build / uv publish

uv run ruff check . --fix
uv run ruff format .
uv run mypy src
uv run pytest -m "not slow" --cov=src
uv run pre-commit install && uv run pre-commit run --all-files
```

```quiz
[
  {
    "question": "Which files belong in version control?",
    "options": [
      "pyproject.toml only",
      "pyproject.toml and uv.lock",
      "pyproject.toml, uv.lock and .venv",
      "requirements.txt and .env"
    ],
    "answer": 1,
    "explanation": "The declaration and the lock are both source of truth; .venv is disposable and .env holds secrets."
  },
  {
    "question": "Why install dependencies before copying source code in a Dockerfile?",
    "options": [
      "It is required by Docker",
      "The dependency layer stays cached when only source changes, so rebuilds are fast",
      "It reduces the final image size",
      "It avoids permission issues"
    ],
    "answer": 1,
    "explanation": "Docker caches layers by content. Dependencies change rarely; source changes constantly. Ordering them this way turns a two-minute rebuild into five seconds."
  },
  {
    "question": "A framework you depend on frequently makes breaking changes. How do you specify it?",
    "options": [
      "framework",
      "framework>=1.0",
      "framework==1.4.0 and upgrade deliberately with tests",
      "Copy its source into your repo"
    ],
    "answer": 2,
    "explanation": "Pinning makes upgrades an explicit, tested decision instead of a surprise in someone's CI run."
  }
]
```

## Summary

- `pyproject.toml` declares, `uv.lock` pins, `uv sync` reproduces — commit both files.
- Dev tooling lives in a dependency group and never ships to production.
- Ruff, mypy, pytest and pre-commit form a fast local gate; CI runs the same commands.
- Pin fast-moving AI frameworks and upgrade on purpose, with tests.

## Next Step

Phase 2 is complete — you can write production Python. Phase 3 starts the data track with
NumPy, the foundation under Pandas, scikit-learn and every tensor library.
