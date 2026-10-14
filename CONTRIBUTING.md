Thank you for your interest in contributing to `saealib`! We welcome contributions from everyone. This document outlines the process for contributing to this project.

### 1. Development Environment Setup

This project uses **[uv](https://github.com/astral-sh/uv)** for dependency management and **[Ruff](https://github.com/astral-sh/ruff)** for linting and formatting.

1. **Install uv** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```


2. **Clone the repository**:
    ```bash
    git clone https://github.com/saealib/saealib.git
    cd saealib
    ```


3. **Install dependencies**:
    ```bash
    uv pip install -e .[dev]
    # Or if you are using a virtual environment managed by uv:
    uv sync
    ```



### 2. Coding Style

We adhere to strict coding standards to ensure maintainability.

* **Linter & Formatter**: We use `ruff`. Please ensure your code passes all checks before submitting a PR.
    ```bash
    ruff check .
    ruff format .
    ```


* **Type Hinting**: Type hints are strongly encouraged for all public APIs.
* **Docstrings**: We follow the **NumPy style** docstrings.

### 3. Pre-1.0 Compatibility Policy

Before 1.0, a beta release may introduce a breaking Python API change. Record
the change and its migration path in the pull request and release notes; do
not add a compatibility shim automatically. Keep a shim when it protects an
established user-facing bridge at a proportionate maintenance cost, and add a
test that describes the retained contract. Remove short-lived implementation
aliases and migration scaffolding once the canonical API is in place.

Checkpoint and other serialized-state compatibility are evaluated separately
from Python API compatibility. Version `0.1.0` is the baseline: its checkpoint
format is version 1, every state key starts at schema version 1, and no reader
for anything written before `0.1.0` is kept. From `0.1.0` onward, preserve
readers when they protect resumability of existing experiments, even when the
corresponding runtime name is no longer part of the public API.

Do not raise the checkpoint format version for a change a current reader can
absorb, such as a new optional field with an unambiguous default, an added
state key, or an added history column. Raise it when the stored form itself
changes incompatibly — a different container, a restructured manifest, or a
field whose meaning changes. When only one state value's representation
changes, prefer raising that state key's schema version over the whole
checkpoint's.

After 1.0, renaming or removing a stable public API normally requires a
documented deprecation cycle. This policy does not make deep implementation
imports stable.

### 4. Testing Policy

We use `pytest` for testing. Since this library involves stochastic evolutionary algorithms, we have a specific testing strategy to distinguish between "expected random variations" and "regressions."

#### Running Tests

```bash
pytest
```

#### Test Categories

1. **Smoke Tests**: Checks if the example code runs without errors.
2. **Unit Tests**: Verifies data structures (`Population`) and deterministic logic (`Repair`, `Constraints`).
3. **Fixed-Seed Integration Tests**: Runs a full optimization with a fixed random seed and asserts the result stays within an acceptable threshold (e.g., `test_integration`). If this fails, check whether your change unintentionally degraded optimization performance.


### 5. Pull Request Guidelines

1. Ensure all tests pass locally.
2. Use the provided Pull Request Template.

### 6. Release Process

#### Versioning

This project follows [Semantic Versioning](https://semver.org/) (`x.y.z`):

* **`x` (major)**: large-scale rewrites or breaking changes to the public API.
* **`y` (minor)**: backward-compatible feature additions (new algorithms, surrogates, strategies, etc.).
* **`z` (patch)**: backward-compatible fixes and small corrections, including documentation-only fixes.

While the major version is `0` (pre-1.0, e.g. `0.1.0`), the public API is not considered stable and may still change between minor versions — this is the standard SemVer meaning of a `0.x` major version, not an exception to it. `0.1.0bN` pre-release identifiers (`b1`, `b2`, ...) mark beta iterations leading up to the `0.1.0` release itself.

#### Tagging and what it triggers

A release is cut by pushing a `vX.Y.Z` (optionally with a `bN` pre-release suffix) tag on `main`. This is the single trigger for:

* Publishing the package to PyPI (`.github/workflows/publish.yml`) — for a tag on `main`, publishing goes through the `pypi` environment; for a tag cut from a `release/vX.Y.x` branch, it goes through `pypi-legacy` (see "Backporting to an old minor series" below). Both environments have required reviewers configured, so publishing waits for a manual approval in either case.
* Building and deploying version-tagged documentation snapshots (`.github/workflows/docs.yml`, `deploy-release` job) — final-release snapshots (`vX.Y.Z/`), the `versions.json` version-switcher list, and the root redirect are all only regenerated here.
* For final releases (`vX.Y.Z`, no pre-release suffix) only: keeping the `release/vX.Y.x` maintenance branch for that minor series in sync with `main` (`create-release-branch` job). The branch is created on the first final release in a minor series, then fast-forwarded on every later final release in the same series — as long as the tag is still an ancestor of `main` (i.e., `main` is still the one shipping patches for that series). Once `main` moves on to a newer minor series, or once `release/vX.Y.x` has taken on maintenance commits of its own (a tag cut directly from the branch rather than from `main`), the job leaves it alone rather than forcing a sync.

Documentation's rolling `dev/` build is deployed separately on every push to `main`, regardless of tags — see "Documentation snapshots" below.

Each final release's minor series (`X.Y`) gets its own `release/vX.Y.x` branch, auto-created and kept in sync with `main` as above for as long as `main` is still shipping that series. `main` itself is not a per-version maintenance branch — once `main` has moved on to a newer minor series, don't try to backdate a patch tag on `main` for an older one. If a mistake is found in an already-released minor series, fix it on the matching `release/vX.Y.x` branch and cut the new patch tag (`vX.Y.Z+1`) from there. Patching an old minor series this way is best-effort, not a maintenance commitment — there's no obligation to backport every fix.

#### Backporting to an old minor series

A tag cut from a `release/vX.Y.x` branch is, by definition, not an ancestor of `main` (`main` has moved on). `publish.yml`'s `check-tag` job detects this and routes the `publish` job through the `pypi-legacy` GitHub Environment instead of `pypi`. Both `pypi` and `pypi-legacy` have required reviewers configured (see repo Settings → Environments), so the build still runs automatically but publishing waits for an explicit manual approval in either case. This is intentional: backporting — whether it's a documentation fix or a real code fix for a security/critical issue in an old series — is a deliberate, infrequent action, not something that should auto-publish just because a `v*` tag was pushed.

#### Documentation snapshots

Only **final** releases (tags matching `vX.Y.Z` with no pre-release suffix) get a permanent, frozen documentation snapshot, published at `https://saealib.github.io/saealib/vX.Y.Z/`, built by the `deploy-release` job when the tag is pushed. Pre-release (beta) tags do not create a new permanent snapshot. Instead, a single rolling `dev/` build is deployed by the separate `deploy-dev` job on every push to `main` (`.github/workflows/docs.yml`), so it always reflects the latest commit on `main`, not just the latest tagged one. `dev/` is named to make clear it may be ahead of (or otherwise differ from) the latest stable release, rather than being confused for "the latest recommended version." This keeps the number of permanently published doc versions tied to actual releases instead of growing with every beta iteration or every commit. The site root redirects to the newest final release once one exists, or to `dev/` before the first `1.0`-track release ships — until the first tag is pushed, `versions.json` and the root redirect don't exist yet, so `dev/` must be reached directly.

#### Documentation languages

`docs/` holds one Markdown tree per language — `docs/en/` and `docs/ja/` — with the same set of pages at the same relative paths. Sphinx builds each tree separately against the shared configuration in `docs/` (`sphinx-build -c docs docs/en ...`), so `conf.py`, `references.bib`, `_static/`, `_templates/` and `_ext/` are written once and used by both. Every version snapshot above (`vX.Y.Z/`, `dev/`) is built once per language, producing a parallel `ja/` tree (`ja/vX.Y.Z/`, `ja/dev/`). A script injected on every page (`docs/_static/js/lang-switch.js`) links to the equivalent page in the other language, and the site root auto-redirects based on the browser's `navigator.language`, falling back to English for crawlers/no-JS clients.

English is canonical: where the two trees disagree about behaviour, the English page is correct. That is a rule for resolving conflicts, not an authoring order — a change may start in either language, and translating a Japanese draft into English is as normal as the reverse.

Both trees carry the same pages, `api/` included. The generated API stubs contain little prose, but they still need a Japanese counterpart: the language switcher rewrites the URL path rather than looking a page up, so a page present in only one tree turns the other tree's switcher into a 404. `scripts/check_docs_languages.py` enforces this and admits no exclusions. A page that is not translated yet gets a stub at the matching path pointing at the original, rather than an exclusion entry.

The Japanese build is not run with `-W`. Sphinx's own Japanese message catalog renders autodoc's "alias of X" boilerplate as ``:py:class:`X`の別名です。``, and RST does not close an inline role that is followed directly by a Japanese particle, so any documented alias emits a warning that no page content can avoid.

Conventions for keeping the trees consistent — what to translate, what to copy verbatim, and the agreed terminology — are in `docs/translation-guide.md`.

To check and build the documentation locally:

```bash
# Page-set parity between the language trees
uv run --group docs python scripts/check_docs_languages.py

# Build each language (English is strict; see below for why Japanese is not)
uv run --group docs sphinx-build -W -b html -c docs docs/en docs/_build/html-en
uv run --group docs sphinx-build -b html -c docs -D language=ja docs/ja docs/_build/html-ja
```

### 7. Public API Governance

`src/saealib/__init__.py` exposes a curated eager root surface (`__all__`)
and a curated lazy root surface (`_LAZY_EXPORTS`). A symbol added to a
subpackage's `__all__` remains public at that namespace; it does not
automatically become a root export. Root additions must be intentional,
represented in `src/saealib/__init__.pyi`, and covered by the export tests.

Use `saealib.core` for framework extension contracts and graph/state
vocabulary. Use `saealib.execution` for runtime provider registration. Modules
such as `saealib.core.compiler.compiler` are implementation paths rather than
canonical extension imports and carry no compatibility guarantee.
