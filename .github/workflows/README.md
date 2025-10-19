````markdown
# GitHub Actions Workflows

This directory contains GitHub Actions workflow definitions for the **800 Picacho Lane LUTs** repository.

## Table of Contents
- [Active Workflows](#active-workflows)
- [Manual Utilities](#manual-utilities)
- [Experimental Workflows](#experimental-workflows)
- [Workflow Configuration](#workflow-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Active Workflows

### `build.yml` — Main Build (Tests + Coverage)
![Build](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/build.yml/badge.svg?branch=main)
  
Runs tests with coverage across Python 3.10–3.12, verifies pinned tools, enforces shim guard, runs mypy with cache, uploads Codecov via OIDC.

**Triggers:** push/pull_request to `main`  
**Key:** pip cache, mypy cache, Codecov v5 OIDC

---

### `ci.yml` — CI (Lean)
![CI (lean)](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/ci.yml/badge.svg?branch=main)

Fast matrix (3.10–3.12) with lean deps → shim guard → sanity imports → pytest.

**Use when:** you want a quick green/red while the main build runs.

---

### `static-analysis.yml` — Static Analysis & Tests
![Static Analysis](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/static-analysis.yml/badge.svg?branch=main)

Style (Black, isort), Flake8 errors-only, Pylint (threshold), mypy (cached), Bandit, Radon CC; then tests+coverage. Most style checks run once on 3.12.

---

### `pylint.yml` — Pylint
![Pylint](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/pylint.yml/badge.svg?branch=main)

Standalone lint with pinned `pylint>=3,<4`, shim guard, mypy cache, and a sanity import step.

---

### `codeql.yml` — CodeQL (Standard)
![CodeQL](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/codeql.yml/badge.svg?branch=main)

Security & quality scanning for Python with weekly schedule.

---

### `codeql-advanced.yml` — CodeQL (Advanced Matrix)
![CodeQL Advanced](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/codeql-advanced.yml/badge.svg?branch=main)

Advanced configuration (multi-language-ready). Includes a Python sanity-import step before analysis.

---

## Manual Utilities

### `shim-sync.yml` — Legacy Shim Sync (Manual)
![shim-sync](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/shim-sync.yml/badge.svg)

One-click generator to **create/update** legacy import shims from tests (e.g., `evolutionary_checkpoint → src.evolutionary`).  
Inputs: `modules` (globs), `dry_run`, `commit_message`.  
> CI enforces `--fail-on-create`; use this workflow to commit required shims first.

---

## Experimental Workflows

### `summary.yml` — AI Issue Summarizer
![Issue Summary](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/summary.yml/badge.svg)

Posts a short OpenAI-generated summary on new issues.  
**Requires:** `OPENAI_API_KEY` secret. Gracefully no-ops without it.

---

## Workflow Configuration

**Caching**
- Prefer `actions/setup-python@v5` with `cache: pip`.
- mypy cache: `.mypy_cache` (restore → run → save). Consider `--sqlite-cache` if you want smaller artifacts.

**Concurrency**
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
````

**Permissions**

* Default minimal: `contents: read`.
* Codecov OIDC: add `id-token: write`.
* Issue comments: `issues: write` or `pull-requests: write` as needed.

**Codecov (OIDC)**

```yaml
permissions:
  id-token: write

- uses: codecov/codecov-action@v5
  with:
    use_oidc: true
    files: ./coverage.xml
    fail_ci_if_error: false
```

**Shim Guard**

* CI step: `python tools/gen_legacy_shims.py --fail-on-create`
* Manual fix: run **shim-sync** → commit shims.

**Sanity Imports**

* Early failure context:

```yaml
- name: Sanity imports
  run: |
    python - <<'PY'
    import importlib, sys
    try:
        importlib.import_module('src.evolutionary')
        importlib.import_module('evolutionary_checkpoint')
        print("✅ import sanity OK")
    except Exception as e:
        print(f"❌ Import sanity failed: {e}")
        sys.exit(2)
    PY
```

---

## Troubleshooting

**“Unrecognized arguments: --cov”** → Install `pytest-cov>=4,<5` (pinned in `requirements-ci.txt`).
**Pylint `--logging-format-style=percent` error** → Use `--logging-format-style=old` with Pylint 3.
**Import errors on `evolutionary_checkpoint`** → Ensure `src/evolutionary.py` + `src/__init__.py` exist and shim is committed.
**CI vs local mismatch** → Check Python versions, secrets, and branch filters.

Handy debug:

```yaml
- name: Debug Environment
  run: |
    python --version
    pip list
    env | sort
    pwd
```

---

## Contributing

1. Add/adjust workflow in `.github/workflows/`
2. Run locally with [`act`](https://github.com/nektos/act) or push a PR
3. Keep pins in `requirements-ci.txt` updated
4. Update this README if the workflow set changes

---
