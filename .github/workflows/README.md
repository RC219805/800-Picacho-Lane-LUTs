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
