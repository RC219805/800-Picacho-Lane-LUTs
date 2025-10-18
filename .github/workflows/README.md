# GitHub Actions Workflows

This directory contains GitHub Actions workflow definitions for the 800 Picacho Lane LUTs repository.

## Table of Contents
- [Active Workflows](#active-workflows)
- [Experimental Workflows](#experimental-workflows)
- [Workflow Configuration](#workflow-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Active Workflows

### `python-app.yml`

**Status**: ✅ Active  
Main CI workflow for Python testing, linting, and coverage.

**Triggers**:
- Push to `main` (and optionally `develop` if enabled in the workflow)
- Pull requests targeting `main`

**Matrix Strategy**:
- Python: 3.10, 3.11, 3.12
- OS: Ubuntu latest

**Key Steps**:
1. Checkout
2. Set up Python **with built-in pip caching**
3. Install dev dependencies
4. Run tests with coverage
5. Upload coverage (artifact + Codecov)

**Config Snippets**:

> **Badge** (replace with your actual file name if different):
```markdown
![Python CI](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/python-app.yml/badge.svg?branch=main)

Concurrency:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

Minimal Permissions (add where needed per job):

permissions:
  contents: read

Built-in pip cache:

- uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
    cache: pip
    cache-dependency-path: |
      **/requirements*.txt

Codecov v5 (OIDC; no token needed):

permissions:
  id-token: write  # required for Codecov OIDC

- name: Upload Coverage to Codecov (OIDC)
  if: always()
  uses: codecov/codecov-action@v5
  with:
    use_oidc: true
    files: ./coverage.xml
    flags: unittests
    fail_ci_if_error: false


⸻

pylint.yml

Status: ✅ Active
Static analysis with Pylint.

Triggers:
	•	Push to main
	•	Pull requests

Configuration:
	•	Uses repository .pylintrc
	•	Example threshold: --fail-under=8.0
	•	Python 3.11 runner

Common Issues:
	•	Too many branches → refactor
	•	Line too long → keep ≤100
	•	Missing docstrings → add module/function docs

⸻

codeql.yml

Status: ✅ Active
Security scanning with GitHub CodeQL.

Triggers:
	•	Push to main
	•	Pull requests
	•	Scheduled weekly scan

Language:
	•	Python

Results: Security tab → Code scanning alerts

⸻

Experimental Workflows

summary.yml — AI Issue Summarizer

Status: ⚠️ Experimental (original version non-functional)

What was broken
	•	The action actions/ai-inference@v2 does not exist; replace with a real API call step.

Working Reference Implementation (OpenAI)

Prerequisites:
	•	Create secret OPENAI_API_KEY

Workflow Step:

- name: Generate AI Summary
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ISSUE_NUMBER: ${{ github.event.issue.number }}
    ISSUE_BODY: ${{ github.event.issue.body }}
  run: |
    python -m pip install --upgrade pip
    pip install openai
    python scripts/summarize_issue.py

scripts/summarize_issue.py (uses $GITHUB_OUTPUT, handles multiline safely):

# scripts/summarize_issue.py
import os
from openai import OpenAI

def main() -> None:
    issue_body = os.environ.get("ISSUE_BODY", "") or "(no body)"
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Keep it short and safe for comments
    prompt = (
        "Summarize the following GitHub issue in 3-5 bullet points, "
        "focusing on problem, context, proposed fix, and blockers. "
        "Return plain text only.\n\n"
        f"{issue_body}"
    )

    # Prefer Responses API; fallback to chat if needed.
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        summary = getattr(resp, "output_text", "").strip() or "Summary unavailable."
    except Exception as exc:  # Why: ensure workflow continues gracefully
        summary = f"Summarization failed: {exc}"

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write("summary<<EOF\n")
            f.write(summary.replace("\r", "") + "\n")
            f.write("EOF\n")
    else:
        print(summary)

if __name__ == "__main__":
    main()

Post a Comment (optional):

- name: Comment summary on the issue
  if: steps.generate.outputs.summary != ''
  env:
    GH_TOKEN: ${{ github.token }}
  run: |
    gh issue comment ${{ github.event.issue.number }} --body "${{ steps.generate.outputs.summary }}"

Graceful Failure:

continue-on-error: true


⸻

Workflow Configuration

Environment Variables

Common, safe defaults:

env:
  PYTHONUNBUFFERED: 1
  PYTHONDONTWRITEBYTECODE: 1
  # Note: do NOT set PIP_NO_CACHE_DIR=1 if using pip cache

Caching Strategy

Preferred: Use actions/setup-python@v5 built-in caching:

- uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
    cache: pip
    cache-dependency-path: |
      **/requirements*.txt

Alternative: actions/cache when you need custom paths (wheels, mypy cache, etc.).

Concurrency Control

Prevent redundant runs:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

Action Pinning

For security, pin actions to a full commit SHA:

uses: actions/checkout@<full-commit-sha>


⸻

Troubleshooting

Workflow Not Triggering
	•	Validate YAML
	•	Ensure branch filters match
	•	Confirm workflow is enabled

CI Fails but Local Passes
	•	Python version mismatch
	•	Missing env vars
	•	Timezone differences
	•	Case-sensitive paths on Linux

Debug block:

- name: Debug Environment
  run: |
    python --version
    pip list
    env | sort
    pwd

Slow Runs
	•	Enable dependency caching
	•	Limit checkout depth fetch-depth: 1
	•	Parallelize tests
	•	Consider pytest -n auto (pytest-xdist)

Permission Errors

permissions:
  contents: read
  issues: write          # required if posting comments
  pull-requests: write   # if commenting on PRs


⸻

Security Best Practices

Secrets
	•	Never commit API keys, tokens, passwords.
	•	Use GitHub Secrets: ${{ secrets.MY_SECRET }}.

Dependency Security
	•	Enable Dependabot:

# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers: ["RC219805"]

Pin Actions

Use full commit SHAs for third-party actions.

⸻

Performance Metrics

(Keep your table here if you track these in CI dashboards.)

⸻

Contributing
	1.	Add workflow under .github/workflows/
	2.	Test locally with act
	3.	Update this README
	4.	Open a PR

Local testing:

act -j test
act pull_request

Naming:
	•	Descriptive (python-tests.yml)
	•	Kebab-case
	•	Specific (pylint-check.yml)

**Why these updates**
- **Badges**: file-based badge URLs are the recommended, robust format.  [oai_citation:1‡GitHub Docs](https://docs.github.com/actions/managing-workflow-runs/adding-a-workflow-status-badge?utm_source=chatgpt.com)  
- **Pip caching**: `actions/setup-python@v5` has first-class `cache: pip`; simpler than manual `actions/cache`.  [oai_citation:2‡GitHub](https://github.com/actions/setup-python?utm_source=chatgpt.com)  
- **Codecov**: v5 supports OIDC; enable with `use_oidc: true` and `permissions: id-token: write`.  [oai_citation:3‡GitHub](https://github.com/codecov/codecov-action?utm_source=chatgpt.com)  
- **`set-output`**: deprecated; use `$GITHUB_OUTPUT` env file instead.  [oai_citation:4‡The GitHub Blog](https://github.blog/changelog/2022-10-10-github-actions-deprecating-save-state-and-set-output-commands/?utm_source=chatgpt.com)  
- **Action pinning**: pin to full commit SHA for supply-chain safety.  [oai_citation:5‡GitHub Docs](https://docs.github.com/en/actions/reference/security/secure-use?utm_source=chatgpt.com)  
- **OpenAI SDK**: modern client usage (`from openai import OpenAI`) documented in the current SDK docs.  [oai_citation:6‡OpenAI Platform](https://platform.openai.com/docs/libraries/python-library?utm_source=chatgpt.com)

**a.** Want me to turn this into a working `summary.yml` + full PR comment step wired to your repo?  
**b.** Should I align README sections to your exact workflow file names and add live badges for each?