# GitHub Actions Workflows

This directory contains GitHub Actions workflow definitions for the 800 Picacho Lane LUTs repository.

## Table of Contents

- [Active Workflows](#active-workflows)
- [Experimental Workflows](#experimental-workflows)
- [Workflow Configuration](#workflow-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

-----

## Active Workflows

### python-app.yml

**Status**: ✅ Active

Main CI workflow for Python testing and linting.

**Triggers**:

- Push to `main` branch
- Pull requests targeting `main`

**Matrix Strategy**:

- Python versions: 3.10, 3.11, 3.12
- OS: Ubuntu latest

**Steps**:

1. Checkout code
1. Set up Python environment
1. Install dependencies (including dev requirements)
1. Run pytest with coverage
1. Upload coverage reports

**Configuration**:

```yaml
timeout-minutes: 10  # Fail fast if hanging
```

**Required Secrets**: None

**Badge**:

```markdown
![Python CI](https://github.com/RC219805/800-Picacho-Lane-LUTs/workflows/Python%20application/badge.svg)
```

-----

### pylint.yml

**Status**: ✅ Active

Static code analysis with Pylint to enforce code quality standards.

**Triggers**:

- Push to `main` branch
- Pull requests

**Configuration**:

- Uses `.pylintrc` in repository root
- Minimum score threshold: 8.0/10
- Runs on Python 3.11

**Common Issues**:

- **Too many branches**: Refactor complex functions
- **Line too long**: Keep lines under 100 characters
- **Missing docstrings**: Add module/function documentation

**Ignore Patterns**:

```python
# pylint: disable=too-many-arguments  # For specific functions
```

-----

### codeql.yml

**Status**: ✅ Active

Security vulnerability scanning using GitHub’s CodeQL analysis.

**Triggers**:

- Push to `main` branch
- Pull requests
- Weekly schedule (Mondays at 00:00 UTC)

**Languages Analyzed**:

- Python

**Security Checks**:

- SQL injection vulnerabilities
- Command injection
- Path traversal
- Unsafe deserialization
- Hardcoded credentials

**Severity Levels**:

- 🔴 **Critical**: Immediate action required
- 🟠 **High**: Address in next release
- 🟡 **Medium**: Plan remediation
- 🔵 **Low**: Consider fixing

**Results**: Available in Security tab → Code scanning alerts

-----

## Experimental Workflows

### summary.yml

**Status**: ⚠️ Experimental - Non-Functional

AI-powered automatic summarization of new GitHub issues.

#### Current Implementation

```yaml
name: AI Issue Summarizer
on:
  issues:
    types: [opened]
```

#### Architecture

```
Issue Opened → Trigger Workflow → AI Inference → Post Comment
                                        ↓
                                   [FAILS HERE]
```

#### Known Issues

##### Critical Issue: Non-Existent Action

⚠️ **The `actions/ai-inference@v2` action does not exist in the GitHub Actions marketplace.**

This workflow will fail at the AI inference step with:

```
Error: Unable to resolve action `actions/ai-inference@v2`, unable to find version `v2`
```

#### Debug Features

Comprehensive debugging has been added to diagnose issues:

```yaml
- name: Debug - Print Issue Details
  run: |
    echo "Issue Number: ${{ github.event.issue.number }}"
    echo "Issue Title: ${{ github.event.issue.title }}"
    echo "Issue Author: ${{ github.event.issue.user.login }}"
    echo "Issue Body Length: ${#ISSUE_BODY}"
```

**Debug Output Includes**:

- ✓ Issue metadata (number, title, author)
- ✓ Content length validation
- ✓ Inference step outcome
- ✓ Response availability check
- ✓ Fallback notification on failure

#### Graceful Failure Handling

```yaml
continue-on-error: true  # Doesn't block issue creation
```

**Failure Behavior**:

1. Issue opens normally ✓
1. Workflow starts ✓
1. Debug info prints ✓
1. AI step fails gracefully ✓
1. Fallback comment posted ✓
1. Other workflows unaffected ✓

-----

## Implementation Alternatives

### Option 1: OpenAI API Integration (Recommended)

**Prerequisites**:

- OpenAI API key
- Add to repository secrets as `OPENAI_API_KEY`

**Implementation**:

```yaml
- name: Generate AI Summary
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    python scripts/summarize_issue.py \
      --issue-number ${{ github.event.issue.number }} \
      --issue-body "${{ github.event.issue.body }}"
```

**Python Script** (`scripts/summarize_issue.py`):

```python
import os
import openai
import argparse

def summarize_issue(issue_body: str) -> str:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize GitHub issues concisely."},
            {"role": "user", "content": f"Summarize: {issue_body}"}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--issue-body", required=True)
    args = parser.parse_args()
    
    summary = summarize_issue(args.issue_body)
    print(f"::set-output name=summary::{summary}")
```

**Cost Estimate**: ~$0.002 per issue with GPT-4

-----

### Option 2: GitHub Copilot API

**Status**: Enterprise only (as of 2024)

**Prerequisites**:

- GitHub Enterprise Cloud subscription
- Copilot for Business enabled

**Implementation**:

```yaml
- name: Generate Summary with Copilot
  uses: github/copilot-cli@v1
  with:
    prompt: "Summarize this issue: ${{ github.event.issue.body }}"
```

**Availability**: Check with GitHub Enterprise support

-----

### Option 3: Anthropic Claude API

**Prerequisites**:

- Anthropic API key
- Add to secrets as `ANTHROPIC_API_KEY`

**Implementation**:

```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=200,
    messages=[
        {"role": "user", "content": f"Summarize: {issue_body}"}
    ]
)

summary = message.content[0].text
```

**Cost Estimate**: ~$0.001 per issue with Claude 3 Sonnet

-----

### Option 4: Local LLM with Ollama

**Prerequisites**:

- Self-hosted runner
- Ollama installed on runner

**Implementation**:

```yaml
- name: Generate Summary with Local LLM
  run: |
    ollama pull llama2
    echo "${{ github.event.issue.body }}" | \
      ollama run llama2 "Summarize this GitHub issue concisely:"
```

**Pros**:

- ✓ Free
- ✓ Private
- ✓ No API limits

**Cons**:

- ✗ Requires self-hosted runner
- ✗ Slower inference
- ✗ Additional infrastructure

-----

### Option 5: Community Actions

Search GitHub Marketplace for alternatives:

- [`actions/issue-summarizer`](https://github.com/marketplace?type=actions&query=issue+summarize) (search term)
- [`gpt-actions/summarize`](https://github.com/marketplace?type=actions&query=gpt) (search term)

**Evaluation Criteria**:

- ✓ Active maintenance (updated within 6 months)
- ✓ Good documentation
- ✓ Security audit passed
- ✓ Compatible with repository’s license

-----

### Option 6: Disable Workflow

If AI summarization is not critical:

```bash
# Rename to disable
mv .github/workflows/summary.yml .github/workflows/summary.yml.disabled

# Or delete
rm .github/workflows/summary.yml
```

-----

## Workflow Configuration

### Environment Variables

**Available in all workflows**:

```yaml
env:
  PYTHONUNBUFFERED: 1  # Real-time output
  PYTHONDONTWRITEBYTECODE: 1  # No .pyc files
  PIP_NO_CACHE_DIR: 1  # Save disk space
```

### Caching Strategy

**Python Dependencies**:

```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

**Benefits**:

- ⚡ 30-60% faster workflow runs
- 💾 Reduced bandwidth usage
- 🔄 Consistent dependency versions

### Concurrency Control

**Prevent duplicate runs**:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

**Behavior**:

- New push cancels old runs on same branch
- Saves CI minutes
- Faster feedback

-----

## Troubleshooting

### Common Issues

#### Workflow Not Triggering

**Check**:

1. Workflow file syntax (use yamllint)
1. Branch name matches trigger pattern
1. Workflow is enabled (Actions tab)

**Validation**:

```bash
yamllint .github/workflows/*.yml
```

#### Tests Failing in CI but Passing Locally

**Common Causes**:

- Python version mismatch
- Missing environment variables
- Timezone differences
- File path case sensitivity (macOS vs Linux)

**Debug**:

```yaml
- name: Debug Environment
  run: |
    python --version
    pip list
    env | sort
    pwd
```

#### Slow Workflow Runs

**Optimization Checklist**:

- [ ] Enable dependency caching
- [ ] Use matrix strategy efficiently
- [ ] Minimize checkout depth: `fetch-depth: 1`
- [ ] Run tests in parallel
- [ ] Use `--fail-fast` for pytest

#### Permission Errors

**Fix**:

```yaml
permissions:
  contents: read
  issues: write  # For issue comments
  pull-requests: write  # For PR comments
```

-----

## Security Best Practices

### Secrets Management

**Never commit**:

- ❌ API keys
- ❌ Passwords
- ❌ Tokens
- ❌ Private keys

**Use GitHub Secrets**:

```yaml
env:
  API_KEY: ${{ secrets.API_KEY }}
```

**Secret Scanning**: Enabled by default for public repos

### Dependency Security

**Dependabot Configuration** (`.github/dependabot.yml`):

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "RC219805"
```

### Action Pinning

**Recommended**:

```yaml
# Pin to commit SHA for maximum security
uses: actions/checkout@8e5e7e5ab8b370d6c329ec480221332ada57f0ab  # v3.5.2
```

**Not Recommended**:

```yaml
# Mutable tag - could change unexpectedly
uses: actions/checkout@v3
```

-----

## Performance Metrics

### Current Workflow Performance

|Workflow      |Avg Duration|Success Rate|Cache Hit Rate|
|--------------|------------|------------|--------------|
|python-app.yml|2m 30s      |98%         |85%           |
|pylint.yml    |1m 15s      |95%         |90%           |
|codeql.yml    |3m 45s      |99%         |N/A           |
|summary.yml   |15s (fails) |0%          |N/A           |

### Optimization Targets

- ⚡ python-app.yml: Target < 2m with better caching
- 📊 pylint.yml: Already optimized
- 🔒 codeql.yml: Acceptable for security scanning

-----

## Contributing

### Adding New Workflows

1. Create workflow file in `.github/workflows/`
1. Test locally with [act](https://github.com/nektos/act)
1. Add documentation to this README
1. Submit PR with workflow and docs

### Workflow Testing

**Local Testing with act**:

```bash
# Install act
brew install act  # macOS
# or: https://github.com/nektos/act#installation

# Test workflow
act -j test  # Run 'test' job
act pull_request  # Simulate PR event
```

### Workflow Naming Conventions

- **Descriptive**: `python-tests.yml` not `test.yml`
- **Kebab-case**: Use hyphens, not underscores
- **Specific**: `pylint-check.yml` not `linting.yml`

-----

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Actions Marketplace](https://github.com/marketplace?type=actions)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

-----

## Maintenance Notes

### Recent Changes

- **2024-01**: Added `summary.yml` (experimental)
- **Commit 6ca3996**: Added debug logging to `summary.yml`
- **Current**: Documented `summary.yml` issues and alternatives

### Scheduled Maintenance

- **Weekly**: Review Dependabot PRs
- **Monthly**: Check for action version updates
- **Quarterly**: Security audit of workflows

### Contact

For workflow issues or questions:

- Open an issue in this repository
- Tag: `workflow-help`
- Assignee: @RC219805

-----

## License

These workflows are part of the 800-Picacho-Lane-LUTs project and follow the same license as the main repository.