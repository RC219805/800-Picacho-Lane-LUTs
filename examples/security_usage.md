# Security Tooling Usage Examples

## Quick Security Scan

Run a quick security audit on the entire codebase:

```bash
python security_audit.py
```

Expected output if no issues:
```
🛡️  Starting security audit...
...
✅ No security issues found!
```

## Emergency Security Update

Update all security-critical dependencies and run comprehensive scan:

```bash
./emergency_security_fixes.sh
```

This will:
1. Update packages from `requirements-security.txt`
2. Scan for vulnerabilities
3. Generate `security_report.json`

## Integrate with CI/CD

Add to `.github/workflows/security.yml`:

```yaml
name: Security Audit

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: pip install -r requirements-security.txt
      
      - name: Run security audit
        run: python security_audit.py
```

## Review Security Report

After running the audit, check the JSON report:

```bash
python -m json.tool security_report.json
```

Or use jq for better formatting:

```bash
jq '.issues[] | select(.severity == "critical")' security_report.json
```

## Fix Common Issues

### Shell Injection (subprocess with shell=True)

**Before:**
```python
import subprocess
subprocess.run(f"ffmpeg -i {input_file} {output_file}", shell=True)
```

**After:**
```python
import subprocess
subprocess.run(
    ["ffmpeg", "-i", input_file, output_file],
    check=True
)
```

### Hardcoded Secrets

**Before:**
```python
API_KEY = "sk-1234567890abcdef"
```

**After:**
```python
import os
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable required")
```

### Unsafe eval()

**Before:**
```python
result = eval(user_input)
```

**After:**
```python
import ast
result = ast.literal_eval(user_input)  # Only for literals
```

## Run Tests

Test the security tooling:

```bash
python tests/test_security_tooling.py
```

Or with pytest:

```bash
pytest tests/test_security_tooling.py -v
```

## Check Specific Files

Audit a specific file:

```bash
# Using grep for quick check
grep -n "shell=True" myfile.py

# Using bandit for detailed analysis
bandit -r myfile.py
```

## Update Dependencies

Keep security packages up to date:

```bash
# See what would be updated
pip list --outdated | grep -E "(cryptography|urllib3|requests|bandit|safety)"

# Update security packages
pip install --upgrade -r requirements-security.txt
```

## Automate with Pre-commit

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: security-audit
        name: Security Audit
        entry: python security_audit.py
        language: system
        pass_filenames: false
```

Install:
```bash
pip install pre-commit
pre-commit install
```

Now security audit runs before every commit!
