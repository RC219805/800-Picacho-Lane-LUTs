# Security Documentation

## Overview

This repository includes comprehensive security tooling to protect against common vulnerabilities in Python applications. The security infrastructure includes:

1. **Emergency Security Fixes Script** (`emergency_security_fixes.sh`)
2. **Security Audit Tool** (`security_audit.py`)
3. **Security Dependencies** (`requirements-security.txt`)

## Quick Start

Run the emergency security fixes script to update dependencies and audit the codebase:

```bash
./emergency_security_fixes.sh
```

This script will:
- ✅ Update all security-critical dependencies
- ✅ Scan for `shell=True` vulnerabilities in subprocess calls
- ✅ Run comprehensive security audit
- ✅ Generate detailed security reports
- ✅ (Optional) Run CodeQL analysis if installed

## Security Tools

### 1. Emergency Security Fixes (`emergency_security_fixes.sh`)

Automated script that performs emergency security updates and scans.

**Usage:**
```bash
chmod +x emergency_security_fixes.sh
./emergency_security_fixes.sh
```

**What it does:**
- Updates all packages in `requirements-security.txt` to their latest secure versions
- Scans for dangerous subprocess patterns (shell injection vulnerabilities)
- Runs the security audit tool
- Creates backups of vulnerable files (`.backup` extension)
- Optionally runs CodeQL analysis if available

### 2. Security Audit Tool (`security_audit.py`)

Comprehensive Python security scanner that checks for common vulnerabilities.

**Usage:**
```bash
python security_audit.py
```

**Checks performed:**

#### Subprocess Shell Injection (Critical)
- Detects `subprocess` calls with `shell=True`
- Shell injection allows arbitrary command execution
- **Fix:** Use `shell=False` and pass commands as lists

```python
# ❌ Vulnerable
subprocess.run(f"ls {user_input}", shell=True)

# ✅ Safe
subprocess.run(["ls", user_input], shell=False)
```

#### Hardcoded Secrets (High)
- Detects hardcoded passwords, API keys, tokens
- Secrets in code can be exposed in version control
- **Fix:** Use environment variables or secure vaults

```python
# ❌ Vulnerable
password = "mySecretPassword123"

# ✅ Safe
password = os.environ.get("PASSWORD")
```

#### Insecure Command Execution (High)
- Detects `os.system()` and `os.popen()` usage
- These are less secure than subprocess module
- **Fix:** Use subprocess with proper argument passing

```python
# ❌ Vulnerable
os.system("ls -la")

# ✅ Safe
subprocess.run(["ls", "-la"], check=True)
```

#### Code Injection (Critical)
- Detects `eval()` and `exec()` usage
- Can execute arbitrary code from user input
- **Fix:** Use safer alternatives like `ast.literal_eval()` for data

```python
# ❌ Vulnerable
result = eval(user_input)

# ✅ Safe
import ast
result = ast.literal_eval(user_input)  # Only for literals
```

#### Unsafe Deserialization (Medium)
- Detects `pickle.load()` and `pickle.loads()`
- Can execute arbitrary code when unpickling untrusted data
- **Fix:** Ensure input is from trusted sources only

```python
# ⚠️ Use with caution
import pickle
data = pickle.load(file)  # Only for trusted files
```

**Exit Codes:**
- `0` - No critical/high issues (may have medium/low)
- `1` - High severity issues found
- `2` - Critical severity issues found

**Output:**
- Console output with color-coded severity levels
- `security_report.json` - Detailed JSON report with all findings

### 3. Security Dependencies (`requirements-security.txt`)

Pinned versions of security-critical packages with known vulnerabilities patched.

**Included packages:**
- `cryptography>=42.0.0` - CVE fixes for cryptographic operations
- `urllib3>=2.2.0` - Security fixes for HTTP handling
- `requests>=2.32.0` - Security patches for HTTP library
- `bandit>=1.7.5` - Python security linter
- `safety>=3.0.0` - Dependency vulnerability checker
- `PyYAML>=6.0.1` - Fixes arbitrary code execution
- `Jinja2>=3.1.3` - Template injection fixes

**Update:**
```bash
pip install --upgrade -r requirements-security.txt
```

## Security Best Practices

### Subprocess Security
Always use `subprocess` with `shell=False` and pass arguments as lists:

```python
import subprocess

# Safe pattern
result = subprocess.run(
    ["ffmpeg", "-i", input_file, output_file],
    capture_output=True,
    text=True,
    check=True
)
```

### Input Validation
Always validate and sanitize user input:

```python
from pathlib import Path

def safe_file_path(user_path):
    """Safely resolve file path preventing directory traversal."""
    base = Path("/allowed/directory")
    target = (base / user_path).resolve()
    
    # Ensure target is within base directory
    if not target.is_relative_to(base):
        raise ValueError("Invalid path")
    
    return target
```

### Secrets Management
Never hardcode secrets. Use environment variables:

```python
import os

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable required")
```

### Dependency Management
Regularly update dependencies and check for vulnerabilities:

```bash
# Update security packages
pip install --upgrade -r requirements-security.txt

# Check for vulnerabilities (requires safety)
safety check
```

## CodeQL Integration

CodeQL is GitHub's semantic code analysis engine. If you have CodeQL installed, the emergency fixes script will automatically run it.

**Install CodeQL:**
1. Download from: https://github.com/github/codeql-cli-binaries/releases
2. Extract and add to PATH
3. Run: `codeql --version` to verify

**Manual CodeQL usage:**
```bash
# Create database
codeql database create codeql-db --language=python

# Analyze
codeql database analyze codeql-db \
    --format=sarif-latest \
    --output=security.sarif \
    python-security-and-quality
```

Results will be in `security.sarif` (SARIF format - viewable in many security tools).

## Continuous Integration

Add security checks to your CI/CD pipeline:

```yaml
# .github/workflows/security.yml
name: Security Scan

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
        run: |
          pip install -r requirements-security.txt
      
      - name: Run security audit
        run: python security_audit.py
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
```

## Troubleshooting

### Script Permission Denied
```bash
chmod +x emergency_security_fixes.sh
```

### Python Import Errors
Ensure dependencies are installed:
```bash
pip install -r requirements-security.txt
```

### False Positives
The security audit tool may flag legitimate uses of patterns (e.g., documentation). Review each finding in context. The tool already excludes itself from scanning.

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [CodeQL for Python](https://codeql.github.com/docs/codeql-language-guides/codeql-for-python/)

## Support

For security issues, please:
1. Review the generated `security_report.json`
2. Check this documentation for mitigation strategies
3. Refer to the official security advisories for affected packages
4. Consider opening a security advisory (not a public issue) for sensitive findings

---

Last updated: 2025-10-19
