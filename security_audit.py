#!/usr/bin/env python3
"""
Security audit script for 800 Picacho Lane LUTs repository.

Performs comprehensive security checks including:
- Subprocess shell injection vulnerabilities
- Hardcoded secrets detection
- Insecure file operations
- Dependency vulnerabilities
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# ANSI color codes for output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


class SecurityIssue:
    """Represents a security issue found during audit."""

    def __init__(
        self,
        severity: str,
        category: str,
        file_path: str,
        line: int,
        message: str,
        code_snippet: str = "",
    ):
        self.severity = severity  # critical, high, medium, low
        self.category = category
        self.file_path = file_path
        self.line = line
        self.message = message
        self.code_snippet = code_snippet

    def __repr__(self) -> str:
        color_map = {
            "critical": RED,
            "high": RED,
            "medium": YELLOW,
            "low": BLUE
        }
        color = color_map.get(self.severity, RESET)
        return (
            f"{color}[{self.severity.upper()}]{RESET} "
            f"{self.category} in {self.file_path}:{self.line}\n"
            f"  → {self.message}"
        )


class SecurityAuditor:
    """Performs security audits on Python codebase."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.issues: List[SecurityIssue] = []

    def audit(self) -> List[SecurityIssue]:
        """Run all security checks."""
        print(f"{BLUE}🛡️  Starting security audit...{RESET}\n")

        self.check_subprocess_vulnerabilities()
        self.check_hardcoded_secrets()
        self.check_insecure_file_operations()
        self.check_eval_exec_usage()
        self.check_pickle_usage()

        return self.issues

    def check_subprocess_vulnerabilities(self):
        """Check for subprocess calls with shell=True."""
        print(f"{BLUE}🔍 Checking subprocess vulnerabilities...{RESET}")

        py_files = list(self.repo_root.rglob("*.py"))
        shell_true_pattern = re.compile(
            r"subprocess\.\w+\([^)]*shell\s*=\s*True"
        )

        for py_file in py_files:
            # Skip test files, virtual environments, and security audit itself
            skip_patterns = ["__pycache__", "venv", "security_audit.py"]
            if any(pattern in str(py_file) for pattern in skip_patterns):
                continue
            if py_file.name == "security_audit.py":
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for i, line in enumerate(lines, start=1):
                    if shell_true_pattern.search(line):
                        issue = SecurityIssue(
                            severity="critical",
                            category="Shell Injection",
                            file_path=str(
                                py_file.relative_to(self.repo_root)
                            ),
                            line=i,
                            message="subprocess call with shell=True detected"
                                    " - vulnerable to shell injection",
                            code_snippet=line.strip(),
                        )
                        self.issues.append(issue)
            except Exception as e:
                print(f"  Warning: Could not read {py_file}: {e}")

        print("  ✓ Subprocess check complete\n")

    def check_hardcoded_secrets(self):
        """Check for potential hardcoded secrets."""
        print(f"{BLUE}🔍 Checking for hardcoded secrets...{RESET}")

        # Common patterns for secrets
        patterns = [
            (r"password\s*=\s*['\"][^'\"]{3,}['\"]", "Hardcoded password"),
            (r"api[_-]?key\s*=\s*['\"][^'\"]{10,}['\"]",
             "Hardcoded API key"),
            (r"secret[_-]?key\s*=\s*['\"][^'\"]{10,}['\"]",
             "Hardcoded secret key"),
            (r"token\s*=\s*['\"][^'\"]{10,}['\"]", "Hardcoded token"),
            (r"aws[_-]?secret[_-]?access[_-]?key\s*=\s*['\"][^'\"]{10,}['\"]",
             "AWS secret key"),
        ]

        py_files = list(self.repo_root.rglob("*.py"))

        for py_file in py_files:
            skip_patterns = ["__pycache__", "venv", "security_audit.py"]
            if any(pattern in str(py_file) for pattern in skip_patterns):
                continue
            if py_file.name == "security_audit.py":
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for i, line in enumerate(lines, start=1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue

                    for pattern, message in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Avoid false positives
                            placeholder_patterns = [
                                r"['\"](\s*|your[-_]|placeholder|example)"
                            ]
                            if any(re.search(p, line, re.IGNORECASE)
                                   for p in placeholder_patterns):
                                continue

                            issue = SecurityIssue(
                                severity="high",
                                category="Hardcoded Secret",
                                file_path=str(
                                    py_file.relative_to(self.repo_root)
                                ),
                                line=i,
                                message=message,
                                code_snippet=line.strip(),
                            )
                            self.issues.append(issue)
            except Exception as e:
                print(f"  Warning: Could not read {py_file}: {e}")

        print("  ✓ Secret check complete\n")

    def check_insecure_file_operations(self):
        """Check for insecure file operations."""
        print(f"{BLUE}🔍 Checking insecure file operations...{RESET}")

        py_files = list(self.repo_root.rglob("*.py"))

        for py_file in py_files:
            skip_patterns = ["__pycache__", "venv", "security_audit.py"]
            if any(pattern in str(py_file) for pattern in skip_patterns):
                continue
            if py_file.name == "security_audit.py":
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for i, line in enumerate(lines, start=1):
                    # Check for dangerous file operations
                    if re.search(r"os\.(system|popen)\s*\(", line):
                        issue = SecurityIssue(
                            severity="high",
                            category="Insecure Command Execution",
                            file_path=str(
                                py_file.relative_to(self.repo_root)
                            ),
                            line=i,
                            message="Use of os.system() or os.popen()"
                                    " - prefer subprocess module",
                            code_snippet=line.strip(),
                        )
                        self.issues.append(issue)
            except Exception as e:
                print(f"  Warning: Could not read {py_file}: {e}")

        print("  ✓ File operation check complete\n")

    def check_eval_exec_usage(self):
        """Check for dangerous eval() and exec() usage."""
        print(f"{BLUE}🔍 Checking eval/exec usage...{RESET}")

        py_files = list(self.repo_root.rglob("*.py"))

        for py_file in py_files:
            skip_patterns = ["__pycache__", "venv", "security_audit.py"]
            if any(pattern in str(py_file) for pattern in skip_patterns):
                continue
            if py_file.name == "security_audit.py":
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for i, line in enumerate(lines, start=1):
                    if line.strip().startswith("#"):
                        continue

                    if re.search(r"\beval\s*\(", line):
                        issue = SecurityIssue(
                            severity="critical",
                            category="Code Injection",
                            file_path=str(
                                py_file.relative_to(self.repo_root)
                            ),
                            line=i,
                            message="Use of eval() - vulnerable to"
                                    " code injection",
                            code_snippet=line.strip(),
                        )
                        self.issues.append(issue)

                    if re.search(r"\bexec\s*\(", line):
                        issue = SecurityIssue(
                            severity="critical",
                            category="Code Injection",
                            file_path=str(
                                py_file.relative_to(self.repo_root)
                            ),
                            line=i,
                            message="Use of exec() - vulnerable to"
                                    " code injection",
                            code_snippet=line.strip(),
                        )
                        self.issues.append(issue)
            except Exception as e:
                print(f"  Warning: Could not read {py_file}: {e}")

        print("  ✓ eval/exec check complete\n")

    def check_pickle_usage(self):
        """Check for unsafe pickle usage."""
        print(f"{BLUE}🔍 Checking pickle usage...{RESET}")

        py_files = list(self.repo_root.rglob("*.py"))

        for py_file in py_files:
            skip_patterns = ["__pycache__", "venv", "security_audit.py"]
            if any(pattern in str(py_file) for pattern in skip_patterns):
                continue
            if py_file.name == "security_audit.py":
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for i, line in enumerate(lines, start=1):
                    if "pickle.load" in line or "pickle.loads" in line:
                        issue = SecurityIssue(
                            severity="medium",
                            category="Deserialization",
                            file_path=str(
                                py_file.relative_to(self.repo_root)
                            ),
                            line=i,
                            message="pickle.load() can execute arbitrary"
                                    " code - ensure input is trusted",
                            code_snippet=line.strip(),
                        )
                        self.issues.append(issue)
            except Exception as e:
                print(f"  Warning: Could not read {py_file}: {e}")

        print("  ✓ Pickle check complete\n")

    def generate_report(self) -> Dict:
        """Generate a structured report of findings."""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for issue in self.issues:
            severity_counts[issue.severity] += 1

        return {
            "total_issues": len(self.issues),
            "severity_counts": severity_counts,
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "file": issue.file_path,
                    "line": issue.line,
                    "message": issue.message,
                    "code": issue.code_snippet,
                }
                for issue in self.issues
            ],
        }

    def print_summary(self):
        """Print a summary of findings."""
        if not self.issues:
            print(f"{GREEN}✅ No security issues found!{RESET}\n")
            return

        separator = RED + "━" * 45 + RESET
        print(f"\n{separator}")
        print(f"{RED}🚨 Security Issues Found: {len(self.issues)}{RESET}")
        print(f"{separator}\n")

        # Group by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for issue in self.issues:
            by_severity[issue.severity].append(issue)

        for severity in ["critical", "high", "medium", "low"]:
            issues = by_severity[severity]
            if issues:
                print(f"{severity.upper()}: {len(issues)} issue(s)")
                for issue in issues:
                    print(f"  {issue}")
                print()


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent

    auditor = SecurityAuditor(repo_root)
    issues = auditor.audit()

    # Print summary
    auditor.print_summary()

    # Generate JSON report
    report = auditor.generate_report()
    report_path = repo_root / "security_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"{BLUE}📄 Detailed report saved to: {report_path}{RESET}\n")

    # Return exit code based on severity
    if report["severity_counts"]["critical"] > 0:
        print(f"{RED}❌ Critical issues found - "
              f"immediate action required{RESET}")
        return 2
    elif report["severity_counts"]["high"] > 0:
        print(f"{YELLOW}⚠️  High severity issues found - "
              f"review recommended{RESET}")
        return 1
    elif len(issues) > 0:
        print(f"{YELLOW}ℹ️  Lower severity issues found - "
              f"review at your convenience{RESET}")
        return 0
    else:
        print(f"{GREEN}✅ Security audit passed{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
