#!/usr/bin/env python3
"""
Test suite for security tooling.

Tests the security_audit.py script and emergency_security_fixes.sh.
"""

import subprocess
import sys
from pathlib import Path


def test_security_audit_runs():
    """Test that security audit script executes successfully."""
    result = subprocess.run(
        [sys.executable, "security_audit.py"],
        capture_output=True,
        text=True,
        check=False
    )
    
    # Should exit with 0 (no critical issues in current codebase)
    assert result.returncode == 0, f"Security audit failed: {result.stderr}"
    print("✓ Security audit runs successfully")


def test_security_report_generated():
    """Test that security report JSON is generated."""
    report_path = Path("security_report.json")
    
    # Run audit first
    subprocess.run([sys.executable, "security_audit.py"], check=True)
    
    assert report_path.exists(), "security_report.json not created"
    
    import json
    with open(report_path, "r") as f:
        data = json.load(f)
    
    assert "total_issues" in data
    assert "severity_counts" in data
    assert "issues" in data
    
    print("✓ Security report generated correctly")


def test_emergency_script_exists():
    """Test that emergency fixes script exists and is executable."""
    script_path = Path("emergency_security_fixes.sh")
    assert script_path.exists(), "emergency_security_fixes.sh not found"
    assert script_path.stat().st_mode & 0o111, "Script is not executable"
    print("✓ Emergency script exists and is executable")


def test_requirements_security_exists():
    """Test that security requirements file exists."""
    req_path = Path("requirements-security.txt")
    assert req_path.exists(), "requirements-security.txt not found"
    
    content = req_path.read_text()
    assert "cryptography" in content
    assert "bandit" in content
    assert "safety" in content
    
    print("✓ Security requirements file exists with expected packages")


def test_no_shell_true_in_main_code():
    """Test that main Python files don't use shell=True."""
    main_files = [
        "luxury_video_master_grader.py",
        "luxury_tiff_batch_processor_cli.py",
    ]
    
    for file_path in main_files:
        path = Path(file_path)
        if not path.exists():
            continue
        
        content = path.read_text()
        # Check for shell=True in non-comment lines
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                continue
            if "shell=True" in line:
                assert False, f"Found shell=True in {file_path}"
    
    print("✓ No shell=True found in main code files")


def main():
    """Run all tests."""
    print("🧪 Testing security tooling...\n")
    
    tests = [
        test_emergency_script_exists,
        test_requirements_security_exists,
        test_no_shell_true_in_main_code,
        test_security_audit_runs,
        test_security_report_generated,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error: {e}")
            failed.append(test.__name__)
    
    print("\n" + "="*50)
    if failed:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1
    else:
        print("✅ All security tooling tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
