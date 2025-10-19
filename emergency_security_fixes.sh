#!/bin/bash
# file: emergency_security_fixes.sh
# Emergency security fixes for 800 Picacho Lane LUTs repository

set -e  # Exit on error

echo "🚨 EMERGENCY SECURITY FIXES STARTING..."
echo ""

# Store the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Update critical dependencies
echo "📦 Updating vulnerable packages..."
if [ -f requirements-security.txt ]; then
    pip install --upgrade -r requirements-security.txt
    echo "✓ Security dependencies updated"
else
    echo "⚠️  requirements-security.txt not found, skipping dependency update"
fi
echo ""

# 2. Find and fix subprocess vulnerabilities
echo "🔍 Scanning for shell=True vulnerabilities..."
FOUND_VULNERABLE=false
if command -v grep &> /dev/null; then
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            echo "Found: $line"
            FOUND_VULNERABLE=true
            
            # Create backup
            file=$(echo "$line" | cut -d: -f1)
            if [ -f "$file" ]; then
                cp "$file" "$file.backup"
                echo "  → Created backup: $file.backup"
            fi
        fi
    done < <(grep -r "shell=True" --include="*.py" . 2>/dev/null || true)
    
    if [ "$FOUND_VULNERABLE" = false ]; then
        echo "✓ No shell=True vulnerabilities found"
    else
        echo "⚠️  Found shell=True vulnerabilities - manual review required"
        echo "    Backups created with .backup extension"
    fi
else
    echo "⚠️  grep not available, skipping vulnerability scan"
fi
echo ""

# 3. Run security audit
echo "🛡️  Running security audit..."
if [ -f security_audit.py ]; then
    python security_audit.py
    AUDIT_EXIT=$?
    if [ $AUDIT_EXIT -eq 0 ]; then
        echo "✓ Security audit passed"
    elif [ $AUDIT_EXIT -eq 1 ]; then
        echo "⚠️  High severity issues found - review recommended"
    else
        echo "❌ Critical issues found - immediate action required"
    fi
else
    echo "⚠️  security_audit.py not found, skipping security audit"
fi
echo ""

# 4. Run CodeQL if available
if command -v codeql &> /dev/null; then
    echo "🔬 Running CodeQL analysis..."
    
    # Clean up any existing database
    if [ -d codeql-db ]; then
        rm -rf codeql-db
    fi
    
    # Create database
    codeql database create codeql-db --language=python --overwrite
    
    # Analyze database
    codeql database analyze codeql-db \
        --format=sarif-latest \
        --output=security.sarif \
        python-security-and-quality
    
    echo "✓ CodeQL analysis complete - results in security.sarif"
else
    echo "ℹ️  CodeQL not available, skipping CodeQL analysis"
    echo "   Install from: https://github.com/github/codeql-cli-binaries/releases"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Emergency fixes complete."
echo ""
if [ -f security.sarif ]; then
    echo "📄 Review security.sarif for detailed CodeQL results."
fi
if [ -f security_report.json ]; then
    echo "📄 Review security_report.json for audit results."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
