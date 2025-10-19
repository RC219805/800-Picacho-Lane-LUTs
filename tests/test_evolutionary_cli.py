"""Tests for src.evolutionary CLI interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_stable_status() -> None:
    """Test CLI with stable status."""
    result = subprocess.run(
        [sys.executable, "-m", "src.evolutionary", "2025-12-31", "migrate/v2", "--today", "2025-12-30"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "STABLE: Current form viable until 2025-12-31" in result.stdout


def test_cli_evolution_required() -> None:
    """Test CLI with evolution required status."""
    result = subprocess.run(
        [sys.executable, "-m", "src.evolutionary", "2025-01-01", "migrate/v3", "--today", "2025-01-02"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "EVOLUTION REQUIRED: Migrate to migrate/v3" in result.stdout


def test_cli_json_output() -> None:
    """Test CLI with JSON output format."""
    result = subprocess.run(
        [sys.executable, "-m", "src.evolutionary", "2025-12-31", "migrate/v2", "--today", "2025-12-30", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["status"] == "stable"
    assert data["horizon"] == "2025-12-31"
    assert data["mutation_path"] == "migrate/v2"
    assert data["today"] == "2025-12-30"
    assert data["due_in_days"] == 1
    assert data["overdue_by_days"] is None


def test_cli_invalid_date_format() -> None:
    """Test CLI with invalid date format."""
    result = subprocess.run(
        [sys.executable, "-m", "src.evolutionary", "not-a-date", "migrate/v2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Invalid date" in result.stderr
