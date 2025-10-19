# file: tests/test_lux_batch_cli_smoke.py
from __future__ import annotations

import importlib
import types

import pytest


def test_cli_help_works():
    typer = pytest.importorskip("typer")
    from typer.testing import CliRunner

    # Import the module
    mod = importlib.import_module("luxury_tiff_batch_processor.cli")
    assert isinstance(mod, types.ModuleType)
    assert hasattr(mod, "app")

    runner = CliRunner()
    # top-level help
    res_root = runner.invoke(mod.app, ["--help"])
    assert res_root.exit_code == 0, res_root.output
    assert "lux-batch" in res_root.output  # app name is set

    # subcommand help
    res_run = runner.invoke(mod.app, ["run", "--help"])
    assert res_run.exit_code == 0, res_run.output
    # basic needles to ensure command registered
    assert "Files and/or directories to process." in res_run.output
