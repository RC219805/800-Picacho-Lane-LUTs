# file: tests/test_evolutionary_shim_guard.py
from __future__ import annotations

import importlib


def test_evolutionary_checkpoint_shim_is_strict():
    sh = importlib.import_module("evolutionary_checkpoint")
    new = importlib.import_module("src.evolutionary")

    # Identity: exact re-exports
    assert sh.EvolutionStatus is new.EvolutionStatus
    assert sh.EvolutionOutcome is new.EvolutionOutcome
    assert sh.EvolutionaryCheckpoint is new.EvolutionaryCheckpoint

    # Public surface: only these three symbols
    public = {n for n in dir(sh) if not n.startswith("_")}
    expected = {"EvolutionStatus", "EvolutionOutcome", "EvolutionaryCheckpoint"}
    assert public == expected, f"Unexpected public names: {sorted(public - expected)}"

    # __all__ must be exactly these names, fixed order (guard accidental edits)
    expected_all = ["EvolutionaryCheckpoint", "EvolutionOutcome", "EvolutionStatus"]
    assert getattr(sh, "__all__", None) == expected_all
