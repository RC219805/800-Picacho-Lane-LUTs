# file: tests/test_evolutionary_shim_lock.py
from __future__ import annotations

import importlib


def test_evolutionary_shim_exports_exist_and_match():
    shim = importlib.import_module("evolutionary_checkpoint")
    new = importlib.import_module("src.evolutionary")

    # Names must exist on the shim
    for name in ("EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionStatus"):
        assert hasattr(shim, name), f"shim missing {name}"

    # Identity equality with canonical types
    assert shim.EvolutionStatus is new.EvolutionStatus
    assert shim.EvolutionOutcome is new.EvolutionOutcome
    assert shim.EvolutionaryCheckpoint is new.EvolutionaryCheckpoint

    # __all__ must be exactly these names (order locked)
    assert getattr(shim, "__all__", None) == [
        "EvolutionaryCheckpoint",
        "EvolutionaryOutcome",
        "EvolutionStatus",
    ]
