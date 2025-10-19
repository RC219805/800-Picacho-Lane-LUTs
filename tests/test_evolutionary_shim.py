# file: tests/test_evolutionary_shim.py
from __future__ import annotations

from datetime import date

from evolutionary_checkpoint import EvolutionaryCheckpoint as EC_legacy, EvolutionaryStatus as ES_legacy
from src.evolutionary import EvolutionaryCheckpoint as EC_new, EvolutionaryStatus as ES_new


def test_shim_exports_identical_objects():
    assert EC_legacy is EC_new
    assert ES_legacy is ES_new


def test_shim_basic_behavior():
    chk = EC_legacy(horizon=date(2025, 12, 31), mutation_path="migrate.md")
    out = chk.evaluate(today=date(2025, 12, 30))
    assert out.status is ES_new.STABLE
    assert "STABLE" in str(out)
