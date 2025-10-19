"""Compatibility shim for legacy imports. Re-exports src.evolutionary."""
from __future__ import annotations

from datetime import date  # noqa: F401 - imported for monkeypatching in tests
from src.evolutionary import (
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)

# Provide legacy aliases with "ary" suffix for backward compatibility
EvolutionaryStatus = EvolutionStatus
EvolutionaryOutcome = EvolutionOutcome

__all__ = ["EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionStatus"]
