# Auto-generated shim by tools/gen_legacy_shims.py; do not edit.
"""Compatibility shim for legacy imports. Re-exports src.evolutionary."""
from src.evolutionary import (
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
    date,
)

# Backward compatibility alias
EvolutionaryOutcome = EvolutionOutcome

__all__ = ["EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionStatus"]
