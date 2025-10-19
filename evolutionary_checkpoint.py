# file: evolutionary_checkpoint.py
"""Legacy shim: re-exports src.evolutionary for backward compatibility."""
from __future__ import annotations

# Re-export from the canonical location
from src.evolutionary import (
    EvolutionaryCheckpoint,
    EvolutionStatus,
    EvolutionOutcome,
)

# Maintain backward compatibility alias
EvolutionaryStatus = EvolutionStatus

__all__ = [
    "EvolutionaryCheckpoint",
    "EvolutionStatus",
    "EvolutionaryStatus",
    "EvolutionOutcome",
]
