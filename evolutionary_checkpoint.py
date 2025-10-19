"""Compatibility shim for legacy imports. Re-exports src.evolutionary."""
from __future__ import annotations

from src.evolutionary import (
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)

# Re-export EvolutionOutcome with both names for compatibility
EvolutionaryOutcome = EvolutionOutcome

# Clean public interface
__all__ = ["EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionStatus"]

# Make only specific items visible in dir()
def __dir__():
    return ["EvolutionaryCheckpoint", "EvolutionOutcome", "EvolutionStatus"]
