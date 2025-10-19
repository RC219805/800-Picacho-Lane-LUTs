"""Compatibility shim for legacy imports. Re-exports src.evolutionary."""

from src.evolutionary import (
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)

__all__ = ["EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionStatus"]


def __getattr__(name):
    """Provide legacy aliases without polluting dir()."""
    if name == "EvolutionaryOutcome":
        return EvolutionOutcome
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
