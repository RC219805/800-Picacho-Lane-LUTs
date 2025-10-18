# file: evolutionary_checkpoint.py  (pre-generate & commit this known shim)
"""Compatibility shim for legacy imports. Re-exports src.evolutionary."""
from __future__ import annotations

from src.evolutionary import (
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)

__all__ = ["EvolutionaryCheckpoint", "EvolutionOutcome", "EvolutionStatus"]
