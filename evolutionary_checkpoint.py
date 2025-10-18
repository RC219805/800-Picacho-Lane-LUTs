# file: evolutionary_checkpoint.py
"""Compatibility shim for legacy imports.

Tests and downstream code may import from `evolutionary_checkpoint`.
This file re-exports the canonical implementations from `src.evolutionary`.
"""
from __future__ import annotations

from src.evolutionary import (  # type: ignore[import-not-found]
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)

__all__ = ["EvolutionStatus", "EvolutionOutcome", "EvolutionaryCheckpoint"]
