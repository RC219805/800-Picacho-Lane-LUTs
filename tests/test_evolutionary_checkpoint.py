"""Tests for :mod:`evolutionary_checkpoint`."""

from __future__ import annotations

from datetime import date

import pytest

from evolutionary_checkpoint import EvolutionaryCheckpoint, EvolutionStatus, EvolutionOutcome


def test_evolution_required_message_when_horizon_has_passed() -> None:
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 1), mutation_path="lux/v2/pipeline"
    )

    message = checkpoint.evolve_or_alert(today=date(2024, 1, 2))

    assert message == "EVOLUTION REQUIRED: Migrate to lux/v2/pipeline"


def test_evolution_not_required_message_when_within_horizon() -> None:
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 10), mutation_path="lux/v3/pipeline"
    )

    message = checkpoint.evolve_or_alert(today=date(2024, 1, 5))

    assert message == "STABLE: Current form viable until 2024-01-10"


class _FrozenDate(date):
    """Helper date that lets us control :func:`date.today`."""

    @classmethod
    def today(cls) -> "_FrozenDate":
        return cls(2024, 1, 9)


def test_today_defaults_to_current_date(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 10), mutation_path="lux/v3/pipeline"
    )

    monkeypatch.setattr("src.evolutionary.date", _FrozenDate)

    message = checkpoint.evolve_or_alert()

    assert message == "STABLE: Current form viable until 2024-01-10"


def test_evolution_outcome_is_required_when_evolution_required() -> None:
    """Test EvolutionOutcome.is_required property."""
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 1), mutation_path="lux/v2/pipeline"
    )
    outcome = checkpoint.evaluate(today=date(2024, 1, 2))
    
    assert outcome.is_required is True
    assert outcome.status is EvolutionStatus.EVOLUTION_REQUIRED


def test_evolution_outcome_not_required_when_stable() -> None:
    """Test EvolutionOutcome.is_required property for stable case."""
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 10), mutation_path="lux/v3/pipeline"
    )
    outcome = checkpoint.evaluate(today=date(2024, 1, 5))
    
    assert outcome.is_required is False
    assert outcome.status is EvolutionStatus.STABLE


def test_due_in_days_when_stable() -> None:
    """Test EvolutionOutcome.due_in_days property."""
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 10), mutation_path="lux/v3/pipeline"
    )
    outcome = checkpoint.evaluate(today=date(2024, 1, 5))
    
    assert outcome.due_in_days == 5


def test_due_in_days_returns_none_when_evolution_required() -> None:
    """Test EvolutionOutcome.due_in_days returns None when evolution required."""
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 1), mutation_path="lux/v2/pipeline"
    )
    outcome = checkpoint.evaluate(today=date(2024, 1, 2))
    
    assert outcome.due_in_days is None


def test_overdue_by_days_when_evolution_required() -> None:
    """Test EvolutionOutcome.overdue_by_days property."""
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 1), mutation_path="lux/v2/pipeline"
    )
    outcome = checkpoint.evaluate(today=date(2024, 1, 5))
    
    assert outcome.overdue_by_days == 4


def test_overdue_by_days_returns_none_when_stable() -> None:
    """Test EvolutionOutcome.overdue_by_days returns None when stable."""
    checkpoint = EvolutionaryCheckpoint(
        horizon=date(2024, 1, 10), mutation_path="lux/v3/pipeline"
    )
    outcome = checkpoint.evaluate(today=date(2024, 1, 5))
    
    assert outcome.overdue_by_days is None


def test_invalid_horizon_type_raises_type_error() -> None:
    """Test that invalid horizon type raises TypeError."""
    with pytest.raises(TypeError, match="horizon must be a datetime.date"):
        EvolutionaryCheckpoint(horizon="2024-01-01", mutation_path="lux/v2/pipeline")  # type: ignore


def test_empty_mutation_path_raises_value_error() -> None:
    """Test that empty mutation_path raises ValueError."""
    with pytest.raises(ValueError, match="mutation_path must be a non-empty string"):
        EvolutionaryCheckpoint(horizon=date(2024, 1, 1), mutation_path="")


def test_whitespace_mutation_path_raises_value_error() -> None:
    """Test that whitespace-only mutation_path raises ValueError."""
    with pytest.raises(ValueError, match="mutation_path must be a non-empty string"):
        EvolutionaryCheckpoint(horizon=date(2024, 1, 1), mutation_path="   ")
