# file: src/evolutionary.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional


class EvolutionStatus(Enum):
    """Machine-friendly status for evolution decisions."""
    STABLE = "stable"
    EVOLUTION_REQUIRED = "evolution_required"


@dataclass(frozen=True, slots=True)
class EvolutionOutcome:
    """Structured outcome of an evolution check."""
    status: EvolutionStatus
    horizon: date
    mutation_path: str
    today: date

    @property
    def is_required(self) -> bool:
        return self.status is EvolutionStatus.EVOLUTION_REQUIRED

    @property
    def due_in_days(self) -> Optional[int]:
        """Days remaining while stable; None if already required."""
        if self.status is EvolutionStatus.STABLE:
            # 0 means "today is the horizon"
            return max((self.horizon - self.today).days, 0)
        return None

    @property
    def overdue_by_days(self) -> Optional[int]:
        """Days overdue if evolution required; None when still stable."""
        if self.status is EvolutionStatus.EVOLUTION_REQUIRED:
            return (self.today - self.horizon).days
        return None

    def message(self) -> str:
        """Human-readable guidance (backwards-compatible wording)."""
        if self.status is EvolutionStatus.EVOLUTION_REQUIRED:
            return f"EVOLUTION REQUIRED: Migrate to {self.mutation_path}"
        return f"STABLE: Current form viable until {self.horizon.isoformat()}"

    def __str__(self) -> str:  # matches prior API behavior for messaging
        return self.message()


@dataclass(frozen=True, slots=True)
class EvolutionaryCheckpoint:
    """Represents an evolutionary deadline for a particular workflow.

    The checkpoint keeps track of a *horizon* (a :class:`~datetime.date` after
    which a migration must be pursued) and the ``mutation_path`` that should be
    followed once the horizon has been crossed.

    Use :meth:`evaluate` for structured results and :meth:`evolve_or_alert`
    for the original string guidance.
    """
    horizon: date
    mutation_path: str

    def __post_init__(self) -> None:
        # Frozen dataclass: use object.__setattr__ to normalize. Avoid silent bad config.
        if not isinstance(self.horizon, date):
            raise TypeError("horizon must be a datetime.date")
        if not isinstance(self.mutation_path, str) or not self.mutation_path.strip():
            raise ValueError("mutation_path must be a non-empty string")
        object.__setattr__(self, "mutation_path", self.mutation_path.strip())

    def evaluate(self, *, today: Optional[date] = None) -> EvolutionOutcome:
        """Return a structured outcome describing the evolution decision.

        Parameters
        ----------
        today:
            Optional date to use instead of :func:`datetime.date.today`.
            Useful for deterministic testing.
        """
        reference_date = today or date.today()
        status = (
            EvolutionStatus.EVOLUTION_REQUIRED
            if reference_date > self.horizon
            else EvolutionStatus.STABLE
        )
        return EvolutionOutcome(
            status=status,
            horizon=self.horizon,
            mutation_path=self.mutation_path,
            today=reference_date,
        )

    def evolve_or_alert(self, *, today: Optional[date] = None) -> str:
        """Backwards-compatible human-readable guidance."""
        # Kept to avoid breaking existing dashboards/CI logs.
        return str(self.evaluate(today=today))


if __name__ == "__main__":
    # Minimal CLI for ad-hoc checks.
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an evolutionary checkpoint.")
    parser.add_argument("horizon", help="Horizon date in YYYY-MM-DD format.")
    parser.add_argument("mutation_path", help="Migration path to follow after the horizon.")
    parser.add_argument(
        "--today",
        help="Override today's date in YYYY-MM-DD (for testing/repro).",
        default=None,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of the message.",
    )
    args = parser.parse_args()

    def parse_iso(d: str) -> date:
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError as exc:
            raise SystemExit(f"Invalid date '{d}'. Expected YYYY-MM-DD.") from exc

    chk = EvolutionaryCheckpoint(
        horizon=parse_iso(args.horizon),
        mutation_path=args.mutation_path,
    )
    today_override = parse_iso(args.today) if args.today else None
    outcome = chk.evaluate(today=today_override)

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "status": outcome.status.value,
                    "horizon": outcome.horizon.isoformat(),
                    "mutation_path": outcome.mutation_path,
                    "today": outcome.today.isoformat(),
                    "due_in_days": outcome.due_in_days,
                    "overdue_by_days": outcome.overdue_by_days,
                    "message": outcome.message(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(outcome)


# file: tests/test_evolutionary.py
from __future__ import annotations

import pytest
from datetime import date, timedelta

from src.evolutionary import (
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)


def test_stable_before_horizon():
    horizon = date(2025, 12, 31)
    today = date(2025, 12, 30)
    chk = EvolutionaryCheckpoint(horizon=horizon, mutation_path="docs/migrate.md")
    out = chk.evaluate(today=today)

    assert out.status is EvolutionStatus.STABLE
    assert out.is_required is False
    assert out.due_in_days == 1
    assert out.overdue_by_days is None
    assert str(out) == "STABLE: Current form viable until 2025-12-31"


def test_stable_on_horizon():
    horizon = date(2025, 12, 31)
    today = date(2025, 12, 31)
    chk = EvolutionaryCheckpoint(horizon=horizon, mutation_path="docs/migrate.md")
    out = chk.evaluate(today=today)

    assert out.status is EvolutionStatus.STABLE
    assert out.due_in_days == 0  # on-horizon is still stable
    assert str(out) == "STABLE: Current form viable until 2025-12-31"


def test_required_after_horizon():
    horizon = date(2025, 12, 31)
    today = date(2026, 1, 2)
    chk = EvolutionaryCheckpoint(horizon=horizon, mutation_path="docs/migrate.md")
    out = chk.evaluate(today=today)

    assert out.status is EvolutionStatus.EVOLUTION_REQUIRED
    assert out.is_required is True
    assert out.due_in_days is None
    assert out.overdue_by_days == 2
    assert str(out) == "EVOLUTION REQUIRED: Migrate to docs/migrate.md"


def test_evolve_or_alert_message_compatibility():
    horizon = date(2025, 6, 1)
    chk = EvolutionaryCheckpoint(horizon=horizon, mutation_path="path/to/plan")

    # Before horizon
    msg1 = chk.evolve_or_alert(today=date(2025, 5, 31))
    assert msg1 == "STABLE: Current form viable until 2025-06-01"

    # After horizon
    msg2 = chk.evolve_or_alert(today=date(2025, 6, 2))
    assert msg2 == "EVOLUTION REQUIRED: Migrate to path/to/plan"


def test_invalid_mutation_path_raises():
    with pytest.raises(ValueError):
        EvolutionaryCheckpoint(horizon=date(2025, 1, 1), mutation_path="  ")


def test_mutation_path_is_stripped():
    chk = EvolutionaryCheckpoint(horizon=date(2025, 1, 1), mutation_path="  plan.md  ")
    assert chk.mutation_path == "plan.md"