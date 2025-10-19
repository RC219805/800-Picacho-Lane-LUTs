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
        if self.status is EvolutionStatus.STABLE:
            return max((self.horizon - self.today).days, 0)
        return None

    @property
    def overdue_by_days(self) -> Optional[int]:
        if self.status is EvolutionStatus.EVOLUTION_REQUIRED:
            return (self.today - self.horizon).days
        return None

    def message(self) -> str:
        if self.status is EvolutionStatus.EVOLUTION_REQUIRED:
            return f"EVOLUTION REQUIRED: Migrate to {self.mutation_path}"
        return f"STABLE: Current form viable until {self.horizon.isoformat()}"

    def __str__(self) -> str:
        return self.message()


@dataclass(frozen=True, slots=True)
class EvolutionaryCheckpoint:
    """Represents an evolutionary deadline for a particular workflow."""
    horizon: date
    mutation_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, date):
            raise TypeError("horizon must be a datetime.date")
        if not isinstance(self.mutation_path, str) or not self.mutation_path.strip():
            raise ValueError("mutation_path must be a non-empty string")
        object.__setattr__(self, "mutation_path", self.mutation_path.strip())

    def evaluate(self, *, today: Optional[date] = None) -> EvolutionOutcome:
        # Support monkeypatching via the public module interface
        import sys
        date_class = getattr(sys.modules.get('evolutionary_checkpoint'), 'date', date)
        reference_date = today or date_class.today()
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
        return str(self.evaluate(today=today))


if __name__ == "__main__":
    # Minimal CLI for ad-hoc checks.
    import argparse, json
    parser = argparse.ArgumentParser(description="Evaluate an evolutionary checkpoint.")
    parser.add_argument("horizon", help="YYYY-MM-DD")
    parser.add_argument("mutation_path")
    parser.add_argument("--today", default=None)
    parser.add_argument("--json", action="store_true")
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
        print(json.dumps(
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
        ))
    else:
        print(outcome)
