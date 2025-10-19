# file: evolutionary_checkpoint.py
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

__all__ = ["EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionaryStatus"]


class EvolutionaryStatus(str, Enum):
    """Lifecycle state of an evolutionary step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvolutionaryOutcome(str, Enum):
    """Result quality of a completed step."""
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"


@dataclass(frozen=True)
class EvolutionaryCheckpoint:
    """
    Immutable record for an evolutionary run step.

    Why frozen: promotes deterministic behavior in pipelines and tests.
    """
    step: int
    status: EvolutionaryStatus
    outcome: Optional[EvolutionaryOutcome] = None
    score: Optional[float] = None
    created_at: datetime = datetime.now(timezone.utc)
    notes: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None  # intentionally plain dict for JSON-friendliness

    def to_dict(self) -> Dict[str, Any]:
        """Stable, JSON-serializable representation."""
        return {
            "step": self.step,
            "status": self.status.value,
            "outcome": self.outcome.value if self.outcome else None,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
            "notes": self.notes,
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionaryCheckpoint":
        """Inverse of to_dict()."""
        return cls(
            step=int(data["step"]),
            status=EvolutionaryStatus(str(data["status"])),
            outcome=(EvolutionaryOutcome(str(data["outcome"])) if data.get("outcome") else None),
            score=(float(data["score"]) if data.get("score") is not None else None),
            created_at=datetime.fromisoformat(str(data["created_at"])) if data.get("created_at") else datetime.now(timezone.utc),
            notes=(str(data["notes"]) if data.get("notes") is not None else None),
            meta=dict(data.get("meta", {}) or {}),
        )

    @property
    def succeeded(self) -> bool:
        """True for completed non-regression results."""
        if self.status != EvolutionaryStatus.COMPLETED or self.outcome is None:
            return False
        return self.outcome in (EvolutionaryOutcome.IMPROVED, EvolutionaryOutcome.UNCHANGED)

    @property
    def failed(self) -> bool:
        """True if run failed or regressed."""
        if self.status == EvolutionaryStatus.FAILED:
            return True
        return self.status == EvolutionaryStatus.COMPLETED and self.outcome == EvolutionaryOutcome.REGRESSED

    def advance(
        self,
        *,
        status: Optional[EvolutionaryStatus] = None,
        outcome: Optional[EvolutionaryOutcome] = None,
        score: Optional[float] = None,
        notes: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "EvolutionaryCheckpoint":
        """
        Return a new checkpoint with updated fields.
        Why: keep immutability; easier reasoning in CI and dashboards.
        """
        return replace(
            self,
            status=status or self.status,
            outcome=outcome if outcome is not None else self.outcome,
            score=score if score is not None else self.score,
            notes=notes if notes is not None else self.notes,
            meta=(dict(meta) if meta is not None else (dict(self.meta) if self.meta else {})),
        )
