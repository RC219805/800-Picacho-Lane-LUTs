from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evolutionary_checkpoint import (
    EvolutionaryCheckpoint,
    EvolutionaryStatus,
    EvolutionaryOutcome,
)


def test_to_dict_from_dict_roundtrip():
    created = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ckpt = EvolutionaryCheckpoint(
        step=1,
        status=EvolutionaryStatus.COMPLETED,
        outcome=EvolutionaryOutcome.IMPROVED,
        score=0.9,
        created_at=created,
        notes="initial",
        meta={"a": 1},
    )

    d = ckpt.to_dict()
    assert d["step"] == 1
    assert d["status"] == "completed"
    assert d["outcome"] == "improved"
    assert d["score"] == 0.9
    assert d["created_at"] == created.isoformat()
    assert d["notes"] == "initial"
    assert d["meta"] == {"a": 1}

    ckpt2 = EvolutionaryCheckpoint.from_dict(d)
    assert ckpt2.step == ckpt.step
    assert ckpt2.status == ckpt.status
    assert ckpt2.outcome == ckpt.outcome
    assert ckpt2.score == ckpt.score
    assert ckpt2.created_at == ckpt.created_at
    assert ckpt2.notes == ckpt.notes
    assert ckpt2.meta == ckpt.meta


def test_advance_immutability_and_updates():
    base = EvolutionaryCheckpoint(step=2, status=EvolutionaryStatus.PENDING, meta=None)
    running = base.advance(status=EvolutionaryStatus.RUNNING, notes="started", meta={"k": "v"})

    # original unchanged
    assert base.status == EvolutionaryStatus.PENDING
    assert base.notes is None
    assert base.meta is None

    # new object updated
    assert running is not base
    assert running.status == EvolutionaryStatus.RUNNING
    assert running.notes == "started"
    assert running.meta == {"k": "v"}

    improved = running.advance(outcome=EvolutionaryOutcome.IMPROVED, score=1.0)
    assert improved.outcome == EvolutionaryOutcome.IMPROVED
    assert improved.score == 1.0
    # earlier object still unchanged
    assert running.outcome is None
    assert running.score is None


def test_meta_serialization_when_none():
    c = EvolutionaryCheckpoint(step=3, status=EvolutionaryStatus.PENDING, meta=None)
    d = c.to_dict()
    # to_dict must emit an empty mapping for meta when None
    assert isinstance(d["meta"], dict)
    assert d["meta"] == {}

    # from_dict must accept missing meta and create empty mapping for internal use
    d2 = {"step": 4, "status": "pending", "created_at": c.created_at.isoformat()}
    c2 = EvolutionaryCheckpoint.from_dict(d2)
    assert c2.meta == {}
