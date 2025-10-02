"""Dreaming pipeline abstractions.

This module models an asynchronous dreaming loop that periodically invents new
techniques while no foreground work is scheduled.  The design is intentionally
light-weight so it can be used in tests and interactive sessions without
requiring any of the large dependencies that power the production rendering
pipelines in this repository.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass
class DreamSequence:
    """A simple container describing the outcome of a generated dream."""

    idea: str
    coherence: float
    failure_modes: List[str] = field(default_factory=list)

    def is_coherent(self, threshold: float = 0.6) -> bool:
        """Return ``True`` when the dream appears actionable."""

        return self.coherence >= threshold


@dataclass
class Technique:
    """Represents a crystallised idea that can be integrated into a pipeline."""

    name: str
    description: str


class StandardPipeline:
    """Collects techniques discovered by the dreaming loop."""

    def __init__(self) -> None:
        self._integrated: List[Technique] = []
        self._pending: List[Technique] = []

    def integrate(self, technique: Technique) -> None:
        """Register a new technique and mark it for future processing."""

        self._integrated.append(technique)
        self._pending.append(technique)

    def active_jobs(self) -> bool:
        """Whether the pipeline has techniques waiting to be processed."""

        return bool(self._pending)

    def complete_next_job(self) -> Technique | None:
        """Pop the next technique awaiting attention."""

        if not self._pending:
            return None
        return self._pending.pop(0)

    @property
    def techniques(self) -> Sequence[Technique]:
        return tuple(self._integrated)


class DreamState:
    """Turns coherent dream sequences into implementable techniques."""

    def crystallize(self, dream_sequence: DreamSequence) -> Technique:
        return Technique(
            name=f"Technique::{dream_sequence.idea}",
            description=(
                "A structured approach distilled from the subconscious "
                f"exploration of {dream_sequence.idea}."
            ),
        )


class InnovationEngine:
    """Generates dream sequences asynchronously."""

    def __init__(self) -> None:
        self._iteration = 0

    async def generate_vision(self) -> DreamSequence:
        # Sleep very briefly so that ``sleep_cycle`` yields control in event
        # loops during tests.
        await asyncio.sleep(0)
        idea = f"concept_{self._iteration}"
        coherence = 0.5 + (self._iteration % 3) * 0.25
        failure_modes: List[str] = []
        if coherence < 0.6:
            failure_modes = [f"insufficient_clarity_{self._iteration}"]
        self._iteration += 1
        return DreamSequence(idea=idea, coherence=coherence, failure_modes=failure_modes)


class BoundaryKnowledge:
    """Tracks the limits discovered through failed dreams."""

    def __init__(self) -> None:
        self._constraints: List[str] = []

    @property
    def constraints(self) -> Sequence[str]:
        return tuple(self._constraints)

    def expand(self, failure_modes: Iterable[str]) -> None:
        for mode in failure_modes:
            if mode not in self._constraints:
                self._constraints.append(mode)


class DreamingPipeline:
    def __init__(self) -> None:
        self.conscious_processor = StandardPipeline()
        self.unconscious_processor = DreamState()
        self.rem_cycles = InnovationEngine()
        self.boundary_knowledge = BoundaryKnowledge()

    def active_jobs(self) -> bool:
        return self.conscious_processor.active_jobs()

    async def sleep_cycle(self) -> None:
        while not self.active_jobs():
            dream_sequence = await self.rem_cycles.generate_vision()

            if dream_sequence.is_coherent():
                new_technique = self.unconscious_processor.crystallize(dream_sequence)
                self.conscious_processor.integrate(new_technique)

            self.boundary_knowledge.expand(dream_sequence.failure_modes)

