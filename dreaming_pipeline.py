# file: dreaming_pipeline.py
"""
Dreaming pipeline abstractions.

Asynchronous dreaming loop that invents techniques when no foreground work exists.
Lightweight for tests and interactive sessions; no heavy production deps required.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, cast


# --------------------------------------------------------------------
# Core domain models
# --------------------------------------------------------------------

@dataclass
class DreamSequence:
    """Outcome of a generated dream."""
    idea: str
    coherence: float
    failure_modes: List[str] = field(default_factory=list)

    def is_coherent(self, threshold: float = 0.6) -> bool:
        return self.coherence >= threshold


@dataclass
class Technique:
    """A crystallised idea that can be integrated into a pipeline."""
    name: str
    description: str


class StandardPipeline:
    """Collects techniques discovered by the dreaming loop."""

    def __init__(self) -> None:
        self._integrated: List[Technique] = []
        self._pending: List[Technique] = []

    def integrate(self, technique: Technique) -> None:
        """Register a technique and mark it pending."""
        self._integrated.append(technique)
        self._pending.append(technique)

    def detect_conflicts(self, technique: Technique) -> List[Technique]:
        """Return techniques that conflict on name."""
        return [t for t in self._integrated if t.name == technique.name]

    def resolve_conflicts(self, technique: Technique) -> None:
        """Remove superseded conflicting techniques by name."""
        self._integrated = [t for t in self._integrated if t.name != technique.name]
        self._pending = [t for t in self._pending if t.name != technique.name]

    def active_jobs(self) -> bool:
        return bool(self._pending)

    def complete_next_job(self) -> Technique | None:
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
        # Yield to event loop; keeps tests snappy.
        await asyncio.sleep(0)
        idea = f"concept_{self._iteration}"
        coherence = 0.5 + (self._iteration % 3) * 0.25
        failure_modes: List[str] = []
        if coherence < 0.6:
            failure_modes = [f"insufficient_clarity_{self._iteration}"]
        self._iteration += 1
        return DreamSequence(idea=idea, coherence=coherence, failure_modes=failure_modes)


class BoundaryKnowledge:
    """Tracks limits discovered through failed dreams."""

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
    """Coordinates dreaming and integration."""

    def __init__(self) -> None:
        self.conscious_processor = StandardPipeline()
        self.unconscious_processor = DreamState()
        self.rem_cycles = InnovationEngine()
        self.boundary_knowledge = BoundaryKnowledge()

    def active_jobs(self) -> bool:
        return self.conscious_processor.active_jobs()

    async def sleep_cycle(
        self,
        *,
        max_cycles: Optional[int] = None,
        idle_backoff: float = 0.0,
    ) -> None:
        """
        Run until a coherent technique is integrated or `max_cycles` elapse.

        Why: prevents unbounded loops in pathological scenarios/tests.
        """
        cycles = 0
        while not self.active_jobs():
            dream_sequence = await self.rem_cycles.generate_vision()
            if dream_sequence.is_coherent():
                new_technique = self.unconscious_processor.crystallize(dream_sequence)
                self.conscious_processor.integrate(new_technique)
            self.boundary_knowledge.expand(dream_sequence.failure_modes)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            if idle_backoff > 0 and not self.active_jobs():
                await asyncio.sleep(idle_backoff)


# --------------------------------------------------------------------
# Optimizer domain
# --------------------------------------------------------------------

@dataclass
class ArchitecturalHypothesis:
    """A speculative pipeline improvement."""
    technique: Technique
    mutation_notes: str
    originating_dream: DreamSequence


@dataclass
class EvaluationResult:
    """Outcome of testing a hypothesis."""
    hypothesis: ArchitecturalHypothesis
    score: float
    diagnostics: Dict[str, float]


class QuantumOptimizer:
    """Transforms the dreaming pipeline into an autonomous optimizer."""

    def __init__(
        self,
        pipeline: DreamingPipeline,
        *,
        max_iterations: int = 25,
        exploration_batch_size: int = 3,
        target_score: float | None = None,
        evaluation_timeout: float | None = None,
        max_concurrency: int | None = None,
        score_reducer: Callable[[Dict[str, float]], float] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.max_iterations = max_iterations
        self.exploration_batch_size = exploration_batch_size
        self.target_score = target_score
        self.evaluation_timeout = evaluation_timeout
        self.max_concurrency = max_concurrency
        self.score_reducer = score_reducer
        self._iteration = 0
        self._best_result: EvaluationResult | None = None
        self._history: List[EvaluationResult] = []
        self._technique_scores: Dict[str, float] = {}

    @property
    def history(self) -> Sequence[EvaluationResult]:
        return tuple(self._history)

    @property
    def best_result(self) -> EvaluationResult | None:
        return self._best_result

    @property
    def technique_scores(self) -> Dict[str, float]:
        return dict(self._technique_scores)

    def convergence_achieved(self) -> bool:
        if (
            self.target_score is not None
            and self._best_result
            and self._best_result.score >= self.target_score
        ):
            return True
        return self._iteration >= self.max_iterations

    async def evolve_pipeline(
        self,
        performance_metrics: Callable[
            [ArchitecturalHypothesis],
            float | Awaitable[float] | Dict[str, float] | Awaitable[Dict[str, float]],
        ],
    ) -> None:
        """Continuously optimise the pipeline until convergence."""
        while not self.convergence_achieved():
            hypotheses = await self.generate_architectural_mutations()
            if not hypotheses:
                break
            results = await self.test_parallel_realities(hypotheses, performance_metrics)
            self.adopt_superior_architecture(results)
            self.crystallize_learning()
            self._iteration += 1

    async def generate_architectural_mutations(self) -> List[ArchitecturalHypothesis]:
        """Expand the search frontier with freshly crystallised techniques."""
        dreams = await asyncio.gather(
            *(self.pipeline.rem_cycles.generate_vision() for _ in range(self.exploration_batch_size))
        )
        hypotheses: List[ArchitecturalHypothesis] = []
        for dream, counter in zip(dreams, itertools.count(1)):
            if not dream.is_coherent():
                self.pipeline.boundary_knowledge.expand(dream.failure_modes)
                continue
            technique = self.pipeline.unconscious_processor.crystallize(dream)
            mutation_notes = f"mutation_{self._iteration}_{counter}: derived from {dream.idea}"
            hypotheses.append(
                ArchitecturalHypothesis(
                    technique=technique, mutation_notes=mutation_notes, originating_dream=dream
                )
            )
        return hypotheses

    async def test_parallel_realities(
        self,
        hypotheses: Sequence[ArchitecturalHypothesis],
        performance_metrics: Callable[
            [ArchitecturalHypothesis],
            float | Awaitable[float] | Dict[str, float] | Awaitable[Dict[str, float]],
        ],
    ) -> List[EvaluationResult]:
        """Evaluate hypotheses concurrently with optional timeouts and concurrency limits."""

        async def _evaluate(hypothesis: ArchitecturalHypothesis) -> EvaluationResult:
            try:
                raw = performance_metrics(hypothesis)
                if inspect.isawaitable(raw):
                    if self.evaluation_timeout is not None:
                        raw = await asyncio.wait_for(raw, timeout=self.evaluation_timeout)
                    else:
                        raw = await raw
                if isinstance(raw, dict):
                    diagnostics = raw
                    reducer = self.score_reducer or (lambda d: sum(d.values()) / max(len(d), 1))
                    score = float(reducer(diagnostics))
                else:
                    score = float(cast(float, raw))
                    diagnostics = {"score": score}
            except asyncio.TimeoutError:
                # Why: make timeouts visible and penalize stalled hypotheses.
                diagnostics = {"timeout": 1.0}
                score = float("-inf")
            except Exception as exc:  # noqa: BLE001 - deliberate catch-all at boundary
                diagnostics = {"error": 1.0}
                # Surface exception class name numerically not meaningful; set worst score.
                score = float("-inf")
            return EvaluationResult(hypothesis=hypothesis, score=score, diagnostics=diagnostics)

        if self.max_concurrency and self.max_concurrency > 0:
            sem = asyncio.Semaphore(self.max_concurrency)

            async def _guarded(h: ArchitecturalHypothesis) -> EvaluationResult:
                async with sem:
                    return await _evaluate(h)

            evaluations = await asyncio.gather(*(_guarded(h) for h in hypotheses))
        else:
            evaluations = await asyncio.gather(*(_evaluate(h) for h in hypotheses))

        self._history.extend(evaluations)
        return evaluations

    def adopt_superior_architecture(self, results: Sequence[EvaluationResult]) -> None:
        """Select and integrate the best performing hypothesis."""
        if not results:
            return
        best_result = max(results, key=lambda r: r.score)
        technique_name = best_result.hypothesis.technique.name
        prior_score = self._technique_scores.get(technique_name)
        if prior_score is not None and prior_score >= best_result.score:
            return
        conflicts = self.pipeline.conscious_processor.detect_conflicts(best_result.hypothesis.technique)
        if conflicts:
            self.pipeline.conscious_processor.resolve_conflicts(best_result.hypothesis.technique)
        self.pipeline.conscious_processor.integrate(best_result.hypothesis.technique)
        self._technique_scores[technique_name] = best_result.score
        self._best_result = best_result

    def crystallize_learning(self) -> None:
        """Record learnings from the latest optimisation step."""
        if not self._best_result:
            return
        dream = self._best_result.hypothesis.originating_dream
        self.pipeline.boundary_knowledge.expand(dream.failure_modes)