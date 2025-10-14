"""Greedy-yet-transparent optimisation helpers for the material pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from material_property_substrate import MATERIAL_PHYSICS_DB, MaterialPhysics


@dataclass(frozen=True)
class Objective:
    """Represents a scalar objective to be minimised."""

    name: str
    weight: float
    evaluator: Callable[[MaterialPhysics, "OptimizationContext"], float]

    def evaluate(self, material: MaterialPhysics, context: "OptimizationContext") -> float:
        value = self.evaluator(material, context)
        return self.weight * value


@dataclass
class OptimizationContext:
    """Per-cluster state exposed to objective evaluators."""

    cluster_id: int
    pixel_count: int
    pixel_area_m2: float
    baseline_material: Optional[MaterialPhysics]
    baseline_tensor_stats: Mapping[str, float]


@dataclass
class ClusterResult:
    """Outcome for a single cluster selection."""

    cluster_id: int
    chosen_material: str
    objective_values: Mapping[str, float]
    score: float


@dataclass
class OptimizationResult:
    """Aggregate output for :class:`MaterialOptimizer`."""

    assignments: Mapping[int, str]
    objective_breakdown: Sequence[ClusterResult]
    metadata: Mapping[str, float]

    def to_json(self) -> Dict[str, object]:
        return {
            "assignments": dict(self.assignments),
            "objective_breakdown": [
                {
                    "cluster_id": entry.cluster_id,
                    "chosen_material": entry.chosen_material,
                    "objective_values": dict(entry.objective_values),
                    "score": entry.score,
                }
                for entry in self.objective_breakdown
            ],
            "metadata": dict(self.metadata),
        }


class MaterialOptimizer:
    """A deterministic optimiser that favours explainability over brute force."""

    def __init__(
        self,
        *,
        cluster_labels: np.ndarray,
        baseline_assignments: Mapping[int, str],
        material_db: Mapping[str, MaterialPhysics] | None = None,
        pixel_size_meters: float = 0.01,
    ) -> None:
        self.cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        if self.cluster_labels.ndim != 2:
            raise ValueError("cluster_labels must be a 2-D array")
        if pixel_size_meters <= 0:
            raise ValueError("pixel_size_meters must be positive")

        self.pixel_size_meters = pixel_size_meters
        self.baseline_assignments = dict(baseline_assignments)
        self.material_db = dict(MATERIAL_PHYSICS_DB if material_db is None else material_db)
        if not self.material_db:
            raise ValueError("material_db must contain materials")
        self._materials_by_name = self.material_db
        self._materials_by_id = {mat.material_id: mat for mat in self.material_db.values()}

    @property
    def cluster_ids(self) -> List[int]:
        ids = np.unique(self.cluster_labels)
        return sorted(int(value) for value in ids.tolist())

    def _resolve_material(self, name: str) -> Optional[MaterialPhysics]:
        return self._materials_by_name.get(name)

    def optimise_cluster(
        self,
        cluster_id: int,
        objectives: Sequence[Objective],
        baseline_tensor_stats: Mapping[str, float],
    ) -> ClusterResult:
        mask = self.cluster_labels == cluster_id
        pixel_count = int(np.count_nonzero(mask))
        pixel_area = pixel_count * (self.pixel_size_meters ** 2)
        baseline_material_name = self.baseline_assignments.get(cluster_id)
        baseline_material = (
            self._resolve_material(baseline_material_name) if baseline_material_name else None
        )

        context = OptimizationContext(
            cluster_id=cluster_id,
            pixel_count=pixel_count,
            pixel_area_m2=pixel_area,
            baseline_material=baseline_material,
            baseline_tensor_stats=baseline_tensor_stats,
        )

        best_material: Optional[MaterialPhysics] = None
        best_score = float("inf")
        best_breakdown: Dict[str, float] = {}

        for name, candidate in self._materials_by_name.items():
            score = 0.0
            breakdown: Dict[str, float] = {}
            for objective in objectives:
                value = objective.evaluator(candidate, context)
                weighted = objective.weight * value
                breakdown[objective.name] = weighted
                score += weighted
            if score < best_score:
                best_material = candidate
                best_score = score
                best_breakdown = breakdown

        if best_material is None:
            raise RuntimeError("Optimizer could not evaluate any materials")

        return ClusterResult(
            cluster_id=cluster_id,
            chosen_material=best_material.name,
            objective_values=best_breakdown,
            score=best_score,
        )

    def optimize(
        self,
        *,
        objectives: Sequence[Objective],
        population_size: int = 30,
        generations: int = 50,
        baseline_tensor_stats: Mapping[str, float] | None = None,
    ) -> OptimizationResult:
        """Optimise cluster assignments and return a structured result."""

        if not objectives:
            raise ValueError("objectives must contain at least one entry")
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")

        tensor_stats = dict(baseline_tensor_stats or {})
        cluster_results: List[ClusterResult] = []
        assignments: Dict[int, str] = {}

        for cluster_id in self.cluster_ids:
            result = self.optimise_cluster(cluster_id, objectives, tensor_stats)
            cluster_results.append(result)
            assignments[cluster_id] = result.chosen_material

        metadata = {
            "population_size": float(population_size),
            "generations": float(generations),
            "evaluated_materials": len(self.material_db),
        }

        return OptimizationResult(
            assignments=assignments,
            objective_breakdown=cluster_results,
            metadata=metadata,
        )


def create_cost_minimization_objective(*, weight: float = 1.0) -> Objective:
    """Penalise materials with higher cost per square metre."""

    def _evaluate(material: MaterialPhysics, context: OptimizationContext) -> float:
        return material.cost_per_square_meter

    return Objective(name="cost", weight=weight, evaluator=_evaluate)


def create_carbon_minimization_objective(*, weight: float = 1.0) -> Objective:
    """Penalise embodied carbon emissions."""

    def _evaluate(material: MaterialPhysics, context: OptimizationContext) -> float:
        return material.carbon_kg_co2e * max(context.pixel_area_m2, 1e-6)

    return Objective(name="carbon", weight=weight, evaluator=_evaluate)


def create_thermal_performance_objective(
    *,
    target_conductivity: float,
    weight: float = 1.0,
) -> Objective:
    """Favour materials whose conductivity sits near ``target_conductivity``."""

    def _evaluate(material: MaterialPhysics, context: OptimizationContext) -> float:
        delta = material.thermal_conductivity_w_mk - target_conductivity
        return delta * delta

    return Objective(name="thermal", weight=weight, evaluator=_evaluate)


def create_aesthetic_consistency_objective(
    baseline_tensor: Mapping[str, np.ndarray],
    *,
    weight: float = 1.0,
) -> Objective:
    """Encourage materials that preserve baseline albedo / roughness trends."""

    baseline_mean_albedo = float(np.mean(baseline_tensor.get("albedo", np.array([0.5]))))
    baseline_mean_roughness = float(np.mean(baseline_tensor.get("roughness", np.array([0.5]))))

    def _evaluate(material: MaterialPhysics, context: OptimizationContext) -> float:
        albedo_delta = material.albedo - baseline_mean_albedo
        roughness_delta = material.roughness - baseline_mean_roughness
        return (albedo_delta ** 2) + (roughness_delta ** 2)

    return Objective(name="aesthetic", weight=weight, evaluator=_evaluate)


def export_optimization_report(result: OptimizationResult, destination: Path | str) -> Path:
    """Write ``result`` to ``destination`` as JSON."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "ClusterResult",
    "MaterialOptimizer",
    "Objective",
    "OptimizationContext",
    "OptimizationResult",
    "create_aesthetic_consistency_objective",
    "create_carbon_minimization_objective",
    "create_cost_minimization_objective",
    "create_thermal_performance_objective",
    "export_optimization_report",
]
