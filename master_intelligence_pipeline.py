"""Unified orchestration for the material intelligence workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

import material_optimizer as optimizer
import material_property_substrate as substrate
import phenomenological_rendering as rendering


DEFAULT_OBJECTIVE_WEIGHTS = {
    "cost": 1.0,
    "carbon": 0.8,
    "thermal": 0.5,
    "aesthetic": 0.6,
}
DEFAULT_GENERATIONS = 100
DEFAULT_POPULATION = 50
DEFAULT_PIXEL_SIZE = 0.01
DEFAULT_TARGET_CONDUCTIVITY = 0.5
DEFAULT_SCENARIOS = rendering.DEFAULT_SCENARIOS


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_id_mask(path: Path) -> np.ndarray:
    """Load a 32-bit ID mask from ``path``."""

    try:
        import tifffile

        array = tifffile.imread(str(path))
    except (ImportError, ModuleNotFoundError):
        from PIL import Image

        image = Image.open(path)
        array = np.array(image.convert("I"))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.uint32, copy=False)


def load_palette(path: Path) -> Tuple[Dict[int, str], Dict[str, object]]:
    """Return ``(assignments, raw_data)`` parsed from ``path``."""

    data = _load_json(path)
    assignments: Dict[int, str] = {}

    if isinstance(data.get("clusters"), Iterable):
        for entry in data["clusters"]:  # type: ignore[index]
            if not isinstance(entry, Mapping):
                continue
            try:
                cluster_id = int(entry["cluster_id"])
                material = str(entry["material"])
            except (KeyError, ValueError, TypeError):
                continue
            assignments[cluster_id] = material
    elif isinstance(data.get("assignments"), Mapping):
        for key, value in data["assignments"].items():  # type: ignore[index]
            try:
                assignments[int(key)] = str(value)
            except (ValueError, TypeError):
                continue

    return assignments, data


def load_baseline_configuration(
    id_mask_path: Path,
    palette_path: Path,
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, object]]:
    """Return cluster labels, baseline assignments, and raw palette data."""

    id_mask = load_id_mask(id_mask_path)
    assignments, palette_data = load_palette(palette_path)

    if not assignments:
        unique_ids = np.unique(id_mask)
        materials = list(substrate.MATERIAL_PHYSICS_DB.keys())
        for idx, cluster_id in enumerate(unique_ids):
            assignments[int(cluster_id)] = materials[idx % len(materials)]

    return id_mask, assignments, palette_data


def _build_objectives(
    baseline_tensor: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] = DEFAULT_OBJECTIVE_WEIGHTS,
    target_conductivity: float = DEFAULT_TARGET_CONDUCTIVITY,
) -> Sequence[optimizer.Objective]:
    return (
        optimizer.create_cost_minimization_objective(weight=weights.get("cost", 1.0)),
        optimizer.create_carbon_minimization_objective(weight=weights.get("carbon", 1.0)),
        optimizer.create_thermal_performance_objective(
            target_conductivity=target_conductivity,
            weight=weights.get("thermal", 1.0),
        ),
        optimizer.create_aesthetic_consistency_objective(
            baseline_tensor,
            weight=weights.get("aesthetic", 1.0),
        ),
    )


def run_optimizer(
    cluster_labels: np.ndarray,
    assignments: Mapping[int, str],
    baseline_tensor: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] = DEFAULT_OBJECTIVE_WEIGHTS,
    generations: int = DEFAULT_GENERATIONS,
    population: int = DEFAULT_POPULATION,
    pixel_size_meters: float = DEFAULT_PIXEL_SIZE,
    target_conductivity: float = DEFAULT_TARGET_CONDUCTIVITY,
) -> optimizer.OptimizationResult:
    """Execute the deterministic optimisation routine."""

    objectives = _build_objectives(
        baseline_tensor,
        weights=weights,
        target_conductivity=target_conductivity,
    )
    opt = optimizer.MaterialOptimizer(
        cluster_labels=cluster_labels,
        baseline_assignments=assignments,
        pixel_size_meters=pixel_size_meters,
    )
    return opt.optimize(
        objectives=objectives,
        generations=generations,
        population_size=population,
        baseline_tensor_stats=substrate.summarise_tensor(baseline_tensor),
    )


def run_complete_pipeline(
    id_mask_path: Path,
    palette_path: Path,
    output_dir: Path,
    *,
    pixel_size_meters: float = DEFAULT_PIXEL_SIZE,
    weights: Mapping[str, float] = DEFAULT_OBJECTIVE_WEIGHTS,
    generations: int = DEFAULT_GENERATIONS,
    population: int = DEFAULT_POPULATION,
    target_conductivity: float = DEFAULT_TARGET_CONDUCTIVITY,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
) -> Dict[str, object]:
    """Run tensor creation, optimisation, and rendering end-to-end."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_labels, baseline_assignments, palette_data = load_baseline_configuration(
        id_mask_path,
        palette_path,
    )

    tensor = substrate.create_property_tensor(
        cluster_labels,
        substrate.MATERIAL_PHYSICS_DB,
        pixel_size_meters=pixel_size_meters,
    )
    tensor_path = substrate.save_property_tensor(tensor, output_dir / "tensors")

    optimisation_result = run_optimizer(
        cluster_labels,
        baseline_assignments,
        tensor,
        weights=weights,
        generations=generations,
        population=population,
        pixel_size_meters=pixel_size_meters,
        target_conductivity=target_conductivity,
    )
    report_path = optimizer.export_optimization_report(
        optimisation_result,
        output_dir / "optimization_report.json",
    )

    render_dir = output_dir / "renders"
    rendering.render_lighting_comparison(
        tensor,
        render_dir,
        scenarios=scenarios,
    )

    summary = {
        "tensor_path": str(tensor_path),
        "optimization_report": str(report_path),
        "render_manifest": str(render_dir / "render_manifest.json"),
        "clusters": len(optimisation_result.assignments),
        "palette_metadata": palette_data,
        "scenarios": list(scenarios),
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "DEFAULT_PIXEL_SIZE",
    "DEFAULT_POPULATION",
    "DEFAULT_TARGET_CONDUCTIVITY",
    "DEFAULT_SCENARIOS",
    "load_baseline_configuration",
    "load_id_mask",
    "load_palette",
    "run_complete_pipeline",
    "run_optimizer",
]
