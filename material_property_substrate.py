"""Foundational material property substrate for the intelligence pipeline.

The repository leans heavily on evocative storytelling and high-level process
descriptions.  This module provides a pragmatic, well-documented baseline that
other components can rely on.  It exposes:

``MATERIAL_PHYSICS_DB``
    A curated mapping of material names to :class:`MaterialPhysics` entries.

``create_property_tensor``
    Converts a 32-bit material identifier mask into a dense stack of property
    layers (density, thermal conductivity, embodied carbon, etc.).

``save_property_tensor``
    Writes the tensor to disk in a NumPy archive alongside a JSON manifest so
    downstream tooling (or curious humans) can inspect the results.

The intent is to offer a lightweight yet believable material substrate that can
slot into optimisation, rendering, or analytic pipelines without requiring a
full-blown PBR database.  The numbers are illustrative—drawn from public
material references—and the helpers prioritise determinism so tests can reason
about their output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

import numpy as np


@dataclass(frozen=True)
class MaterialPhysics:
    """Physical parameters that downstream modules can reason about."""

    material_id: int
    name: str
    density_kg_m3: float
    cost_per_square_meter: float
    carbon_kg_co2e: float
    thermal_conductivity_w_mk: float
    albedo: float
    roughness: float
    notes: str = ""

    def to_metadata(self) -> Dict[str, float | int | str]:
        """Return a JSON-serialisable summary of the material."""

        return asdict(self)


def _build_material_db() -> Dict[str, MaterialPhysics]:
    """Return the canonical set of materials for the baseline substrate."""

    return {
        "limestone": MaterialPhysics(
            material_id=1,
            name="Limestone",
            density_kg_m3=2400.0,
            cost_per_square_meter=85.0,
            carbon_kg_co2e=38.0,
            thermal_conductivity_w_mk=1.3,
            albedo=0.55,
            roughness=0.45,
            notes="Neutral limestone facade cladding used across Picacho Lane",
        ),
        "oak_plank": MaterialPhysics(
            material_id=2,
            name="Oak Plank",
            density_kg_m3=720.0,
            cost_per_square_meter=135.0,
            carbon_kg_co2e=12.0,
            thermal_conductivity_w_mk=0.17,
            albedo=0.38,
            roughness=0.52,
            notes="Engineered wide-plank oak flooring with satin finish",
        ),
        "bronze_panel": MaterialPhysics(
            material_id=3,
            name="Bronze Panel",
            density_kg_m3=8700.0,
            cost_per_square_meter=420.0,
            carbon_kg_co2e=95.0,
            thermal_conductivity_w_mk=43.0,
            albedo=0.32,
            roughness=0.28,
            notes="Oil-rubbed bronze mullions and hardware",
        ),
        "glass curtain": MaterialPhysics(
            material_id=4,
            name="Glass Curtain",
            density_kg_m3=2500.0,
            cost_per_square_meter=210.0,
            carbon_kg_co2e=61.0,
            thermal_conductivity_w_mk=0.9,
            albedo=0.62,
            roughness=0.1,
            notes="Triple-glazed low-iron curtain wall",
        ),
        "linen_drape": MaterialPhysics(
            material_id=5,
            name="Linen Drape",
            density_kg_m3=410.0,
            cost_per_square_meter=48.0,
            carbon_kg_co2e=8.0,
            thermal_conductivity_w_mk=0.04,
            albedo=0.67,
            roughness=0.6,
            notes="Hand-loomed Belgian linen drapery",
        ),
        "pool_mosaic": MaterialPhysics(
            material_id=6,
            name="Pool Mosaic",
            density_kg_m3=2200.0,
            cost_per_square_meter=160.0,
            carbon_kg_co2e=42.0,
            thermal_conductivity_w_mk=1.6,
            albedo=0.41,
            roughness=0.35,
            notes="Glass mosaic tiles used in water features",
        ),
    }


MATERIAL_PHYSICS_DB: Dict[str, MaterialPhysics] = _build_material_db()


def _ensure_uint_mask(id_mask: np.ndarray | Iterable[int]) -> np.ndarray:
    """Normalise ``id_mask`` into a 2-D ``np.uint32`` array."""

    mask = np.asarray(id_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError("ID mask must be a 2-D array")
    return mask.astype(np.uint32, copy=False)


def _material_lookup(materials: Mapping[str, MaterialPhysics]) -> Dict[int, MaterialPhysics]:
    """Return a mapping from ``material_id`` to the descriptor."""

    by_id: Dict[int, MaterialPhysics] = {}
    for entry in materials.values():
        by_id.setdefault(entry.material_id, entry)
    return by_id


def create_property_tensor(
    id_mask: np.ndarray | Iterable[int],
    material_db: Mapping[str, MaterialPhysics] | None = None,
    *,
    pixel_size_meters: float = 0.01,
) -> MutableMapping[str, np.ndarray]:
    """Convert ``id_mask`` into a dense stack of material properties.

    Parameters
    ----------
    id_mask:
        Two-dimensional array of integers.  Each value corresponds to a
        ``material_id`` entry within ``material_db``.
    material_db:
        Optional override for the material database.  Defaults to
        :data:`MATERIAL_PHYSICS_DB`.
    pixel_size_meters:
        Physical size represented by each pixel.  Used to compute thickness and
        derived energy terms.
    """

    if pixel_size_meters <= 0:
        raise ValueError("pixel_size_meters must be positive")

    database = MATERIAL_PHYSICS_DB if material_db is None else dict(material_db)
    if not database:
        raise ValueError("material_db must contain at least one entry")

    id_mask_arr = _ensure_uint_mask(id_mask)
    material_by_id = _material_lookup(database)

    shape = id_mask_arr.shape
    density = np.zeros(shape, dtype=np.float32)
    cost = np.zeros(shape, dtype=np.float32)
    carbon = np.zeros(shape, dtype=np.float32)
    thermal = np.zeros(shape, dtype=np.float32)
    albedo = np.zeros(shape, dtype=np.float32)
    roughness = np.zeros(shape, dtype=np.float32)
    thickness = np.full(shape, pixel_size_meters, dtype=np.float32)

    for mat_id, descriptor in material_by_id.items():
        mask = id_mask_arr == np.uint32(mat_id)
        if not np.any(mask):
            continue
        density[mask] = descriptor.density_kg_m3
        cost[mask] = descriptor.cost_per_square_meter
        carbon[mask] = descriptor.carbon_kg_co2e
        thermal[mask] = descriptor.thermal_conductivity_w_mk
        albedo[mask] = descriptor.albedo
        roughness[mask] = descriptor.roughness

    tensor: MutableMapping[str, np.ndarray] = {
        "material_id": id_mask_arr,
        "density": density,
        "cost": cost,
        "carbon": carbon,
        "thermal_conductivity": thermal,
        "albedo": albedo,
        "roughness": roughness,
        "thickness": thickness,
    }
    return tensor


def summarise_tensor(tensor: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Return scalar diagnostics for ``tensor``.

    For area-based totals (carbon, cost), scale by pixel area derived from mean thickness.
    """

    summary: Dict[str, float] = {}
    pixel_area_m2 = None
    if "thickness" in tensor:
        # Assume square pixels, pixel size in meters is mean thickness
        pixel_size_meters = float(np.mean(tensor["thickness"]))
        pixel_area_m2 = pixel_size_meters ** 2
    if "density" in tensor:
        summary["mean_density"] = float(np.mean(tensor["density"]))
    if "carbon" in tensor:
        total_carbon = float(np.sum(tensor["carbon"]))
        if pixel_area_m2 is not None:
            total_carbon *= pixel_area_m2
        summary["total_carbon"] = total_carbon
    if "cost" in tensor:
        total_cost = float(np.sum(tensor["cost"]))
        if pixel_area_m2 is not None:
            total_cost *= pixel_area_m2
        summary["total_cost"] = total_cost
    if "thermal_conductivity" in tensor:
        summary["mean_thermal_conductivity"] = float(np.mean(tensor["thermal_conductivity"]))
    return summary


def save_property_tensor(tensor: Mapping[str, np.ndarray], output_dir: Path | str) -> Path:
    """Persist ``tensor`` to ``output_dir`` and return the archive path."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "material_property_tensor.npz"
    np.savez_compressed(archive_path, **{k: np.asarray(v) for k, v in tensor.items()})

    manifest = {
        "layers": sorted(tensor.keys()),
        "summary": summarise_tensor(tensor),
    }
    manifest_path = out_dir / "material_property_tensor.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return archive_path


__all__ = [
    "MaterialPhysics",
    "MATERIAL_PHYSICS_DB",
    "create_property_tensor",
    "save_property_tensor",
    "summarise_tensor",
]
