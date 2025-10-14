"""Enhanced material substrate with perceptual microstructure channels."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping

import numpy as np

from material_property_substrate import (
    MATERIAL_PHYSICS_DB as BASE_DB,
    MaterialPhysics,
    create_property_tensor as base_create_property_tensor,
    save_property_tensor as base_save_property_tensor,
)


# Copy the baseline DB and sprinkle additional luxurious materials.
MATERIAL_PHYSICS_DB: Mapping[str, MaterialPhysics] = {
    **BASE_DB,
    "white_plaster": MaterialPhysics(
        material_id=8,
        name="White Plaster",
        density_kg_m3=1900.0,
        cost_per_square_meter=72.0,
        carbon_kg_co2e=29.0,
        thermal_conductivity_w_mk=0.7,
        albedo=0.7,
        roughness=0.42,
        notes="Luminous lime plaster with marble dust aggregate",
    ),
    "travertine": MaterialPhysics(
        material_id=9,
        name="Travertine",
        density_kg_m3=2500.0,
        cost_per_square_meter=195.0,
        carbon_kg_co2e=56.0,
        thermal_conductivity_w_mk=1.1,
        albedo=0.48,
        roughness=0.33,
        notes="Veined Italian travertine for spa suites",
    ),
}


def _microstructure_response(id_mask: np.ndarray) -> np.ndarray:
    """Generate a pseudo microstructure field based on ID gradients."""

    grad_y, grad_x = np.gradient(id_mask.astype(np.float32))
    magnitude = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    if magnitude.size == 0:
        return magnitude
    magnitude_max = magnitude.max()
    magnitude = magnitude / (magnitude_max if magnitude_max != 0 else 1.0)
    return magnitude.astype(np.float32)


def create_property_tensor(
    id_mask: np.ndarray,
    material_db: Mapping[str, MaterialPhysics] | None = None,
    *,
    pixel_size_meters: float = 0.01,
) -> MutableMapping[str, np.ndarray]:
    """Return the enhanced property tensor with microstructure guidance."""

    db = MATERIAL_PHYSICS_DB if material_db is None else material_db
    tensor = base_create_property_tensor(
        id_mask,
        db,
        pixel_size_meters=pixel_size_meters,
    )

    tensor["microstructure"] = _microstructure_response(tensor["material_id"])

    # A light-weight perceptual brightness hint (albedo smoothed across neighbours)
    albedo = tensor.get("albedo")
    if albedo is not None:
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
        kernel /= kernel.sum()
        padded = np.pad(albedo, 1, mode="edge")
        brightness = np.zeros_like(albedo)
        for i in range(brightness.shape[0]):
            for j in range(brightness.shape[1]):
                brightness[i, j] = np.sum(padded[i : i + 3, j : j + 3] * kernel)
        tensor["perceptual_brightness"] = brightness
    return tensor


def save_property_tensor(tensor: Mapping[str, np.ndarray], output_dir: str | Path) -> Path:
    """Proxy through to the baseline saver (keeps manifest generation in sync)."""

    return base_save_property_tensor(tensor, output_dir)


__all__ = [
    "MATERIAL_PHYSICS_DB",
    "MaterialPhysics",
    "create_property_tensor",
    "save_property_tensor",
]
