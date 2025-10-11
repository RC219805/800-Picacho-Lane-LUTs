"""Tests for :mod:`material_response_optimizer`."""

from __future__ import annotations

from pathlib import Path

import pytest

from material_response_optimizer import RenderEnhancementPlanner


@pytest.fixture(scope="module")
def blueprint() -> dict:
    planner = RenderEnhancementPlanner.from_json(Path("material_response_report.json"))
    return planner.build_blueprint()


def test_pool_requires_targeted_luminance(blueprint: dict) -> None:
    pool_entry = next(
        target for target in blueprint["luminance_strategy"]["targets"] if target["scene"] == "pool"
    )
    assert pytest.approx(pool_entry["current"], rel=1e-3) == 0.23
    assert pool_entry["target"] <= 0.32
    assert "specular_pool_reflections" in pool_entry["focus_areas"]
    assert "architectural_whites" in pool_entry["focus_areas"]


def test_awe_alignment_sets_explicit_targets(blueprint: dict) -> None:
    actions = {action["scene"]: action for action in blueprint["awe_alignment"]["actions"]}
    assert actions["great_room"]["target"] == 0.85
    assert actions["pool"]["target"] == 0.75


def test_comfort_reduction_defined_for_primary_suite(blueprint: dict) -> None:
    comfort = blueprint["comfort_realignment"]
    assert comfort["scene"] == "primary_bedroom"
    assert comfort["target"] == 0.85
    assert any("shadow" in move for move in comfort["moves"])


def test_hero_surface_texture_targets_present(blueprint: dict) -> None:
    hero_targets = blueprint["texture_dimension_strategy"]["hero_targets"]
    surfaces = {entry["surface"] for entry in hero_targets}
    assert {"island_waterfall_edge", "stone_feature_wall", "headboard_textile_panel"} <= surfaces
    assert all(entry["target"] == 2.25 for entry in hero_targets)


def test_lux_strategy_detects_low_deltas(blueprint: dict) -> None:
    lux_entries = blueprint["lux_version_strategy"]["remedy"]
    assert any(entry["scene"] == "aerial" for entry in lux_entries)
    aerial_entry = next(entry for entry in lux_entries if entry["scene"] == "aerial")
    assert aerial_entry["delta"] == pytest.approx(0.011, rel=1e-3)
    assert "golden hour" in " ".join(aerial_entry["actions"])


def test_scene_specific_targets_raise_luxury_indices(blueprint: dict) -> None:
    aerial_plan = blueprint["scene_specific_enhancements"]["aerial"]
    assert aerial_plan["target"] == pytest.approx(0.7, rel=1e-3)
    assert any("coastline" in move for move in aerial_plan["moves"])
