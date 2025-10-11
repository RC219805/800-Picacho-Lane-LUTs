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
    assert pool_entry["target"] == pytest.approx(0.265, rel=1e-3)
    assert pool_entry["midtone_lift_percent"] == 20
    assert "water_surface_speculars" in pool_entry["priority_regions"]
    specular_gain = pool_entry["highlight_overlays"][0]
    assert specular_gain["amount"] == pytest.approx(0.4, rel=1e-3)


def test_awe_alignment_sets_explicit_targets(blueprint: dict) -> None:
    actions = {action["scene"]: action for action in blueprint["awe_alignment"]["actions"]}
    assert actions["great_room"]["target"] == 0.85
    assert actions["great_room"]["god_rays"]["count"] == 4
    assert actions["pool"]["target"] == pytest.approx(0.74, rel=1e-3)
    assert actions["pool"]["twilight_transformation"]["grade"]["highlights"] == "+15_orange"


def test_comfort_reduction_defined_for_primary_suite(blueprint: dict) -> None:
    comfort = blueprint["comfort_realignment"]
    assert comfort["scene"] == "primary_bedroom"
    assert comfort["target"] == 0.85
    adjustment_types = {entry["type"] for entry in comfort["adjustments"]}
    assert {"shadow_enrichment", "view_activation", "material_complexity"} <= adjustment_types
    view_activation = next(entry for entry in comfort["adjustments"] if entry["type"] == "view_activation")
    assert "ship lights" in view_activation["detail"]


def test_hero_surface_texture_targets_present(blueprint: dict) -> None:
    hero_targets = blueprint["texture_dimension_strategy"]["hero_targets"]
    surfaces = {entry["surface"] for entry in hero_targets}
    assert {"island_waterfall_edge", "stone_feature_wall", "headboard_textile_panel"} <= surfaces
    push_values = {entry["push_to"] for entry in hero_targets}
    assert push_values == {2.25}
    for entry in hero_targets:
        assert entry["target_range"] == [2.2, 2.3]
    assert blueprint["texture_dimension_strategy"]["target_range"] == [2.2, 2.3]
    assert "frequency separation" in blueprint["texture_dimension_strategy"]["technique"]


def test_lux_strategy_detects_low_deltas(blueprint: dict) -> None:
    lux_entries = blueprint["lux_version_strategy"]["remedy"]
    assert any(entry["scene"] == "aerial" for entry in lux_entries)
    aerial_entry = next(entry for entry in lux_entries if entry["scene"] == "aerial")
    assert aerial_entry["delta"] == pytest.approx(0.011, rel=1e-3)
    assert any("blue hour" in action for action in aerial_entry["actions"])
    protocol_steps = blueprint["lux_version_strategy"]["true_lux_transformation_protocol"]
    assert any(step["step"] == "time_shift" for step in protocol_steps)


def test_algorithmic_formula_tracks_scene_targets(blueprint: dict) -> None:
    aerial_formula = blueprint["algorithmic_enhancement_formula"]["aerial"]
    assert aerial_formula["targets"]["luxury_index"] == pytest.approx(0.68, rel=1e-3)
    assert any(step["step"] == "future_elements" for step in aerial_formula["protocol"])
    pool_formula = blueprint["algorithmic_enhancement_formula"]["pool"]
    assert pool_formula["targets"]["awe"] == pytest.approx(0.74, rel=1e-3)
    assert any("floating candles" in action for action in pool_formula["protocol"][1]["actions"])
