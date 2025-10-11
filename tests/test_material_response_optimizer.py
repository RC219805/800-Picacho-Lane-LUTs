"""Tests for :mod:`material_response_optimizer`."""

from __future__ import annotations

from pathlib import Path

import pytest

from material_response_optimizer import RenderEnhancementPlanner


@pytest.fixture(scope="module")
def blueprint() -> dict:
    planner = RenderEnhancementPlanner.from_json(Path("material_response_report.json"))
    return planner.build_blueprint()


def test_aerial_and_pool_luminance_strategy_shift(blueprint: dict) -> None:
    aerial_entry = next(
        target for target in blueprint["luminance_strategy"]["targets"] if target["scene"] == "aerial"
    )
    assert aerial_entry["status"] == "dial_back_luminance"
    assert aerial_entry["target"] == pytest.approx(0.307, rel=1e-3)
    assert aerial_entry["reduction_percent"] == 20
    assert aerial_entry["atmospheric_haze"]["opacity"] == pytest.approx(0.2, rel=1e-3)
    pool_entry = next(
        target for target in blueprint["luminance_strategy"]["targets"] if target["scene"] == "pool"
    )
    assert pool_entry["status"] == "reconceptualize"
    assert "obsidian" in pool_entry["conceptual_shift"]["directives"][0]
    assert "mirror" in pool_entry["surface_directive"]
    histogram = blueprint["luminance_strategy"]["histogram_expansion"]
    assert histogram["highlight_push"] == 150000


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
    assert {"architectural_severity", "material_injection", "soft_goods_reduction"} <= adjustment_types
    severity = next(entry for entry in comfort["adjustments"] if entry["type"] == "architectural_severity")
    assert "sharp-edged" in severity["actions"][0]
    soft_goods = next(entry for entry in comfort["adjustments"] if entry["type"] == "soft_goods_reduction")
    assert soft_goods["remove_percent"] == 30


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


def test_lux_strategy_embraces_shadow_theater(blueprint: dict) -> None:
    lux_entries = blueprint["lux_version_strategy"]["remedy"]
    aerial_entry = next(entry for entry in lux_entries if entry["scene"] == "aerial")
    assert aerial_entry["delta"] == pytest.approx(0.0, abs=1e-6)
    assert aerial_entry["shadow_priority"] is True
    assert any("shadow theater" in action for action in aerial_entry["actions"])
    protocol_steps = {step["step"] for step in blueprint["lux_version_strategy"]["true_lux_transformation_protocol"]}
    assert "night_pivot" in protocol_steps
    assert "specular_control" in protocol_steps


def test_algorithmic_formula_tracks_scene_targets(blueprint: dict) -> None:
    aerial_formula = blueprint["algorithmic_enhancement_formula"]["aerial"]
    assert aerial_formula["targets"]["luminance"] == pytest.approx(0.307, rel=1e-3)
    assert any(step["step"] == "distance_restoration" for step in aerial_formula["protocol"])
    pool_formula = blueprint["algorithmic_enhancement_formula"]["pool"]
    assert pool_formula["targets"]["luxury_index"] == pytest.approx(0.75, rel=1e-3)
    assert any("obsidian" in action for action in pool_formula["protocol"][1]["actions"])
    kitchen_formula = blueprint["algorithmic_enhancement_formula"]["kitchen"]
    assert any("cookbook" in action for action in kitchen_formula["protocol"][1]["actions"])
    primary_formula = blueprint["algorithmic_enhancement_formula"]["primary_bedroom"]
    assert any(step["step"] == "negative_space_creation" for step in primary_formula["protocol"])


def test_emotional_amplification_reframes_value(blueprint: dict) -> None:
    amplification = blueprint["emotional_amplification"]
    assert amplification["aerial_distance_restoration"]["plateau_detected"] is True
    assert amplification["pool_water_feature_reboot"]["target"] == pytest.approx(0.75, rel=1e-3)
    primary_plan = amplification["primary_bedroom_power_reframe"]
    assert primary_plan["goal"] == pytest.approx(0.85, rel=1e-3)
    assert any("leather" in move for move in primary_plan["moves"])
    conflict_protocol = amplification["conflict_resolution_protocol"]
    assert "resolve conflicts" in conflict_protocol["mandate"].lower()
