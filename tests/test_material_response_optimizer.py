"""Tests for :mod:`material_response_optimizer`."""

from __future__ import annotations

from pathlib import Path

import pytest

from material_response_optimizer import RenderEnhancementPlanner


@pytest.fixture(scope="module")
def blueprint() -> dict:
    planner = RenderEnhancementPlanner.from_json(Path("material_response_report.json"))
    return planner.build_blueprint()


def test_luminance_strategy_handles_histogram_conflict(blueprint: dict) -> None:
    luminance = blueprint["luminance_strategy"]
    aerial_entry = next(target for target in luminance["targets"] if target["scene"] == "aerial")
    assert aerial_entry["status"] == "gradient_regrade"
    assert aerial_entry["target_gradient"]["foreground"] == pytest.approx(0.45, rel=1e-3)
    assert aerial_entry["target_gradient"]["background"] == pytest.approx(0.25, rel=1e-3)
    assert "uniform" in aerial_entry["uniformity_warning"].lower()
    pool_entry = next(target for target in luminance["targets"] if target["scene"] == "pool")
    assert pool_entry["status"] == "composition_shift"
    directive = pool_entry["composition_directive"]
    assert directive["water_ratio_target"] == pytest.approx(0.3, rel=1e-3)
    assert "obsidian" in pool_entry["color_regrade"]
    histogram = luminance["histogram_expansion"]
    assert histogram["highlight_push"] == 155000
    assert histogram["shadow_floor"] == 4500
    commitment = luminance["histogram_narrative_commitment"]
    assert commitment["scene"] == "kitchen"
    assert "mandate" in commitment


def test_awe_alignment_sets_explicit_targets(blueprint: dict) -> None:
    actions = {action["scene"]: action for action in blueprint["awe_alignment"]["actions"]}
    assert actions["great_room"]["target"] == 0.85
    assert actions["great_room"]["god_rays"]["count"] == 4
    assert actions["pool"]["target"] == pytest.approx(0.74, rel=1e-3)
    assert actions["pool"]["twilight_transformation"]["grade"]["highlights"] == "+15_orange"


def test_primary_suite_focus_plan_balances_comfort(blueprint: dict) -> None:
    comfort = blueprint["comfort_realignment"]
    assert comfort["scene"] == "primary_bedroom"
    assert comfort["targets"]["comfort"] == pytest.approx(0.95, rel=1e-3)
    assert comfort["targets"]["focus"] == pytest.approx(0.7, rel=1e-3)
    adjustment_types = {entry["type"] for entry in comfort["adjustments"]}
    assert {"purpose_workspace", "capability_signal", "material_injection", "negative_space"} <= adjustment_types
    capability = next(entry for entry in comfort["adjustments"] if entry["type"] == "capability_signal")
    assert any("histogram" in action for action in capability["actions"])


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
    assert "texture-poor" in blueprint["texture_dimension_strategy"]["source_limitations"]


def test_lux_strategy_demands_dynamic_range(blueprint: dict) -> None:
    lux_entries = blueprint["lux_version_strategy"]["summary"]
    aerial_entry = next(entry for entry in lux_entries if entry["scene"] == "aerial")
    assert aerial_entry["shadow_priority"] is True
    assert any("dynamic range" in action for action in aerial_entry["actions"])
    protocol_steps = {step["step"] for step in blueprint["lux_version_strategy"]["true_lux_transformation_protocol"]}
    assert {"night_pivot", "shadow_theater", "specular_control"} <= protocol_steps
    assert blueprint["lux_version_strategy"]["warmth_program"]["lux"] == "mysterious warm/cool interplay"


def test_algorithmic_formula_tracks_scene_targets(blueprint: dict) -> None:
    aerial_formula = blueprint["algorithmic_enhancement_formula"]["aerial"]
    assert aerial_formula["targets"]["luminance_gradient"]["foreground"] == pytest.approx(0.45, rel=1e-3)
    assert any("gradient" in action for action in aerial_formula["protocol"][0]["actions"])
    pool_formula = blueprint["algorithmic_enhancement_formula"]["pool"]
    assert pool_formula["targets"]["luxury_index"] == pytest.approx(0.75, rel=1e-3)
    assert any("crop" in action for action in pool_formula["protocol"][0]["actions"])
    kitchen_formula = blueprint["algorithmic_enhancement_formula"]["kitchen"]
    assert any("cookbook" in action for action in kitchen_formula["protocol"][1]["actions"])
    assert any("void" in action for action in kitchen_formula["protocol"][2]["actions"])
    primary_formula = blueprint["algorithmic_enhancement_formula"]["primary_bedroom"]
    assert any(step["step"] == "purpose_activation" for step in primary_formula["protocol"])
    assert primary_formula["targets"]["focus"] == pytest.approx(0.7, rel=1e-3)


def test_emotional_amplification_reframes_value(blueprint: dict) -> None:
    amplification = blueprint["emotional_amplification"]
    assert amplification["aerial_distance_restoration"]["plateau_detected"] is True
    assert any("gradient" in directive for directive in amplification["aerial_distance_restoration"]["directives"])
    assert amplification["pool_water_feature_reboot"]["target"] == pytest.approx(0.75, rel=1e-3)
    primary_plan = amplification["primary_bedroom_power_reframe"]
    assert primary_plan["goal"] == pytest.approx(0.85, rel=1e-3)
    assert any("leather" in move for move in primary_plan["moves"])
    conflict_protocol = amplification["conflict_resolution_protocol"]
    assert "resolve conflicts" in conflict_protocol["mandate"].lower()
    warmth_variance = amplification["warmth_diffusion_variance"]
    assert "diffusion_range" in warmth_variance["targets"]
