"""Tests for board_material_aerial_enhancer module.

Validates clustering, material assignment, texture blending, and end-to-end
aerial enhancement workflow for MBAR board material application.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from board_material_aerial_enhancer import (
    ClusterStats,
    MaterialRule,
    apply_materials,
    assign_materials,
    build_material_rules,
    enhance_aerial,
)


def _dummy_rule(name: str, texture_path: Path) -> MaterialRule:
    """Create a MaterialRule for testing that always scores 1.0."""
    def score(_: ClusterStats) -> float:
        return 1.0

    return MaterialRule(name=name, texture=str(texture_path), blend=1.0, score_fn=score)


def test_assign_materials_prefers_high_scores(tmp_path: Path) -> None:
    """Test that material assignment selects clusters with highest scores."""
    stats = [
        ClusterStats(
            label=0, count=100, mean_rgb=np.array([0.9, 0.9, 0.85]),
            mean_hsv=np.array([0.1, 0.05, 0.9]), std_rgb=np.zeros(3)
        ),
        ClusterStats(
            label=1, count=80, mean_rgb=np.array([0.75, 0.65, 0.55]),
            mean_hsv=np.array([0.09, 0.25, 0.65]), std_rgb=np.zeros(3)
        ),
        ClusterStats(
            label=2, count=60, mean_rgb=np.array([0.5, 0.48, 0.46]),
            mean_hsv=np.array([0.02, 0.1, 0.5]), std_rgb=np.zeros(3)
        ),
        ClusterStats(
            label=3, count=40, mean_rgb=np.array([0.28, 0.22, 0.18]),
            mean_hsv=np.array([0.06, 0.2, 0.28]), std_rgb=np.zeros(3)
        ),
    ]

    # Create dummy texture files for all materials
    material_names = ("plaster", "stone", "cladding", "screens", "equitone", "roof", "bronze", "shade")
    textures = {name: (tmp_path / f"{name}.png") for name in material_names}
    for tex in textures.values():
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tex)

    rules = build_material_rules(textures)
    assignments = assign_materials(stats, rules)

    # Verify assignments are valid
    assert set(assignments.keys()) <= {stat.label for stat in stats}

    # Verify that high-scoring materials are assigned
    assert any(rule.name == "plaster" for rule in assignments.values())
    assert any(rule.name == "stone" for rule in assignments.values())


def test_apply_materials_blends_texture(tmp_path: Path) -> None:
    """Test that texture blending produces expected color shifts."""
    # Create gray base image
    base = np.full((4, 4, 3), 0.5, dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.uint8)

    # Create red texture
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(texture_path)

    # Apply red texture with full blend
    rule = MaterialRule(name="test", texture=str(texture_path), blend=1.0, score_fn=lambda _: 1.0)
    output = apply_materials(base, labels, {0: rule})

    # Verify red channel is dominant (with some tolerance for soft masking)
    assert np.isclose(output[..., 0].mean(), 1.0, atol=0.05)
    assert np.isclose(output[..., 1:].mean(), 0.0, atol=0.1)


def test_enhance_aerial_creates_output(tmp_path: Path) -> None:
    """Test end-to-end aerial enhancement workflow."""
    width, height = 160, 120

    # Create synthetic aerial with distinct regions
    image = Image.new("RGB", (width, height), (160, 160, 150))
    for x in range(width):
        for y in range(height):
            if x < width // 3:
                image.putpixel((x, y), (225, 215, 200))  # plaster region
            elif x < 2 * width // 3:
                image.putpixel((x, y), (190, 170, 150))  # stone region
            else:
                image.putpixel((x, y), (60, 50, 40))  # bronze/windows region

    input_path = tmp_path / "input.png"
    image.save(input_path)

    # Create texture files with distinct colors
    material_names = ("plaster", "stone", "cladding", "screens", "equitone", "roof", "bronze", "shade")
    textures = {name: (tmp_path / f"{name}.png") for name in material_names}
    colors = {
        "plaster": (240, 230, 215),
        "stone": (210, 190, 170),
        "cladding": (200, 170, 140),
        "screens": (150, 145, 140),
        "equitone": (100, 100, 105),
        "roof": (170, 170, 175),
        "bronze": (80, 60, 50),
        "shade": (240, 240, 240),
    }
    for name, tex_path in textures.items():
        Image.new("RGB", (16, 16), colors[name]).save(tex_path)

    output_path = tmp_path / "output.png"
    enhance_aerial(
        input_path,
        output_path,
        analysis_max_dim=128,
        k=4,
        seed=1,
        target_width=256,
        textures=textures,
    )

    # Verify output exists and has correct dimensions
    assert output_path.exists()
    enhanced = Image.open(output_path)
    assert enhanced.size[0] == 256

    # Verify enhancement modified the image
    assert np.asarray(enhanced).mean() != np.asarray(image.resize(enhanced.size)).mean()
