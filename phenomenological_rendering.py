"""Lightweight phenomenological renderer for lighting scenario comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_SCENARIOS = ("golden_hour", "noon", "overcast", "dusk")


def _scenario_lighting_bias(scenario: str) -> float:
    return {
        "golden_hour": 1.15,
        "noon": 1.0,
        "overcast": 0.85,
        "dusk": 0.75,
        "midnight": 0.45,
    }.get(scenario, 1.0)


def _scenario_tint(scenario: str) -> tuple[int, int, int]:
    palette = {
        "golden_hour": (255, 196, 112),
        "noon": (210, 228, 255),
        "overcast": (200, 210, 220),
        "dusk": (160, 180, 255),
        "midnight": (80, 100, 180),
    }
    return palette.get(scenario, (220, 220, 220))


def _prepare_canvas(base_color: tuple[int, int, int], intensity: float, size: int = 512) -> Image.Image:
    r, g, b = base_color
    scale = np.clip(intensity, 0.2, 1.4)
    colour = (int(r * scale) % 256, int(g * scale) % 256, int(b * scale) % 256)
    return Image.new("RGB", (size, size), colour)


def render_lighting_comparison(
    tensor: Mapping[str, np.ndarray],
    output_dir: Path | str,
    *,
    scenarios: Sequence[str] | None = None,
) -> MutableMapping[str, Path]:
    """Render colour-coded squares representing different lighting scenarios."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = tuple(scenarios or DEFAULT_SCENARIOS)

    base_albedo = float(np.mean(tensor.get("albedo", np.array([0.5]))))
    base_roughness = float(np.mean(tensor.get("roughness", np.array([0.5]))))

    outputs: MutableMapping[str, Path] = {}

    for scenario in scenarios:
        tint = _scenario_tint(scenario)
        intensity = base_albedo * _scenario_lighting_bias(scenario)
        canvas = _prepare_canvas(tint, intensity)
        draw = ImageDraw.Draw(canvas)
        text = f"{scenario}\nAlbedo {base_albedo:.2f}\nRoughness {base_roughness:.2f}"
        draw.multiline_text((16, 16), text, fill=(20, 20, 20))
        file_path = out_dir / f"render_{scenario}.png"
        canvas.save(file_path)
        outputs[scenario] = file_path

    metadata = {
        "scenarios": list(scenarios),
        "base_albedo": base_albedo,
        "base_roughness": base_roughness,
        "outputs": {scenario: str(path) for scenario, path in outputs.items()},
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return outputs


__all__ = [
    "DEFAULT_SCENARIOS",
    "render_lighting_comparison",
]
