"""Blueprint generation for enhancing Material Response renderings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class MetricSnapshot:
    """Bundle of quantitative scores for a single rendering version."""

    luminance: float
    awe: float
    comfort: float
    texture_dimension: float
    future_alignment: float
    luxury_index: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MetricSnapshot":
        """Create a snapshot from a JSON-compatible mapping."""

        return cls(
            luminance=float(data["luminance"]),
            awe=float(data["awe"]),
            comfort=float(data["comfort"]),
            texture_dimension=float(data["texture_dimension"]),
            future_alignment=float(data["future_alignment"]),
            luxury_index=float(data["luxury_index"]),
        )


@dataclass(frozen=True)
class SceneReport:
    """Collection of scores for each deliverable version."""

    name: str
    versions: Mapping[str, MetricSnapshot]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SceneReport":
        versions = {
            version_name: MetricSnapshot.from_mapping(version_metrics)
            for version_name, version_metrics in data.get("versions", {}).items()
        }
        return cls(name=str(data["name"]), versions=versions)

    def metric(self, version: str, metric_name: str) -> float:
        """Return ``metric_name`` for ``version`` raising ``KeyError`` on failure."""

        snapshot = self.versions[version]
        return getattr(snapshot, metric_name)


@dataclass(frozen=True)
class MaterialResponseReport:
    """Structured representation of ``material_response_report.json``."""

    generated: str
    analysis_version: str
    scenes: Mapping[str, SceneReport]

    @classmethod
    def load(cls, path: str | Path) -> "MaterialResponseReport":
        with Path(path).open("r", encoding="utf-8") as fp:
            raw: dict[str, Any] = json.load(fp)

        scenes = {
            scene_data["name"]: SceneReport.from_mapping(scene_data)
            for scene_data in raw.get("scenes", [])
        }

        return cls(
            generated=str(raw.get("generated", "")),
            analysis_version=str(raw.get("analysis_version", "")),
            scenes=scenes,
        )

    def iter_scenes(self) -> Iterable[SceneReport]:
        """Yield scenes in the report."""

        return self.scenes.values()

    def top_scene(self, metric_name: str, *, version: str = "regular") -> SceneReport:
        """Return the scene with the highest ``metric_name`` for ``version``."""

        return max(self.iter_scenes(), key=lambda scene: scene.metric(version, metric_name))


class RenderEnhancementPlanner:
    """Translate report metrics into an actionable upgrade blueprint."""

    def __init__(self, report: MaterialResponseReport):
        self.report = report

    @classmethod
    def from_json(cls, path: str | Path) -> "RenderEnhancementPlanner":
        return cls(MaterialResponseReport.load(path))

    def build_blueprint(self) -> dict[str, Any]:
        """Return a nested dictionary describing rendering upgrades."""

        luminance_strategy = self._derive_luminance_strategy()
        awe_alignment = self._derive_awe_alignment()
        comfort_plan = self._derive_comfort_plan()
        texture_dimensions = self._derive_texture_strategy()
        future_alignment = self._derive_future_alignment_strategy()
        lux_plan = self._derive_lux_strategy()
        scene_specific = self._derive_scene_specific_upgrades()

        return {
            "generated": self.report.generated,
            "analysis_version": self.report.analysis_version,
            "luminance_strategy": luminance_strategy,
            "awe_alignment": awe_alignment,
            "comfort_realignment": comfort_plan,
            "texture_dimension_strategy": texture_dimensions,
            "future_alignment": future_alignment,
            "lux_version_strategy": lux_plan,
            "scene_specific_enhancements": scene_specific,
            "narrative": (
                "transcends conventional luxury through orchestrated tension "
                "between photonic drama, tactile richness, and future-forward quiet tech."
            ),
            "technical_workflow": {
                "layer_stack": [
                    "base_render",
                    "targeted_luminance_curves",
                    "hero_surface_texture_pass",
                    "atmospheric_depth_layers",
                    "floating_future_elements",
                    "golden_hour_color_grade",
                ]
            },
            "variation_sets": [
                {
                    "name": "morning_precision",
                    "focus": "heightened focus, tempered comfort",
                    "metrics_bias": {"comfort": -0.05, "awe": +0.03},
                },
                {
                    "name": "twilight_awe",
                    "focus": "maximum awe and luxury allure",
                    "metrics_bias": {"awe": +0.1, "luxury_index": +0.08},
                },
                {
                    "name": "night_future_alignment",
                    "focus": "visionary tech-forward ambience",
                    "metrics_bias": {"future_alignment": +0.12},
                },
            ],
            "ab_testing_framework": {
                "variants": [
                    {"name": "baseline", "delta": 0.0},
                    {"name": "elevated", "delta": +0.15},
                    {"name": "targeted_room_bias", "delta": "per-metric"},
                ],
                "goal": "orchestrate an emotional crescendo instead of uniform gains",
            },
        }

    def _derive_luminance_strategy(self) -> Dict[str, Any]:
        hierarchy_targets = []
        for scene in self.report.iter_scenes():
            base_luminance = scene.metric("regular", "luminance")
            if scene.name in {"pool", "aerial"} and base_luminance < 0.3:
                target = round(min(0.32, base_luminance * 1.18), 4)
                hierarchy_targets.append(
                    {
                        "scene": scene.name,
                        "current": base_luminance,
                        "target": target,
                        "focus_areas": [
                            "specular_pool_reflections" if scene.name == "pool" else "roofline_glow",
                            "interior_window_bloom" if scene.name == "aerial" else "architectural_whites",
                        ],
                        "approach": "sculpted masks and dodge layers to avoid uniform brightening",
                    }
                )

        return {
            "reference_luminance": 0.31,
            "notes": "0.30-0.32 scores correlate with top luxury perception. Maintain hierarchy.",
            "targets": hierarchy_targets,
        }

    def _derive_awe_alignment(self) -> Dict[str, Any]:
        kitchen_awe = self.report.scenes["kitchen"].metric("regular", "awe")
        actions = []

        great_room = self.report.scenes.get("great_room")
        if great_room is not None:
            actions.append(
                {
                    "scene": "great_room",
                    "current": great_room.metric("regular", "awe"),
                    "target": 0.85,
                    "moves": [
                        "introduce volumetric sunset shaft through skylight",
                        "amplify contrast on stone wall relief",
                    ],
                }
            )

        pool = self.report.scenes.get("pool")
        if pool is not None:
            actions.append(
                {
                    "scene": "pool",
                    "current": pool.metric("regular", "awe"),
                    "target": 0.75,
                    "moves": [
                        "activate underwater lighting with geometric caustics",
                        "layer fire feature reflections across water surface",
                    ],
                }
            )

        return {
            "benchmark_scene": "kitchen",
            "benchmark_awe": kitchen_awe,
            "actions": actions,
        }

    def _derive_comfort_plan(self) -> Dict[str, Any]:
        bedroom = self.report.scenes.get("primary_bedroom")
        if bedroom is None:
            return {}

        comfort = bedroom.metric("regular", "comfort")
        return {
            "scene": "primary_bedroom",
            "current": comfort,
            "target": 0.85,
            "moves": [
                "fold in subtle corner shadows to reintroduce tension",
                "cool ambient color temperature by ~250K",
                "animate sheer curtains for micro-motion",
            ],
        }

    def _derive_texture_strategy(self) -> Dict[str, Any]:
        hero_surfaces = {
            "kitchen": "island_waterfall_edge",
            "great_room": "stone_feature_wall",
            "primary_bedroom": "headboard_textile_panel",
        }

        hero_targets = []
        for scene_name, surface in hero_surfaces.items():
            scene = self.report.scenes.get(scene_name)
            if scene is None:
                continue
            current = scene.metric("regular", "texture_dimension")
            hero_targets.append(
                {
                    "scene": scene_name,
                    "surface": surface,
                    "current": current,
                    "target": 2.25,
                    "method": "microcontrast maps + procedural detail passes",
                }
            )

        return {
            "baseline": 1.9,
            "hero_targets": hero_targets,
            "guardrails": "Maintain supporting surfaces at 1.9 to avoid noise accumulation.",
        }

    def _derive_future_alignment_strategy(self) -> Dict[str, Any]:
        adjustments = []
        for scene in self.report.iter_scenes():
            current = scene.metric("regular", "future_alignment")
            if current < 0.7:
                adjustments.append(
                    {
                        "scene": scene.name,
                        "current": current,
                        "target": 0.72,
                        "interventions": [
                            "float linear LED reveals detached from architecture",
                            "introduce high-polish reflections for spatial ambiguity",
                            "embed discreet sensor-like pin lights",
                        ],
                    }
                )

        return {
            "summary": "Current readings imply contemporary comfort. Layer visionary cues to exceed 0.70.",
            "adjustments": adjustments,
        }

    def _derive_lux_strategy(self) -> Dict[str, Any]:
        entries = []
        for scene in self.report.iter_scenes():
            regular = scene.metric("regular", "luxury_index")
            lux = scene.metric("lux", "luxury_index")
            delta = lux - regular
            if delta < 0.05:
                entries.append(
                    {
                        "scene": scene.name,
                        "delta": round(delta, 4),
                        "actions": [
                            "global golden hour regrade",
                            "prismatic highlights in glass and water",
                            "boost natural material saturation by 20%",
                            "layer atmospheric haze for multi-plane depth",
                        ],
                    }
                )

        return {
            "observation": "Lux variants only marginally outperform baseline.",
            "remedy": entries,
        }

    def _derive_scene_specific_upgrades(self) -> Dict[str, Any]:
        return {
            "aerial": {
                "current_luxury": self.report.scenes["aerial"].metric("regular", "luxury_index"),
                "target": 0.7,
                "moves": [
                    "paint champagne sunset across pool",
                    "project architectural light patterns onto landscaping",
                    "hint at distant coastline haze",
                ],
            },
            "pool": {
                "current_luxury": self.report.scenes["pool"].metric("regular", "luxury_index"),
                "target": 0.72,
                "moves": [
                    "introduce spa steam plumes",
                    "cast caustic light dances on retaining walls",
                    "stage floating floral candles",
                ],
            },
            "great_room": {
                "current_luxury": self.report.scenes["great_room"].metric("regular", "luxury_index"),
                "target": 0.75,
                "moves": [
                    "intensify fire feature for kinetic shadow play",
                    "suspend dust motes inside skylight beam",
                    "animate curtain sway for breathable movement",
                ],
            },
        }


__all__ = [
    "MaterialResponseReport",
    "MetricSnapshot",
    "RenderEnhancementPlanner",
    "SceneReport",
]
