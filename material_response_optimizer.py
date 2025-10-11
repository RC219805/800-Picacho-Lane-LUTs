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
            raw: Dict[str, Any] = json.load(fp)

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

    def build_blueprint(self) -> Dict[str, Any]:
        """Return a nested dictionary describing rendering upgrades."""

        luminance_strategy = self._derive_luminance_strategy()
        awe_alignment = self._derive_awe_alignment()
        comfort_plan = self._derive_comfort_plan()
        texture_dimensions = self._derive_texture_strategy()
        future_alignment = self._derive_future_alignment_strategy()
        lux_plan = self._derive_lux_strategy()
        algorithmic_formula = self._derive_algorithmic_formula()
        scene_specific = self._summarize_scene_specific_enhancements(algorithmic_formula)
        phase_program = self._derive_phase_program()
        measurement_loop = self._derive_measurement_loop()
        emotional_amplification = self._derive_emotional_amplification()

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
            "algorithmic_enhancement_formula": algorithmic_formula,
            "emotional_amplification": emotional_amplification,
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
            "phase_program": phase_program,
            "measurement_loop": measurement_loop,
        }

    def _derive_luminance_strategy(self) -> Dict[str, Any]:
        hierarchy_targets = []
        for scene in self.report.iter_scenes():
            base_luminance = scene.metric("regular", "luminance")
            if scene.name == "aerial":
                target_value = round(base_luminance * 0.8, 3)
                hierarchy_targets.append(
                    {
                        "scene": scene.name,
                        "current": base_luminance,
                        "target": target_value,
                        "status": "dial_back_luminance",
                        "reduction_percent": 20,
                        "aspirational_distance": "Restore aspirational distance by allowing shadows to veil access points.",
                        "atmospheric_haze": {
                            "color": "#F2CFA0",
                            "opacity": 0.2,
                            "placement": "ridge_line_and_pool_edge",
                        },
                        "light_trail_directive": "thread single surveillance drone path to imply monitored exclusivity.",
                        "notes": "Brightness surge solved visibility but collapsed mystique—pull exposure to reintroduce longing.",
                    }
                )
            elif scene.name == "pool":
                hierarchy_targets.append(
                    {
                        "scene": scene.name,
                        "current": base_luminance,
                        "target": base_luminance,
                        "status": "reconceptualize",
                        "conceptual_shift": {
                            "goal": "water_feature_over_pool",
                            "directives": [
                                "swap to obsidian or midnight blue tile",
                                "erase visible mechanical equipment",
                                "craft infinity edge spillway aligned with horizon",
                            ],
                        },
                        "surface_directive": "treat water as mirror with <2mm ripple tolerance",
                        "stillness_protocol": "deploy wind baffles + circulation pause for captures",
                        "notes": "Luminance tweaks cannot fix conceptual poverty—design drama through form and material.",
                    }
                )

        return {
            "reference_luminance": 0.31,
            "global_normalization": {"threshold": 0.25, "target_range": [0.26, 0.28]},
            "notes": "Preserve hierarchy but protect mystique where distance sells exclusivity.",
            "histogram_expansion": {
                "highlight_push": 150000,
                "strategy": "Sculpt specular peaks beyond the 90k plateau while guarding midtone nuance.",
            },
            "aspirational_distance": "Dial exposure only where access should feel earned.",
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
                    "god_rays": {"count": 4, "angles": "staggered", "dust_motes": True},
                    "stone_wall_treatment": {"clarity": "+30%", "uplighting": "gradient_from_floor"},
                    "fire_feature": {"reflection_mode": "color_dodge", "ceiling_kick": True},
                }
            )

        pool = self.report.scenes.get("pool")
        if pool is not None:
            actions.append(
                {
                    "scene": "pool",
                    "current": pool.metric("regular", "awe"),
                    "target": 0.74,
                    "twilight_transformation": {
                        "grade": {"highlights": "+15_orange", "shadows": "-10_blue"},
                        "underwater_geometry": "procedural_caustic_light_patterns",
                        "floating_elements": "candle_cluster_5_to_7",
                    },
                    "architectural_glow": {
                        "window_temperature": "warm_amber_#FFA500",
                        "falloff": "exponential",
                        "fire_reflections": "screen_blend_over_water",
                    },
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
            "adjustments": [
                {
                    "type": "architectural_severity",
                    "actions": [
                        "introduce sharp-edged furniture silhouettes",
                        "carve negative space between bed and seating",
                        "reposition lighting to cast angular contrast lines",
                    ],
                },
                {
                    "type": "material_injection",
                    "material_insertions": ["patinated_leather", "brushed_blackened_steel"],
                    "notes": "Trade plush upholstery for performance-grade surfaces.",
                },
                {
                    "type": "soft_goods_reduction",
                    "remove_percent": 30,
                    "method": "strip non-essential pillows, throws, and florals",
                },
                {
                    "type": "shadow_enrichment",
                    "delta_luminance": -0.15,
                    "placement": "perimeter_corners",
                    "geometry": "knife-edge flags to sculpt tension pockets",
                },
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
                    "push_to": 2.25,
                    "target_range": [2.2, 2.3],
                    "method": "frequency_separation + procedural detail passes",
                }
            )

        return {
            "baseline": 1.9,
            "target_range": [2.2, 2.3],
            "hero_targets": hero_targets,
            "guardrails": "Maintain supporting surfaces at 1.9 to avoid noise accumulation.",
            "technique": "frequency separation to decouple detail from color",
        }

    def _derive_future_alignment_strategy(self) -> Dict[str, Any]:
        adjustments = []
        impossible_elements = []
        for scene in self.report.iter_scenes():
            current = scene.metric("regular", "future_alignment")
            if current < 0.7:
                element = "floating_led_reveal" if scene.name in {"aerial", "pool"} else "cantilevered_shadow_gap"
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
                impossible_elements.append({"scene": scene.name, "concept": element})

        return {
            "summary": "Current readings imply contemporary comfort. Layer visionary cues to exceed 0.70.",
            "adjustments": adjustments,
            "impossible_elements": impossible_elements,
        }

    def _derive_emotional_amplification(self) -> Dict[str, Any]:
        """Outline drama-forward upgrades once technical fixes plateau."""

        plan: Dict[str, Any] = {}

        primary = self.report.scenes.get("primary_bedroom")
        if primary is not None:
            plan["primary_bedroom_power_reframe"] = {
                "current_comfort": primary.metric("regular", "comfort"),
                "goal": 0.85,
                "moves": [
                    "swap soft benches for angular, low-profile seating",
                    "introduce leather-and-metal nightstands to telegraph capability",
                    "pull drapery back to expose negative space and sharpen sightlines",
                ],
            }

        aerial = self.report.scenes.get("aerial")
        if aerial is not None:
            regular = aerial.versions["regular"]
            lux = aerial.versions["lux"]
            plateau_detected = abs(lux.luxury_index - regular.luxury_index) < 1e-3
            plan["aerial_distance_restoration"] = {
                "luxury_delta": lux.luxury_index - regular.luxury_index,
                "plateau_detected": plateau_detected,
                "directives": [
                    "drop overall luminance by ~20% to reinstate aspirational separation",
                    "layer atmospheric haze between property and viewer",
                    "spotlight only key thresholds to suggest controlled access",
                ],
            }

        pool = self.report.scenes.get("pool")
        if pool is not None:
            plan["pool_water_feature_reboot"] = {
                "current_luxury": pool.metric("regular", "luxury_index"),
                "target": 0.75,
                "interventions": [
                    "convert basin to obsidian mirror with infinity spill",
                    "purge pool hardware from sightlines and decks",
                    "stage ceremonial fire or sculpture to anchor reflection",
                ],
            }

        lux_variants = {
            scene.name: (scene.metric("lux", "luxury_index") - scene.metric("regular", "luxury_index"))
            for scene in self.report.iter_scenes()
        }
        stagnant = {scene: delta for scene, delta in lux_variants.items() if abs(delta) < 0.02}
        if stagnant:
            plan["lux_version_shadow_theater"] = {
                "scenes": sorted(stagnant),
                "strategy": [
                    "abandon exposure boosts in favor of chiaroscuro storytelling",
                    "treat lux sequences as after-dark exclusivity events",
                    "let pools, fireplaces, and concealed LEDs be primary emitters",
                ],
            }

        plan["conflict_resolution_protocol"] = {
            "mandate": "Identify and resolve conflicts between luminance retreat and perceived luxury before final approval.",
            "steps": [
                "flag scenes where lowered luminance drags luxury below target",
                "counterbalance with material richness or narrative cues instead of re-brightening",
                "document adjustments to maintain mystique without sacrificing value signaling",
            ],
        }

        return plan

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
                            "pivot to after-dark shadow theater instead of brighter copies",
                            "let water, fire, and concealed LEDs become sole emitters",
                            "withhold fill light to keep corners in mystery",
                            "introduce reflective surfaces that catch pinpoint highlights only",
                            "stage human absence to heighten exclusivity",
                        ],
                        "shadow_priority": True,
                    }
                )

        return {
            "observation": "Lux variants mirror regular frames rather than transform them.",
            "remedy": entries,
            "true_lux_transformation_protocol": [
                {
                    "step": "night_pivot",
                    "description": "Switch every lux frame into twilight/night narratives with deep shadow wells.",
                },
                {
                    "step": "emitters_only",
                    "description": "Limit light sources to curated emitters—fire, water, architectural reveals, and art lighting.",
                },
                {
                    "step": "specular_control",
                    "description": "Push speculars into the 150k histogram range while leaving midtones restrained.",
                },
                {
                    "step": "presence_management",
                    "description": "Remove casual lifestyle cues; treat lux frames as impossible private shows.",
                },
            ],
        }

    def _derive_algorithmic_formula(self) -> Dict[str, Any]:
        def current_metrics(scene_name: str) -> Dict[str, float]:
            snapshot = self.report.scenes[scene_name].versions["regular"]
            return {
                "luminance": snapshot.luminance,
                "awe": snapshot.awe,
                "comfort": snapshot.comfort,
                "texture_dimension": snapshot.texture_dimension,
                "future_alignment": snapshot.future_alignment,
                "luxury_index": snapshot.luxury_index,
            }

        return {
            "aerial": {
                "current": current_metrics("aerial"),
                "targets": {
                    "luminance": round(self.report.scenes["aerial"].metric("regular", "luminance") * 0.8, 3),
                    "luxury_index": 0.7,
                    "awe": 0.72,
                },
                "protocol": [
                    {
                        "step": "distance_restoration",
                        "actions": [
                            "drop global exposure by approximately 20%",
                            "feather vignettes around property perimeter",
                            "retain highlight integrity on signature architectural lines",
                        ],
                    },
                    {
                        "step": "elevated_haze",
                        "layers": [
                            {"type": "atmospheric_haze", "color": "#F2CFA0", "opacity": 0.2},
                            {"type": "gradient_fog", "direction": "foreground_to_background", "density": 0.18},
                        ],
                    },
                    {
                        "step": "threshold_highlighting",
                        "actions": [
                            "spotlight arrival path and helipad only",
                            "paint champagne sunset reflection narrowly across pool",
                            "leave majority of landscaping in controlled shadow",
                        ],
                    },
                    {
                        "step": "tech_aura",
                        "actions": [
                            "add single drone light trail",
                            "trace 1px LED seams with restrained outer glow to signal surveillance",
                        ],
                    },
                ],
            },
            "pool": {
                "current": current_metrics("pool"),
                "targets": {
                    "luxury_index": 0.75,
                    "future_alignment": 0.7,
                    "awe": 0.76,
                },
                "protocol": [
                    {
                        "step": "infinity_edge_illusion",
                        "actions": [
                            "raise water level to meet deck plane",
                            "blend far edge with horizon using spillway plate",
                            "extend deck reflections via mirror-finish coping",
                        ],
                    },
                    {
                        "step": "obsidian_tile_conversion",
                        "actions": [
                            "swap visible tile to obsidian or midnight blue glass mosaic",
                            "black out pool floor fittings and drains",
                            "introduce minimal perimeter lighting to kiss the edge",
                        ],
                    },
                    {
                        "step": "stillness_capture",
                        "actions": [
                            "pause circulation pumps during capture",
                            "deploy wind baffles for mirror calm",
                            "add ceremonial fire or sculpture for reflection anchor",
                        ],
                    },
                    {
                        "step": "narrative_extension",
                        "actions": [
                            "stage minimal seating to imply private performance",
                            "run concealed LED ribbon beneath coping for glow",
                        ],
                    },
                ],
            },
            "great_room": {
                "current": current_metrics("great_room"),
                "targets": {"luxury_index": 0.72, "awe": 0.85},
                "protocol": [
                    {
                        "step": "god_ray_multiplication",
                        "parameters": {"shafts": 4, "variation": "multi-angle", "dust_motes": True},
                    },
                    {
                        "step": "stone_wall_hero",
                        "actions": [
                            "increase clarity by 30%",
                            "add subtle uplighting gradient from floor",
                        ],
                    },
                    {
                        "step": "fire_feature_activation",
                        "actions": [
                            "paint flame reflections on adjacent surfaces",
                            "apply orange color dodge on ceiling near fireplace",
                        ],
                    },
                ],
            },
            "kitchen": {
                "current": current_metrics("kitchen"),
                "targets": {"luxury_index": 0.73, "awe": 0.977},
                "protocol": [
                    {
                        "step": "precision_polish",
                        "actions": [
                            "remove chromatic aberration to enhance sharpness",
                            "boost metallic luminance by 15% selectively",
                        ],
                    },
                    {
                        "step": "deliberate_flaw",
                        "actions": [
                            "leave a single cookbook open to chef's notes",
                            "fog one window corner with residual steam",
                            "stage a living herb pot with imperfect leaves",
                        ],
                    },
                ],
            },
            "primary_bedroom": {
                "current": current_metrics("primary_bedroom"),
                "targets": {"comfort": 0.85, "luxury_index": 0.71},
                "protocol": [
                    {
                        "step": "architectural_severity",
                        "actions": [
                            "replace soft seating with angular silhouettes",
                            "introduce metal inlays along headboard wall",
                            "carve light into razor lines across cabinetry",
                        ],
                    },
                    {
                        "step": "material_rearmament",
                        "actions": [
                            "swap plush textiles for patinated leather",
                            "introduce brushed blackened steel surfaces",
                            "limit palette to disciplined neutrals",
                        ],
                    },
                    {
                        "step": "negative_space_creation",
                        "actions": [
                            "remove 30% of soft furnishings",
                            "expose floor zone around bed for circulation cues",
                            "reinforce shadow pockets in perimeter",
                        ],
                    },
                ],
            },
        }

    def _summarize_scene_specific_enhancements(self, formula: Mapping[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for scene, plan in formula.items():
            targets = dict(plan.get("targets", {}))
            summary[scene] = {
                "current_luxury": plan["current"].get("luxury_index"),
                "target_luxury": targets.get("luxury_index", plan["current"].get("luxury_index")),
                "supporting_targets": {k: v for k, v in targets.items() if k != "luxury_index"},
                "headline_steps": [step["step"] for step in plan.get("protocol", [])],
            }
        return summary

    def _derive_phase_program(self) -> Iterable[Dict[str, Any]]:
        return [
            {
                "phase": 1,
                "name": "luminance_normalization",
                "objective": "Raise all sub-0.25 luminance scenes into the 0.26-0.28 band.",
                "actions": [
                    "batch parametric curves for aerial and pool midtone lifts (18-20%)",
                    "protect shadows via luminosity masks",
                    "validate pool/patio highlight masks before moving on",
                ],
            },
            {
                "phase": 2,
                "name": "hero_surface_protocol",
                "objective": "Push one hero surface per room into the 2.2-2.3 texture range.",
                "actions": [
                    "apply frequency separation passes to hero materials",
                    "restrain supporting surfaces to texture 1.9",
                    "document before/after microcontrast for island, stone wall, headboard",
                ],
            },
            {
                "phase": 3,
                "name": "awe_gap_correction",
                "objective": "Ensure every scene lands between 0.65 and 0.85 awe, led by kitchen benchmark.",
                "actions": [
                    "execute god ray and fire feature upgrades in great room",
                    "deploy twilight transformation and caustics in pool court",
                    "tune volumetric depth until awe scores converge",
                ],
            },
            {
                "phase": 4,
                "name": "future_alignment_leap",
                "objective": "Add one seemingly impossible element per scene to exceed 0.70 future alignment.",
                "actions": [
                    "float LED reveals off the architecture",
                    "introduce frameless reflections to imply hidden tech",
                    "audit for cantilever illusions and sensor light cues",
                ],
            },
        ]

    def _derive_measurement_loop(self) -> Dict[str, Any]:
        return {
            "analysis_refresh": "Re-run material response analysis after every enhancement pass.",
            "delta_guardrail": 0.15,
            "target_gain_window": [0.08, 0.12],
            "steps": [
                "Document metric deltas scene-by-scene after each iteration.",
                "Rollback or soften adjustments that exceed the 0.15 change guardrail.",
                "Lock in gains only when they fall within the 0.08-0.12 window.",
            ],
            "nuclear_option": {
                "description": "Time-shift the full set into dawn, day, dusk, and night series if incremental moves stall.",
                "targets": {
                    "dawn": {"focus": "maximum_focus", "goal_metrics": {"focus": 0.7}},
                    "day": {"focus": "balanced_metrics"},
                    "dusk": {"focus": "maximum_awe", "goal_metrics": {"awe": 0.8}},
                    "night": {"focus": "maximum_future_alignment", "goal_metrics": {"future_alignment": 0.75}},
                },
            },
        }


__all__ = [
    "MaterialResponseReport",
    "MetricSnapshot",
    "RenderEnhancementPlanner",
    "SceneReport",
]
