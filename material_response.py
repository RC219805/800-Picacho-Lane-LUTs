"""Material Response principle support utilities.

The marketing material for the 800 Picacho Lane collection repeatedly alludes
to a proprietary "Material Response" technology.  The production codebase does
not actually perform physically based rendering, but we can still document what
the principle *means* so tests and documentation have a concrete artefact to
reference.  This module provides such an artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MaterialResponseExample:
    """Structured example describing how the principle manifests."""

    material: str
    lighting: str
    challenge: str
    response: str
    outcome: str

    def as_dict(self) -> Dict[str, str]:
        """Return the example in a serialisable dictionary form."""

        return {
            "material": self.material,
            "lighting": self.lighting,
            "challenge": self.challenge,
            "response": self.response,
            "outcome": self.outcome,
        }


class MaterialResponsePrinciple:
    """Captures the intent behind the Material Response marketing claims.

    The principle is intentionally high-level: it frames how surface-specific
    adjustments should be reasoned about even when a pipeline primarily applies
    tone and colour transforms.  Tests that refer to this principle can inspect
    the structured data returned here without coupling to any particular image
    processing implementation.
    """

    name: str = "Material Response"
    focus: str = (
        "Honor the unique light interaction of each surface instead of applying "
        "purely global transforms."
    )

    tenets: List[str] = [
        "Respect energy conservation in highlights so reflective materials retain believable sheen.",
        "Preserve midtone texture to keep organic materials tactile and dimensional.",
        "Blend transitions between adjacent materials so adjustments feel authored rather than procedural.",
    ]

    def describe(self) -> str:
        """Return a human readable description of the principle."""

        return f"{self.name}: {self.focus}"

    def guidelines(self) -> List[str]:
        """Return the set of guiding tenets."""

        return list(self.tenets)

    def generate_examples(self) -> List[Dict[str, str]]:
        """Produce canonical examples demonstrating the principle in action.

        The examples intentionally cover a variety of materials.  Returning
        dictionaries keeps the data ergonomic for documentation tooling and
        avoids a dependency on this module's dataclass by callers.
        """

        examples: Iterable[MaterialResponseExample] = [
            MaterialResponseExample(
                material="polished marble foyer",
                lighting="late-afternoon sun grazing through clerestory windows",
                challenge="Specular highlights risk clipping and flattening the stone veining.",
                response="Apply highlight recovery before clarity so micro-contrast is boosted only after energy is preserved.",
                outcome="Marble retains luminous sheen with detailed veining rather than a white patch.",
            ),
            MaterialResponseExample(
                material="brushed brass fixtures",
                lighting="mixed tungsten sconces and cool daylight",
                challenge="Mixed colour temperatures create muddy warm-cool interference on the anisotropic metal.",
                response="Balance tint locally while moderating saturation to keep the brushed texture defined.",
                outcome="Brass reads as intentionally warm while grooves stay articulated.",
            ),
            MaterialResponseExample(
                material="velvet chaise",
                lighting="soft bounced fill with narrow specular kicker",
                challenge="Global contrast lift would crush the velvet's directionality and make it plastic.",
                response="Target midtone contrast and vibrance to maintain pile shading while enriching dye density.",
                outcome="Velvet keeps tactile depth and directional sheen without waxy highlights.",
            ),
        ]

        return [example.as_dict() for example in examples]


_STOP_WORDS = {
    "the",
    "and",
    "or",
    "so",
    "that",
    "a",
    "an",
    "to",
    "of",
    "in",
    "between",
    "with",
    "for",
    "be",
    "is",
    "are",
    "as",
    "keep",
    "so",
    "rather",
}


def _extract_keywords(text: str) -> List[str]:
    """Return a list of meaningful lowercase keywords from ``text``."""

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in _STOP_WORDS and len(token) > 2]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Return ``value`` limited to the inclusive ``[minimum, maximum]`` range."""

    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class MaterialAestheticProfile:
    """Structured description of how a material should be perceived."""

    name: str
    texture: str
    rarity: float
    craftsmanship: float
    innovation: float

    def __post_init__(self) -> None:
        for attribute in ("rarity", "craftsmanship", "innovation"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )


@dataclass(frozen=True)
class LightingProfile:
    """Simplified representation of the lighting environment."""

    warmth: float
    intensity: float
    diffusion: float

    def __post_init__(self) -> None:
        for attribute in ("warmth", "intensity", "diffusion"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )


@dataclass(frozen=True)
class ViewerProfile:
    """Describes the viewer in terms of aesthetic preferences."""

    cultural_background: str
    novelty_preference: float
    heritage_affinity: float

    def __post_init__(self) -> None:
        for attribute in ("novelty_preference", "heritage_affinity"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )

        if not self.cultural_background:
            raise ValueError("cultural_background must be a non-empty string")


@dataclass(frozen=True)
class EmotionalResonance:
    """Container capturing the viewer's predicted emotional response."""

    awe: float
    comfort: float
    focus: float
    cultural_background: str

    def __post_init__(self) -> None:
        for attribute in ("awe", "comfort", "focus"):
            value = getattr(self, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{attribute} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )

        if not self.cultural_background:
            raise ValueError("cultural_background must be a non-empty string")

    def as_dict(self) -> Dict[str, float]:
        """Return the resonance as a serialisable mapping."""

        return {
            "awe": self.awe,
            "comfort": self.comfort,
            "focus": self.focus,
        }


@dataclass(frozen=True)
class ContextualResonance:
    """Emotion scores contextualised for a cultural narrative."""

    scores: Dict[str, float]
    narrative: str

    def __post_init__(self) -> None:
        required_keys = {"awe", "comfort", "focus"}
        missing = required_keys.difference(self.scores)
        if missing:
            raise ValueError(
                f"scores must include {sorted(required_keys)}; missing keys: {sorted(missing)}"
            )

        for key, value in self.scores.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Score {key!r} must be between 0.0 and 1.0 inclusive; received {value!r}"
                )

        if not self.narrative:
            raise ValueError("narrative must be a non-empty string")


class NeuroAestheticEngine:
    """Lightweight psychophysical model for interpreting material descriptors."""

    _TEXTURE_AFFECTIONS: Mapping[str, Mapping[str, float]] = {
        "velvet": {"comfort": 0.18, "awe": 0.07},
        "marble": {"awe": 0.2, "focus": 0.08},
        "brushed": {"focus": 0.12, "awe": 0.04},
        "polished": {"awe": 0.15},
        "matte": {"focus": -0.05, "comfort": 0.05},
        "handcrafted": {"comfort": 0.12, "awe": 0.05},
        "organic": {"comfort": 0.1},
        "lacquer": {"awe": 0.06, "focus": 0.05},
    }

    def predict_limbic_response(
        self, texture: str, warmth: float, cultural_background: str
    ) -> EmotionalResonance:
        """Return an :class:`EmotionalResonance` derived from qualitative inputs."""

        if not 0.0 <= warmth <= 1.0:
            raise ValueError("warmth must be between 0.0 and 1.0 inclusive")

        tokens = re.findall(r"[a-z]+", texture.lower())
        awe = 0.45
        comfort = 0.45
        focus = 0.45

        for token in tokens:
            if token in self._TEXTURE_AFFECTIONS:
                weights = self._TEXTURE_AFFECTIONS[token]
                awe += weights.get("awe", 0.0)
                comfort += weights.get("comfort", 0.0)
                focus += weights.get("focus", 0.0)

        # Warmth emphasises comfort, whereas cooler setups heighten focus.
        comfort += 0.2 * (1.0 - abs(warmth - 0.6))
        focus += 0.18 * (0.5 - abs(warmth - 0.4))
        awe += 0.15 * warmth

        # Cultural familiarity slightly reinforces comfort and awe.
        cultural_modifier = 0.02 if cultural_background else 0.0
        comfort += cultural_modifier
        awe += cultural_modifier

        return EmotionalResonance(
            awe=_clamp(awe),
            comfort=_clamp(comfort),
            focus=_clamp(focus),
            cultural_background=cultural_background.lower(),
        )


class GlobalLuxurySemantics:
    """Contextualises emotion scores for regional expectations of luxury."""

    _CULTURAL_WEIGHTS: Mapping[str, Mapping[str, float]] = {
        "mediterranean": {"comfort": 0.06, "awe": 0.05},
        "scandinavian": {"comfort": 0.08, "focus": 0.07, "awe": -0.03},
        "japanese": {"focus": 0.09, "comfort": -0.02, "awe": 0.04},
        "middle eastern": {"awe": 0.08, "comfort": 0.04},
        "american": {"focus": 0.03, "comfort": 0.02},
    }

    def recontextualize(
        self, material: MaterialAestheticProfile, resonance: EmotionalResonance
    ) -> ContextualResonance:
        """Return resonance tuned to cultural and material narratives."""

        background = resonance.cultural_background.lower()
        weights = self._CULTURAL_WEIGHTS.get(background, {})

        awe = _clamp(resonance.awe + weights.get("awe", 0.0) + 0.12 * material.rarity)
        comfort = _clamp(
            resonance.comfort
            + weights.get("comfort", 0.0)
            + 0.1 * material.craftsmanship
        )
        focus = _clamp(
            resonance.focus
            + weights.get("focus", 0.0)
            + 0.08 * (1.0 - material.innovation)
        )

        narrative = (
            f"{material.name} channels a {background or 'global'} sensibility by "
            f"balancing awe ({awe:.2f}), comfort ({comfort:.2f}) and focus ({focus:.2f})."
        )

        return ContextualResonance(scores={"awe": awe, "comfort": comfort, "focus": focus}, narrative=narrative)


class FutureStatePredictor:
    """Projects how a material treatment will age alongside design trends."""

    def project(
        self, material: MaterialAestheticProfile, resonance: ContextualResonance
    ) -> float:
        """Return a 0-1 score indicating forward-looking relevance."""

        awe = resonance.scores["awe"]
        focus = resonance.scores["focus"]

        rarity_weight = 0.35 * material.rarity
        innovation_weight = 0.4 * material.innovation
        emotional_weight = 0.25 * (0.6 * awe + 0.4 * focus)

        projection = rarity_weight + innovation_weight + emotional_weight
        return _clamp(projection)


class CognitiveMaterialResponse:
    """High-level facade orchestrating the psychophysical subsystems."""

    def __init__(self) -> None:
        self.perception_model = NeuroAestheticEngine()
        self.cultural_context = GlobalLuxurySemantics()
        self.temporal_relevance = FutureStatePredictor()

    def process(
        self,
        material: MaterialAestheticProfile,
        lighting: LightingProfile,
        viewer_profile: ViewerProfile,
    ) -> Dict[str, object]:
        """Return a holistic appraisal of the material treatment."""

        # Not just physics - but psychophysics
        emotional_resonance = self.perception_model.predict_limbic_response(
            material.texture,
            lighting.warmth,
            viewer_profile.cultural_background,
        )

        return self.optimize_for_consciousness(material, emotional_resonance)

    def optimize_for_consciousness(
        self, material: MaterialAestheticProfile, emotional_resonance: EmotionalResonance
    ) -> Dict[str, object]:
        """Blend cultural and temporal heuristics into actionable guidance."""

        contextualized = self.cultural_context.recontextualize(material, emotional_resonance)
        future_alignment = self.temporal_relevance.project(material, contextualized)
        luxury_index = self._composite_index(material, contextualized, future_alignment)

        recommendations = self._recommendations(material, contextualized)

        return {
            "material": material.name,
            "texture": material.texture,
            "emotional_resonance": contextualized.scores,
            "narrative": contextualized.narrative,
            "luxury_index": luxury_index,
            "future_alignment": future_alignment,
            "recommendations": recommendations,
        }

    @staticmethod
    def _composite_index(
        material: MaterialAestheticProfile,
        resonance: ContextualResonance,
        future_alignment: float,
    ) -> float:
        craftsmanship_weight = 0.35 * material.craftsmanship
        rarity_weight = 0.25 * material.rarity
        emotional_weight = 0.25 * (0.5 * resonance.scores["awe"] + 0.5 * resonance.scores["comfort"])
        future_weight = 0.15 * future_alignment

        return _clamp(craftsmanship_weight + rarity_weight + emotional_weight + future_weight)

    @staticmethod
    def _recommendations(
        material: MaterialAestheticProfile, resonance: ContextualResonance
    ) -> List[str]:
        recommendations: List[str] = []
        awe = resonance.scores["awe"]
        comfort = resonance.scores["comfort"]
        focus = resonance.scores["focus"]

        if awe < 0.6:
            recommendations.append(
                "Introduce controlled specular accents to elevate perceived grandeur."
            )
        if comfort < 0.55:
            recommendations.append(
                "Blend warmer fill lighting or tactile styling to soften the presentation."
            )
        if focus < 0.5:
            recommendations.append(
                "Shape negative space to emphasise the material's structural rhythm."
            )

        if material.innovation > 0.65 and awe >= 0.6:
            recommendations.append(
                "Document the treatment narrative for launch collateral while momentum is high."
            )

        if not recommendations:
            recommendations.append("Maintain current treatment; responses align with luxury objectives.")

        return recommendations


def violates(decision: str, tenet: str) -> bool:
    """Heuristically determine whether ``decision`` conflicts with ``tenet``."""

    decision_lower = decision.lower()
    tenet_lower = tenet.lower()

    # Specific heuristics for the three marketing tenets.  These catch the most
    # common contradictions that show up in the architectural documentation and
    # associated tests.
    tenet_specific_checks = (
        (
            {"energy", "highlight"},
            {
                "clip the highlights",
                "blow out the highlights",
                "overexpose highlights",
                "ignore energy conservation",
                "treat highlights as unlimited",
            },
        ),
        (
            {"midtone", "texture"},
            {
                "flatten the midtones",
                "blur midtone texture",
                "remove midtone detail",
                "plastic midtones",
                "eliminate texture",
            },
        ),
        (
            {"blend", "transitions"},
            {
                "hard mask transition",
                "no blending between materials",
                "treat each material in isolation",
                "apply a binary mask",
            },
        ),
    )

    for keywords, forbidden_phrases in tenet_specific_checks:
        if keywords.issubset(set(_extract_keywords(tenet_lower))):
            if any(phrase in decision_lower for phrase in forbidden_phrases):
                return True

    # Generic negation handling: if the decision explicitly ignores or bypasses
    # aspects that the tenet cares about we treat it as a violation.
    negating_words = {
        "ignore",
        "bypass",
        "discard",
        "skip",
        "flatten",
        "eliminate",
        "remove",
        "crush",
        "erase",
    }
    keywords = _extract_keywords(tenet_lower)

    for negation in negating_words:
        if negation in decision_lower:
            if any(f"{negation} {keyword}" in decision_lower for keyword in keywords):
                return True
            # Fall back to a proximity check if the explicit "negation keyword"
            # pattern was not matched.  This keeps the heuristic resilient when
            # the decision references the concept in a different grammatical
            # order (e.g. "midtone texture gets flattened")
            for keyword in keywords:
                if keyword in decision_lower:
                    return True

    return False


class MarketingClaimValidator:
    """Validates that implementation decisions align with marketing principles."""

    def assess_decision(
        self, decision: str, principle: MaterialResponsePrinciple
    ) -> bool:
        """Return ``True`` when ``decision`` honours ``principle``'s tenets."""

        for tenet in principle.guidelines():
            if violates(decision, tenet):
                return False
        return True


__all__ = [
    "MaterialResponsePrinciple",
    "MaterialResponseExample",
    "MaterialAestheticProfile",
    "LightingProfile",
    "ViewerProfile",
    "EmotionalResonance",
    "ContextualResonance",
    "NeuroAestheticEngine",
    "GlobalLuxurySemantics",
    "FutureStatePredictor",
    "CognitiveMaterialResponse",
    "MarketingClaimValidator",
    "MaterialResponseValidator",
    "violates",
]


class MaterialResponseValidator:
    """Quantitative heuristics for validating material treatments.

    The validator deliberately keeps the implementations lightweight and
    dependency-free beyond :mod:`numpy`.  The goal is to surface signal that is
    directionally correct for tests rather than to provide production-grade
    analysis of BRDFs or fractal geometry.
    """

    def measure_specular_preservation(
        self, before: Sequence[Sequence[float]], after: Sequence[Sequence[float]]
    ) -> float:
        """Return the energy ratio for the high-frequency Fourier band.

        ``before`` and ``after`` are expected to be array-like objects that can
        be coerced into :class:`numpy.ndarray` instances.  The method computes the
        sum of squared magnitudes in the high-frequency region of the Fourier
        spectrum and reports ``after / before``.  When the reference energy is
        zero the ratio gracefully falls back to ``1.0`` so tests can reason about
        a neutral baseline.
        """

        return self._fourier_energy_ratio(before, after, band="high")

    def measure_texture_dimensionality(
        self, surface: Sequence[Sequence[float]]
    ) -> float:
        """Approximate the fractal (Hausdorff) dimension via box counting."""

        return self._calculate_hausdorff_dimension(surface)

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _fourier_energy_ratio(
        before: Sequence[Sequence[float]],
        after: Sequence[Sequence[float]],
        *,
        band: str,
    ) -> float:
        """Return the ratio of Fourier-band energy between ``after`` and ``before``.

        The helper performs minimal validation and defaults to returning ``1.0``
        if the reference energy is zero to keep the metric stable for synthetic
        fixtures used in the tests.
        """

        if band not in {"high", "low"}:
            raise ValueError("band must be 'high' or 'low'")

        before_arr = np.asarray(before, dtype=float)
        after_arr = np.asarray(after, dtype=float)

        if before_arr.shape != after_arr.shape:
            raise ValueError("before and after arrays must share the same shape")

        fft_before = np.fft.fftn(before_arr)
        fft_after = np.fft.fftn(after_arr)

        band_mask = MaterialResponseValidator._frequency_band_mask(fft_before.shape, band)

        energy_before = np.sum(np.abs(fft_before) ** 2 * band_mask)
        energy_after = np.sum(np.abs(fft_after) ** 2 * band_mask)

        if np.isclose(energy_before, 0.0):
            if np.isclose(energy_after, 0.0):
                return 1.0
            else:
                raise ValueError(
                    "Cannot compute Fourier energy ratio: energy_before is zero but energy_after is not. "
                    f"energy_before={energy_before}, energy_after={energy_after}, band={band}"
                )

        return float(energy_after / energy_before)

    @staticmethod
    def _frequency_band_mask(
        shape: Sequence[int], 
        band: str, 
        cutoff: float = None
    ) -> np.ndarray:
        """
        Return a boolean mask isolating the requested frequency band.

        By default, the median radial frequency is used as the cutoff to separate
        high and low frequency bands. This provides a balanced split for typical
        material textures, but can be overridden by specifying the `cutoff`
        parameter.
        """

        grids = np.meshgrid(
            *[np.fft.fftfreq(n, d=1.0) for n in shape],
            indexing="ij",
        )
        radial_freq = np.sqrt(np.sum(np.square(grids), axis=0))
        if cutoff is None:
            cutoff = np.median(radial_freq)

        if band == "high":
            return radial_freq >= cutoff
        return radial_freq <= cutoff

    @staticmethod
    def _calculate_hausdorff_dimension(surface: Sequence[Sequence[float]]) -> float:
        """Estimate fractal dimension using a simple box-counting approach."""

        data = np.asarray(surface, dtype=float)
        if data.ndim == 1:
            # Promote 1D signals to 2D by treating them as a single-row image.
            data = data[np.newaxis, :]
        elif data.ndim != 2:
            raise ValueError("surface must be 1D or 2D array-like")

        data = data - np.min(data)
        max_val = np.max(data)
        if not np.isclose(max_val, 0.0):
            data = data / max_val

        # Binarise around the median to highlight structure while reducing
        # sensitivity to absolute intensity.
        threshold = np.median(data)
        binary = data > threshold

        # Determine box sizes as powers of two that fit the smallest dimension.
        min_dim = min(binary.shape)
        max_exponent = int(np.floor(np.log2(min_dim)))
        if max_exponent <= 1:
            # If the grid is extremely small the slope degenerates; return a
            # neutral dimension of 1.0 to keep tests stable.
            return 1.0

        sizes = 2 ** np.arange(1, max_exponent)
        counts = []

        for size in sizes:
            counts.append(MaterialResponseValidator._boxcount(binary, size))

        # Perform linear regression in log-log space.  A tiny epsilon guards
        # against logarithms of zero when synthetic fixtures have no structure.
        eps = 1e-9
        coeffs = np.polyfit(np.log(sizes + eps), np.log(np.asarray(counts) + eps), 1)
        dimension = -coeffs[0]
        return float(max(dimension, 0.0))

    @staticmethod
    def _boxcount(binary: np.ndarray, size: int) -> int:
        """Count non-empty boxes of the given ``size`` for ``binary`` data."""

        shape = binary.shape
        new_shape = (shape[0] // size, size, shape[1] // size, size)
        trimmed = binary[: new_shape[0] * size, : new_shape[2] * size]
        reshaped = trimmed.reshape(new_shape)
        # Collapse the size axes and count how many boxes contain at least one
        # "True" cell.
        occupied = reshaped.any(axis=(1, 3))
        return int(np.count_nonzero(occupied))

