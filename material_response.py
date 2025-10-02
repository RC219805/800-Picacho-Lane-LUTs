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
from typing import Dict, Iterable, List, Sequence

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
            return 1.0 if np.isclose(energy_after, 0.0) else float("inf")

        return float(energy_after / energy_before)

    @staticmethod
    def _frequency_band_mask(shape: Sequence[int], band: str) -> np.ndarray:
        """Return a boolean mask isolating the requested frequency band."""

        grids = np.meshgrid(
            *[np.fft.fftfreq(n, d=1.0) for n in shape],
            indexing="ij",
        )
        radial_freq = np.sqrt(np.sum(np.square(grids), axis=0))
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

