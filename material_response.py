"""Material Response principle support utilities.

The marketing material for the 800 Picacho Lane collection repeatedly alludes
to a proprietary "Material Response" technology.  The production codebase does
not actually perform physically based rendering, but we can still document what
the principle *means* so tests and documentation have a concrete artefact to
reference.  This module provides such an artefact.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple


def _is_sequence(value: object) -> bool:
    """Return ``True`` when ``value`` should be treated as a sequence."""

    return isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def _coerce_matrix(data: Sequence[Sequence[float]]) -> List[List[float]]:
    """Convert ``data`` into a rectangular list-of-lists of floats."""

    if not _is_sequence(data):
        raise TypeError("material data must be a sequence")

    if len(data) == 0:
        raise ValueError("material data cannot be empty")

    if _is_sequence(data[0]):
        rows = []
        row_length = None
        for row in data:
            if not _is_sequence(row):
                raise TypeError("material rows must be sequences")
            coerced_row = [float(value) for value in row]
            if row_length is None:
                row_length = len(coerced_row)
                if row_length == 0:
                    raise ValueError("material rows cannot be empty")
            elif len(coerced_row) != row_length:
                raise ValueError("material data must form a rectangular grid")
            rows.append(coerced_row)
        return rows

    coerced = [float(value) for value in data]
    if len(coerced) == 0:
        raise ValueError("material data cannot be empty")
    return [coerced]


def _flatten(matrix: Sequence[Sequence[float]]) -> List[float]:
    """Return a flattened representation of ``matrix``."""

    return [value for row in matrix for value in row]


def _median(values: Sequence[float]) -> float:
    """Compute the median of ``values``."""

    ordered = sorted(values)
    length = len(ordered)
    if length == 0:
        raise ValueError("median of empty sequence is undefined")
    middle = length // 2
    if length % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2.0
    return ordered[middle]


def _fft_frequency(index: int, size: int) -> float:
    """Return the normalised FFT frequency for ``index``."""

    if size <= 0:
        raise ValueError("size must be positive")
    half = size // 2
    if index <= half:
        return index / size
    return (index - size) / size


def _dft2(matrix: Sequence[Sequence[float]]) -> List[List[complex]]:
    """Compute the 2D discrete Fourier transform for ``matrix``."""

    rows = len(matrix)
    cols = len(matrix[0])
    result = [[0j for _ in range(cols)] for _ in range(rows)]

    for u in range(rows):
        for v in range(cols):
            total = 0j
            for x in range(rows):
                for y in range(cols):
                    angle = -2.0 * math.pi * ((u * x / rows) + (v * y / cols))
                    total += matrix[x][y] * complex(math.cos(angle), math.sin(angle))
            result[u][v] = total

    return result


def _energy_by_band(
    dft: Sequence[Sequence[complex]],
    band: str,
    cutoff: float,
    radii: Sequence[Sequence[float]],
) -> float:
    """Return the total squared magnitude energy for the requested band."""

    total = 0.0
    rows = len(dft)
    cols = len(dft[0])

    for u in range(rows):
        for v in range(cols):
            radius = radii[u][v]
            in_band = radius >= cutoff if band == "high" else radius <= cutoff
            if in_band:
                coefficient = dft[u][v]
                total += (coefficient.real ** 2) + (coefficient.imag ** 2)

    return total


def _radial_frequency_grid(shape: Tuple[int, int]) -> List[List[float]]:
    """Return the radial frequency grid for ``shape``."""

    rows, cols = shape
    grid = []
    for u in range(rows):
        row = []
        freq_u = _fft_frequency(u, rows)
        for v in range(cols):
            freq_v = _fft_frequency(v, cols)
            row.append(math.hypot(freq_u, freq_v))
        grid.append(row)
    return grid


def _linear_regression(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Return the slope and intercept for ``xs``/``ys``."""

    if len(xs) != len(ys):
        raise ValueError("xs and ys must have matching lengths")
    n = len(xs)
    if n == 0:
        raise ValueError("linear regression requires data")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)

    if math.isclose(denominator, 0.0):
        return 0.0, mean_y

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


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

    The validator deliberately keeps the implementations lightweight and free of
    heavy numerical dependencies.  The goal is to surface signal that is
    directionally correct for tests rather than to provide production-grade
    analysis of BRDFs or fractal geometry.
    """

    def measure_specular_preservation(
        self, before: Sequence[Sequence[float]], after: Sequence[Sequence[float]]
    ) -> float:
        """Return the energy ratio for the high-frequency Fourier band.

        ``before`` and ``after`` are expected to be array-like objects that can
        be coerced into rectangular grids of floats.  The method computes the sum
        of squared magnitudes in the high-frequency region of the Fourier
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

        before_matrix = _coerce_matrix(before)
        after_matrix = _coerce_matrix(after)

        if len(before_matrix) != len(after_matrix) or len(before_matrix[0]) != len(after_matrix[0]):
            raise ValueError("before and after arrays must share the same shape")

        dft_before = _dft2(before_matrix)
        dft_after = _dft2(after_matrix)

        radii = _radial_frequency_grid((len(before_matrix), len(before_matrix[0])))
        cutoff = _median(_flatten(radii))

        energy_before = _energy_by_band(dft_before, band, cutoff, radii)
        energy_after = _energy_by_band(dft_after, band, cutoff, radii)

        if math.isclose(energy_before, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            if math.isclose(energy_after, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                return 1.0
            raise ValueError(
                "Cannot compute Fourier energy ratio: energy_before is zero but energy_after is not. "
                f"energy_before={energy_before}, energy_after={energy_after}, band={band}"
            )

        return float(energy_after / energy_before)

    @staticmethod
    def _calculate_hausdorff_dimension(surface: Sequence[Sequence[float]]) -> float:
        """Estimate fractal dimension using a simple box-counting approach."""

        matrix = _coerce_matrix(surface)
        if len(matrix) == 0 or len(matrix[0]) == 0:
            raise ValueError("surface must contain data")

        min_value = min(_flatten(matrix))
        normalised = [[value - min_value for value in row] for row in matrix]

        max_value = max(_flatten(normalised))
        if not math.isclose(max_value, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            normalised = [[value / max_value for value in row] for row in normalised]

        threshold = _median(_flatten(normalised))
        binary = [[value > threshold for value in row] for row in normalised]

        rows = len(binary)
        cols = len(binary[0])
        min_dim = min(rows, cols)
        if min_dim <= 0:
            raise ValueError("surface must contain data")

        max_exponent = int(math.floor(math.log2(min_dim)))
        if max_exponent <= 1:
            return 1.0

        sizes = [2 ** exponent for exponent in range(1, max_exponent)]
        counts = [MaterialResponseValidator._boxcount(binary, size) for size in sizes]

        eps = 1e-9
        xs = [math.log(size + eps) for size in sizes]
        ys = [math.log(count + eps) for count in counts]
        slope, _ = _linear_regression(xs, ys)
        dimension = max(-slope, 0.0)
        return float(dimension)

    @staticmethod
    def _boxcount(binary: Sequence[Sequence[bool]], size: int) -> int:
        """Count non-empty boxes of the given ``size`` for ``binary`` data."""

        if size <= 0:
            raise ValueError("size must be positive")

        rows = len(binary)
        cols = len(binary[0]) if rows else 0
        if rows == 0 or cols == 0:
            return 0

        trimmed_rows = rows - (rows % size)
        trimmed_cols = cols - (cols % size)
        if trimmed_rows == 0 or trimmed_cols == 0:
            return 0

        count = 0
        for row in range(0, trimmed_rows, size):
            for col in range(0, trimmed_cols, size):
                occupied = False
                for dr in range(size):
                    if occupied:
                        break
                    for dc in range(size):
                        if binary[row + dr][col + dc]:
                            occupied = True
                            break
                if occupied:
                    count += 1

        return count

