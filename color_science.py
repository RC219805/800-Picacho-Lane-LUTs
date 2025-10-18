# path: color_space_contract.py
"""Utilities for reasoning about color space metadata coherence.

Lightweight representations of color spaces and a ``ColorSpaceContract`` that
assesses whether pixel data and tagged metadata align. Designed to be simple,
testable, and FFprobe-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

# Common aliases across tools (FFmpeg/ffprobe, Resolve, camera tags).
_COLOR_SPACE_ALIASES = {
    # Color space names
    "rec709": "bt709",
    "rec.709": "bt709",
    "bt.709": "bt709",
    "srgb": "bt709",
    "rec2020": "bt2020",
    "rec.2020": "bt2020",
    "bt.2020": "bt2020",
    "bt2020ncl": "bt2020",
    "bt.2020nc": "bt2020",
    "bt2020nc": "bt2020",
    "dcip3": "dci-p3",
    "dci-p3": "dci-p3",
    "acescg": "acescg",
    # Primaries (token layer often mirrors name)
    "bt709": "bt709",
    "bt2020": "bt2020",
    "dci-p3-d65": "dci-p3",
    # Transfer functions
    "iec61966-2-1": "srgb",
    "gamma22": "gamma2.2",
    "gamma2.2": "gamma2.2",
    "gamma28": "gamma2.8",
    "gamma2.8": "gamma2.8",
    "linear": "linear",
    "pq": "smpte2084",
    "st2084": "smpte2084",
    "smpte2084": "smpte2084",
    "hlg": "arib-std-b67",
    "arib-std-b67": "arib-std-b67",
    # Matrices
    "bt601": "bt601",
    "smpte170m": "bt601",
    "bt709-matrix": "bt709",
    "bt.709-matrix": "bt709",
    "bt2020nc-matrix": "bt2020",
}

def _normalise_token(value: Optional[str]) -> Optional[str]:
    """Lowercase + alias map; ``None``/empty → ``None``."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return _COLOR_SPACE_ALIASES.get(cleaned, cleaned)

def _normalise_tuple(values: Optional[Iterable[str]]) -> Tuple[str, ...] | None:
    """Normalise a sequence of tokens; strings are treated as single tokens."""
    if values is None:
        return None
    if isinstance(values, str):
        values = (values,)
    normalised = tuple(
        t for t in (_normalise_token(v) for v in values) if t is not None
    )
    return normalised or None


@dataclass(frozen=True)
class ColorSpace:
    """Minimal description of a color space for compatibility checks.

    Attributes align with ffprobe tags: *primaries*, *transfer*, *matrix*.
    """

    name: Optional[str]
    primaries: Optional[str]
    transfer_function: Optional[str]
    matrix_coefficients: Optional[str]

    @staticmethod
    def from_tokens(
        *,
        name: Optional[str] = None,
        primaries: Optional[str] = None,
        transfer: Optional[str] = None,
        matrix: Optional[str] = None,
    ) -> "ColorSpace":
        """Create from loose string tokens."""
        return ColorSpace(
            name=_normalise_token(name),
            primaries=_normalise_token(primaries),
            transfer_function=_normalise_token(transfer),
            matrix_coefficients=_normalise_token(matrix),
        )

    @staticmethod
    def from_ffprobe_tags(tags: Mapping[str, str]) -> "ColorSpace":
        """Create from ffprobe dict (e.g., ``stream`` fields)."""
        return ColorSpace.from_tokens(
            name=_normalise_token(tags.get("color_space") or tags.get("space")),
            primaries=_normalise_token(tags.get("color_primaries") or tags.get("primaries")),
            transfer=_normalise_token(tags.get("color_transfer") or tags.get("transfer_characteristics")),
            matrix=_normalise_token(tags.get("color_space_matrix") or tags.get("matrix_coefficients")),
        )

    def compatible_with(self, other: "ColorSpace") -> bool:
        """True if spaces are compatible by pragmatic heuristics."""
        self_name = _normalise_token(self.name)
        other_name = _normalise_token(other.name)
        if self_name and other_name and self_name == other_name:
            return True  # name match dominates

        # Compare individual tokens; None acts as wildcard
        for a, b in (
            (_normalise_token(self.primaries), _normalise_token(other.primaries)),
            (_normalise_token(self.transfer_function), _normalise_token(other.transfer_function)),
            (_normalise_token(self.matrix_coefficients), _normalise_token(other.matrix_coefficients)),
        ):
            if a is not None and b is not None and a != b:
                return False
        return True

    def mismatch_reasons(self, other: "ColorSpace") -> Tuple[str, ...]:
        """Explain incompatibility; empty tuple means compatible."""
        reasons: list[str] = []
        self_name = _normalise_token(self.name)
        other_name = _normalise_token(other.name)
        # If names exist and mismatch, still check granular attributes for better diagnostics.
        pairs = (
            ("primaries", _normalise_token(self.primaries), _normalise_token(other.primaries)),
            ("transfer", _normalise_token(self.transfer_function), _normalise_token(other.transfer_function)),
            ("matrix", _normalise_token(self.matrix_coefficients), _normalise_token(other.matrix_coefficients)),
        )
        for label, a, b in pairs:
            if a is not None and b is not None and a != b:
                reasons.append(f"{label} mismatch: {a} vs {b}")
        if not reasons and self_name and other_name and self_name != other_name:
            reasons.append(f"name mismatch: {self_name} vs {other_name}")
        return tuple(reasons)


@dataclass(frozen=True)
class ColorSpaceContract:
    """Expectation that pixel content matches its metadata tags."""

    content_space: ColorSpace
    tagged_space: ColorSpace
    confidence: float

    def validate_coherence(self, *, minimum_confidence: float = 0.75) -> bool:
        """True when confidence is high enough and spaces are compatible."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.confidence < minimum_confidence:
            return False
        return self.content_space.compatible_with(self.tagged_space)

    def coherence_report(self, *, minimum_confidence: float = 0.75) -> dict:
        """Machine-readable report with reasons when not coherent."""
        ok = self.validate_coherence(minimum_confidence=minimum_confidence)
        report = {
            "coherent": ok,
            "confidence": self.confidence,
            "minimum_confidence": minimum_confidence,
            "reasons": [],
        }
        if not ok and self.confidence >= minimum_confidence:
            report["reasons"] = list(self.content_space.mismatch_reasons(self.tagged_space))
        return report


__all__ = [
    "ColorSpace",
    "ColorSpaceContract",
]