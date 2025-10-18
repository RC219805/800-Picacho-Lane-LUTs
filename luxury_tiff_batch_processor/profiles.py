# path: luxury_tiff_batch_processor/profiles.py
"""Processing profile definitions for balancing fidelity and throughput."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class ProcessingProfile:
    """Describe processing trade-offs for a pipeline run.

    - `glow_multiplier` / `chroma_denoise_multiplier`: scale user settings.
    - `target_bit_depth`: output bit depth. None → preserve source dtype;
       8 → uint8, 16 → uint16, 32 → float32 (why: HDR/analysis workflows).
    - `compression`: preferred compression name (overrides caller if set).
    """

    name: str
    glow_multiplier: float = 1.0
    chroma_denoise_multiplier: float = 1.0
    target_bit_depth: Optional[int] = None  # None | 8 | 16 | 32
    compression: Optional[str] = None

    def __post_init__(self) -> None:
        # why: keep invalid configs from slipping into runtime paths
        if self.target_bit_depth is not None and self.target_bit_depth not in (8, 16, 32):
            raise ValueError("target_bit_depth must be one of {8, 16, 32} or None")
        # Clamp multipliers to sane non-negative values
        gm = max(0.0, float(self.glow_multiplier))
        cd = max(0.0, float(self.chroma_denoise_multiplier))
        if gm != self.glow_multiplier or cd != self.chroma_denoise_multiplier:
            object.__setattr__(self, "glow_multiplier", gm)
            object.__setattr__(self, "chroma_denoise_multiplier", cd)

    # --- mapping helpers -----------------------------------------------------

    def resolve_glow(self, value: float) -> float:
        """Return adjusted glow amount."""
        return value * self.glow_multiplier

    def resolve_chroma_denoise(self, value: float) -> float:
        """Return adjusted chroma denoise amount."""
        return value * self.chroma_denoise_multiplier

    def target_dtype(self, source_dtype: np.dtype) -> np.dtype:
        """Return dtype to use for saving results."""
        if self.target_bit_depth is None:
            return np.dtype(source_dtype)
        if self.target_bit_depth == 32:
            return np.dtype(np.float32)
        if self.target_bit_depth >= 16:
            return np.dtype(np.uint16)
        return np.dtype(np.uint8)

    def resolve_compression(self, requested: str) -> str:
        """Return compression honoring the profile preference when set."""
        return self.compression or requested

    # --- misc ----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Lightweight serialization for logs/debugging."""
        return {
            "name": self.name,
            "glow_multiplier": self.glow_multiplier,
            "chroma_denoise_multiplier": self.chroma_denoise_multiplier,
            "target_bit_depth": self.target_bit_depth,
            "compression": self.compression,
        }


DEFAULT_PROFILE_NAME = "quality"


PROCESSING_PROFILES: Dict[str, ProcessingProfile] = {
    "quality": ProcessingProfile(
        name="quality",
        glow_multiplier=1.0,
        chroma_denoise_multiplier=1.0,
        target_bit_depth=None,   # preserve source dtype
        compression=None,
    ),
    "balanced": ProcessingProfile(
        name="balanced",
        glow_multiplier=0.6,
        chroma_denoise_multiplier=0.5,
        target_bit_depth=16,     # prefer 16-bit outputs
        compression=None,
    ),
    "performance": ProcessingProfile(
        name="performance",
        glow_multiplier=0.0,
        chroma_denoise_multiplier=0.0,
        target_bit_depth=8,      # fast uint8 outputs
        compression="tiff_jpeg",
    ),
    # Optional HDR/analysis profile example (kept but unused by default):
    # "analysis": ProcessingProfile(
    #     name="analysis", glow_multiplier=1.0, chroma_denoise_multiplier=1.0,
    #     target_bit_depth=32, compression=None
    # ),
}


def get_profile(name: str) -> ProcessingProfile:
    """Return a profile by name with a clear error on unknown keys."""
    try:
        return PROCESSING_PROFILES[name]
    except KeyError as exc:  # why: surface available options to caller
        available = ", ".join(sorted(PROCESSING_PROFILES))
        raise KeyError(f"Unknown profile {name!r}. Available: {available}") from exc


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "PROCESSING_PROFILES",
    "ProcessingProfile",
]