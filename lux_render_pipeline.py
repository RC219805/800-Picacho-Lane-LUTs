#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Luxury Real Estate Render Refinement Pipeline (CI-friendly).

This module exposes lightweight, deterministic utilities that can be imported
in tests without optional heavy dependencies. CLI features (via `typer`) are
available only when executed as a script.

Exports:
- apply_material_response_finishing(image, **kwargs) -> PIL.Image.Image
"""

from __future__ import annotations

# ---------- Optional CLI dependency (guarded so tests can import the module) ----------
try:  # keep tests importable without typer
    import typer  # type: ignore
except Exception:  # pragma: no cover

    class _TyperShim:
        def __getattr__(self, _):
            raise RuntimeError(
                "CLI features require the optional dependency 'typer'. "
                "Library functions can be imported without it."
            )

    typer = _TyperShim()  # type: ignore
# -------------------------------------------------------------------------------------

# --- stdlib ---
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

# --- third-party (kept lean) ---
import numpy as np
from PIL import Image, ImageEnhance

__all__ = [
    "apply_material_response_finishing",
]


def _to_numpy_rgb(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Convert PIL.Image or ndarray to float32 RGB array in [0,1]."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[-1] != 3:
            raise ValueError("Expected 3 channels (RGB).")
    return np.clip(arr, 0.0, 1.0)


def _from_numpy_rgb(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    arr8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr8, mode="RGB").convert(mode)


def apply_material_response_finishing(
    image: Union[Image.Image, np.ndarray, str, Path],
    *,
    exposure: float = 0.0,  # stops; + means brighter
    contrast: float = 1.0,  # 1.0 no change
    saturation: float = 1.0,  # 1.0 no change
    clamp_low: float = 0.0,  # 0..1
    clamp_high: float = 1.0,  # 0..1
    out_mode: str = "RGB",
) -> Image.Image:
    """
    Deterministic, CPU-only finishing pass suitable for CI:
    - exposure adjustment (gamma-like via multiply in linear-ish space)
    - contrast & saturation via PIL enhancers
    - hard clamp to [clamp_low, clamp_high]
    """
    # Load if path given
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = _from_numpy_rgb(_to_numpy_rgb(image))

    # Exposure (approximate linear multiply)
    arr = _to_numpy_rgb(pil)
    mul = float(2.0**exposure)
    arr = np.clip(arr * mul, 0.0, 1.0)

    # Clamp (in normalized space)
    lo = float(np.clip(clamp_low, 0.0, 1.0))
    hi = float(np.clip(clamp_high, 0.0, 1.0))
    if hi < lo:
        hi = lo
    arr = np.clip((arr - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)

    pil = _from_numpy_rgb(arr)

    # Contrast & saturation via PIL (deterministic)
    if contrast != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(float(contrast))
    if saturation != 1.0:
        pil = ImageEnhance.Color(pil).enhance(float(saturation))

    return pil.convert(out_mode)


# --------------------------
# CLI (only built when executed as a script)
# --------------------------


def _build_cli_app():
    # Local import ensures `typer` is only required when using the CLI
    import typer as _typer

    app = _typer.Typer(add_completion=False)

    @app.command("finish")
    def finish_cmd(
        input_path: Path,
        output_path: Path,
        exposure: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        clamp_low: float = 0.0,
        clamp_high: float = 1.0,
        out_mode: str = "RGB",
    ) -> None:
        """
        Apply a deterministic finishing pass and write the result.
        """
        out = apply_material_response_finishing(
            input_path,
            exposure=exposure,
            contrast=contrast,
            saturation=saturation,
            clamp_low=clamp_low,
            clamp_high=clamp_high,
            out_mode=out_mode,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path)

    return app


if __name__ == "__main__":  # pragma: no cover
    _build_cli_app()()
