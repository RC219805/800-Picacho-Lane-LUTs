# path: coastal_estate_render.py
"""Utilities for rendering the Montecito coastal estate hero frame.

This helper wraps :mod:`lux_render_pipeline` with curated defaults from the
Material Response review. The heavy diffusion stack is imported lazily so tests
can stub the pipeline and environments without ML/GPU deps can import safely.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Mapping, Optional


COASTAL_ESTATE_PROMPT: str = (
    "luxury coastal estates Montecito golden hour, Spanish Mediterranean "
    "architecture, terracotta roofs glowing in sunset, emerald manicured lawns, "
    "Pacific Ocean with crystalline water gradients, palm trees casting long shadows, "
    "ultra-high-end real estate photography, architectural digest quality, "
    "material response technology, each surface expressing unique photonic signature"
)
"""Prompt tuned for the Montecito golden hour coastline brief."""

# Managed option keys that this helper is authoritative for.
# (Used only to detect conflicting keys in `extra_options`.)
_MANAGED_OPTION_PARAMS: Mapping[str, str] = {
    "input": "input_image",
    "out": "output_dir",
    "prompt": "prompt",
    "width": "width",
    "height": "height",
    "strength": "strength",
    "gs": "guidance_scale",
    "w4k": "export_4k",
    "use_realesrgan": "use_realesrgan",
    "brand_text": "brand_text",
    "logo": "brand_logo",
    "seed": "seed",
    "neg": "negative_prompt",
}


def _load_pipeline_module() -> ModuleType:
    """Lazily import :mod:`lux_render_pipeline` with a clear error path."""
    try:
        return import_module("lux_render_pipeline")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "lux_render_pipeline is not installed/available. "
            'Install the ML extras (e.g., `pip install -e ".[ml]"`) and ensure '
            "its dependencies (diffusers / torch) are present."
        ) from exc


def _resolve_pipeline_main(mod: ModuleType) -> Callable[..., Any]:
    """Resolve a callable pipeline entrypoint named `main`."""
    fn = getattr(mod, "main", None)
    if not callable(fn):
        raise RuntimeError(
            "lux_render_pipeline.main(...) was not found or is not callable. "
            "Ensure your pipeline exposes a `main(**kwargs)` entrypoint."
        )
    return fn


def _validate_options(
    *,
    width: int,
    height: int,
    strength: float,
    guidance_scale: float,
) -> None:
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive int, got {width!r}")
    if not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive int, got {height!r}")
    if not (0.0 <= float(strength) <= 1.0):
        raise ValueError(f"strength must be in [0.0, 1.0], got {strength!r}")
    if float(guidance_scale) <= 0.0:
        raise ValueError(f"guidance_scale must be > 0, got {guidance_scale!r}")


def render_coastal_estate(
    input_image: str,
    output_dir: str = "./transcended",
    *,
    prompt: str = COASTAL_ESTATE_PROMPT,
    width: int = 1536,
    height: int = 1024,
    strength: float = 0.35,
    guidance_scale: float = 8.0,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    export_4k: bool = False,
    use_realesrgan: bool = False,
    brand_text: Optional[str] = None,
    brand_logo: Optional[str] = None,
    extra_options: Optional[Mapping[str, Any]] = None,
) -> None:
    """Run the coastal estate diffusion render with curated defaults.

    Parameters
    ----------
    input_image:
        Path or glob pattern for the base aerial capture (forwarded to pipeline).
    output_dir:
        Destination directory (pipeline `--out`).
    prompt:
        Positive prompt describing the coastal estate aesthetic.
    width, height:
        Diffusion canvas resolution (pixels).
    strength:
        Deviation from input capture (0..1).
    guidance_scale:
        Classifier-free guidance scale.
    seed:
        Optional deterministic seed.
    negative_prompt:
        Optional negative prompt override.
    export_4k:
        Enable latent 4K upscale pass.
    use_realesrgan:
        Enable Real-ESRGAN for additional polish.
    brand_text, brand_logo:
        Caption and logo forwarded to finishing stage.
    extra_options:
        Extra kwargs forwarded verbatim to the pipeline `main`. Collisions with
        managed options are rejected to keep this helper authoritative.
    """
    _validate_options(
        width=width, height=height, strength=strength, guidance_scale=guidance_scale
    )

    # Build base options using pipeline's expected CLI-style keys.
    options: dict[str, Any] = {
        "input": input_image,
        "out": output_dir,
        "prompt": prompt,
        "width": width,
        "height": height,
        "strength": strength,
        "gs": guidance_scale,
        "w4k": export_4k,
        "use_realesrgan": use_realesrgan,
        "brand_text": brand_text,
        "logo": brand_logo,
        "seed": seed,
        "neg": negative_prompt,
    }
    # Drop None values to avoid surprising **kwargs errors downstream.
    options = {k: v for k, v in options.items() if v is not None}

    if extra_options:
        conflicts = sorted(set(options).intersection(extra_options.keys()))
        if conflicts:
            formatted = ", ".join(f"'{k}'" for k in conflicts)
            raise ValueError(
                f"extra_options contains managed key(s): {formatted}. "
                "Pass only non-managed options; this helper is the source of truth "
                "for curated defaults."
            )
        # Additional guard using the known managed map for future-proofing
        managed_aliases = set(_MANAGED_OPTION_PARAMS.keys())
        alias_conflicts = sorted(managed_aliases.intersection(extra_options.keys()))
        if alias_conflicts:
            formatted = ", ".join(f"'{k}'" for k in alias_conflicts)
            raise ValueError(
                f"extra_options uses reserved alias key(s): {formatted}."
            )
        options.update(extra_options)

    pipeline_module = _load_pipeline_module()
    pipeline_main = _resolve_pipeline_main(pipeline_module)
    pipeline_main(**options)


__all__ = ["COASTAL_ESTATE_PROMPT", "render_coastal_estate"]