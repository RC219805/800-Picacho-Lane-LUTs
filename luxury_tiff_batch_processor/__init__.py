# path: luxury_tiff_batch_processor/__init__.py
"""Luxury TIFF batch processing toolkit.

Modules:
- adjustments: color science primitives, presets, and array ops.
- io_utils: capability detection, float conversion, TIFF I/O.
- cli: Typer-based batch CLI wiring.
- pipeline/profiles: optional orchestration & profiles (if present).

Top-level exports are lazily loaded to minimize import time while remaining
visible to type checkers and IDEs.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any
import logging

# Attach a NullHandler by default; applications may configure logging as needed.
LOGGER = logging.getLogger("luxury_tiff_batch_processor")
LOGGER.addHandler(logging.NullHandler())

# Resolve package version from distribution metadata (best-effort).
try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("luxury_tiff_batch_processor")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Public API surface (kept stable for consumers).
__all__ = [
    # adjustments
    "AdjustmentSettings",
    "LUXURY_PRESETS",
    "apply_adjustments",
    "gaussian_blur",
    "gaussian_kernel",
    "gaussian_kernel_cached",
    # cli
    "app",
    "lux_batch",
    "main",
    # io_utils
    "FloatDynamicRange",
    "ImageToFloatResult",
    "LuxuryGradeException",
    "ProcessingCapabilities",
    "ProcessingContext",
    "float_to_dtype_array",
    "image_to_float",
    "save_image",
    # compatibility (legacy argparse-style API)
    "parse_args",
    "run_pipeline",
    # meta
    "__version__",
]

# Map export name → (module path, attribute name)
_EXPORTS = {
    # adjustments
    "AdjustmentSettings": ("luxury_tiff_batch_processor.adjustments", "AdjustmentSettings"),
    "LUXURY_PRESETS": ("luxury_tiff_batch_processor.adjustments", "LUXURY_PRESETS"),
    "apply_adjustments": ("luxury_tiff_batch_processor.adjustments", "apply_adjustments"),
    "gaussian_blur": ("luxury_tiff_batch_processor.adjustments", "gaussian_blur"),
    "gaussian_kernel": ("luxury_tiff_batch_processor.adjustments", "gaussian_kernel"),
    "gaussian_kernel_cached": ("luxury_tiff_batch_processor.adjustments", "gaussian_kernel_cached"),
    # cli (real symbols)
    "app": ("luxury_tiff_batch_processor.cli", "app"),
    "lux_batch": ("luxury_tiff_batch_processor.cli", "lux_batch"),
    "main": ("luxury_tiff_batch_processor.cli", "main"),
    # io_utils
    "FloatDynamicRange": ("luxury_tiff_batch_processor.io_utils", "FloatDynamicRange"),
    "ImageToFloatResult": ("luxury_tiff_batch_processor.io_utils", "ImageToFloatResult"),
    "LuxuryGradeException": ("luxury_tiff_batch_processor.io_utils", "LuxuryGradeException"),
    "ProcessingCapabilities": ("luxury_tiff_batch_processor.io_utils", "ProcessingCapabilities"),
    "ProcessingContext": ("luxury_tiff_batch_processor.io_utils", "ProcessingContext"),
    "float_to_dtype_array": ("luxury_tiff_batch_processor.io_utils", "float_to_dtype_array"),
    "image_to_float": ("luxury_tiff_batch_processor.io_utils", "image_to_float"),
    "save_image": ("luxury_tiff_batch_processor.io_utils", "save_image"),
    # compat (legacy argparse-style API)
    "parse_args": ("luxury_tiff_batch_processor.compat", "parse_args"),
    "run_pipeline": ("luxury_tiff_batch_processor.compat", "run_pipeline"),
    # meta
    "__version__": (__name__, "__version__"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute loader for top-level exports."""
    try:
        module_path, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    """Offer a helpful attribute list in REPL/IDEs."""
    return sorted(set(globals().keys()) | set(__all__))


# Static imports for type checkers only; no runtime side effects.
if TYPE_CHECKING:  # pragma: no cover
    from .adjustments import (  # noqa: F401
        AdjustmentSettings,
        LUXURY_PRESETS,
        apply_adjustments,
        gaussian_blur,
        gaussian_kernel,
        gaussian_kernel_cached,
    )
    from .cli import (  # noqa: F401
        app,
        lux_batch,
        main,
    )
    from .compat import (  # noqa: F401
        parse_args,
        run_pipeline,
    )
    from .io_utils import (  # noqa: F401
        FloatDynamicRange,
        ImageToFloatResult,
        LuxuryGradeException,
        ProcessingCapabilities,
        ProcessingContext,
        float_to_dtype_array,
        image_to_float,
        save_image,
    )
