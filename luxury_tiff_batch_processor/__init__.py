# path: luxury_tiff_batch_processor/__init__.py
"""Luxury TIFF batch processing toolkit.

The package is organised into focused modules:

- :mod:`luxury_tiff_batch_processor.adjustments` encapsulates the color science
  primitives, presets, and mathematical helpers for manipulating image arrays.
- :mod:`luxury_tiff_batch_processor.io_utils` hosts capability detection,
  floating-point conversion helpers, and low-level TIFF I/O routines.
- :mod:`luxury_tiff_batch_processor.cli` wires these pieces together for the
  command-line interface while remaining lightweight for library consumers.
- :mod:`luxury_tiff_batch_processor.pipeline` provides orchestration helpers
  used by both the CLI and programmatic integrations.

This package exposes a flat, convenient API at the top level. Symbols are
**lazily imported** to minimise import time and avoid circular dependencies,
while remaining fully visible to type checkers and IDEs.
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
    # Optional backport; only used on very old Pythons if present.
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("luxury_tiff_batch_processor")
except PackageNotFoundError:  # Local checkout / not installed as a dist
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
    "build_adjustments",
    "default_output_folder",
    "main",
    "parse_args",
    "run_pipeline",
    # io_utils
    "FloatDynamicRange",
    "ImageToFloatResult",
    "LuxuryGradeException",
    "ProcessingCapabilities",
    "ProcessingContext",
    "float_to_dtype_array",
    "image_to_float",
    "save_image",
    # pipeline
    "_PROGRESS_WRAPPER",
    "_wrap_with_progress",
    "collect_images",
    "ensure_output_path",
    "process_image",
    "process_single_image",
    # profiles
    "DEFAULT_PROFILE_NAME",
    "PROCESSING_PROFILES",
    "ProcessingProfile",
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
    # cli
    "build_adjustments": ("luxury_tiff_batch_processor.cli", "build_adjustments"),
    "default_output_folder": ("luxury_tiff_batch_processor.cli", "default_output_folder"),
    "main": ("luxury_tiff_batch_processor.cli", "main"),
    "parse_args": ("luxury_tiff_batch_processor.cli", "parse_args"),
    "run_pipeline": ("luxury_tiff_batch_processor.cli", "run_pipeline"),
    # io_utils
    "FloatDynamicRange": ("luxury_tiff_batch_processor.io_utils", "FloatDynamicRange"),
    "ImageToFloatResult": ("luxury_tiff_batch_processor.io_utils", "ImageToFloatResult"),
    "LuxuryGradeException": ("luxury_tiff_batch_processor.io_utils", "LuxuryGradeException"),
    "ProcessingCapabilities": ("luxury_tiff_batch_processor.io_utils", "ProcessingCapabilities"),
    "ProcessingContext": ("luxury_tiff_batch_processor.io_utils", "ProcessingContext"),
    "float_to_dtype_array": ("luxury_tiff_batch_processor.io_utils", "float_to_dtype_array"),
    "image_to_float": ("luxury_tiff_batch_processor.io_utils", "image_to_float"),
    "save_image": ("luxury_tiff_batch_processor.io_utils", "save_image"),
    # pipeline
    "_PROGRESS_WRAPPER": ("luxury_tiff_batch_processor.pipeline", "_PROGRESS_WRAPPER"),
    "_wrap_with_progress": ("luxury_tiff_batch_processor.pipeline", "_wrap_with_progress"),
    "collect_images": ("luxury_tiff_batch_processor.pipeline", "collect_images"),
    "ensure_output_path": ("luxury_tiff_batch_processor.pipeline", "ensure_output_path"),
    "process_image": ("luxury_tiff_batch_processor.pipeline", "process_image"),
    "process_single_image": ("luxury_tiff_batch_processor.pipeline", "process_single_image"),
    # profiles
    "DEFAULT_PROFILE_NAME": ("luxury_tiff_batch_processor.profiles", "DEFAULT_PROFILE_NAME"),
    "PROCESSING_PROFILES": ("luxury_tiff_batch_processor.profiles", "PROCESSING_PROFILES"),
    "ProcessingProfile": ("luxury_tiff_batch_processor.profiles", "ProcessingProfile"),
    # meta
    "__version__": (__name__, "__version__"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute loader for top-level exports."""
    try:
        module_path, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_path)  # Heavy modules loaded on demand.
    value = getattr(module, attr)

    # Cache the resolved attribute on this module for faster subsequent access.
    globals()[name] = value
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
        build_adjustments,
        default_output_folder,
        main,
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
    from .pipeline import (  # noqa: F401
        _PROGRESS_WRAPPER,
        _wrap_with_progress,
        collect_images,
        ensure_output_path,
        process_image,
        process_single_image,
    )
    from .profiles import (  # noqa: F401
        DEFAULT_PROFILE_NAME,
        PROCESSING_PROFILES,
        ProcessingProfile,
    )