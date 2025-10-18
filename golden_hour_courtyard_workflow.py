# file: golden_hour_courtyard.py
"""Golden Hour Courtyard helper utilities.

Wraps :mod:`luxury_tiff_batch_processor` with the preset + parameter overrides
requested for the Montecito coastal courtyard aerial. Provides the same CLI
surface while exposing a Python API for notebooks/orchestration.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import luxury_tiff_batch_processor as ltiff


@dataclass(frozen=True)
class _AdjustmentFlag:
    """Mapping between adjustment attribute names and CLI flags."""
    attribute: str
    flag: str


_ADJUSTMENT_FLAGS: Sequence[_AdjustmentFlag] = (
    _AdjustmentFlag("exposure", "--exposure"),
    _AdjustmentFlag("white_balance_temp", "--white-balance-temp"),
    _AdjustmentFlag("white_balance_tint", "--white-balance-tint"),
    _AdjustmentFlag("shadow_lift", "--shadow-lift"),
    _AdjustmentFlag("highlight_recovery", "--highlight-recovery"),
    _AdjustmentFlag("midtone_contrast", "--midtone-contrast"),
    _AdjustmentFlag("vibrance", "--vibrance"),
    _AdjustmentFlag("saturation", "--saturation"),
    _AdjustmentFlag("clarity", "--clarity"),
    _AdjustmentFlag("chroma_denoise", "--chroma-denoise"),
    _AdjustmentFlag("glow", "--luxury-glow"),
)

_DEFAULT_GOLDEN_HOUR_OVERRIDES: Dict[str, float] = {
    "exposure": 0.08,
    "shadow_lift": 0.24,
    "highlight_recovery": 0.18,
    "vibrance": 0.28,
    "clarity": 0.20,
    "glow": 0.12,
    "white_balance_temp": 5600.0,
    "midtone_contrast": 0.10,
}


def _format_value(value: float) -> str:
    """Render numeric overrides for the CLI while preserving precision."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text in ("", "-0", "+0"):  # avoid '-0'
        return "0"
    return text


def _merge_overrides(overrides: Mapping[str, float | None] | None) -> Dict[str, float]:
    merged: MutableMapping[str, float] = dict(_DEFAULT_GOLDEN_HOUR_OVERRIDES)
    if overrides is None:
        return dict(merged)

    valid_attributes = {entry.attribute for entry in _ADJUSTMENT_FLAGS}
    for attribute, value in overrides.items():
        if attribute not in valid_attributes:
            raise ValueError(f"Unknown adjustment override '{attribute}'")
        if value is None:
            merged.pop(attribute, None)
        else:
            f = float(value)
            if not math.isfinite(f):
                raise ValueError(f"Override '{attribute}' must be a finite number")
            merged[attribute] = f

    return dict(merged)


def _validate_resize_long_edge(resize_long_edge: int | None) -> int | None:
    if resize_long_edge is None:
        return None
    try:
        n = int(resize_long_edge)
    except (TypeError, ValueError):
        raise ValueError("resize_long_edge must be an integer") from None
    if n <= 0:
        raise ValueError("resize_long_edge must be > 0")
    return n


def _validate_suffix(suffix: str) -> str:
    if not isinstance(suffix, str) or suffix == "":
        raise ValueError("suffix must be a non-empty string")
    return suffix


def _build_cli_vector(
    input_dir: Path | str,
    output_dir: Path | str | None,
    *,
    recursive: bool,
    overwrite: bool,
    dry_run: bool,
    suffix: str,
    compression: str,
    resize_long_edge: int | None,
    log_level: str,
    overrides: Mapping[str, float | None] | None,
) -> list[str]:
    vector: list[str] = [str(input_dir)]
    if output_dir is not None:
        vector.append(str(output_dir))

    vector.extend(["--preset", "golden_hour_courtyard"])

    if recursive:
        vector.append("--recursive")
    if overwrite:
        vector.append("--overwrite")
    if dry_run:
        vector.append("--dry-run")

    _validate_suffix(suffix)
    if suffix != "_lux":
        vector.extend(["--suffix", suffix])

    if compression != "tiff_lzw":
        vector.extend(["--compression", compression])

    n = _validate_resize_long_edge(resize_long_edge)
    if n is not None:
        vector.extend(["--resize-long-edge", str(n)])

    lvl = (log_level or "").upper()
    if lvl != "INFO":
        vector.extend(["--log-level", lvl])

    merged_overrides = _merge_overrides(overrides)
    flag_lookup = {entry.attribute: entry.flag for entry in _ADJUSTMENT_FLAGS}

    # Deterministic: emit in the order defined by _ADJUSTMENT_FLAGS.
    for entry in _ADJUSTMENT_FLAGS:
        if entry.attribute in merged_overrides:
            vector.extend([flag_lookup[entry.attribute], _format_value(merged_overrides[entry.attribute])])

    return vector


def format_cli(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    **kwargs,
) -> str:
    """Return a shell-friendly CLI string for preview/debugging."""
    args = _build_cli_vector(
        input_dir,
        output_dir,
        recursive=bool(kwargs.get("recursive", False)),
        overwrite=bool(kwargs.get("overwrite", False)),
        dry_run=bool(kwargs.get("dry_run", False)),
        suffix=str(kwargs.get("suffix", "_lux")),
        compression=str(kwargs.get("compression", "tiff_lzw")),
        resize_long_edge=kwargs.get("resize_long_edge", None),
        log_level=str(kwargs.get("log_level", "INFO")),
        overrides=kwargs.get("overrides", None),
    )
    return " ".join(shlex.quote(a) for a in args)


def process_courtyard_scene(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    recursive: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    suffix: str = "_lux",
    compression: str = "tiff_lzw",
    resize_long_edge: int | None = None,
    log_level: str = "INFO",
    overrides: Mapping[str, float | None] | None = None,
) -> int:
    """Process a folder of TIFFs using the Golden Hour Courtyard recipe.

    Parameters
    ----------
    input_dir:
        Directory that contains the source TIFF files.
    output_dir:
        Destination directory. When ``None`` writes to ``<input>_lux``.
    recursive:
        Mirror the input folder tree recursively when ``True``.
    overwrite:
        Permit overwriting of existing files in the destination.
    dry_run:
        Print the proposed work without writing files when ``True``.
    suffix:
        Filename suffix appended before the extension for processed files.
    compression:
        TIFF compression passed through to Pillow/tifffile.
    resize_long_edge:
        Optional long-edge clamp applied before grading (pixels, > 0).
    log_level:
        Logging verbosity understood by :mod:`luxury_tiff_batch_processor`.
    overrides:
        Mapping of adjustment attribute names to override values; ``None``
        removes that override and uses the preset default.

    Returns
    -------
    int
        Number of successfully processed images as reported by
        :func:`luxury_tiff_batch_processor.run_pipeline`.
    """
    cli_vector = _build_cli_vector(
        input_dir,
        output_dir,
        recursive=recursive,
        overwrite=overwrite,
        dry_run=dry_run,
        suffix=suffix,
        compression=compression,
        resize_long_edge=resize_long_edge,
        log_level=log_level,
        overrides=overrides,
    )

    capabilities = ltiff.ProcessingCapabilities()
    capabilities.assert_luxury_grade()

    args = ltiff.parse_args(cli_vector)
    return ltiff.run_pipeline(args)


__all__ = ["process_courtyard_scene", "format_cli"]


# --------------------------- tests (pytest) ---------------------------

# file: test_golden_hour_courtyard.py
def test_format_value_trims_and_normalizes_zero():
    from golden_hour_courtyard import _format_value
    assert _format_value(0.0) == "0"
    assert _format_value(-0.0) == "0"
    assert _format_value(1.2345678) == "1.234568"
    assert _format_value(1.200000) == "1.2"

def test_merge_overrides_unknown_and_nonfinite():
    import math, pytest
    from golden_hour_courtyard import _merge_overrides
    with pytest.raises(ValueError):
        _merge_overrides({"nope": 1.0})
    with pytest.raises(ValueError):
        _merge_overrides({"exposure": math.inf})
    with pytest.raises(ValueError):
        _merge_overrides({"exposure": math.nan})

def test_merge_overrides_none_removes_default():
    from golden_hour_courtyard import _merge_overrides, _DEFAULT_GOLDEN_HOUR_OVERRIDES
    m = _merge_overrides({"clarity": None})
    assert "clarity" not in m
    # others remain
    assert "exposure" in m and m["exposure"] == _DEFAULT_GOLDEN_HOUR_OVERRIDES["exposure"]

def test_build_cli_vector_deterministic_order():
    from golden_hour_courtyard import _build_cli_vector
    # Intentionally scramble keys
    overrides = {"glow": 0.5, "exposure": 0.1, "white_balance_temp": 5500.0}
    args = _build_cli_vector(
        "in", None, recursive=False, overwrite=False, dry_run=False,
        suffix="_lux", compression="tiff_lzw", resize_long_edge=None,
        log_level="INFO", overrides=overrides,
    )
    # Expect flags appear in the fixed order defined by _ADJUSTMENT_FLAGS
    glow_idx = args.index("--luxury-glow")
    exp_idx  = args.index("--exposure")
    wb_idx   = args.index("--white-balance-temp")
    assert exp_idx < wb_idx < glow_idx or exp_idx < glow_idx < wb_idx  # exposure first

def test_validate_resize_and_suffix_and_cli_preview():
    import pytest
    from golden_hour_courtyard import format_cli
    with pytest.raises(ValueError):
        # invalid: empty suffix
        format_cli("in", suffix="")
    with pytest.raises(ValueError):
        # invalid: non-positive resize
        format_cli("in", resize_long_edge=0)
    # Quoted preview for spaces
    s = format_cli("in dir", output_dir="out dir", overrides={"clarity": 0.5})
    assert "'in dir'" in s and "'out dir'" in s