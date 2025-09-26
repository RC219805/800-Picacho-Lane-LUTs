codex/create-batch-script-for-tiff-image-editing
"""Luxury TIFF batch enhancer.

This script performs batch processing on high-resolution TIFF images to produce
polished deliverables suitable for ultra-luxury digital marketing campaigns.

The pipeline preserves bit depth where possible, keeps metadata, and applies
selectable presets that can be fine-tuned per project. Adjustments include
white balance refinement, tonal sculpting, vibrance, clarity, chroma denoising,
and an optional diffusion glow for an elevated aesthetic.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

try:  # Optional high-fidelity TIFF writer
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tifffile = None


LOGGER = logging.getLogger("luxury_tiff_batch_processor")


@dataclasses.dataclass
class AdjustmentSettings:
    """Holds the image adjustment parameters for a processing run."""

    exposure: float = 0.0
    white_balance_temp: Optional[float] = None
    white_balance_tint: float = 0.0
    shadow_lift: float = 0.0
    highlight_recovery: float = 0.0
    midtone_contrast: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    clarity: float = 0.0
    chroma_denoise: float = 0.0
    glow: float = 0.0


# Signature looks tailored to the 800 Picacho Lane collection.
LUXURY_PRESETS = {
    "signature": AdjustmentSettings(
        exposure=0.12,
        white_balance_temp=6500,
        white_balance_tint=4.0,
        shadow_lift=0.18,
        highlight_recovery=0.15,
        midtone_contrast=0.08,
        vibrance=0.18,
        saturation=0.06,
        clarity=0.16,
        chroma_denoise=0.08,
        glow=0.05,
    ),
    "architectural": AdjustmentSettings(
        exposure=0.08,
        white_balance_temp=6200,
        white_balance_tint=2.0,
        shadow_lift=0.12,
        highlight_recovery=0.1,
        midtone_contrast=0.05,
        vibrance=0.12,
        saturation=0.04,
        clarity=0.22,
        chroma_denoise=0.05,
        glow=0.02,
    ),
    "twilight": AdjustmentSettings(
        exposure=0.05,
        white_balance_temp=5400,
        white_balance_tint=8.0,
        shadow_lift=0.24,
        highlight_recovery=0.18,
        midtone_contrast=0.1,
        vibrance=0.24,
        saturation=0.08,
        clarity=0.12,
        chroma_denoise=0.1,
        glow=0.12,
    ),
}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch enhance TIFF files for ultra-luxury marketing output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Folder that contains source TIFF files")
    parser.add_argument("output", type=Path, help="Folder where processed files will be written")
    parser.add_argument(
        "--preset",
        default="signature",
        choices=sorted(LUXURY_PRESETS.keys()),
        help="Adjustment preset that provides a starting point",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process folders recursively and mirror the directory tree in the output",
    )
    parser.add_argument(
        "--suffix",
        default="_lux",
        help="Filename suffix appended before the extension for processed files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the destination",
    )
    parser.add_argument(
        "--compression",
        default="tiff_lzw",
        help="TIFF compression to use when saving (as understood by Pillow)",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=None,
        help="Optionally resize the longest image edge to this many pixels while preserving aspect ratio",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview the work without writing any files")

    # Fine control overrides.
    parser.add_argument("--exposure", type=float, default=None, help="Exposure adjustment in stops")
    parser.add_argument(
        "--white-balance-temp",
        type=float,
        default=None,
        dest="white_balance_temp",
        help="Target color temperature in Kelvin",
    )
    parser.add_argument(
        "--white-balance-tint",
        type=float,
        default=None,
        dest="white_balance_tint",
        help="Green-magenta tint compensation (positive skews magenta)",
    )
    parser.add_argument("--shadow-lift", type=float, default=None, help="Shadow recovery strength (0-1)")
    parser.add_argument(
        "--highlight-recovery",
        type=float,
        default=None,
        help="Highlight compression strength (0-1)",
    )
    parser.add_argument(
        "--midtone-contrast", type=float, default=None, dest="midtone_contrast", help="Midtone contrast strength"
    )
    parser.add_argument("--vibrance", type=float, default=None, help="Vibrance strength (0-1)")
    parser.add_argument("--saturation", type=float, default=None, help="Additional saturation multiplier delta")
    parser.add_argument("--clarity", type=float, default=None, help="Local contrast boost strength (0-1)")
    parser.add_argument(
        "--chroma-denoise",
        type=float,
        default=None,
        dest="chroma_denoise",
        help="Chrominance denoising amount (0-1)",
    )
    parser.add_argument(
        "--luxury-glow", type=float, default=None, dest="glow", help="Diffusion glow strength (0-1)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    return args


def build_adjustments(args: argparse.Namespace) -> AdjustmentSettings:
    base = dataclasses.replace(LUXURY_PRESETS[args.preset])
    for field in dataclasses.fields(AdjustmentSettings):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(base, field.name, value)
    LOGGER.debug("Using adjustments: %s", base)
    return base


def collect_images(folder: Path, recursive: bool) -> Iterator[Path]:
    patterns: List[str] = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    if recursive:
        for pattern in patterns:
            yield from folder.rglob(pattern)
    else:
        for pattern in patterns:
            yield from folder.glob(pattern)


def ensure_output_path(input_root: Path, output_root: Path, source: Path, suffix: str, recursive: bool) -> Path:
    relative = source.relative_to(input_root) if recursive else Path(source.name)
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    new_name = destination.stem + suffix + destination.suffix
    return destination.with_name(new_name)


def image_to_float(image: Image.Image) -> Tuple[np.ndarray, np.dtype, Optional[np.ndarray]]:
    """Converts a PIL image to an RGB float32 array in the 0-1 range."""

    if image.mode not in {"RGB", "RGBA", "I;16", "I;16L", "I;16B", "I;16S", "L"}:
        image = image.convert("RGBA" if "A" in image.mode else "RGB")

    arr = np.array(image)
    alpha_channel: Optional[np.ndarray] = None

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        alpha_channel = arr[:, :, 3]
        arr = arr[:, :, :3]

    dtype_max = float(np.iinfo(arr.dtype).max) if arr.dtype.kind in {"u", "i"} else 1.0
    arr_float = arr.astype(np.float32) / dtype_max

    if alpha_channel is not None:
        alpha_channel = alpha_channel.astype(np.float32) / dtype_max

    return arr_float, arr.dtype, alpha_channel


main
def float_to_dtype_array(
    arr: np.ndarray,
    dtype: np.dtype,
    alpha: Optional[np.ndarray],
) -> np.ndarray:
    """Convert a float array back to the original image dtype without bias."""

    arr = np.clip(arr, 0.0, 1.0)
    np_dtype = np.dtype(dtype)
codex/create-batch-script-for-tiff-image-editing
    dtype_info = np.iinfo(np_dtype) if np.issubdtype(np_dtype, np.integer) else None
    if dtype_info:
        dtype_max = float(dtype_info.max)
        arr_int = np.round(arr * dtype_max).astype(np_dtype)
    else:
        # Preserve floating-point sample formats (e.g. 32-bit float TIFF)
        arr_int = arr.astype(np_dtype, copy=False)

    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
        if dtype_info:
            alpha_int = np.round(alpha * dtype_max).astype(np_dtype)
        else:
            alpha_int = alpha.astype(np_dtype, copy=False)
        arr_int = np.concatenate([arr_int, alpha_int[:, :, None]], axis=2)

    return np.ascontiguousarray(arr_int)


def compression_for_tifffile(compression: str) -> Optional[str]:
    comp = compression.lower()
    mapping = {
        "tiff_lzw": "lzw",
        "lzw": "lzw",
        "tiff_adobe_deflate": "deflate",
        "adobe_deflate": "deflate",
        "deflate": "deflate",
        "tiff_zip": "deflate",
        "zip": "deflate",
        "tiff_jpeg": "jpeg",
        "jpeg": "jpeg",
        "tiff_none": None,
        "none": None,
        "raw": None,
    }
    return mapping.get(comp, comp)


SAFE_TIFF_TAGS = {
    270,  # ImageDescription
    271,  # Make
    272,  # Model
    274,  # Orientation
    282,  # XResolution
    283,  # YResolution
    296,  # ResolutionUnit
    305,  # Software
    306,  # DateTime
    315,  # Artist
    316,  # HostComputer
}


def sanitize_tiff_metadata(raw_metadata: Optional[Any]) -> Optional[Dict[int, Any]]:
    if raw_metadata is None:
        return None
    safe: Dict[int, Any] = {}
    try:
        for tag in raw_metadata:
            if tag in SAFE_TIFF_TAGS:
                safe[tag] = raw_metadata[tag]
    except Exception:  # pragma: no cover - metadata best effort
        LOGGER.debug("Unable to sanitise TIFF metadata", exc_info=True)
        return None
    return safe or None


def save_image(
    destination: Path,
    arr_int: np.ndarray,
    dtype: np.dtype,
    metadata,
    icc_profile: Optional[bytes],
    compression: str,
) -> None:
    metadata = sanitize_tiff_metadata(metadata)
    dtype_info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    bits = dtype_info.bits if dtype_info else 0
main

    if np.issubdtype(np_dtype, np.integer):
        dtype_info = np.iinfo(np_dtype)
        dtype_max = float(dtype_info.max)
        arr_out = np.rint(arr * dtype_max).astype(np_dtype)
        if alpha is not None:
            alpha_out = np.rint(np.clip(alpha, 0.0, 1.0) * dtype_max).astype(np_dtype)
            alpha_out = alpha_out[:, :, None]
            arr_out = np.concatenate([arr_out, alpha_out], axis=2)
    elif np.issubdtype(np_dtype, np.floating):
        arr_out = arr.astype(np_dtype, copy=False)
        if alpha is not None:
            alpha_out = np.clip(alpha, 0.0, 1.0).astype(np_dtype, copy=False)
            alpha_out = alpha_out[:, :, None]
            arr_out = np.concatenate([arr_out, alpha_out], axis=2)
    else:
        # For uncommon dtypes, fall back to float32 to avoid surprises.
        arr_out = arr.astype(np.float32)
        if alpha is not None:
            alpha_out = np.clip(alpha, 0.0, 1.0).astype(np.float32)
            alpha_out = alpha_out[:, :, None]
            arr_out = np.concatenate([arr_out, alpha_out], axis=2)

    return np.ascontiguousarray(arr_out)