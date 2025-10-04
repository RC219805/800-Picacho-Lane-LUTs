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
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Iterable, Iterator, List, Optional, Tuple

Any = _Any
Dict = _Dict

import numpy as np
from PIL import Image, TiffImagePlugin

try:  # Optional high-fidelity TIFF writer
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tifffile = None


class LuxuryGradeException(RuntimeError):
    """Raised when the processing environment cannot meet luxury standards."""


class ProcessingCapabilities:
    """Introspects optional dependencies to describe processing fidelity."""

    def __init__(self, tifffile_module: Any | None = None) -> None:
        """Initialise capability detection.

        Parameters
        ----------
        tifffile_module:
            Optional dependency override primarily used by tests.  When ``None``
            the globally imported :mod:`tifffile` module is consulted.
        """

        self._tifffile = tifffile_module if tifffile_module is not None else tifffile
        self.bit_depth = 16 if self._supports_16_bit_output() else 8
        self.hdr_capable = self._detect_hdr_support()

    def _supports_16_bit_output(self) -> bool:
        """Return ``True`` when a writer capable of 16-bit output is available."""

        return bool(getattr(self._tifffile, "imwrite", None))

    def _detect_hdr_support(self) -> bool:
        """Best-effort check for HDR support given optional dependencies."""

        if not self._supports_16_bit_output():
            return False

        supports_hdr = getattr(self._tifffile, "supports_hdr", True)
        try:
            return bool(supports_hdr)
        except Exception:  # pragma: no cover - defensive fallback
            return False

    def assert_luxury_grade(self) -> None:
        """
        Validates that the processing environment meets luxury-grade requirements:
        - 16-bit precision
        - HDR capability
        Raises LuxuryGradeException if requirements are not met.
        """
        if self.bit_depth < 16:
            raise LuxuryGradeException(
                "Material Response requires 16-bit precision. "
                "Install tifffile to unlock quantum color depth."
            )
        if not self.hdr_capable:
            raise LuxuryGradeException(
                "Luxury-grade processing requires HDR capability. "
                "Install tifffile or ensure your environment supports HDR."
            )


LOGGER = logging.getLogger("luxury_tiff_batch_processor")

RESAMPLING_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


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
    "golden_hour_courtyard": AdjustmentSettings(
        exposure=0.08,
        white_balance_temp=5600,
        white_balance_tint=5.0,
        shadow_lift=0.24,
        highlight_recovery=0.18,
        midtone_contrast=0.10,
        vibrance=0.28,
        saturation=0.05,
        clarity=0.20,
        chroma_denoise=0.06,
        glow=0.12,
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
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Folder where processed files will be written. Defaults to '<input>_lux' next to the input folder.",
    )
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
    if args.output is None:
        args.output = default_output_folder(args.input)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    return args


def default_output_folder(input_folder: Path) -> Path:
    """Return the default output folder for a given input directory."""

    # input_folder is already a Path object
    if input_folder.name:
        return input_folder.parent / f"{input_folder.name}_lux"
    return input_folder / "luxury_output"


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


def _normalise_to_unit_range(array: np.ndarray) -> np.ndarray:
    """Normalise an array to the ``[0, 1]`` range as ``float32``."""

    if array.dtype == np.bool_:
        return array.astype(np.float32)

    if np.issubdtype(array.dtype, np.integer):
        dtype_info = np.iinfo(array.dtype)
        scale = float(dtype_info.max - dtype_info.min)
        if scale == 0:
            return np.zeros(array.shape, dtype=np.float32)
        array32 = array.astype(np.float32)
        normalised = (array32 - dtype_info.min) / scale
        return np.clip(normalised, 0.0, 1.0)

    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array.astype(np.float32), 0.0, 1.0)

    raise TypeError(f"Unsupported dtype for normalisation: {array.dtype}")


def _extrasamples_for_channels(num_channels: int) -> Optional[Iterable[str]]:
    if num_channels in (2, 4):
        return ["UNASSALPHA"]
    if num_channels > 4:
        return ["UNASSALPHA"] * (num_channels - 3)
    return None


def _prepare_tifffile_image(array: np.ndarray) -> Tuple[np.ndarray, str, Optional[Iterable[str]]]:
    if array.ndim == 2:
        return array, "minisblack", None
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            return array[:, :, 0], "minisblack", None
        return array, "rgb", _extrasamples_for_channels(channels)
    raise ValueError("Unexpected array shape for image data")


def _split_alpha_channel(array: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if array.ndim != 3:
        return array, None
    channels = array.shape[2]
    if channels == 2:
        return array[:, :, :1], array[:, :, 1]
    if channels >= 4:
        return array[:, :, :3], array[:, :, 3]
    return array, None


def _write_with_tifffile(
    destination: Path,
    array: np.ndarray,
    metadata: Optional[Dict[int, Any]],
    icc_profile: Optional[bytes],
    compression: str,
) -> None:
    array_for_write, photometric, extrasamples = _prepare_tifffile_image(array)
    tiff_kwargs: Dict[str, Any] = {
        "photometric": photometric,
        "compression": compression_for_tifffile(compression),
        "metadata": None,
    }

    if extrasamples:
        tiff_kwargs["extrasamples"] = extrasamples

    extratags = []
    if icc_profile:
        extratags.append((34675, "B", len(icc_profile), icc_profile, False))

    if metadata:
        # tifffile expects JSON-serialisable metadata; sanitised metadata already
        # contains primitive types so we can pass it through.
        tiff_kwargs["metadata"] = {tag: metadata[tag] for tag in metadata}

    if extratags:
        tiff_kwargs["extratags"] = extratags

    tifffile.imwrite(destination, array_for_write, **tiff_kwargs)


def image_to_float(image: Image.Image) -> Tuple[np.ndarray, np.dtype, Optional[np.ndarray], int]:
    """Convert a PIL image into a float representation in the 0-1 range."""

    if image.mode not in {"RGB", "RGBA", "I", "I;16", "I;16L", "I;16B", "I;16S", "L", "LA"}:
        image = image.convert("RGBA" if "A" in image.mode else "RGB")

    arr = np.array(image)
    base_channels = 3
    alpha_channel: Optional[np.ndarray] = None

    if arr.ndim == 2:
        base_channels = 1
        base = arr[:, :, None]
    else:
        channels = arr.shape[2]
        if channels == 1:
            base_channels = 1
            base = arr
        elif channels == 2:
            base_channels = 1
            base = arr[:, :, :1]
            alpha_channel = arr[:, :, 1]
        else:
            base_channels = min(3, channels)
            base = arr[:, :, :base_channels]
            if channels > base_channels:
                alpha_channel = arr[:, :, base_channels]

    base_dtype = base.dtype

    base_float = _normalise_to_unit_range(base)
    if base_float.ndim == 3 and base_float.shape[2] == 1:
        base_float = np.repeat(base_float, 3, axis=2)

    if alpha_channel is not None:
        alpha_float = _normalise_to_unit_range(alpha_channel)
    else:
        alpha_float = None

    return base_float.astype(np.float32, copy=False), base_dtype, alpha_float, base_channels


def float_to_dtype_array(
    arr: np.ndarray,
    dtype: np.dtype,
    alpha: Optional[np.ndarray],

    base_channels: int,
) -> np.ndarray:
    """Convert a float array back to the requested dtype, restoring alpha."""

    arr = np.clip(arr, 0.0, 1.0)
    target_dtype = np.dtype(dtype)

    if base_channels == 1 and arr.ndim == 3:
        arr = arr[:, :, :1]

    if np.issubdtype(target_dtype, np.floating):
        result = arr.astype(target_dtype, copy=False)
    elif np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        normalised = np.clip(arr, 0.0, 1.0)
        if info.min < 0:
            scaled = normalised * (info.max - info.min) + info.min
        else:
            scaled = normalised * info.max
        scaled = np.clip(scaled, info.min, info.max)
        result = np.floor(scaled + 0.5, dtype=np.float64).astype(target_dtype)
    elif np.issubdtype(target_dtype, np.bool_):
        result = arr >= 0.5
    else:
        raise TypeError(f"Unsupported dtype for conversion: {target_dtype}")

    if result.ndim == 3 and result.shape[2] == 1:
        result = result[:, :, 0]

    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            if info.min < 0:
                alpha_scaled = alpha * (info.max - info.min) + info.min
            else:
                alpha_scaled = alpha * info.max
            alpha_scaled = np.clip(alpha_scaled, info.min, info.max)
            alpha_int = np.floor(alpha_scaled + 0.5, dtype=np.float64).astype(target_dtype)
        elif np.issubdtype(target_dtype, np.floating):
            alpha_int = alpha.astype(target_dtype, copy=False)
        elif np.issubdtype(target_dtype, np.bool_):
            alpha_int = alpha >= 0.5
        else:
            raise TypeError(f"Unsupported dtype for alpha conversion: {target_dtype}")

        if result.ndim == 2:
            result = result[:, :, None]
        if alpha_int.ndim == 2:
            alpha_int = alpha_int[:, :, None]
        result = np.concatenate([result, alpha_int], axis=2)

    return np.ascontiguousarray(result)


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


def _cumsum_with_pad(arr: np.ndarray, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (1, 0)
    return np.pad(np.cumsum(arr, axis=axis), pad_width, mode="constant")


def box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Simple box blur that preserves array shape using reflective padding."""

    if radius <= 0:
        return arr

    kernel_size = radius * 2 + 1
    pad_width = [(radius, radius), (radius, radius)]
    pad_width.extend([(0, 0)] * (arr.ndim - 2))
    padded = np.pad(arr, pad_width, mode="reflect")

    summed_axis0 = _cumsum_with_pad(padded, axis=0)
    window_axis0 = summed_axis0[kernel_size:, ...] - summed_axis0[:-kernel_size, ...]

    summed_axis1 = _cumsum_with_pad(window_axis0, axis=1)
    window_axis1 = summed_axis1[:, kernel_size:, ...] - summed_axis1[:, :-kernel_size, ...]

    return window_axis1 / float(kernel_size * kernel_size)


def kelvin_to_rgb(temp: float) -> np.ndarray:
    """Approximate RGB scaling factors for a given color temperature."""

    if temp is None:
        temp = 6500.0

    temp = max(1000.0, min(40000.0, float(temp))) / 100.0

    if temp <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19.0 else 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        blue = 255.0

    rgb = np.array([red, green, blue], dtype=np.float32) / 255.0
    return np.clip(rgb, 0.0, 4.0)


def apply_white_balance(arr: np.ndarray, temp: Optional[float], tint: float) -> np.ndarray:
    if temp is None and abs(tint) < 1e-6:
        return arr

    reference = kelvin_to_rgb(6500.0)
    target = kelvin_to_rgb(temp if temp is not None else 6500.0)
    scale = target / np.clip(reference, 1e-6, None)

    if abs(tint) > 1e-6:
        # Positive tint introduces magenta by reducing green slightly.
        tint_scale = np.array([
            1.0 + tint * 0.002,
            1.0 - tint * 0.004,
            1.0 + tint * 0.002,
        ], dtype=np.float32)
        scale *= tint_scale

    return np.clip(arr * scale, 0.0, 1.0)


def apply_exposure(arr: np.ndarray, exposure: float) -> np.ndarray:
    if abs(exposure) < 1e-6:
        return arr
    factor = 2.0 ** exposure
    return np.clip(arr * factor, 0.0, 1.0)


def apply_shadow_highlight(
    arr: np.ndarray, shadow_lift: float, highlight_recovery: float
) -> np.ndarray:
    if abs(shadow_lift) < 1e-6 and abs(highlight_recovery) < 1e-6:
        return arr

    lum = np.dot(arr, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))

    if shadow_lift:
        shadow_mask = 1.0 - np.clip(lum / 0.6, 0.0, 1.0)
        arr = arr + shadow_mask[..., None] * shadow_lift * 0.8

    if highlight_recovery:
        highlight_mask = np.clip((lum - 0.65) / 0.35, 0.0, 1.0)
        compression = 1.0 - highlight_recovery * 0.7 * highlight_mask[..., None]
        arr = arr * compression

    return np.clip(arr, 0.0, 1.0)


def apply_midtone_contrast(arr: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-6:
        return arr

    midpoint = 0.5
    contrast = 1.0 + amount * 1.6
    return np.clip((arr - midpoint) * contrast + midpoint, 0.0, 1.0)


def apply_vibrance(arr: np.ndarray, vibrance: float) -> np.ndarray:
    if abs(vibrance) < 1e-6:
        return arr

    maxc = arr.max(axis=2, keepdims=True)
    minc = arr.min(axis=2, keepdims=True)
    sat = maxc - minc
    sat_norm = sat / (sat.max() + 1e-6)
    mean = arr.mean(axis=2, keepdims=True)
    factor = 1.0 + vibrance * (1.0 - sat_norm)
    return np.clip(mean + (arr - mean) * factor, 0.0, 1.0)


def apply_saturation(arr: np.ndarray, saturation: float) -> np.ndarray:
    if abs(saturation) < 1e-6:
        return arr

    factor = 1.0 + saturation
    mean = arr.mean(axis=2, keepdims=True)
    return np.clip(mean + (arr - mean) * factor, 0.0, 1.0)


def apply_clarity(arr: np.ndarray, clarity: float) -> np.ndarray:
    if clarity <= 1e-6:
        return arr

    radius = max(1, int(round(clarity * 6)))
    blurred = box_blur(arr, radius)
    detail = arr - blurred
    return np.clip(arr + detail * clarity * 1.5, 0.0, 1.0)


def rgb_to_ycbcr(arr: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        dtype=np.float32,
    )
    offset = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    return arr @ matrix.T + offset


def ycbcr_to_rgb(arr: np.ndarray) -> np.ndarray:
    y = arr[..., 0]
    cb = arr[..., 1] - 0.5
    cr = arr[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


def apply_chroma_denoise(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 1e-6:
        return arr

    ycbcr = rgb_to_ycbcr(arr)
    radius = max(1, int(round(amount * 6)))
    cb = box_blur(ycbcr[..., 1][..., None], radius)[..., 0]
    cr = box_blur(ycbcr[..., 2][..., None], radius)[..., 0]
    ycbcr[..., 1] = cb
    ycbcr[..., 2] = cr
    return np.clip(ycbcr_to_rgb(ycbcr), 0.0, 1.0)


def apply_glow(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 1e-6:
        return arr

    radius = max(1, int(round(2 + amount * 10)))
    softened = box_blur(arr, radius)
    lum = np.dot(arr, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    mask = np.clip((lum - 0.55) / 0.45, 0.0, 1.0)
    blend = arr + (softened - arr) * (mask[..., None] * amount)
    return np.clip(blend, 0.0, 1.0)


def apply_adjustments(arr: np.ndarray, settings: AdjustmentSettings) -> np.ndarray:
    arr = arr.astype(np.float32, copy=True)
    arr = apply_exposure(arr, settings.exposure)
    arr = apply_white_balance(arr, settings.white_balance_temp, settings.white_balance_tint)
    arr = apply_shadow_highlight(arr, settings.shadow_lift, settings.highlight_recovery)
    arr = apply_midtone_contrast(arr, settings.midtone_contrast)
    arr = apply_vibrance(arr, settings.vibrance)
    arr = apply_saturation(arr, settings.saturation)
    arr = apply_clarity(arr, settings.clarity)
    arr = apply_chroma_denoise(arr, settings.chroma_denoise)
    arr = apply_glow(arr, settings.glow)
    return np.clip(arr, 0.0, 1.0)


def resize_image(image: Image.Image, target_long_edge: Optional[int]) -> Image.Image:
    if not target_long_edge:
        return image

    width, height = image.size
    current_long_edge = max(width, height)
    if current_long_edge <= target_long_edge:
        return image

    scale = target_long_edge / float(current_long_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    LOGGER.debug("Resizing image from %s to %s", (width, height), new_size)
    return image.resize(new_size, RESAMPLING_LANCZOS)


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
        LOGGER.debug("Unable to sanitize TIFF metadata", exc_info=True)
        return None
    return safe or None


def save_image(
    destination: Path,
    arr_int: np.ndarray,
    dtype: np.dtype,
    metadata: Optional[Any],
    icc_profile: Optional[bytes],
    compression: str,
) -> None:
    metadata = sanitize_tiff_metadata(metadata)
    dtype = np.dtype(dtype)
    arr_out = np.ascontiguousarray(arr_int)

    # Pillow expects 2D arrays for single-channel images. Avoid keeping a trailing
    # singleton channel which can appear after concatenating alpha data upstream.
    if arr_out.ndim == 3 and arr_out.shape[2] == 1:
        arr_out = arr_out[:, :, 0]

    # Prefer tifffile for high bit-depth or floating point data when available so
    # we can retain metadata and compression support.
    use_tifffile = False
    if tifffile is not None:
        if dtype.kind == "f":
            use_tifffile = True
        elif np.issubdtype(dtype, np.integer):
            try:
                info = np.iinfo(dtype)
            except ValueError:
                info = None
            else:
                use_tifffile = info.bits > 8

    if use_tifffile and tifffile is not None:
        try:
            _write_with_tifffile(destination, arr_out, metadata, icc_profile, compression)
        except Exception as exc:
            if "imagecodecs" in str(exc).lower() and compression_for_tifffile(compression) is not None:
                LOGGER.warning(
                    "Compression '%s' requires imagecodecs; retrying without compression",
                    compression,
                )
                _write_with_tifffile(destination, arr_out, metadata, icc_profile, "tiff_none")
            else:
                raise
        return

    if dtype.kind == "f" and tifffile is None:
        LOGGER.warning(
            "Saving floating point TIFF without 'tifffile'; converting to 16-bit before writing with Pillow."
        )
        base, alpha_channel = _split_alpha_channel(arr_out)
        base_float = base.astype(np.float32, copy=False)
        alpha_float = alpha_channel.astype(np.float32, copy=False) if alpha_channel is not None else None
        base_channel_count = 1 if base_float.ndim == 2 else base_float.shape[2]
        arr_out = float_to_dtype_array(base_float, np.uint16, alpha_float, base_channel_count)

    image = Image.fromarray(arr_out)

    save_kwargs: Dict[str, Any] = {}
    if compression:
        save_kwargs["compression"] = compression
    if icc_profile is not None:
        save_kwargs["icc_profile"] = icc_profile
    if metadata:
        info = TiffImagePlugin.ImageFileDirectory_v2()
        for tag, value in metadata.items():
            info[tag] = value
        save_kwargs["tiffinfo"] = info

    image.save(destination, **save_kwargs)


def process_single_image(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    compression: str,
    resize_long_edge: Optional[int],
) -> None:
    LOGGER.info("Processing %s -> %s", source, destination)

    with Image.open(source) as image:
        image.load()
        metadata = getattr(image, "tag_v2", None)
        icc_profile = image.info.get("icc_profile") if hasattr(image, "info") else None

        frame_count = getattr(image, "n_frames", 1)
        if frame_count > 1:
            LOGGER.warning("%s contains multiple frames; only the first frame will be processed", source)

        image = resize_image(image, resize_long_edge)

        arr_float, dtype, alpha, base_channels = image_to_float(image)
        adjusted = apply_adjustments(arr_float, adjustments)
        arr_int = float_to_dtype_array(adjusted, dtype, alpha, base_channels)

    save_image(destination, arr_int, dtype, metadata, icc_profile, compression)


def describe_plan(
    source: Path,
    destination: Path,
    resize_long_edge: Optional[int],
    adjustments: AdjustmentSettings,
) -> str:
    pieces = [f"{source.name} -> {destination.name}"]
    if resize_long_edge:
        pieces.append(f"resize≤{resize_long_edge}px")
    pieces.append(f"preset={adjustments}")
    return ", ".join(pieces)


def run_pipeline(args: argparse.Namespace) -> int:
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input folder '{input_root}' does not exist or is not a directory")

    def _contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
        except ValueError:
            return False
        return True

    if input_root == output_root:
        raise SystemExit("Output folder must be different from the input folder to avoid self-overwrites.")
    if _contains(input_root, output_root):
        raise SystemExit(
            "Output folder cannot be located inside the input folder; choose a sibling or separate directory."
        )
    if _contains(output_root, input_root):
        raise SystemExit(
            "Input folder cannot be located inside the output folder; choose non-overlapping directories."
        )

    adjustments = build_adjustments(args)
    images = sorted(collect_images(input_root, args.recursive))
    if not images:
        LOGGER.warning("No TIFF images found in %s", input_root)
        return 0

    processed = 0
    for source in images:
        destination = ensure_output_path(input_root, output_root, source, args.suffix, args.recursive)

        if destination.exists() and not args.overwrite:
            LOGGER.warning("Skipping %s because %s already exists", source, destination)
            continue

        if args.dry_run:
            LOGGER.info("DRY RUN: %s", describe_plan(source, destination, args.resize_long_edge, adjustments))
            continue

        try:
            process_single_image(source, destination, adjustments, args.compression, args.resize_long_edge)
            processed += 1
        except Exception:
            LOGGER.exception("Failed to process %s", source)

    LOGGER.info("Finished processing %d image(s)", processed)
    return processed


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
