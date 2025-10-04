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
from typing import Iterable, Iterator, List, Literal, Optional, Tuple, Union

Any = _Any
Dict = _Dict

import numpy as np
from PIL import Image

try:  # Optional high-fidelity TIFF writer
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tifffile = None

try:  # Optional progress bar for batch runs
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None


def _tqdm_progress(
    iterable: Iterable[Any], *, total: Optional[int], description: Optional[str]
) -> Iterable[Any]:
    """Wrap *iterable* with :mod:`tqdm` if available."""

    if _tqdm is None:  # pragma: no cover - defensive fallback
        return iterable
    return _tqdm(iterable, total=total, desc=description, unit="image")


_PROGRESS_WRAPPER = _tqdm_progress if _tqdm is not None else None


class LuxuryGradeException(RuntimeError):
    """Raised when the processing environment cannot meet luxury standards."""


class ProcessingCapabilities:
    """Introspects optional dependencies to describe processing fidelity."""

    _SENTINEL = object()

    def __init__(self, tifffile_module: Any | None | object = _SENTINEL) -> None:
        """Initialise capability detection.

        Parameters
        ----------
        tifffile_module:
            Optional dependency override primarily used by tests.  When omitted
            (the default) the globally imported :mod:`tifffile` module is
            consulted.  Passing ``None`` explicitly simulates the dependency not
            being available at all.
        """

        if tifffile_module is self._SENTINEL:
            self._tifffile = tifffile
        else:
            self._tifffile = tifffile_module
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


__all__ = [
    "AdjustmentSettings",
    "FloatDynamicRange",
    "ImageToFloatResult",
    "LUXURY_PRESETS",
    "ProcessingCapabilities",
    "apply_adjustments",
    "build_adjustments",
    "collect_images",
    "float_to_dtype_array",
    "image_to_float",
    "main",
    "parse_args",
    "process_image",
    "process_single_image",
    "run_pipeline",
    "save_image",
]


def _wrap_with_progress(
    iterable: Iterable[Any],
    *,
    total: Optional[int],
    description: str,
    enabled: bool,
) -> Iterable[Any]:
    """Return an iterable wrapped with a progress helper when available."""

    if not enabled:
        return iterable

    helper = _PROGRESS_WRAPPER
    if helper is None:
        LOGGER.debug("Progress helper not available; install tqdm for progress reporting.")
        return iterable

    try:
        return helper(iterable, total=total, description=description)
    except Exception:  # pragma: no cover - defensive fallback
        LOGGER.exception("Progress helper failed; continuing without progress display.")
        return iterable


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

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        def ensure_range(name: str, value: float, minimum: float, maximum: float) -> None:
            if not (minimum <= value <= maximum):
                raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")

        ensure_range("exposure", self.exposure, -5.0, 5.0)

        if self.white_balance_temp is not None:
            ensure_range("white_balance_temp", self.white_balance_temp, 1500.0, 20000.0)

        ensure_range("white_balance_tint", self.white_balance_tint, -150.0, 150.0)
        ensure_range("shadow_lift", self.shadow_lift, 0.0, 1.0)
        ensure_range("highlight_recovery", self.highlight_recovery, 0.0, 1.0)
        ensure_range("midtone_contrast", self.midtone_contrast, -1.0, 1.0)
        ensure_range("vibrance", self.vibrance, -1.0, 1.0)
        ensure_range("saturation", self.saturation, -1.0, 1.0)
        ensure_range("clarity", self.clarity, -1.0, 1.0)
        ensure_range("chroma_denoise", self.chroma_denoise, 0.0, 1.0)
        ensure_range("glow", self.glow, 0.0, 1.0)


# --- Float handling primitives -------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FloatDynamicRange:
    """Describes how floating point image data was normalised."""

    offset: np.ndarray
    scale: np.ndarray
    scale_recip: np.ndarray

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Optional["FloatDynamicRange"]:
        """Create a descriptor capturing per-channel offset and scale."""

        if arr.size == 0:
            return None

        if arr.ndim == 2:
            working = arr[:, :, None]
        else:
            working = arr

        channels = working.shape[-1]
        flattened = working.reshape(-1, channels)

        offsets = np.zeros(channels, dtype=np.float32)
        scales = np.ones(channels, dtype=np.float32)
        saw_finite = False

        for idx in range(channels):
            channel = flattened[:, idx]
            finite = channel[np.isfinite(channel)]
            if finite.size == 0:
                offsets[idx] = 0.0
                scales[idx] = 1.0
                continue

            saw_finite = True
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
            offsets[idx] = np.float32(min_val)
            diff = float(max_val - min_val)
            if diff <= 0.0 or not math.isfinite(diff):
                scales[idx] = 1.0
            else:
                scales[idx] = np.float32(diff)

        if not saw_finite:
            return None

        scale_recip = np.where(scales == 0.0, 1.0, 1.0 / scales).astype(np.float32)
        return cls(offset=offsets, scale=scales.astype(np.float32), scale_recip=scale_recip)

    @staticmethod
    def _prepare(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
        working = np.asarray(arr, dtype=np.float32)
        squeezed = False
        if working.ndim == 2:
            working = working[:, :, None]
            squeezed = True
        return working, squeezed

    def normalise(self, arr: np.ndarray) -> np.ndarray:
        working, squeezed = self._prepare(arr)
        offset = self.offset.reshape((1, 1, -1))
        scale = self.scale_recip.reshape((1, 1, -1))
        normalised = (working - offset) * scale
        if squeezed:
            return normalised[:, :, 0]
        return normalised

    def denormalise(self, arr: np.ndarray) -> np.ndarray:
        working, squeezed = self._prepare(arr)
        offset = self.offset.reshape((1, 1, -1))
        scale = self.scale.reshape((1, 1, -1))
        restored = working * scale + offset
        if squeezed:
            return restored[:, :, 0]
        return restored


@dataclasses.dataclass(frozen=True)
class ImageToFloatResult:
    """Container for :func:`image_to_float` results with metadata."""

    array: np.ndarray
    dtype: np.dtype
    alpha: Optional[np.ndarray]
    base_channels: int
    float_normalisation: Optional[FloatDynamicRange] = None

    def __iter__(self):
        yield self.array
        yield self.dtype
        yield self.alpha
        yield self.base_channels

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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting (useful for minimal or non-interactive environments)",
    )

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
            base._validate()
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


def ensure_output_path(
    input_root: Path,
    output_root: Path,
    source: Path,
    suffix: str,
    recursive: bool,
    *,
    create: bool = True,
) -> Path:
    relative = source.relative_to(input_root) if recursive else Path(source.name)
    destination = output_root / relative
    if create:
        destination.parent.mkdir(parents=True, exist_ok=True)
    new_name = destination.stem + suffix + destination.suffix
    return destination.with_name(new_name)


def image_to_float(
    image: Image.Image,
    return_format: Literal["tuple3", "tuple4", "object"] = "object",
) -> Union[
    Tuple[np.ndarray, np.dtype, Optional[np.ndarray]],
    Tuple[np.ndarray, np.dtype, Optional[np.ndarray], int],
    ImageToFloatResult,
]:
    """Converts a PIL image to an RGB float32 array in the 0-1 range."""

    supported_modes = {
        "RGB",
        "RGBA",
        "I",
        "I;16",
        "I;16L",
        "I;16B",
        "I;16S",
        "F",
        "L",
        "LA",
    }
    if image.mode not in supported_modes:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    arr = np.array(image)
    alpha_channel: Optional[np.ndarray] = None

    if arr.ndim == 2:
        base_channels = 1
        color_data = arr[:, :, None]
    else:
        if arr.shape[2] == 4:
            alpha_channel = arr[:, :, 3]
            base_channels = 3
            color_data = arr[:, :, :3]
        elif arr.shape[2] == 2:
            alpha_channel = arr[:, :, 1]
            base_channels = 1
            color_data = arr[:, :, :1]
        else:
            base_channels = arr.shape[2]
            color_data = arr[:, :, :base_channels]

    color_float = color_data.astype(np.float32, copy=False)
    alpha_float = alpha_channel.astype(np.float32, copy=False) if alpha_channel is not None else None

    float_norm: Optional[FloatDynamicRange] = None
    if np.issubdtype(color_data.dtype, np.integer):
        dtype_info = np.iinfo(color_data.dtype)
        scale = dtype_info.max - dtype_info.min
        if scale == 0:
            normalised = np.zeros_like(color_float, dtype=np.float32)
            if alpha_float is not None:
                alpha_float = np.zeros_like(alpha_float, dtype=np.float32)
        else:
            offset = float(dtype_info.min)
            inv_scale = 1.0 / float(scale)
            normalised = (color_float - offset) * inv_scale
            if alpha_float is not None:
                alpha_float = (alpha_float - offset) * inv_scale
        normalised = np.clip(normalised, 0.0, 1.0)
        if alpha_float is not None:
            alpha_float = np.clip(alpha_float, 0.0, 1.0)
    else:
        float_norm = FloatDynamicRange.from_array(color_float)
        if float_norm is not None:
            normalised = float_norm.normalise(color_float)
        else:
            normalised = color_float
        normalised = np.clip(normalised, 0.0, 1.0)
        if alpha_float is not None:
            alpha_float = np.clip(alpha_float, 0.0, 1.0)

    if base_channels == 1:
        working = np.repeat(normalised, 3, axis=2)
    else:
        working = normalised

    result = ImageToFloatResult(
        array=np.ascontiguousarray(working, dtype=np.float32),
        dtype=np.dtype(color_data.dtype),
        alpha=None if alpha_float is None else np.ascontiguousarray(alpha_float, dtype=np.float32),
        base_channels=base_channels,
        float_normalisation=float_norm,
    )

    if return_format == "tuple3":
        return result.array, result.dtype, result.alpha
    if return_format == "tuple4":
        return result.array, result.dtype, result.alpha, result.base_channels
    allowed_formats = ("object", "tuple3", "tuple4")
    if return_format not in allowed_formats:
        raise ValueError(f"Unsupported return_format: {return_format}. Allowed values are: {allowed_formats}")
    return result


def float_to_dtype_array(
    arr: np.ndarray,
    dtype: np.dtype,
    alpha: Optional[np.ndarray],
    base_channels: Optional[int] = None,
    *,
    float_normalisation: Optional[FloatDynamicRange] = None,
) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        working = arr[:, :, None]
    else:
        working = arr

    if base_channels is None:
        base_channels = working.shape[2]
    color = working[:, :, :base_channels]

    np_dtype = np.dtype(dtype)
    dtype_info = np.iinfo(np_dtype) if np.issubdtype(np_dtype, np.integer) else None
    if dtype_info:
        scale = float(dtype_info.max - dtype_info.min)
        if scale == 0:
            color_int = np.full_like(color, dtype_info.min, dtype=np_dtype)
        else:
            color_int = np.round(color * scale + dtype_info.min).astype(np_dtype)
    else:
        if float_normalisation is not None and np.issubdtype(np_dtype, np.floating):
            color = float_normalisation.denormalise(color)
        color_int = color.astype(np_dtype, copy=False)

    channels: list[np.ndarray] = [color_int]

    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
        if dtype_info:
            scale = float(dtype_info.max - dtype_info.min)
            if scale == 0:
                alpha_int = np.full_like(alpha, dtype_info.min, dtype=np_dtype)
            else:
                alpha_int = np.round(alpha * scale + dtype_info.min).astype(np_dtype)
        else:
            alpha_int = alpha.astype(np_dtype, copy=False)
        channels.append(alpha_int[:, :, None])

    result = np.concatenate(channels, axis=2) if len(channels) > 1 else color_int
    if result.shape[2] == 1:
        result = result[:, :, 0]
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



def sanitize_tiff_metadata(raw_metadata: Optional[Any]) -> Optional[Dict[int, Any]]:
    if raw_metadata is None:
        return None
    safe: Dict[int, Any] = {}
    forbidden_tags = {256, 257, 273, 279, 322, 323, 324, 325}
    try:
        for tag in raw_metadata:
            if tag in forbidden_tags:
                continue
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
    np_dtype = np.dtype(dtype)
    dtype_info = np.iinfo(np_dtype) if np.issubdtype(np_dtype, np.integer) else None

    use_tifffile = tifffile is not None and (
        (dtype_info is not None and dtype_info.bits >= 16)
        or np.issubdtype(np_dtype, np.floating)
    )

    array_to_write = arr_int
    if array_to_write.ndim == 2:
        photometric = "minisblack"
    else:
        photometric = "rgb" if array_to_write.shape[2] >= 3 else "minisblack"

    if use_tifffile:
        tiff_kwargs = {
            "photometric": photometric,
            "compression": compression_for_tifffile(compression),
            "metadata": None,
        }
        if array_to_write.ndim == 3 and array_to_write.shape[2] > 3:
            tiff_kwargs["extrasamples"] = "unassoc"
        extratags = []
        if icc_profile:
            extratags.append((34675, "B", len(icc_profile), icc_profile, False))
        if metadata:
            try:
                tiff_kwargs["metadata"] = {tag: metadata[tag] for tag in metadata}
            except Exception:  # pragma: no cover - best-effort metadata copy
                LOGGER.debug("Unable to serialise TIFF metadata", exc_info=True)
        if extratags:
            tiff_kwargs["extratags"] = extratags
        tifffile.imwrite(destination, array_to_write, **tiff_kwargs)
        return

    if dtype_info and dtype_info.bits == 16:
        LOGGER.warning(
            "Falling back to Pillow for 16-bit save; output will be 8-bit. Install 'tifffile' for full 16-bit support."
        )
        scale = dtype_info.max / 255.0 if dtype_info.max else 1.0
        rgb = np.clip(array_to_write[..., :3], 0, dtype_info.max).astype(np.float32) / scale
        rgb8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        if array_to_write.ndim == 3 and array_to_write.shape[2] > 3:
            alpha = np.clip(array_to_write[..., 3], 0, dtype_info.max).astype(np.float32) / scale
            alpha8 = np.clip(np.round(alpha), 0, 255).astype(np.uint8)
            array_to_write = np.concatenate([rgb8, alpha8[:, :, None]], axis=2)
        else:
            array_to_write = rgb8
    elif np.issubdtype(np_dtype, np.floating):
        rgb = np.clip(array_to_write, 0.0, 1.0)
        array_to_write = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    else:
        array_to_write = array_to_write.astype(np.uint8)

    if array_to_write.ndim == 2:
        mode = "L"
    elif array_to_write.shape[2] == 4:
        mode = "RGBA"
    elif array_to_write.shape[2] == 3:
        mode = "RGB"
    else:
        mode = "L"

    image = Image.fromarray(array_to_write, mode=mode)
    save_kwargs = {"compression": compression}
    if metadata is not None:
        save_kwargs["tiffinfo"] = metadata
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    image.save(destination, format="TIFF", **save_kwargs)


def apply_exposure(arr: np.ndarray, stops: float) -> np.ndarray:
    if stops == 0:
        return arr
    factor = float(2.0 ** stops)
    LOGGER.debug("Applying exposure: %s stops (factor %.3f)", stops, factor)
    return arr * factor


def kelvin_to_rgb(temperature: float) -> np.ndarray:
    temp = temperature / 100.0
    if temp <= 0:
        temp = 0.1

    if temp <= 66:
        red = 1.0
        green = np.clip(0.39008157876901960784 * math.log(temp) - 0.63184144378862745098, 0, 1)
        blue = 0 if temp <= 19 else np.clip(0.54320678911019607843 * math.log(temp - 10) - 1.19625408914, 0, 1)
    else:
        red = np.clip(1.29293618606274509804 * (temp - 60) ** -0.1332047592, 0, 1)
        green = np.clip(1.12989086089529411765 * (temp - 60) ** -0.0755148492, 0, 1)
        blue = 1.0
    return np.array([red, green, blue], dtype=np.float32)


def apply_white_balance(arr: np.ndarray, temperature: Optional[float], tint: float) -> np.ndarray:
    result = arr
    if temperature is not None:
        ref = kelvin_to_rgb(6500.0)
        target = kelvin_to_rgb(temperature)
        scale = target / ref
        LOGGER.debug("Applying temperature: %sK scale=%s", temperature, scale)
        result = result * scale.reshape((1, 1, 3))
    if tint:
        tint_scale = np.array([1.0 + tint * 0.0015, 1.0, 1.0 - tint * 0.0015], dtype=np.float32)
        LOGGER.debug("Applying tint scale=%s", tint_scale)
        result = result * tint_scale.reshape((1, 1, 3))
    return result


def luminance(arr: np.ndarray) -> np.ndarray:
    return arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722


def apply_shadow_lift(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    gamma = 1.0 / (1.0 + amount * 3.0)
    lum = luminance(arr)
    lifted = np.power(np.clip(lum, 0.0, 1.0), gamma)
    LOGGER.debug("Shadow lift amount=%s gamma=%.3f", amount, gamma)
    return arr + (lifted - lum)[..., None]


def apply_highlight_recovery(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    gamma = 1.0 + amount * 2.0
    lum = luminance(arr)
    compressed = np.power(np.clip(lum, 0.0, 1.0), gamma)
    LOGGER.debug("Highlight recovery amount=%s gamma=%.3f", amount, gamma)
    return arr + (compressed - lum)[..., None]


def apply_midtone_contrast(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return arr
    lum = luminance(arr)
    contrasted = 0.5 + (lum - 0.5) * (1.0 + amount)
    LOGGER.debug("Midtone contrast amount=%s", amount)
    return arr + (contrasted - lum)[..., None]


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    diff = maxc - minc

    hue = np.zeros_like(maxc)
    mask = diff != 0
    rc = np.zeros_like(maxc)
    gc = np.zeros_like(maxc)
    bc = np.zeros_like(maxc)

    rc[mask] = ((maxc - r) / diff)[mask]
    gc[mask] = ((maxc - g) / diff)[mask]
    bc[mask] = ((maxc - b) / diff)[mask]

    hue[maxc == r] = (bc - gc)[maxc == r]
    hue[maxc == g] = 2.0 + (rc - bc)[maxc == g]
    hue[maxc == b] = 4.0 + (gc - rc)[maxc == b]
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maxc)
    saturation[maxc != 0] = diff[maxc != 0] / maxc[maxc != 0]

    value = maxc
    return np.stack([hue, saturation, value], axis=-1)


def hsv_to_rgb(arr: np.ndarray) -> np.ndarray:
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    shape = h.shape + (3,)
    rgb = np.zeros(shape, dtype=np.float32)

    conditions = [
        (i_mod == 0, np.stack([v, t, p], axis=-1)),
        (i_mod == 1, np.stack([q, v, p], axis=-1)),
        (i_mod == 2, np.stack([p, v, t], axis=-1)),
        (i_mod == 3, np.stack([p, q, v], axis=-1)),
        (i_mod == 4, np.stack([t, p, v], axis=-1)),
        (i_mod == 5, np.stack([v, p, q], axis=-1)),
    ]
    for condition, value in conditions:
        rgb[condition] = value[condition]
    return rgb


def apply_vibrance(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return arr
    hsv = rgb_to_hsv(arr)
    saturation = hsv[..., 1]
    hsv[..., 1] = np.clip(saturation + amount * (1.0 - saturation) * np.sqrt(saturation), 0.0, 1.0)
    LOGGER.debug("Vibrance amount=%s", amount)
    return hsv_to_rgb(hsv)


def apply_saturation(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return arr
    hsv = rgb_to_hsv(arr)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + amount), 0.0, 1.0)
    LOGGER.debug("Saturation delta=%s", amount)
    return hsv_to_rgb(hsv)


def gaussian_kernel(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    if radius <= 0:
        return np.array([1.0], dtype=np.float32)
    sigma = sigma or max(radius / 3.0, 1e-6)
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def separable_convolve(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * arr.ndim
    k = kernel.size // 2
    pad_width[axis] = (k, k)
    padded = np.pad(arr, pad_width, mode="reflect")
    convolved = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=axis, arr=padded)
    return convolved.astype(np.float32)


def gaussian_blur(arr: np.ndarray, radius: int, sigma: Optional[float] = None) -> np.ndarray:
    kernel = gaussian_kernel(radius, sigma)
    blurred = separable_convolve(arr, kernel, axis=0)
    blurred = separable_convolve(blurred, kernel, axis=1)
    return blurred


def apply_clarity(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    radius = max(1, int(round(1 + amount * 5)))
    blurred = gaussian_blur(arr, radius)
    high_pass = arr - blurred
    LOGGER.debug("Clarity amount=%s radius=%s", amount, radius)
    return np.clip(arr + high_pass * (0.6 + amount * 0.8), 0.0, 1.0)


def rgb_to_yuv(arr: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [
            [0.2126, 0.7152, 0.0722],
            [-0.1146, -0.3854, 0.5000],
            [0.5000, -0.4542, -0.0458],
        ],
        dtype=np.float32,
    )
    yuv = arr @ matrix.T
    yuv[..., 1:] += 0.5
    return yuv


def yuv_to_rgb(arr: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [
            [1.0, 0.0, 1.5748],
            [1.0, -0.1873, -0.4681],
            [1.0, 1.8556, 0.0],
        ],
        dtype=np.float32,
    )
    rgb = arr.copy()
    rgb[..., 1:] -= 0.5
    rgb = rgb @ matrix.T
    return rgb


def apply_chroma_denoise(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    yuv = rgb_to_yuv(arr)
    radius = max(1, int(round(1 + amount * 4)))
    for channel in (1, 2):
        channel_data = yuv[..., channel]
        blurred = gaussian_blur(channel_data[..., None], radius)[:, :, 0]
        yuv[..., channel] = channel_data * (1.0 - amount) + blurred * amount
    LOGGER.debug("Chroma denoise amount=%s radius=%s", amount, radius)
    return np.clip(yuv_to_rgb(yuv), 0.0, 1.0)


def apply_glow(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    radius = max(2, int(round(6 + amount * 20)))
    softened = gaussian_blur(arr, radius)
    LOGGER.debug("Glow amount=%s radius=%s", amount, radius)
    return np.clip(arr * (1.0 - amount) + softened * amount, 0.0, 1.0)


def resize_bilinear(arr: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    height, width = arr.shape[:2]
    if width == new_width and height == new_height:
        return arr
    x = np.linspace(0, width - 1, new_width, dtype=np.float32)
    y = np.linspace(0, height - 1, new_height, dtype=np.float32)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x_weight = (x - x0).astype(np.float32).reshape(1, -1, 1)
    y_weight = (y - y0).astype(np.float32).reshape(-1, 1, 1)

    Ia = arr[np.ix_(y0, x0)]
    Ib = arr[np.ix_(y0, x1)]
    Ic = arr[np.ix_(y1, x0)]
    Id = arr[np.ix_(y1, x1)]

    top = Ia * (1.0 - x_weight) + Ib * x_weight
    bottom = Ic * (1.0 - x_weight) + Id * x_weight
    return (top * (1.0 - y_weight) + bottom * y_weight).astype(np.float32)


def resize_array(arr: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    if arr.ndim == 2:
        expanded = arr[:, :, None]
        resized = resize_bilinear(expanded, new_width, new_height)
        return resized[:, :, 0]
    if arr.ndim == 3:
        return resize_bilinear(arr, new_width, new_height)
    raise ValueError("Unsupported array shape for resizing")


def resize_long_edge_array(arr: np.ndarray, target: int) -> np.ndarray:
    height, width = arr.shape[:2]
    long_edge = max(width, height)
    if long_edge <= target:
        return arr
    scale = target / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    LOGGER.debug("Resizing from %sx%s to %s", width, height, (new_width, new_height))
    return resize_array(arr, new_width, new_height)


def apply_adjustments(arr: np.ndarray, adjustments: AdjustmentSettings) -> np.ndarray:
    arr = apply_white_balance(arr, adjustments.white_balance_temp, adjustments.white_balance_tint)
    arr = apply_exposure(arr, adjustments.exposure)
    arr = apply_shadow_lift(arr, adjustments.shadow_lift)
    arr = apply_highlight_recovery(arr, adjustments.highlight_recovery)
    arr = apply_midtone_contrast(arr, adjustments.midtone_contrast)
    arr = np.clip(arr, 0.0, 1.0)
    arr = apply_chroma_denoise(arr, adjustments.chroma_denoise)
    arr = apply_vibrance(arr, adjustments.vibrance)
    arr = apply_saturation(arr, adjustments.saturation)
    arr = apply_clarity(arr, adjustments.clarity)
    arr = apply_glow(arr, adjustments.glow)
    return np.clip(arr, 0.0, 1.0)


def _coerce_resize_target(
    resize_long_edge: Optional[int], resize_target: Optional[int]
) -> Optional[int]:
    """Normalise legacy ``resize_target`` parameter usages."""

    if resize_long_edge is None:
        return resize_target
    if resize_target is None or resize_target == resize_long_edge:
        return resize_long_edge
    raise ValueError("Conflicting resize targets provided; choose one value")


def process_single_image(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    *,
    compression: str,
    resize_long_edge: Optional[int] = None,
    resize_target: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    LOGGER.info("Processing %s -> %s", source, destination)
    if destination.exists() and not dry_run and not destination.is_file():
        if destination.is_dir():
            path_type = "directory"
        elif destination.is_symlink():
            path_type = "symlink"
        else:
            path_type = "non-file"
        raise ValueError(f"Destination path exists but is a {path_type}: {destination}")

    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        metadata = getattr(image, "tag_v2", None)
        icc_profile = image.info.get("icc_profile") if isinstance(image.info, dict) else None
        float_result = image_to_float(image, return_format="object")
        arr = float_result.array
        dtype = float_result.dtype
        alpha = float_result.alpha
        base_channels = float_result.base_channels
        float_norm = float_result.float_normalisation
        adjusted = apply_adjustments(arr, adjustments)
        target = _coerce_resize_target(resize_long_edge, resize_target)
        if target is not None:
            adjusted = resize_long_edge_array(adjusted, target)
            if alpha is not None:
                alpha = resize_long_edge_array(alpha, target)
        arr_int = float_to_dtype_array(
            adjusted,
            dtype,
            alpha,
            base_channels,
            float_normalisation=float_norm,
        )
        if dry_run:
            LOGGER.info("Dry run enabled, skipping save for %s", destination)
            return
        save_image(destination, arr_int, dtype, metadata, icc_profile, compression)


# Backwards compatibility shim for older integrations expecting the previous helper name.
process_image = process_single_image


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


def _ensure_non_overlapping(input_root: Path, output_root: Path) -> None:
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


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the batch processor with the provided arguments."""

    adjustments = build_adjustments(args)
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    if not input_root.is_dir():
        raise SystemExit(f"Input folder '{input_root}' does not exist or is not a directory")

    _ensure_non_overlapping(input_root, output_root)

    images = sorted(collect_images(input_root, args.recursive))
    if not images:
        LOGGER.warning("No TIFF images found in %s", input_root)
        return 0

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Found %s image(s) to process", len(images))
    processed = 0

    progress_iterable = _wrap_with_progress(
        images,
        total=len(images),
        description="Processing images",
        enabled=not getattr(args, "no_progress", False),
    )

    for image_path in progress_iterable:
        destination = ensure_output_path(
            input_root,
            output_root,
            image_path,
            args.suffix,
            args.recursive,
            create=not args.dry_run,
        )
        if destination.exists() and not args.overwrite and not args.dry_run:
            LOGGER.warning("Skipping %s (exists, use --overwrite to replace)", destination)
            continue
        if args.dry_run:
            LOGGER.info("Dry run: would process %s -> %s", image_path, destination)
        process_single_image(
            image_path,
            destination,
            adjustments,
            compression=args.compression,
            resize_long_edge=args.resize_long_edge,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            processed += 1

    return processed


if __name__ == "__main__":  # pragma: no cover
    main()
