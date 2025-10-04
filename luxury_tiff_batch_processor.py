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
from PIL import Image

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


def image_to_float(image: Image.Image) -> Tuple[np.ndarray, np.dtype, Optional[np.ndarray], int]:
    """Convert a PIL image into a float32 RGB array in the 0-1 domain."""

    supported_modes = {
        "RGB",
        "RGBA",
        "I",
        "I;16",
        "I;16L",
        "I;16B",
        "I;16S",
        "L",
        "LA",
        "F",
    }
    if image.mode not in supported_modes:
        image = image.convert("RGBA" if "A" in image.mode else "RGB")

    arr = np.array(image)
    alpha_channel: Optional[np.ndarray] = None

    if arr.ndim == 2:
        base_channels = 1
        arr = np.stack([arr] * 3, axis=-1)
    else:
        base_channels = arr.shape[2]
        if arr.shape[2] == 4:
            alpha_channel = arr[:, :, 3]
            arr = arr[:, :, :3]
            base_channels = 3
        elif arr.shape[2] == 2:
            alpha_channel = arr[:, :, 1]
            arr = np.stack([arr[:, :, 0]] * 3, axis=-1)
            base_channels = 1

    np_dtype = arr.dtype
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        min_value = float(info.min)
        max_value = float(info.max)
        scale = max_value - min_value
        if scale <= 0:
            arr_float = np.zeros_like(arr, dtype=np.float32)
        else:
            arr_float = (arr.astype(np.float32) - min_value) / scale
        if alpha_channel is not None:
            alpha_channel = (
                (alpha_channel.astype(np.float32) - min_value) / scale if scale > 0 else np.zeros_like(alpha_channel)
            )
    else:
        arr_float = arr.astype(np.float32)
        if alpha_channel is not None:
            alpha_channel = alpha_channel.astype(np.float32)

    arr_float = np.clip(arr_float, 0.0, 1.0)
    if alpha_channel is not None:
        alpha_channel = np.clip(alpha_channel, 0.0, 1.0)

    return arr_float, np_dtype, alpha_channel, base_channels


def float_to_dtype_array(
    arr: np.ndarray,
    dtype: np.dtype,
    alpha: Optional[np.ndarray],
    base_channels: Optional[int] = None,
) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    np_dtype = np.dtype(dtype)
    dtype_info = np.iinfo(np_dtype) if np.issubdtype(np_dtype, np.integer) else None
    if dtype_info:
        range_span = float(dtype_info.max) - float(dtype_info.min)
        if range_span <= 0:
            arr_scaled = np.zeros_like(arr, dtype=np.float32)
        else:
            arr_scaled = arr * range_span + float(dtype_info.min)
        arr_int = np.round(arr_scaled).astype(np_dtype)
    else:
        # Preserve floating-point sample formats (e.g. 32-bit float TIFF)
        arr_int = arr.astype(np_dtype, copy=False)

    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
        if dtype_info:
            range_span = float(dtype_info.max) - float(dtype_info.min)
            if range_span <= 0:
                alpha_scaled = np.zeros_like(alpha, dtype=np.float32)
            else:
                alpha_scaled = alpha * range_span + float(dtype_info.min)
            alpha_int = np.round(alpha_scaled).astype(np_dtype)
        else:
            alpha_int = alpha.astype(np_dtype, copy=False)
        arr_int = np.concatenate([arr_int, alpha_int[:, :, None]], axis=2)

    if base_channels is not None and arr_int.ndim == 3 and base_channels < arr_int.shape[2]:
        arr_int = arr_int[:, :, :base_channels + (1 if alpha is not None else 0)]

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



def sanitize_tiff_metadata(raw_metadata: Optional[Any]) -> Optional[Dict[int, Any]]:
    if raw_metadata is None:
        return None
    safe: Dict[int, Any] = {}
    try:
        forbidden = {256, 257, 273, 279, 322, 323, 324, 325}
        for tag in raw_metadata:
            if tag in forbidden:
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
    bits = dtype_info.bits if dtype_info else 0

    save_arr = arr_int
    if tifffile is not None:
        arr_for_tifffile = save_arr
        if arr_for_tifffile.ndim == 3 and arr_for_tifffile.shape[2] == 1:
            arr_for_tifffile = arr_for_tifffile[:, :, 0]
        tiff_kwargs = {
            "photometric": "rgb" if (arr_for_tifffile.ndim == 3 and arr_for_tifffile.shape[2] >= 3) else "minisblack",
            "compression": compression_for_tifffile(compression),
            "metadata": None,
        }
        if arr_for_tifffile.ndim == 3 and arr_for_tifffile.shape[2] > 3:
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
        tifffile.imwrite(destination, arr_for_tifffile, **tiff_kwargs)
        return

    arr_uint8: np.ndarray
    if dtype_info and bits == 16:
        LOGGER.warning(
            "Falling back to Pillow for 16-bit save; output will be 8-bit. Install 'tifffile' for full 16-bit support."
        )
        rgb = np.clip(save_arr[..., :3], 0, dtype_info.max).astype(np.float32) / (dtype_info.max / 255.0)
        rgb8 = rgb.astype(np.uint8)
        if save_arr.ndim == 3 and save_arr.shape[2] > 3:
            alpha = np.clip(save_arr[:, :, 3], 0, dtype_info.max).astype(np.float32) / (dtype_info.max / 255.0)
            alpha8 = alpha.astype(np.uint8)
            arr_uint8 = np.concatenate([rgb8, alpha8[:, :, None]], axis=2)
        else:
            arr_uint8 = rgb8
    elif np.issubdtype(np_dtype, np.floating):
        rgb = np.clip(save_arr[..., :3], 0.0, 1.0)
        rgb16 = np.round(rgb * 65535.0).astype(np.uint16)
        if save_arr.ndim == 3 and save_arr.shape[2] > 3:
            alpha = np.clip(save_arr[:, :, 3], 0.0, 1.0)
            alpha16 = np.round(alpha * 65535.0).astype(np.uint16)
            arr_uint8 = np.concatenate([rgb16, alpha16[:, :, None]], axis=2)
        else:
            arr_uint8 = rgb16
    else:
        arr_uint8 = save_arr.astype(np.uint8)

    mode = "RGB"
    if arr_uint8.ndim == 2:
        mode = "L"
    elif arr_uint8.shape[2] == 4:
        mode = "RGBA"
    elif arr_uint8.shape[2] == 1:
        mode = "L"
        arr_uint8 = arr_uint8[:, :, 0]

    image = Image.fromarray(arr_uint8, mode=mode)
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
    muted_weight = np.sqrt(np.clip(saturation, 0.0, 1.0))
    boost = amount * (1.0 - saturation) * muted_weight
    hsv[..., 1] = np.clip(saturation + boost, 0.0, 1.0)
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


def process_single_image(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    *,
    compression: str,
    resize_long_edge: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    LOGGER.info("Processing %s -> %s", source, destination)
    if destination.exists() and not dry_run:
        if not destination.is_file():
            raise ValueError(f"Destination path is not a file: {destination}")
        LOGGER.debug("Destination exists")
    with Image.open(source) as image:
        metadata = getattr(image, "tag_v2", None)
        icc_profile = image.info.get("icc_profile") if isinstance(image.info, dict) else None
        arr, dtype, alpha, base_channels = image_to_float(image)
        adjusted = apply_adjustments(arr, adjustments)
        if resize_long_edge is not None:
            adjusted = resize_long_edge_array(adjusted, resize_long_edge)
            if alpha is not None:
                alpha = resize_long_edge_array(alpha, resize_long_edge)
        arr_int = float_to_dtype_array(adjusted, dtype, alpha, base_channels)
        if dry_run:
            LOGGER.info("Dry run enabled, skipping save for %s", destination)
            return
        save_image(destination, arr_int, dtype, metadata, icc_profile, compression)


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
    adjustments = build_adjustments(args)

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input folder '{input_root}' does not exist or is not a directory")

    _ensure_non_overlapping(input_root, output_root)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    images = sorted(collect_images(input_root, args.recursive))
    if not images:
        LOGGER.warning("No TIFF images found in %s", input_root)
        return 0

    LOGGER.info("Found %s image(s) to process", len(images))
    processed = 0

    for image_path in images:
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
        process_single_image(
            image_path,
            destination,
            adjustments,
            compression=args.compression,
            resize_long_edge=args.resize_long_edge,
            dry_run=args.dry_run,
        )
        processed += 0 if args.dry_run else 1

    return processed


if __name__ == "__main__":  # pragma: no cover
    main()
