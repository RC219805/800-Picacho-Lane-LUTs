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
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

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


def _normalise_to_unit_range(array: np.ndarray) -> np.ndarray:
    """Return ``array`` scaled to the [0, 1] range as ``float32``."""

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


def image_to_float(image: Image.Image) -> Tuple[np.ndarray, np.dtype, Optional[np.ndarray]]:
    """Converts a PIL image to an RGB float32 array in the 0-1 range."""

    if image.mode not in {"RGB", "RGBA", "I", "I;16", "I;16L", "I;16B", "I;16S", "L", "LA"}:
        image = image.convert("RGBA" if "A" in image.mode else "RGB")

    arr = np.array(image)
    alpha_channel: Optional[np.ndarray] = None

    if arr.ndim == 2:
        base = arr[:, :, None]
    elif arr.shape[2] == 1:
        base = arr
    elif arr.shape[2] == 2:
        base = arr[:, :, :1]
        alpha_channel = arr[:, :, 1]
    else:
        base = arr[:, :, :3]
        if arr.shape[2] > 3:
            alpha_channel = arr[:, :, 3]

    base_float = _normalise_to_unit_range(base)
    if base_float.shape[2] == 1:
        base_float = np.repeat(base_float, 3, axis=2)

    alpha_float: Optional[np.ndarray]
    if alpha_channel is not None:
        alpha_float = _normalise_to_unit_range(alpha_channel)
    else:
        alpha_float = None

    return base_float.astype(np.float32, copy=False), arr.dtype, alpha_float


def float_to_dtype_array(
    arr: np.ndarray,
    dtype: np.dtype,
    alpha: Optional[np.ndarray],
) -> np.ndarray:
    """Convert a float array in ``[0, 1]`` into the requested dtype."""

    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("float_to_dtype_array expects a floating point source array")

    arr = np.clip(arr, 0.0, 1.0)
    np_dtype = np.dtype(dtype)

    if np.issubdtype(np_dtype, np.floating):
        result = arr.astype(np_dtype, copy=False)
        if alpha is None:
            return np.ascontiguousarray(result)

        alpha_arr = np.clip(alpha, 0.0, 1.0).astype(np_dtype, copy=False)
        if result.ndim == 2:
            result = result[:, :, None]
        if alpha_arr.ndim == 2:
            alpha_arr = alpha_arr[:, :, None]
        combined = np.concatenate([result, alpha_arr], axis=2)
        return np.ascontiguousarray(combined)

    if not np.issubdtype(np_dtype, np.integer):
        raise TypeError(f"Unsupported dtype: {np_dtype}")

    dtype_info = np.iinfo(np_dtype)
    normalised = np.clip(arr, 0.0, 1.0)
    if dtype_info.min < 0:
        scaled = normalised * (dtype_info.max - dtype_info.min) + dtype_info.min
    else:
        scaled = normalised * dtype_info.max
    scaled = np.clip(scaled, dtype_info.min, dtype_info.max)
    rounded = np.floor(scaled + 0.5, dtype=np.float64).astype(np_dtype)

    if alpha is None:
        return np.ascontiguousarray(rounded)

    alpha_arr = np.clip(alpha, 0.0, 1.0)
    if dtype_info.min < 0:
        alpha_scaled = alpha_arr * (dtype_info.max - dtype_info.min) + dtype_info.min
    else:
        alpha_scaled = alpha_arr * dtype_info.max
    alpha_scaled = np.clip(alpha_scaled, dtype_info.min, dtype_info.max)
    alpha_int = np.floor(alpha_scaled + 0.5, dtype=np.float64).astype(np_dtype)

    rounded_base = rounded
    if rounded_base.ndim == 2:
        rounded_base = rounded_base[:, :, None]
    if alpha_int.ndim == 2:
        alpha_int = alpha_int[:, :, None]
    combined = np.concatenate([rounded_base, alpha_int], axis=2)
    return np.ascontiguousarray(combined)


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


def _write_with_tifffile(
    destination: Path,
    array: np.ndarray,
    metadata,
    icc_profile: Optional[bytes],
    compression: str,
) -> None:
    array_for_write, photometric, extrasamples = _prepare_tifffile_image(array)
    tiff_kwargs = {
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
        try:
            tiff_kwargs["metadata"] = {tag: metadata[tag] for tag in metadata}
        except Exception:  # pragma: no cover - best-effort metadata copy
            LOGGER.debug("Unable to serialise TIFF metadata", exc_info=True)
    if extratags:
        tiff_kwargs["extratags"] = extratags
    tifffile.imwrite(destination, array_for_write, **tiff_kwargs)


def save_image(
    destination: Path,
    arr_int: np.ndarray,
    dtype: np.dtype,
    metadata,
    icc_profile: Optional[bytes],
    compression: str,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    target_dtype = np.dtype(dtype)
    array = np.ascontiguousarray(arr_int)

    if np.issubdtype(target_dtype, np.floating):
        if tifffile is not None:
            _write_with_tifffile(destination, array, metadata, icc_profile, compression)
            return
        LOGGER.warning(
            "Saving floating point TIFF without 'tifffile'; converting to 16-bit before writing with Pillow."
        )
        base, alpha_channel = _split_alpha_channel(array)
        array = float_to_dtype_array(base.astype(np.float32, copy=False), np.uint16, alpha_channel)
        target_dtype = array.dtype

    dtype_info = np.iinfo(target_dtype) if np.issubdtype(target_dtype, np.integer) else None

    if dtype_info and dtype_info.bits > 8 and tifffile is not None:
        _write_with_tifffile(destination, array, metadata, icc_profile, compression)
        return

    pillow_array = array
    pillow_mode = None
    if pillow_array.ndim == 3:
        channels = pillow_array.shape[2]
        if channels == 1:
            pillow_array = pillow_array[:, :, 0]
        elif channels == 2:
            pillow_mode = "LA"
        elif channels == 3:
            pillow_mode = "RGB"
        elif channels == 4:
            pillow_mode = "RGBA"
        else:
            raise ValueError("Unsupported channel count for Pillow save")

    pillow_compression = None
    if compression == "deflate":
        pillow_compression = "tiff_deflate"
    elif compression is not None:
        pillow_compression = compression

    image = Image.fromarray(pillow_array, mode=pillow_mode)
    save_kwargs = {}
    if pillow_compression is not None:
        save_kwargs["compression"] = pillow_compression
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
    hsv[..., 1] = np.clip(saturation + amount * (1.0 - saturation) * saturation, 0.0, 1.0)
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
    kernel = kernel.astype(np.float32, copy=False)
    squeeze_channel = False
    working = arr
    if working.ndim == 2:
        working = working[:, :, None]
        squeeze_channel = True

    moved = np.moveaxis(working, axis, 0)
    k = kernel.size // 2
    pad_shape = ((k, k),) + ((0, 0),) * (moved.ndim - 1)
    padded = np.pad(moved, pad_shape, mode="reflect")
    windows = sliding_window_view(padded, window_shape=kernel.size, axis=0)
    convolved = np.tensordot(windows, kernel, axes=([windows.ndim - 1], [0]))
    convolved = np.moveaxis(convolved, 0, axis)

    if squeeze_channel:
        convolved = convolved[:, :, 0]

    return convolved.astype(np.float32, copy=False)


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


def process_image(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    compression: str,
    resize_target: Optional[int],
    dry_run: bool,
) -> None:
    LOGGER.info("Processing %s -> %s", source, destination)
    if destination.exists() and not dry_run:
        if not destination.is_file():
            raise ValueError(f"Destination path is not a file: {destination}")
        LOGGER.debug("Destination exists")
    with Image.open(source) as image:
        metadata = getattr(image, "tag_v2", None)
        icc_profile = image.info.get("icc_profile") if isinstance(image.info, dict) else None
        arr, dtype, alpha = image_to_float(image)
        adjusted = apply_adjustments(arr, adjustments)
        if resize_target is not None:
            adjusted = resize_long_edge_array(adjusted, resize_target)
            if alpha is not None:
                alpha = resize_long_edge_array(alpha, resize_target)
        arr_int = float_to_dtype_array(adjusted, dtype, alpha)
        if dry_run:
            LOGGER.info("Dry run enabled, skipping save for %s", destination)
            return
        save_image(destination, arr_int, dtype, metadata, icc_profile, compression)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    adjustments = build_adjustments(args)

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    images = sorted(collect_images(input_root, args.recursive))
    if not images:
        LOGGER.warning("No TIFF images found in %s", input_root)
        return

    LOGGER.info("Found %s image(s) to process", len(images))

    for image_path in images:
        destination = ensure_output_path(input_root, output_root, image_path, args.suffix, args.recursive)
        if destination.exists() and not args.overwrite and not args.dry_run:
            LOGGER.warning("Skipping %s (exists, use --overwrite to replace)", destination)
            continue
        process_image(
            image_path,
            destination,
            adjustments,
            args.compression,
            args.resize_long_edge,
            args.dry_run,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
