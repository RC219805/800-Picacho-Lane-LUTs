# path: luxury_tiff_batch_processor/adjustments.py
"""Color adjustments, presets, and supporting image math.

All functions accept (H, W, 3) or (N, H, W, 3) float32 RGB in [0, 1].
Some steps may operate outside [0, 1]; outputs are clipped where needed.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

from .profiles import ProcessingProfile

LOGGER = logging.getLogger("luxury_tiff_batch_processor")


# --- Configuration & helpers -------------------------------------------------


def _ensure_rgb_float32(arr: np.ndarray) -> np.ndarray:
    if arr.ndim not in (3, 4) or arr.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) RGB array; got shape {arr.shape!r}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


# --- Public API --------------------------------------------------------------


@dataclasses.dataclass(slots=True)
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


def kelvin_to_rgb(temperature: float) -> np.ndarray:
    temp = temperature / 100.0
    if temp <= 0:
        temp = 0.1
    if temp <= 66:
        red = 1.0
        green = np.clip(0.3900815787690196 * math.log(temp) - 0.6318414437886275, 0, 1)
        blue = 0 if temp <= 19 else np.clip(
            0.5432067891101961 * math.log(temp - 10) - 1.19625408914, 0, 1
        )
    else:
        red = np.clip(1.2929361860627451 * (temp - 60) ** -0.1332047592, 0, 1)
        green = np.clip(1.1298908608952941 * (temp - 60) ** -0.0755148492, 0, 1)
        blue = 1.0
    return np.array([red, green, blue], dtype=np.float32)


def apply_exposure(arr: np.ndarray, stops: float) -> np.ndarray:
    if stops == 0:
        return arr
    factor = float(2.0**stops)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Applying exposure: %s stops (factor %.3f)", stops, factor)
    return arr * factor


def apply_white_balance(arr: np.ndarray, temperature: Optional[float], tint: float) -> np.ndarray:
    result = arr
    if temperature is not None:
        ref = kelvin_to_rgb(6500.0)
        target = kelvin_to_rgb(temperature)
        scale = target / ref
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Applying temperature: %sK scale=%s", temperature, scale)
        result = result * scale  # broadcast on last channel
    if tint:
        tint_scale = np.array([1.0 + tint * 0.0015, 1.0, 1.0 - tint * 0.0015], dtype=np.float32)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Applying tint scale=%s", tint_scale)
        result = result * tint_scale
    return result


def luminance(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def apply_shadow_lift(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    gamma = 1.0 / (1.0 + amount * 3.0)
    lum = luminance(arr)
    lifted = np.power(np.clip(lum, 0.0, 1.0), gamma)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Shadow lift amount=%s gamma=%.3f", amount, gamma)
    return arr + (lifted - lum)[..., None]  # type: ignore[no-any-return]


def apply_highlight_recovery(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    gamma = 1.0 + amount * 2.0
    lum = luminance(arr)
    compressed = np.power(np.clip(lum, 0.0, 1.0), gamma)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Highlight recovery amount=%s gamma=%.3f", amount, gamma)
    return arr + (compressed - lum)[..., None]  # type: ignore[no-any-return]


def apply_midtone_contrast(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return arr
    lum = luminance(arr)
    contrasted = 0.5 + (lum - 0.5) * (1.0 + amount)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Midtone contrast amount=%s", amount)
    return arr + (contrasted - lum)[..., None]  # type: ignore[no-any-return]


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    diff = maxc - minc

    hue = np.zeros_like(maxc, dtype=np.float32)
    mask = diff != 0

    rc = np.zeros_like(maxc, dtype=np.float32)
    gc = np.zeros_like(maxc, dtype=np.float32)
    bc = np.zeros_like(maxc, dtype=np.float32)

    np.copyto(rc, maxc)
    rc -= r
    np.divide(rc, diff, out=rc, where=mask)

    np.copyto(gc, maxc)
    gc -= g
    np.divide(gc, diff, out=gc, where=mask)

    np.copyto(bc, maxc)
    bc -= b
    np.divide(bc, diff, out=bc, where=mask)

    sel_r = maxc == r
    sel_g = maxc == g
    sel_b = maxc == b

    hue[sel_r] = (bc - gc)[sel_r]
    hue[sel_g] = 2.0 + (rc - bc)[sel_g]
    hue[sel_b] = 4.0 + (gc - rc)[sel_b]
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maxc, dtype=np.float32)
    non_zero = maxc != 0
    saturation[non_zero] = diff[non_zero] / maxc[non_zero]

    value = maxc.astype(np.float32, copy=False)
    return np.stack([hue, saturation, value], axis=-1)


def hsv_to_rgb(arr: np.ndarray) -> np.ndarray:
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    shape = h.shape + (3,)
    rgb = np.zeros(shape, dtype=np.float32)

    candidates = [
        (i_mod == 0, np.stack([v, t, p], axis=-1)),
        (i_mod == 1, np.stack([q, v, p], axis=-1)),
        (i_mod == 2, np.stack([p, v, t], axis=-1)),
        (i_mod == 3, np.stack([p, q, v], axis=-1)),
        (i_mod == 4, np.stack([t, p, v], axis=-1)),
        (i_mod == 5, np.stack([v, p, q], axis=-1)),
    ]
    for cond, val in candidates:
        rgb[cond] = val[cond]
    return rgb


def apply_vibrance(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return arr
    hsv = rgb_to_hsv(arr)
    saturation = hsv[..., 1]
    hsv[..., 1] = np.clip(
        saturation + amount * (1.0 - saturation) * np.sqrt(np.clip(saturation, 0.0, 1.0)),
        0.0,
        1.0,
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Vibrance amount=%s", amount)
    return hsv_to_rgb(hsv)


def apply_saturation(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return arr
    hsv = rgb_to_hsv(arr)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + amount), 0.0, 1.0)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Saturation delta=%s", amount)
    return hsv_to_rgb(hsv)


@functools.lru_cache(maxsize=32)
def _gaussian_kernel_cached(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    if radius <= 0:
        cached = np.array([1.0], dtype=np.float32)
        cached.setflags(write=False)
        return cached
    sigma = sigma or max(radius / 3.0, 1e-6)
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    cached = kernel.astype(np.float32)
    cached.setflags(write=False)
    return cached  # type: ignore[no-any-return]


def gaussian_kernel(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    return _gaussian_kernel_cached(radius, sigma).copy()  # type: ignore[no-any-return]


def gaussian_kernel_cached(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    return _gaussian_kernel_cached(radius, sigma)


gaussian_kernel.cache_clear = _gaussian_kernel_cached.cache_clear  # type: ignore[attr-defined]
gaussian_kernel.cache_info = _gaussian_kernel_cached.cache_info  # type: ignore[attr-defined]


def separable_convolve(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * arr.ndim
    k = kernel.size // 2
    pad_width[axis] = (k, k)
    padded = np.pad(arr, pad_width, mode="reflect")
    convolved = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=axis, arr=padded)
    return convolved.astype(np.float32, copy=False)


def gaussian_blur(arr: np.ndarray, radius: int, sigma: Optional[float] = None) -> np.ndarray:
    """Apply separable Gaussian blur over spatial axes (H, W)."""
    kernel = gaussian_kernel_cached(radius, sigma)
    if arr.ndim == 3:
        axes = (0, 1)
    elif arr.ndim == 4:
        axes = (1, 2)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {arr.ndim}D")
    blurred = separable_convolve(arr, kernel, axis=axes[0])
    blurred = separable_convolve(blurred, kernel, axis=axes[1])
    return blurred


def apply_clarity(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    radius = max(1, int(round(1 + amount * 5)))
    blurred = gaussian_blur(arr, radius)
    high_pass = arr - blurred
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Clarity amount=%s radius=%s", amount, radius)
    return np.clip(arr + high_pass * (0.6 + amount * 0.8), 0.0, 1.0)  # type: ignore[no-any-return]


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
    return yuv  # type: ignore[no-any-return]


def yuv_to_rgb(arr: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [
            [1.0, 0.0, 1.28033],
            [1.0, -0.21482, -0.38059],
            [1.0, 2.12798, 0.0],
        ],
        dtype=np.float32,
    )
    rgb = arr.copy()
    rgb[..., 1:] -= 0.5
    rgb = rgb @ matrix.T
    return rgb  # type: ignore[no-any-return]


def apply_chroma_denoise(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    yuv = rgb_to_yuv(arr)
    radius = max(1, int(round(1 + amount * 4)))
    for channel in (1, 2):
        channel_data = yuv[..., channel]
        blurred = gaussian_blur(channel_data[..., None], radius)[..., 0]
        yuv[..., channel] = channel_data * (1.0 - amount) + blurred * amount
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Chroma denoise amount=%s radius=%s", amount, radius)
    return np.clip(yuv_to_rgb(yuv), 0.0, 1.0)


def apply_glow(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return arr
    radius = max(2, int(round(6 + amount * 20)))
    softened = gaussian_blur(arr, radius)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Glow amount=%s radius=%s", amount, radius)
    return np.clip(arr * (1.0 - amount) + softened * amount, 0.0, 1.0)


def apply_adjustments(
    arr: np.ndarray,
    adjustments: AdjustmentSettings,
    *,
    profile: ProcessingProfile | None = None,
) -> np.ndarray:
    """Apply adjustments to (H,W,3) or (N,H,W,3); returns float32 clipped to [0,1]."""
    arr = _ensure_rgb_float32(arr)

    arr = apply_white_balance(arr, adjustments.white_balance_temp, adjustments.white_balance_tint)
    arr = apply_exposure(arr, adjustments.exposure)
    arr = apply_shadow_lift(arr, adjustments.shadow_lift)
    arr = apply_highlight_recovery(arr, adjustments.highlight_recovery)
    arr = apply_midtone_contrast(arr, adjustments.midtone_contrast)
    arr = np.clip(arr, 0.0, 1.0)

    chroma_amount = adjustments.chroma_denoise
    glow_amount = adjustments.glow
    if profile is not None:
        chroma_amount = profile.resolve_chroma_denoise(chroma_amount)
        glow_amount = profile.resolve_glow(glow_amount)

    arr = apply_chroma_denoise(arr, chroma_amount)
    arr = apply_vibrance(arr, adjustments.vibrance)
    arr = apply_saturation(arr, adjustments.saturation)
    arr = apply_clarity(arr, adjustments.clarity)
    arr = apply_glow(arr, glow_amount)
    return np.clip(arr, 0.0, 1.0)


# --- Batch API ---------------------------------------------------------------


def _apply_adjustments_single(
    payload: Tuple[int, np.ndarray, AdjustmentSettings, Optional[ProcessingProfile]]
) -> Tuple[int, np.ndarray]:
    idx, img, adj, prof = payload
    out = apply_adjustments(img, adj, profile=prof)
    return idx, out


def batch_apply_adjustments(
    arr: np.ndarray,
    adjustments: AdjustmentSettings | Sequence[AdjustmentSettings],
    *,
    profile: ProcessingProfile | None = None,
    method: Literal["auto", "vectorized", "multiprocessing"] = "auto",
    workers: Optional[int] = None,
) -> np.ndarray:
    """Batch processing for (N,H,W,3).

    - Single `AdjustmentSettings` → vectorized 4D processing.
    - Sequence of per-frame settings → serial or multiprocessing per image.

    Parameters
    ----------
    arr : np.ndarray
        Input array shaped (N,H,W,3), float32 preferred.
    adjustments : AdjustmentSettings | Sequence[AdjustmentSettings]
        Single settings for all frames, or per-frame sequence of length N.
    profile : ProcessingProfile | None
        Optional profile to modulate strengths.
    method : {"auto","vectorized","multiprocessing"}
        Strategy selection. "auto" picks "multiprocessing" when N is large.
    workers : int | None
        Process count for multiprocessing. Defaults to CPU count.

    Returns
    -------
    np.ndarray
        Output array (N,H,W,3), float32 clipped to [0,1].
    """
    arr = _ensure_rgb_float32(arr)
    if arr.ndim != 4:
        raise ValueError(f"`arr` must be 4D (N,H,W,3); got shape {arr.shape!r}")
    n = arr.shape[0]

    if isinstance(adjustments, AdjustmentSettings):
        if method not in ("auto", "vectorized"):
            # Why: single settings benefits most from true 4D vectorized path.
            LOGGER.debug("Overriding method=%s to 'vectorized' for single settings", method)
        return apply_adjustments(arr, adjustments, profile=profile)

    if not isinstance(adjustments, Sequence) or len(adjustments) != n:
        raise ValueError("When passing a sequence of adjustments, its length must equal N")

    if method == "vectorized":
        # Not feasible to vectorize different presets efficiently; do serial.
        method = "auto"

    if method == "auto":
        threshold = 8
        method = "multiprocessing" if n >= threshold else "vectorized"

    if method == "multiprocessing":
        max_workers = workers or max(1, (os.cpu_count() or 1))
        tasks: Iterable[Tuple[int, np.ndarray, AdjustmentSettings, Optional[ProcessingProfile]]] = (
            (i, arr[i], adjustments[i], profile) for i in range(n)
        )
        results: list[Optional[np.ndarray]] = [None] * n
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_apply_adjustments_single, t) for t in tasks]
                for fut in as_completed(futs):
                    idx, out = fut.result()
                    results[idx] = out
        except Exception as exc:
            LOGGER.warning("Multiprocessing failed (%s); falling back to serial.", exc)
            results = [apply_adjustments(arr[i], adjustments[i], profile=profile) for i in range(n)]
        return np.stack([r for r in results if r is not None], axis=0).astype(np.float32, copy=False)

    # "vectorized" here means efficient serial loop (different presets).
    outputs = [apply_adjustments(arr[i], adjustments[i], profile=profile) for i in range(n)]
    return np.stack(outputs, axis=0).astype(np.float32, copy=False)


__all__ = [
    "AdjustmentSettings",
    "LUXURY_PRESETS",
    "apply_adjustments",
    "apply_chroma_denoise",
    "apply_clarity",
    "apply_exposure",
    "apply_glow",
    "apply_highlight_recovery",
    "apply_midtone_contrast",
    "apply_saturation",
    "apply_shadow_lift",
    "apply_vibrance",
    "apply_white_balance",
    "gaussian_blur",
    "gaussian_kernel",
    "gaussian_kernel_cached",
    "kelvin_to_rgb",
    "luminance",
    "rgb_to_hsv",
    "hsv_to_rgb",
    "batch_apply_adjustments",
]