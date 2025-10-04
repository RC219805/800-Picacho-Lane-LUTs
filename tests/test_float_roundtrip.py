from pathlib import Path
import sys
from typing import Optional

import numpy as np
from PIL import Image

try:
    import tifffile
except Exception:  # pragma: no cover - optional dependency
    tifffile = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luxury_tiff_batch_processor import (
    float_to_dtype_array,
    gaussian_blur,
    gaussian_kernel,
    image_to_float,
    save_image,
)


def test_float_to_dtype_array_preserves_float_values():
    gradient = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
    rgb = np.stack([gradient, gradient ** 2, np.sqrt(gradient)], axis=-1)
    result = float_to_dtype_array(rgb, np.float32, None)
    assert result.dtype == np.float32
    assert np.allclose(result, rgb)


def test_save_image_retains_float_tonal_range(tmp_path):
    width, height = 32, 16
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    gradient = (x + y) / 2.0
    rgb = np.stack([gradient, gradient ** 1.5, np.clip(gradient * 1.2, 0.0, 1.0)], axis=-1)

    float_data = float_to_dtype_array(rgb, np.float32, None)
    output_path = tmp_path / "float_image.tiff"
    save_image(output_path, float_data, np.dtype(np.float32), metadata=None, icc_profile=None, compression="deflate")

    if tifffile is not None:
        saved_array = tifffile.imread(output_path)
    else:
        with Image.open(output_path) as saved:
            saved_array = np.array(saved)

    flat = saved_array.reshape(-1)
    assert np.unique(flat).size > 10


def test_image_to_float_roundtrip_signed_int_image():
    gradient = np.linspace(-5000, 5000, 49, dtype=np.int32).reshape(7, 7)
    image = Image.fromarray(gradient)

    rgb_float, dtype, alpha = image_to_float(image)

    assert dtype == np.int32
    assert alpha is None
    assert rgb_float.dtype == np.float32
    assert np.all((rgb_float >= 0.0) & (rgb_float <= 1.0))

    restored = float_to_dtype_array(rgb_float, dtype, alpha)
    assert restored.dtype == np.int32
    max_diff = np.max(np.abs(restored[:, :, 0] - gradient))
    assert max_diff <= 256
    for channel in range(1, restored.shape[2]):
        assert np.array_equal(restored[:, :, channel], restored[:, :, 0])


def _reference_gaussian_blur(arr: np.ndarray, radius: int, sigma: Optional[float] = None) -> np.ndarray:
    kernel = gaussian_kernel(radius, sigma)

    working = arr
    squeeze = False
    if working.ndim == 2:
        working = working[:, :, None]
        squeeze = True

    pad = kernel.size // 2
    padded = np.pad(working, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    vertical = np.empty_like(working, dtype=np.float32)
    for y in range(working.shape[0]):
        window = padded[y : y + kernel.size]
        vertical[y] = np.tensordot(kernel, window, axes=(0, 0))

    padded_h = np.pad(vertical, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    blurred = np.empty_like(working, dtype=np.float32)
    for x in range(working.shape[1]):
        window = padded_h[:, x : x + kernel.size]
        blurred[:, x] = np.tensordot(kernel, window, axes=(0, 1))

    if squeeze:
        return blurred[:, :, 0]
    return blurred


def test_gaussian_blur_matches_reference():
    rng = np.random.default_rng(42)
    data = rng.random((12, 10, 3), dtype=np.float32)
    radius = 3

    optimised = gaussian_blur(data, radius)
    reference = _reference_gaussian_blur(data, radius)

    assert np.allclose(optimised, reference, atol=1e-6)
