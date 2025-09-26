from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image, TiffImagePlugin
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from luxury_tiff_batch_processor import (
    AdjustmentSettings,
    apply_adjustments,
    parse_args,
    process_single_image,
    run_pipeline,
)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    return maxc - minc


def test_apply_adjustments_respects_exposure_clamp():
    arr = np.full((4, 4, 3), 0.25, dtype=np.float32)
    settings = AdjustmentSettings(exposure=2.0)

    out = apply_adjustments(arr, settings)

    assert out.shape == arr.shape
    assert np.allclose(out, 1.0, atol=1e-4)


def test_apply_adjustments_vibrance_boosts_muted_colors_more():
    neutral = np.full((1, 1, 3), 0.5, dtype=np.float32)
    muted = np.array([[[0.5, 0.35, 0.3]]], dtype=np.float32)
    saturated = np.array([[[0.9, 0.2, 0.2]]], dtype=np.float32)
    arr = np.concatenate([neutral, muted, saturated], axis=0)

    settings = AdjustmentSettings(vibrance=0.8)
    out = apply_adjustments(arr, settings)

    sat_before = _saturation(arr)
    sat_after = _saturation(out)

    # Neutral remains essentially neutral
    assert sat_after[0, 0] == pytest.approx(sat_before[0, 0], abs=1e-4)
    # Muted pixel receives a larger relative boost than already saturated pixel
    muted_gain = sat_after[1, 0] - sat_before[1, 0]
    saturated_gain = sat_after[2, 0] - sat_before[2, 0]
    assert muted_gain > saturated_gain


def test_process_single_image_handles_resize_and_metadata(tmp_path: Path):
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()

    arr = np.linspace(0, 255, 4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    image = Image.fromarray(arr, mode="RGB")
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = "Luxury scene"
    source_path = source_dir / "frame.tif"
    image.save(source_path, tiffinfo=info)

    dest_path = output_dir / "frame_processed.tif"

    process_single_image(
        source_path,
        dest_path,
        AdjustmentSettings(),
        compression="tiff_lzw",
        resize_long_edge=2,
    )

    assert dest_path.exists()
    with Image.open(dest_path) as processed:
        assert processed.size == (2, 2)
        tags = getattr(processed, "tag_v2", {})
        assert tags.get(270) == "Luxury scene"
        assert np.array(processed).dtype == np.uint8


def test_run_pipeline_dry_run_creates_no_outputs(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    image = Image.new("RGB", (2, 2), color=(128, 128, 128))
    source_path = input_dir / "sample.tif"
    image.save(source_path)

    args = parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--dry-run",
        ]
    )

    processed = run_pipeline(args)

    assert processed == 0
    assert not any(output_dir.rglob("*.tif"))

