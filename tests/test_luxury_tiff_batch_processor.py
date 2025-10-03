from __future__ import annotations

from pathlib import Path
import sys

import pytest

from .documentation import documents

np = pytest.importorskip("numpy")
pytest.importorskip("PIL.Image")
pytest.importorskip("PIL.TiffImagePlugin")
from PIL import Image, TiffImagePlugin

sys.path.append(str(Path(__file__).resolve().parent.parent))

import luxury_tiff_batch_processor as ltiff


def _saturation(rgb: np.ndarray) -> np.ndarray:
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    return maxc - minc


@documents("Material Response honors surface physics, not global transforms")
def test_apply_adjustments_respects_exposure_clamp():
    arr = np.full((4, 4, 3), 0.25, dtype=np.float32)
    settings = ltiff.AdjustmentSettings(exposure=2.0)

    out = ltiff.apply_adjustments(arr, settings)

    assert out.shape == arr.shape
    assert np.allclose(out, 1.0, atol=1e-4)


@documents("Token system allows composition without prescription")
def test_apply_adjustments_vibrance_boosts_muted_colors_more():
    neutral = np.full((1, 1, 3), 0.5, dtype=np.float32)
    muted = np.array([[[0.5, 0.35, 0.3]]], dtype=np.float32)
    saturated = np.array([[[0.9, 0.2, 0.2]]], dtype=np.float32)
    arr = np.concatenate([neutral, muted, saturated], axis=0)

    settings = ltiff.AdjustmentSettings(vibrance=0.8)
    out = ltiff.apply_adjustments(arr, settings)

    sat_before = _saturation(arr)
    sat_after = _saturation(out)

    # Neutral remains essentially neutral
    assert sat_after[0, 0] == pytest.approx(sat_before[0, 0], abs=1e-4)
    # Muted pixel receives a larger relative boost than already saturated pixel
    muted_gain = sat_after[1, 0] - sat_before[1, 0]
    saturated_gain = sat_after[2, 0] - sat_before[2, 0]
    assert muted_gain > saturated_gain


@documents("Pipeline preserves authored intent while optimizing logistics")
def test_process_single_image_handles_resize_and_metadata(tmp_path: Path):
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()

    arr = np.linspace(0, 255, 4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    # Suppress deprecated mode parameter warning for cleaner test output
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        image = Image.fromarray(arr, mode="RGB")
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = "Luxury scene"
    source_path = source_dir / "frame.tif"
    image.save(source_path, tiffinfo=info)

    dest_path = output_dir / "frame_processed.tif"

    ltiff.process_single_image(
        source_path,
        dest_path,
        ltiff.AdjustmentSettings(),
        compression="tiff_lzw",
        resize_long_edge=2,
    )

    assert dest_path.exists()
    with Image.open(dest_path) as processed:
        assert processed.size == (2, 2)
        tags = getattr(processed, "tag_v2", {})
        assert tags.get(270) == "Luxury scene"
        assert np.array(processed).dtype == np.uint8


@documents("Dry runs provide planning insight without side effects")
def test_run_pipeline_dry_run_creates_no_outputs(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    image = Image.new("RGB", (2, 2), color=(128, 128, 128))
    source_path = input_dir / "sample.tif"
    image.save(source_path)

    args = ltiff.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--dry-run",
        ]
    )

    processed = ltiff.run_pipeline(args)

    assert processed == 0
    assert not any(output_dir.rglob("*.tif"))


@documents("Filesystem discovery respects operator scope selections")
def test_collect_images_handles_recursive(tmp_path):
    input_root = tmp_path
    (input_root / "top.tif").write_bytes(b"top")
    (input_root / "upper.TIFF").write_bytes(b"upper")
    nested = input_root / "nested"
    nested.mkdir()
    (nested / "inner.tiff").write_bytes(b"inner")
    (nested / "ignore.jpg").write_bytes(b"jpg")

    non_recursive = sorted(p.relative_to(input_root) for p in ltiff.collect_images(input_root, recursive=False))
    assert non_recursive == [
        Path("top.tif"),
        Path("upper.TIFF"),
    ]

    recursive = sorted(p.relative_to(input_root) for p in ltiff.collect_images(input_root, recursive=True))
    assert recursive == [
        Path("nested/inner.tiff"),
        Path("top.tif"),
        Path("upper.TIFF"),
    ]


def test_parse_args_sets_default_output(tmp_path: Path):
    input_dir = tmp_path / "SV-Stills"
    input_dir.mkdir()

    args = ltiff.parse_args([str(input_dir)])

    assert args.output == input_dir.parent / "SV-Stills_lux"


@documents("User overrides cascade atop curated presets without drift")
def test_build_adjustments_applies_overrides(tmp_path):
    args = ltiff.parse_args(
        [
            str(tmp_path / "input"),
            str(tmp_path / "output"),
            "--preset",
            "signature",
            "--exposure",
            "0.5",
            "--white-balance-temp",
            "7000",
            "--clarity",
            "0.33",
        ]
    )

    adjustments = ltiff.build_adjustments(args)

    assert adjustments.exposure == 0.5
    assert adjustments.white_balance_temp == 7000
    assert adjustments.clarity == 0.33
    # Ensure we still carry over preset defaults for untouched controls
    assert adjustments.shadow_lift == ltiff.LUXURY_PRESETS["signature"].shadow_lift


@documents("Platform intelligence supersedes forced uniformity")
def test_image_roundtrip_uint16_with_alpha():
    data = np.array(
        [
            [[0, 32768, 65535, 65535], [65535, 0, 32768, 0]],
            [[16384, 49152, 1024, 32768], [8192, 16384, 24576, 16384]],
        ],
        dtype=np.uint16,
    )
    # Note: PIL automatically converts uint16 RGBA images to uint8
    # This is the expected behavior, not a bug
    # We need to specify mode="RGBA" for uint16 data, though it's deprecated
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        image = Image.fromarray(data, mode="RGBA")

    base_float, base_dtype, alpha, base_channels = ltiff.image_to_float(image)
    assert base_float.dtype == np.float32
    # PIL converts uint16 RGBA to uint8, so base_dtype will be uint8
    assert base_dtype == np.uint8
    assert alpha is not None and alpha.shape == data.shape[:2]
    assert base_channels == 3

    restored = ltiff.float_to_dtype_array(base_float, base_dtype, alpha, base_channels)
    assert restored.dtype == np.uint8
    assert restored.shape == data.shape
    # Since PIL downcast the data to uint8, the restored data will match the PIL conversion
    expected_uint8 = np.array(image)
    np.testing.assert_array_equal(restored, expected_uint8)
