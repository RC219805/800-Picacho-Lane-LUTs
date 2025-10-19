# file: tests/test_board_material_aerial_enhancer.py
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import board_material_aerial_enhancer as mbar


# ------------------------------ helpers ------------------------------------

def _mk_img(color=(64, 128, 96), size=(32, 32)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

def _mk_grad(size=(64, 64)) -> Image.Image:
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    arr = np.stack([xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(arr, mode="RGB")

def _open_rgb01(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0

def _delta_e_ab(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    lab1 = mbar._rgb_to_lab(a01)
    lab2 = mbar._rgb_to_lab(b01)
    d = lab1 - lab2
    return np.sqrt(np.sum(d * d, axis=-1))


# ------------------------------- tests -------------------------------------

def test_enhance_flat_color_k1_low_variance(tmp_path: Path):
    im = _mk_img(color=(80, 160, 120), size=(48, 48))
    inp = tmp_path / "in.png"
    out = tmp_path / "out.jpg"
    im.save(inp)
    mbar.enhance_aerial(
        input_path=inp,
        output_path=out,
        analysis_max_dim=64,
        k=1,
        seed=0,
        target_width=128,
        strength=0.8,
        jpeg_quality=100,
        jpeg_subsampling=0,
        show_progress=False,
        respect_icc=True,
        palette=None,
    )
    assert out.exists() and out.stat().st_size > 0
    oarr = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32)
    assert oarr.std() < 8.0  # mostly uniform even after JPEG

def test_kmeans_degenerate_no_crash():
    X = np.tile(np.array([[0.33, 0.66, 0.10]], dtype=np.float64), (200, 1))
    res = mbar._kmeans(X, k=3, seed=42)
    assert res.centers.shape == (3, 3)
    assert res.inertia == pytest.approx(0.0, abs=1e-9)

def test_tiny_image_k1(tmp_path: Path):
    im = _mk_img(color=(10, 10, 10), size=(1, 1))
    inp = tmp_path / "tiny.png"
    out = tmp_path / "tiny_out.jpg"
    im.save(inp)
    mbar.enhance_aerial(
        inp,
        out,
        analysis_max_dim=64,
        k=1,
        seed=1,
        target_width=32,
        strength=0.5,
        jpeg_quality=95,
        show_progress=False,
    )
    assert out.exists() and out.stat().st_size > 0

def test_invalid_palette_hex_raises(tmp_path: Path):
    im = _mk_img()
    inp = tmp_path / "in.png"
    out = tmp_path / "out.jpg"
    im.save(inp)
    with pytest.raises(ValueError):
        mbar.enhance_aerial(inp, out, palette=["#GG0000"], show_progress=False)

def test_cli_smoke(tmp_path: Path):
    typer = pytest.importorskip("typer")
    from typer.testing import CliRunner

    inp = tmp_path / "cli_in.png"
    out = tmp_path / "cli_out.jpg"
    _mk_grad().save(inp)

    runner = CliRunner()
    result = runner.invoke(
        mbar.main,
        [
            "enhance",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--analysis-max-dim",
            "64",
            "--k",
            "2",
            "--seed",
            "7",
            "--target-width",
            "128",
            "--strength",
            "0.7",
            "--jpeg-quality",
            "95",
            "--respect-icc",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 0

def test_handles_embedded_icc_without_crashing(tmp_path: Path):
    ImageCms = pytest.importorskip("PIL.ImageCms")
    im = _mk_grad(size=(48, 48))
    srgb = ImageCms.createProfile("sRGB")
    icc_bytes = ImageCms.ImageCmsProfile(srgb).tobytes()
    im.info["icc_profile"] = icc_bytes
    inp = tmp_path / "icc_in.jpg"
    im.save(inp, quality=95, icc_profile=icc_bytes)

    out_respect = tmp_path / "icc_out_respect.jpg"
    out_ignore = tmp_path / "icc_out_ignore.jpg"
    mbar.enhance_aerial(inp, out_respect, analysis_max_dim=64, k=2, seed=3, target_width=96,
                        strength=0.6, jpeg_quality=95, show_progress=False, respect_icc=True)
    mbar.enhance_aerial(inp, out_ignore, analysis_max_dim=64, k=2, seed=3, target_width=96,
                        strength=0.6, jpeg_quality=95, show_progress=False, respect_icc=False)
    assert out_respect.exists() and out_ignore.exists()

# --------------------------- golden ΔE*ab tests ----------------------------

def _golden_dir() -> Path:
    return Path(__file__).parent / "golden"

def _ensure_or_skip_golden(source_out: Path, golden_name: str):
    gdir = _golden_dir()
    gpath = gdir / golden_name
    update = os.getenv("UPDATE_GOLDEN", "") == "1"
    if not gpath.exists():
        if update:
            gdir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_out, gpath)
            pytest.skip(f"Golden created: {gpath.name}")
        pytest.skip(f"Golden missing: {gpath.name}. Set UPDATE_GOLDEN=1 to create.")
    return gpath

@pytest.mark.parametrize(
    "name,make,params",
    [
        ("grad", _mk_grad, dict(analysis_max_dim=64, k=3, seed=11, target_width=128, strength=0.75)),
        ("flat", lambda: _mk_img(color=(90, 140, 120), size=(64, 64)), dict(analysis_max_dim=64, k=1, seed=2, target_width=128, strength=0.9)),
    ],
)
def test_golden_delta_e_ab(tmp_path: Path, name, make, params):
    inp = tmp_path / f"{name}_in.png"
    out = tmp_path / f"{name}_out.jpg"
    make().save(inp)
    mbar.enhance_aerial(
        inp,
        out,
        jpeg_quality=100,
        jpeg_subsampling=0,
        show_progress=False,
        respect_icc=True,
        **params,
    )
    gpath = _ensure_or_skip_golden(out, f"{name}_golden.jpg")

    a = _open_rgb01(out)
    b = _open_rgb01(gpath)
    assert a.shape == b.shape
    dE = _delta_e_ab(a, b)
    mean_dE = float(dE.mean())
    max_dE = float(dE.max())
    # JPEG/library drift guard; keep thresholds tight but resilient.
    assert mean_dE <= 1.0, f"mean ΔE too high: {mean_dE:.3f}"
    assert max_dE <= 3.0, f"max ΔE too high: {max_dE:.3f}"
