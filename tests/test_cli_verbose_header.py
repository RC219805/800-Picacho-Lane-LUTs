# file: tests/test_cli_verbose_header.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import pytest

import board_material_aerial_enhancer as mbar


def _mk_grad(size: Tuple[int, int] = (24, 24)) -> Image.Image:
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    arr = np.stack([xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(arr, mode="RGB")


@pytest.mark.parametrize("verbose_flag, expect_icc_line", [(False, False), (True, True)])
def test_cli_verbose_header_toggles(tmp_path: Path, verbose_flag: bool, expect_icc_line: bool):
    # Use Typer's runner against mbar.main (works since it executes the app internally).
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    inp = tmp_path / "in.png"
    out = tmp_path / "out.jpg"
    _mk_grad().save(inp)

    args = [
        "enhance",
        "--input",
        str(inp),
        "--output",
        str(out),
        "--analysis-max-dim",
        "32",
        "--k",
        "1",
        "--seed",
        "0",
        "--target-width",
        "32",
        "--strength",
        "0.1",
        "--jpeg-quality",
        "90",
        "--no-progress",
        "--respect-icc",
    ]
    if verbose_flag:
        args.append("--verbose-header")

    runner = CliRunner()
    result = runner.invoke(mbar.main, args)
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 0

    # Legacy header always present
    assert "Resolution: 4K (4096px width)" in result.output

    # Verbose line only when flag is passed
    assert ("ICC handling:" in result.output) is expect_icc_line
