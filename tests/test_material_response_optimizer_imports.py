# file: tests/test_material_response_optimizer_imports.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from material_response_optimizer import (
    RenderEnhancementPlanner,
    MaterialAwareEnhancementPlanner,
)


def _mk_grad(size: Tuple[int, int] = (32, 32)) -> Image.Image:
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    arr = np.stack([xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def test_imports_exist_and_are_callable():
    base = RenderEnhancementPlanner()
    aware = MaterialAwareEnhancementPlanner()
    assert hasattr(base, "plan") and callable(base.plan)
    assert hasattr(aware, "apply_image") and callable(aware.apply_image)


def test_apply_image_roundtrip_small_image():
    aware = MaterialAwareEnhancementPlanner(k=2, seed=7, strength=0.5)
    im = _mk_grad((24, 24))
    out = aware.apply_image(im)
    assert isinstance(out, Image.Image)
    assert out.size == im.size
    assert out.mode == "RGB"


def test_plan_writes_output(tmp_path: Path):
    base = RenderEnhancementPlanner(k=1, seed=0, strength=0.2)
    inp = tmp_path / "in.png"
    out = tmp_path / "out.jpg"
    _mk_grad((24, 24)).save(inp)
    base.plan(
        input_path=inp,
        output_path=out,
        target_width=32,
        jpeg_quality=90,
        jpeg_subsampling=1,
        show_progress=False,
        respect_icc=False,
    )
    assert out.exists() and out.stat().st_size > 0
