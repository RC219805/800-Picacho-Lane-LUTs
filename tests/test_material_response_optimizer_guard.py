# file: tests/test_material_response_optimizer_guard.py
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _mk_grad(size: Tuple[int, int] = (16, 16)) -> Image.Image:
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    arr = np.stack([xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def test_material_response_optimizer_public_surface_and_methods(tmp_path: Path):
    mod = importlib.import_module("material_response_optimizer")

    # Public API must expose exactly these two classes (no extras).
    expected = {"RenderEnhancementPlanner", "MaterialAwareEnhancementPlanner"}
    assert hasattr(mod, "__all__"), "material_response_optimizer must define __all__"
    assert set(mod.__all__) == expected and len(mod.__all__) == 2, f"Unexpected __all__: {mod.__all__}"

    # Fetch classes
    RenderEnhancementPlanner = getattr(mod, "RenderEnhancementPlanner")
    MaterialAwareEnhancementPlanner = getattr(mod, "MaterialAwareEnhancementPlanner")

    # Instantiate
    base = RenderEnhancementPlanner(k=1, seed=0, strength=0.2)
    aware = MaterialAwareEnhancementPlanner(k=2, seed=1, strength=0.4)

    # Methods must be callable
    assert callable(base.plan)
    assert callable(aware.apply_image)

    # __call__ must alias plan (method identity via underlying function)
    assert hasattr(base, "__call__")
    assert base.__call__.__func__ is base.plan.__func__  # type: ignore[attr-defined]

    # apply_image: minimal smoke call
    im = _mk_grad((20, 20))
    out_im = aware.apply_image(im)
    assert isinstance(out_im, Image.Image)
    assert out_im.size == im.size

    # plan: write tiny output
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.jpg"
    im.save(inp)
    base.plan(
        input_path=inp,
        output_path=outp,
        target_width=32,
        jpeg_quality=90,
        jpeg_subsampling=1,
        show_progress=False,
        respect_icc=False,
    )
    assert outp.exists() and outp.stat().st_size > 0
