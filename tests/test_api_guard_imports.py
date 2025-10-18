# file: tests/test_api_guard_imports.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# Shim and new API must both be importable
from evolutionary_checkpoint import EvolutionaryCheckpoint as EC_legacy, EvolutionStatus as ES_legacy
from src.evolutionary import EvolutionaryCheckpoint as EC_new, EvolutionStatus as ES_new

# Planners module surface must exist
from material_response_optimizer import (
    RenderEnhancementPlanner,
    MaterialAwareEnhancementPlanner,
)


def _mk_grad(size: Tuple[int, int] = (16, 16)) -> Image.Image:
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    arr = np.stack([xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def test_evolutionary_shim_points_to_new_types():
    assert EC_legacy is EC_new
    assert ES_legacy is ES_new
    chk = EC_legacy(horizon=date(2025, 12, 31), mutation_path="migrate.md")
    out = chk.evaluate(today=date(2025, 12, 30))
    assert out.status is ES_new.STABLE


def test_planners_import_and_minimal_ops(tmp_path: Path):
    # in-memory path
    aware = MaterialAwareEnhancementPlanner(k=2, seed=1, strength=0.4)
    im = _mk_grad()
    out_im = aware.apply_image(im)
    assert isinstance(out_im, Image.Image)
    assert out_im.size == im.size

    # file path
    base = RenderEnhancementPlanner(k=1, seed=0, strength=0.2)
    inp = tmp_path / "in.png"
    outp = tmp_path / "out.jpg"
    _mk_grad((20, 20)).save(inp)
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
