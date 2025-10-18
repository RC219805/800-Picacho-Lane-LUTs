# file: evolutionary_checkpoint.py
from __future__ import annotations

# Back-compat shim for older imports used by legacy tests.
from src.evolutionary import (  # type: ignore[import-not-found]
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)

__all__ = ["EvolutionStatus", "EvolutionOutcome", "EvolutionaryCheckpoint"]


# file: src/__init__.py
# Makes 'src' an explicit package for tools that dislike implicit namespace pkgs.


# file: material_response_optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

from board_material_aerial_enhancer import (
    enhance_aerial,
    _kmeans,
    _palette_rgb01,
    _pairwise_sq_dists,
    _rgb_to_lab,
)

__all__ = ["RenderEnhancementPlanner", "MaterialAwareEnhancementPlanner"]


@dataclass
class RenderEnhancementPlanner:
    """Baseline planner around the aerial enhancer; kept for compatibility."""
    k: int = 8
    seed: int = 22
    strength: float = 0.85
    palette: Optional[Sequence[str]] = None
    analysis_max_dim: int = 1280

    def plan(
        self,
        input_path: Path | str,
        output_path: Path | str,
        *,
        target_width: int = 4096,
        jpeg_quality: int = 95,
        jpeg_subsampling: int = 1,
        show_progress: bool = False,
        respect_icc: bool = True,
    ) -> Path:
        return enhance_aerial(
            input_path=input_path,
            output_path=output_path,
            analysis_max_dim=self.analysis_max_dim,
            k=self.k,
            seed=self.seed,
            target_width=target_width,
            strength=self.strength,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            show_progress=show_progress,
            respect_icc=respect_icc,
            palette=self.palette,
        )

    def optimize(self, input_path: Path | str, output_path: Path | str, **kwargs) -> Path:
        return self.plan(input_path, output_path, **kwargs)

    __call__ = plan  # convenience


@dataclass
class MaterialAwareEnhancementPlanner(RenderEnhancementPlanner):
    """Material-aware variant exposing in-memory ops used by tests."""

    def _fit_centers(self, arr01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        flat = arr01.reshape(-1, 3).astype(np.float64, copy=False)
        sample_n = min(flat.shape[0], 250_000)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(flat.shape[0], size=sample_n, replace=False)
        sample = flat[idx]

        km = _kmeans(sample, k=self.k, seed=self.seed)
        centers_rgb = km.centers  # (k,3)

        pal_rgb = _palette_rgb01(self.palette)
        centers_lab = _rgb_to_lab(centers_rgb)
        pal_lab = _rgb_to_lab(pal_rgb)

        diff2 = _pairwise_sq_dists(centers_lab, pal_lab)  # (k, m)
        nearest_idx = np.argmin(diff2, axis=1)
        target_rgb = pal_rgb[nearest_idx]                 # (k,3)
        deltas = target_rgb - centers_rgb                 # (k,3)

        if self.k > 1:
            cd2 = _pairwise_sq_dists(centers_rgb, centers_rgb)
            iu = np.triu_indices(self.k, 1)
            pd = float(np.sqrt(cd2[iu].mean())) if iu[0].size > 0 else 0.25
        else:
            pd = 0.25
        sigma2 = max((0.5 * pd) ** 2, 1e-8)
        return centers_rgb, deltas, sigma2

    def apply_array(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype in (np.uint8, np.uint16, np.int32):
            arr01 = (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
        else:
            arr01 = arr.astype(np.float32, copy=False).clip(0.0, 1.0)

        H, W, _ = arr01.shape
        centers_rgb, deltas, sigma2 = self._fit_centers(arr01)
        rows_target = max(128, min(H, int(1_000_000 / max(1, W))))
        out = np.empty_like(arr01, dtype=np.float32)

        inv_two_sigma2 = 1.0 / (2.0 * sigma2)
        for y0 in range(0, H, rows_target):
            y1 = min(H, y0 + rows_target)
            chunk = arr01[y0:y1].reshape(-1, 3).astype(np.float64, copy=False)
            dist2 = _pairwise_sq_dists(chunk, centers_rgb)  # (n,k)
            weights = np.exp(-dist2 * inv_two_sigma2)
            weights /= (weights.sum(axis=1, keepdims=True) + 1e-12)
            blended = weights @ deltas
            new_chunk = np.clip(chunk + self.strength * blended, 0.0, 1.0).astype(np.float32)
            out[y0:y1] = new_chunk.reshape(y1 - y0, W, 3)
        return out

    def apply_image(self, image: Image.Image) -> Image.Image:
        im = ImageOps.exif_transpose(image).convert("RGB")
        arr01 = np.asarray(im, dtype=np.float32) / 255.0
        out01 = self.apply_array(arr01)
        return Image.fromarray((out01 * 255.0 + 0.5).astype(np.uint8), mode="RGB")


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
    assert "Resolution: 4K (4096px width)" in result.output
    assert ("ICC handling:" in result.output) is expect_icc_line


# file: tests/test_api_guard_imports.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from evolutionary_checkpoint import EvolutionaryCheckpoint as EC_legacy, EvolutionStatus as ES_legacy
from src.evolutionary import EvolutionaryCheckpoint as EC_new, EvolutionStatus as ES_new

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
    aware = MaterialAwareEnhancementPlanner(k=2, seed=1, strength=0.4)
    im = _mk_grad()
    out_im = aware.apply_image(im)
    assert isinstance(out_im, Image.Image)
    assert out_im.size == im.size

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
