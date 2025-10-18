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

    __call__ = plan


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
