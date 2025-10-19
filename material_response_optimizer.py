# file: material_response_optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

# Only import the public entrypoint; vendor internal math locally.
from board_material_aerial_enhancer import enhance_aerial  # type: ignore


__all__ = ["RenderEnhancementPlanner", "MaterialAwareEnhancementPlanner"]


# -------------------------- minimal color/kmeans utils -----------------------

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    lin = _srgb_to_linear(rgb.astype(np.float64))
    return np.tensordot(lin, M.T, axes=1)

def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    eps = (6 / 29) ** 3
    k = (29 / 3) ** 2 / 3
    def f(t):
        return np.where(t > eps, np.cbrt(t), k * t + 4 / 29)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return _xyz_to_lab(_rgb_to_xyz(rgb))

def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    C = C.astype(np.float64, copy=False)
    X2 = np.einsum("ij,ij->i", X, X)[:, None]
    C2 = np.einsum("ij,ij->i", C, C)[None, :]
    XC = X @ C.T
    return np.maximum(X2 + C2 - 2.0 * XC, 0.0)

def _hex_to_rgb01(code: str) -> Tuple[float, float, float]:
    s = code.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6 or any(ch not in "0123456789aAbBcCdDeEfF" for ch in s):
        raise ValueError(f"Invalid HEX color: {code!r}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)

_DEFAULT_MBAR_8 = [
    "#F5F2EC", "#D8C7A0", "#B5A58B", "#9A7957",
    "#6A6D6F", "#3A3C3E", "#CFD6DF", "#244A5A",
]

def _palette_rgb01(palette: Optional[Sequence[str]]) -> np.ndarray:
    src = _DEFAULT_MBAR_8 if palette is None or len(palette) == 0 else list(palette)
    return np.asarray([_hex_to_rgb01(h) for h in src], dtype=np.float64)

class _KMeansResult(Tuple[np.ndarray, float]):  # not used directly; just for clarity
    ...

def _kmeans_plus_plus_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    idx0 = rng.integers(0, n)
    centers = [data[idx0]]
    dist2 = np.full(n, np.inf, dtype=np.float64)
    for _ in range(1, k):
        d = data - centers[-1]
        cur = np.einsum("ij,ij->i", d, d)
        dist2 = np.minimum(dist2, cur)
        total = float(dist2.sum())
        if not np.isfinite(total) or total <= 1e-12:
            idx = rng.integers(0, n)
        else:
            probs = dist2 / total
            idx = rng.choice(n, p=probs)
        centers.append(data[idx])
    return np.asarray(centers, dtype=np.float64)

def _kmeans(data: np.ndarray, k: int, seed: int, max_iter: int = 25, tol: float = 1e-4) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if n < k:
        raise ValueError(f"Not enough samples ({n}) for k={k}")
    centers = _kmeans_plus_plus_init(data, k, rng=rng)
    last_inertia = np.inf
    for _ in range(max_iter):
        dist2 = _pairwise_sq_dists(data, centers)
        labels = np.argmin(dist2, axis=1)
        inertia = float(dist2[np.arange(n), labels].sum())
        if abs(last_inertia - inertia) <= tol * max(1.0, last_inertia):
            last_inertia = inertia
            break
        last_inertia = inertia
        new_centers = np.empty_like(centers)
        for j in range(k):
            mask = labels == j
            new_centers[j] = data[rng.integers(0, n)] if not np.any(mask) else data[mask].mean(axis=0)
        centers = new_centers
    return centers, float(last_inertia)


# ------------------------------- planners ------------------------------------

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
        # Note: jpeg_subsampling, show_progress, and respect_icc are accepted for API compatibility
        # but not currently used by enhance_aerial
        return enhance_aerial(
            input_path=input_path,
            output_path=output_path,
            analysis_max_dim=self.analysis_max_dim,
            k=self.k,
            seed=self.seed,
            target_width=target_width,
            strength=self.strength,
            jpeg_quality=jpeg_quality,
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

        centers_rgb, _inertia = _kmeans(sample, k=self.k, seed=self.seed)
        pal_rgb = _palette_rgb01(self.palette)

        centers_lab = _rgb_to_lab(centers_rgb)
        pal_lab = _rgb_to_lab(pal_rgb)

        diff2 = _pairwise_sq_dists(centers_lab, pal_lab)
        nearest_idx = np.argmin(diff2, axis=1)
        target_rgb = pal_rgb[nearest_idx]
        deltas = target_rgb - centers_rgb

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
            dist2 = _pairwise_sq_dists(chunk, centers_rgb)
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
