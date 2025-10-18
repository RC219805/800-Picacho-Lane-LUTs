# file: board_material_aerial_enhancer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# ------------------------------- color utils -------------------------------

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)

def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    # sRGB D65
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
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883  # D65
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    eps = (6 / 29) ** 3
    k = (29 / 3) ** 2 / 3
    def f(t):  # piecewise
        return np.where(t > eps, np.cbrt(t), k * t + 4 / 29)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return _xyz_to_lab(_rgb_to_xyz(rgb))

def _hex_to_rgb01(code: str) -> Tuple[float, float, float]:
    s = code.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)

# ------------------------------- palettes ----------------------------------

_DEFAULT_MBAR_8 = [
    "#F5F2EC",  # warm white
    "#D8C7A0",  # sand
    "#B5A58B",  # stone
    "#9A7957",  # oak
    "#6A6D6F",  # slate
    "#3A3C3E",  # charcoal
    "#CFD6DF",  # sky haze
    "#244A5A",  # deep teal
]

def _palette_rgb01(palette: Optional[Sequence[str]]) -> np.ndarray:
    src = _DEFAULT_MBAR_8 if palette is None else list(palette)
    return np.asarray([_hex_to_rgb01(h) for h in src], dtype=np.float64)

# -------------------------------- k-means ----------------------------------

@dataclass(frozen=True)
class KMeansResult:
    centers: np.ndarray  # (k,3) in RGB01
    inertia: float

def _kmeans_plus_plus_init(data: np.ndarray, k: int, *, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    idx0 = rng.integers(0, n)
    centers = [data[idx0]]
    dist2 = np.full(n, np.inf, dtype=np.float64)
    for _ in range(1, k):
        d = data - centers[-1]
        dist2 = np.minimum(dist2, np.einsum("ij,ij->i", d, d))
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers.append(data[idx])
    return np.asarray(centers, dtype=np.float64)

def _kmeans(data: np.ndarray, k: int, *, seed: int, max_iter: int = 25, tol: float = 1e-4) -> KMeansResult:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if n < k:
        raise ValueError(f"Not enough samples ({n}) for k={k}")
    centers = _kmeans_plus_plus_init(data, k, rng=rng)
    last_inertia = np.inf
    for _ in range(max_iter):
        # assign
        # distances to centers: (n,k)
        d = data[:, None, :] - centers[None, :, :]
        dist2 = np.einsum("nik,nik->ni", d, d)
        labels = np.argmin(dist2, axis=1)
        # update
        new_centers = np.empty_like(centers)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # re-seed empty center
                new_centers[j] = data[rng.integers(0, n)]
            else:
                new_centers[j] = data[mask].mean(axis=0)
        centers = new_centers
        inertia = float(dist2[np.arange(n), labels].sum())
        if abs(last_inertia - inertia) <= tol * max(1.0, last_inertia):
            break
        last_inertia = inertia
    return KMeansResult(centers=centers, inertia=last_inertia)

# ---------------------------- aerial enhancer ------------------------------

def enhance_aerial(
    input_path: Path | str,
    output_path: Path | str,
    *,
    analysis_max_dim: int = 1280,
    k: int = 8,
    seed: int = 22,
    target_width: int = 4096,
    strength: float = 0.85,
    jpeg_quality: int = 95,
    palette: Optional[Sequence[str]] = None,
) -> Path:
    """Apply MBAR palette transfer to aerial; return Path to saved file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    # load
    with Image.open(input_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w0, h0 = im.size
        arr = np.asarray(im, dtype=np.float32) / 255.0

    # analysis image
    scale = min(1.0, analysis_max_dim / max(w0, h0)) if analysis_max_dim > 0 else 1.0
    if scale < 1.0:
        im_small = im.resize((int(round(w0 * scale)), int(round(h0 * scale))), Image.LANCZOS)
        arr_small = np.asarray(im_small, dtype=np.float32) / 255.0
    else:
        arr_small = arr

    # sample for k-means
    flat = arr_small.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(seed)
    sample_n = min(flat.shape[0], 250_000)
    idx = rng.choice(flat.shape[0], size=sample_n, replace=False)
    sample = flat[idx]

    km = _kmeans(sample, k=k, seed=seed)
    centers_rgb = km.centers  # (k,3) in RGB01

    # map centers to palette in Lab
    pal_rgb = _palette_rgb01(palette)
    centers_lab = _rgb_to_lab(centers_rgb)
    pal_lab = _rgb_to_lab(pal_rgb)
    # nearest palette color for each center
    diff = centers_lab[:, None, :] - pal_lab[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    nearest_idx = np.argmin(d2, axis=1)  # (k,)
    target_rgb = pal_rgb[nearest_idx]  # (k,3)

    # per-center RGB delta
    deltas = target_rgb - centers_rgb  # (k,3)

    # sigma from centroid spread
    if k > 1:
        pd = np.sqrt(
            np.maximum(
                1e-8,
                np.mean(
                    [
                        np.sum((centers_rgb[i] - centers_rgb[j]) ** 2)
                        for i in range(k)
                        for j in range(i + 1, k)
                    ]
                )
            )
        )
    else:
        pd = 0.25
    sigma2 = (0.5 * pd) ** 2

    # apply to full-res in chunks
    H, W, _ = arr.shape
    rows_target = max(128, min(H, int(1_000_000 / max(1, W))))  # ~1M px per chunk
    out = np.empty_like(arr, dtype=np.float32)

    for y0 in tqdm(range(0, H, rows_target), desc="Applying palette", unit="rows"):
        y1 = min(H, y0 + rows_target)
        chunk = arr[y0:y1].reshape(-1, 3).astype(np.float64)  # (n,3)
        # distances to centers -> weights
        dd = chunk[:, None, :] - centers_rgb[None, :, :]  # (n,k,3)
        dist2 = np.einsum("nkj,nkj->nk", dd, dd)
        weights = np.exp(-dist2 / (2 * sigma2 + 1e-12))  # (n,k)
        ws = weights.sum(axis=1, keepdims=True) + 1e-12
        weights /= ws
        # blended delta
        blended = weights @ deltas  # (n,3)
        new_chunk = np.clip(chunk + strength * blended, 0.0, 1.0).astype(np.float32)
        out[y0:y1] = new_chunk.reshape(y1 - y0, W, 3)

    # resize to target width
    if target_width and W != target_width:
        new_h = int(round(H * (target_width / W)))
        out_img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB").resize(
            (target_width, new_h), Image.LANCZOS
        )
    else:
        out_img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path, format="JPEG", quality=int(jpeg_quality), subsampling=1, optimize=True)
    return output_path

# ---------------------------------- CLI ------------------------------------

def _format_bytes(n: int) -> str:
    return f"{n / (1024**2):.2f} MB"

def _print_header(inp: Path, out: Path, k: int) -> None:
    print(f"Processing: {inp.name}")
    print(f"Output: {out.name}")
    print(f"Resolution: 4K (4096px width)")
    print(f"Materials: MBAR-approved palette ({k} materials)\n")

def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        import typer  # lazy import for optional dep
    except Exception as e:
        print("This CLI requires 'typer'. Install with: pip install typer", flush=True)
        return 2

    app = typer.Typer(add_completion=False, no_args_is_help=True, help="MBAR aerial enhancer")

    @app.command("enhance")
    def _enhance(
        input_path: Path = typer.Option(..., "--input", "-i", exists=True, readable=True, dir_okay=False, file_okay=True),
        output_path: Path = typer.Option(..., "--output", "-o"),
        analysis_max_dim: int = typer.Option(1280, help="Max dimension for clustering preview."),
        k: int = typer.Option(8, min=1, max=16, help="Number of color clusters."),
        seed: int = typer.Option(22, help="Random seed for clustering."),
        target_width: int = typer.Option(4096, help="Final output width."),
        strength: float = typer.Option(0.85, min=0.0, max=1.0, help="Blend strength toward palette."),
        jpeg_quality: int = typer.Option(95, min=70, max=100, help="JPEG quality."),
    ) -> None:
        _print_header(input_path, output_path, k)
        result = enhance_aerial(
            input_path=input_path,
            output_path=output_path,
            analysis_max_dim=analysis_max_dim,
            k=k,
            seed=seed,
            target_width=target_width,
            strength=strength,
            jpeg_quality=jpeg_quality,
        )
        print(f"✅ Enhanced aerial saved to: {result}")
        try:
            print(f"✅ File size: {_format_bytes(result.stat().st_size)}")
        except Exception:
            pass

    return app(standalone_mode=True)

if __name__ == "__main__":
    raise SystemExit(main())