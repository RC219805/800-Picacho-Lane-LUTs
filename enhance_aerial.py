# file: mbar_enhance.py
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Dict, Iterable, Tuple, List, Optional

import math
import numpy as np
from PIL import Image, ImageFilter

# --- small compat for Pillow resampling ---
_RESAMPLING = getattr(Image, "Resampling", Image)
_NEAREST = _RESAMPLING.NEAREST
_BILINEAR = _RESAMPLING.BILINEAR
_BICUBIC = _RESAMPLING.BICUBIC

# ---------- Color utilities (sRGB <-> Lab) ----------

def _hex_to_rgb01(hex_str: str) -> np.ndarray:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    # piecewise inverse EOTF
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def _rgb01_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1], returns Lab with D65."""
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    lin = _srgb_to_linear(rgb)
    # sRGB -> XYZ (D65), ref white
    M = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = np.dot(lin, M.T)
    # Normalize by D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    def _f(t: np.ndarray) -> np.ndarray:
        delta = 6 / 29
        return np.where(t > (delta ** 3), np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


# ---------- Image helpers ----------

def _image_to_array01(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _array01_to_image(arr: np.ndarray) -> Image.Image:
    arr8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGB")


def _compute_average_rgb01(img: Image.Image) -> np.ndarray:
    arr = _image_to_array01(img)
    return arr.reshape(-1, 3).mean(axis=0)


def _tile_image_to_size(tex: Image.Image, size: Tuple[int, int]) -> Image.Image:
    tw, th = tex.size
    W, H = size
    if tw == 0 or th == 0:
        raise ValueError("Texture has zero dimension.")
    reps_x = math.ceil(W / tw)
    reps_y = math.ceil(H / th)
    canvas = Image.new("RGB", (tw * reps_x, th * reps_y))
    for y in range(reps_y):
        for x in range(reps_x):
            canvas.paste(tex, (x * tw, y * th))
    return canvas.crop((0, 0, W, H))


def _resize_preserve_max(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img.copy()
    scale = max_dim / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), _BILINEAR)


def _deterministic_noise_texture(size: Tuple[int, int], color: np.ndarray, rng: np.random.Generator) -> Image.Image:
    """Why: provides texture when none supplied while keeping palette tint."""
    H, W = size[1], size[0]
    noise = rng.random((H, W, 1), dtype=np.float32)
    base = np.ones((H, W, 3), dtype=np.float32) * color[None, None, :]
    # mild variation; centered around 1.0
    mod = 0.15 * (noise - 0.5)
    out = np.clip(base * (1.0 + mod), 0.0, 1.0)
    return _array01_to_image(out)


def _recolor_texture_to_target(tex: Image.Image, target_rgb01: np.ndarray, strength: float = 0.6) -> Image.Image:
    """
    Channel-wise gain toward target; blend back for stability.
    Why: simple, fast color transfer without heavy deps.
    """
    tex_arr = _image_to_array01(tex)
    mean = tex_arr.reshape(-1, 3).mean(axis=0)
    eps = 1e-5
    gains = np.clip((target_rgb01 + eps) / (mean + eps), 0.5, 1.8)
    recol = np.clip(tex_arr * gains[None, None, :], 0.0, 1.0)
    blended = np.clip((1.0 - strength) * tex_arr + strength * recol, 0.0, 1.0)
    return _array01_to_image(blended)


# ---------- K-means (deterministic, k-means++) ----------

def _kmeans_pp_init(samples: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = samples.shape[0]
    idx0 = int(rng.integers(0, n))
    centers = [samples[idx0]]
    # choose remaining centers
    for _ in range(1, k):
        d2 = np.min(((samples[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        cum = np.cumsum(probs)
        r = rng.random()
        idx = int(np.searchsorted(cum, r))
        centers.append(samples[idx])
    return np.stack(centers, axis=0)


def _kmeans(samples: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 30, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    n = samples.shape[0]
    k = max(1, min(k, n))
    centers = _kmeans_pp_init(samples, k, rng)
    labels = np.zeros((n,), dtype=np.int32)

    for _ in range(max_iter):
        # assign
        dists = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        # recompute
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = new_labels == i
            if not np.any(mask):
                # re-seed empty cluster
                new_centers[i] = samples[int(rng.integers(0, n))]
            else:
                new_centers[i] = samples[mask].mean(axis=0)
        shift = np.linalg.norm(new_centers - centers) / (np.linalg.norm(centers) + 1e-8)
        centers, labels = new_centers, new_labels
        if shift < tol:
            break
    return centers, labels


# ---------- Palette ----------

_DEFAULT_MBAR_PALETTE: Dict[str, str] = {
    "asphalt": "#4a4a4a",
    "vegetation": "#3a6b3a",
    "soil": "#9c6b3f",
    "water": "#2f5f9e",
    "roof": "#a0503a",
    "concrete": "#b8b8b8",
    "sand": "#d8c39a",
    "shadow": "#1e1e1e",
}


def _load_textures_or_defaults(
    textures: Optional[Mapping[str, Path]],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Image.Image], Dict[str, np.ndarray]]:
    """Returns (material_name -> texture_image, material_name -> palette_rgb01)."""
    tex_images: Dict[str, Image.Image] = {}
    palette_rgb: Dict[str, np.ndarray] = {}

    if textures:
        for name, p in textures.items():
            if not isinstance(p, Path):
                continue
            if not p.exists():
                continue
            img = Image.open(p).convert("RGB")
            tex_images[name] = img
            palette_rgb[name] = _compute_average_rgb01(img)
        if tex_images:
            return tex_images, palette_rgb
        # fallback to defaults if mapping was empty/invalid

    for name, hx in _DEFAULT_MBAR_PALETTE.items():
        rgb01 = _hex_to_rgb01(hx)
        # synth noise texture 512x512
        tex_images[name] = _deterministic_noise_texture((512, 512), rgb01, rng)
        palette_rgb[name] = rgb01
    return tex_images, palette_rgb


def _assign_clusters_to_materials(
    centers_rgb01: np.ndarray,
    palette_rgb01: Dict[str, np.ndarray],
) -> List[Tuple[int, str]]:
    """Greedy nearest neighbor in Lab; returns list of (cluster_index, material_name)."""
    centers_lab = _rgb01_to_lab(centers_rgb01)
    names = list(palette_rgb01.keys())
    pal_rgb = np.stack([palette_rgb01[n] for n in names], axis=0)
    pal_lab = _rgb01_to_lab(pal_rgb)

    assignments: List[Tuple[int, str]] = []
    for i, c in enumerate(centers_lab):
        d = np.linalg.norm(pal_lab - c[None, :], axis=1)
        j = int(np.argmin(d))
        assignments.append((i, names[j]))
    return assignments


def enhance_aerial(
    input_path: Path,
    output_path: Path,
    *,
    analysis_max_dim: int = 1280,
    k: int = 8,
    seed: int = 22,
    target_width: int = 4096,
    textures: Mapping[str, Path] | None = None,
) -> Path:
    """
    Enhance an aerial image by clustering colors, assigning MBAR board materials,
    and blending high-res textures to approximate the approved palette.

    Args:
        input_path: Path to input aerial image.
        output_path: Path to save enhanced image.
        analysis_max_dim: Max dimension for clustering analysis.
        k: Number of clusters.
        seed: RNG seed for reproducibility.
        target_width: Output width in pixels.
        textures: Optional mapping of material names to texture paths.

    Returns:
        Path to saved enhanced image.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    rng = np.random.default_rng(seed)

    # Load input
    src = Image.open(input_path).convert("RGB")
    src_analysis = _resize_preserve_max(src, analysis_max_dim)

    # Prepare samples
    small_arr = _image_to_array01(src_analysis)
    samples = small_arr.reshape(-1, 3)

    # K-means clustering
    centers_rgb01, labels_small = _kmeans(samples, k=max(1, k), rng=rng)
    labels_small = labels_small.reshape(small_arr.shape[:2]).astype(np.int32)

    # Palette from textures or defaults
    tex_images, palette_rgb = _load_textures_or_defaults(textures, rng)

    # Assign clusters -> materials (Lab nearest)
    assignments = _assign_clusters_to_materials(centers_rgb01, palette_rgb)
    # maps
    cluster_to_material: Dict[int, str] = {ci: name for ci, name in assignments}

    # Compute output size
    src_w, src_h = src.size
    scale = target_width / float(src_w)
    out_w = int(round(target_width))
    out_h = max(1, int(round(src_h * scale)))

    # Upscale base and label map
    base = src.resize((out_w, out_h), _BICUBIC)
    label_img = Image.fromarray(labels_small, mode="I")
    labels_up = label_img.resize((out_w, out_h), _NEAREST)
    labels_up_arr = np.array(labels_up, dtype=np.int32)

    # Composite
    out = base.copy()
    blend_alpha = 0.6  # texture influence
    feather = 1.2      # px

    unique_clusters = sorted(int(c) for c in np.unique(labels_small))
    for cluster_idx in unique_clusters:
        mat = cluster_to_material.get(cluster_idx)
        if mat is None:
            continue
        # get texture + recolor to palette mean
        palette_color = palette_rgb[mat]
        tex = tex_images[mat]
        tex_tiled = _tile_image_to_size(tex, (out_w, out_h))
        tex_tinted = _recolor_texture_to_target(tex_tiled, palette_color, strength=0.6)

        # region mask
        mask_bin = (labels_up_arr == cluster_idx).astype(np.uint8) * 255
        mask = Image.fromarray(mask_bin, mode="L")
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

        # blend over whole frame, then paste by mask (fast & simple)
        blended = Image.blend(out, tex_tinted, alpha=blend_alpha)
        out.paste(blended, mask=mask)

    # Persist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return output_path