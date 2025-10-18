# file: board_material_aerial_enhancer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Pillow resampling compat
_RESAMPLING = getattr(Image, "Resampling", Image)
_NEAREST = _RESAMPLING.NEAREST
_BILINEAR = _RESAMPLING.BILINEAR
_BICUBIC = _RESAMPLING.BICUBIC

# ------------------------- Data models -------------------------

@dataclass(frozen=True)
class MaterialRule:
    """Material parameters for blending."""
    name: str
    target_rgb01: np.ndarray
    blend: float = 0.6       # 0..1, how strong to overlay the texture
    tint: Tuple[int, int, int] | None = None  # optional swatch for legend/extra tint
    tint_strength: float = 0.0               # 0..1

@dataclass(frozen=True)
class ClusterStat:
    """Per-cluster summary used for rule assignment/reporting."""
    label: int
    count: int
    mean_rgb: Tuple[float, float, float]   # 0..1
    mean_hsv: Tuple[float, float, float]   # H 0..1, S 0..1, V 0..1

# ------------------------- Color utils -------------------------

def _hex_to_rgb01(hex_str: str) -> np.ndarray:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

def _rgb01_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    lin = _srgb_to_linear(rgb)
    M = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32
    )
    xyz = lin @ M.T
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[..., 0] / Xn, xyz[..., 1] / Yn, xyz[..., 2] / Zn

    delta = 6 / 29
    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)

def _rgb01_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB(0..1) -> HSV(0..1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin + 1e-12
    h = np.zeros_like(cmax)
    mask = delta > 1e-12
    r_is_max = (cmax == r) & mask
    g_is_max = (cmax == g) & mask
    b_is_max = (cmax == b) & mask
    h[r_is_max] = ((g - b)[r_is_max] / delta[r_is_max]) % 6
    h[g_is_max] = (b - r)[g_is_max] / delta[g_is_max] + 2
    h[b_is_max] = (r - g)[b_is_max] / delta[b_is_max] + 4
    h = (h / 6.0) % 1.0
    s = np.where(cmax <= 1e-12, 0.0, delta / (cmax + 1e-12))
    v = cmax
    return np.stack([h, s, v], axis=-1).astype(np.float32)

# ------------------------- Image helpers -------------------------

def _image_to_array01(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

def _array01_to_image(arr: np.ndarray) -> Image.Image:
    arr8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGB")

def _downsample_image(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img.copy()
    s = max_dim / float(m)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return img.resize((nw, nh), _BILINEAR)

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

def _recolor_texture_to_target(tex: Image.Image, target_rgb01: np.ndarray, strength: float = 0.6) -> Image.Image:
    """Channel-wise gain toward target; blended for stability."""
    tex_arr = _image_to_array01(tex)
    mean = tex_arr.reshape(-1, 3).mean(axis=0)
    eps = 1e-5
    gains = np.clip((target_rgb01 + eps) / (mean + eps), 0.5, 1.8)
    recol = np.clip(tex_arr * gains[None, None, :], 0.0, 1.0)
    blended = np.clip((1.0 - strength) * tex_arr + strength * recol, 0.0, 1.0)
    return _array01_to_image(blended)

def _deterministic_noise_texture(size: Tuple[int, int], color: np.ndarray, rng: np.random.Generator) -> Image.Image:
    H, W = size[1], size[0]
    noise = rng.random((H, W, 1), dtype=np.float32)
    base = np.ones((H, W, 3), dtype=np.float32) * color[None, None, :]
    mod = 0.15 * (noise - 0.5)
    out = np.clip(base * (1.0 + mod), 0.0, 1.0)
    return _array01_to_image(out)

# ------------------------- K-means -------------------------

def _kmeans_pp_init(samples: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = samples.shape[0]
    idx0 = int(rng.integers(0, n))
    centers = [samples[idx0]]
    for _ in range(1, k):
        d2 = np.min(((samples[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        cum = np.cumsum(probs)
        r = rng.random()
        idx = int(np.searchsorted(cum, r))
        centers.append(samples[idx])
    return np.stack(centers, axis=0)

def _kmeans(samples: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 30, tol: float = 1e-4) -> np.ndarray:
    """
    Deterministic K-means returning centroids only (RGB in 0..1).
    Why: matches your script's expected signature.
    """
    n = samples.shape[0]
    k = max(1, min(k, n))
    centers = _kmeans_pp_init(samples, k, rng)
    labels = np.zeros((n,), dtype=np.int32)

    for _ in range(max_iter):
        dists = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = new_labels == i
            if not np.any(mask):
                new_centers[i] = samples[int(rng.integers(0, n))]
            else:
                new_centers[i] = samples[mask].mean(axis=0)
        shift = np.linalg.norm(new_centers - centers) / (np.linalg.norm(centers) + 1e-8)
        centers, labels = new_centers, new_labels
        if shift < tol:
            break
    return centers

def _assign_full_image(image_rgb01: np.ndarray, centroids_rgb01: np.ndarray) -> np.ndarray:
    """Return labels [H,W] assigning each pixel to nearest centroid."""
    H, W, _ = image_rgb01.shape
    px = image_rgb01.reshape(-1, 3)
    dists = ((px[:, None, :] - centroids_rgb01[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dists, axis=1).astype(np.uint8)
    return labels.reshape(H, W)

# ------------------------- Stats & rules -------------------------

_DEFAULT_PALETTE: Dict[str, str] = {
    "plaster":   "#dcc9b4",
    "stone":     "#b49a7e",
    "cladding":  "#9a7a5f",
    "screens":   "#8c8a86",
    "equitone":  "#2a2c32",
    "roof":      "#9b9b9b",
    "bronze":    "#463a30",
    "shade":     "#f2f2f2",
}

DEFAULT_TEXTURES: Dict[str, Path] = {
    # Optional: user can fill these with real file paths; enhancer falls back to procedural textures.
    # "plaster": Path("textures/plaster.jpg"),
    # "stone": Path("textures/stone.jpg"),
    # ...
}

def _cluster_stats(image_rgb01: np.ndarray, labels: np.ndarray) -> List[ClusterStat]:
    """Compute per-cluster counts and mean RGB/HSV."""
    H, W, _ = image_rgb01.shape
    n = H * W
    flat = image_rgb01.reshape(-1, 3)
    unique = sorted(int(i) for i in np.unique(labels))
    stats: List[ClusterStat] = []
    hsv = _rgb01_to_hsv(flat).reshape(H * W, 3)
    for lbl in unique:
        mask = (labels.reshape(-1) == lbl)
        cnt = int(mask.sum())
        if cnt == 0:
            mean_rgb = (0.0, 0.0, 0.0)
            mean_hsv = (0.0, 0.0, 0.0)
        else:
            m_rgb = flat[mask].mean(axis=0)
            m_hsv = hsv[mask].mean(axis=0)
            mean_rgb = (float(m_rgb[0]), float(m_rgb[1]), float(m_rgb[2]))
            mean_hsv = (float(m_hsv[0]), float(m_hsv[1]), float(m_hsv[2]))
        stats.append(ClusterStat(label=lbl, count=cnt, mean_rgb=mean_rgb, mean_hsv=mean_hsv))
    return stats

def build_material_rules(textures: Mapping[str, Path] | None = None) -> Dict[str, MaterialRule]:
    """Create default MBAR-ish material rules. Textures are optional (used at render time)."""
    rules: Dict[str, MaterialRule] = {}
    for name, hx in _DEFAULT_PALETTE.items():
        rgb01 = _hex_to_rgb01(hx)
        # Conservative defaults; specific tweaks can be changed here
        blend = 0.6 if name not in {"roof", "shade"} else (0.55 if name == "roof" else 0.45)
        tint = None
        tint_strength = 0.0
        rules[name] = MaterialRule(name=name, target_rgb01=rgb01, blend=blend, tint=tint, tint_strength=tint_strength)
    return rules

def assign_materials(stats: Sequence[ClusterStat], rules: Mapping[str, MaterialRule]) -> Dict[int, MaterialRule]:
    """Greedy nearest-in-Lab assignment of each cluster to the closest rule target."""
    # Build palette Lab
    names = list(rules.keys())
    pal_rgb = np.stack([rules[n].target_rgb01 for n in names], axis=0)
    pal_lab = _rgb01_to_lab(pal_rgb)
    assignments: Dict[int, MaterialRule] = {}
    for s in stats:
        c_rgb = np.array(s.mean_rgb, dtype=np.float32)
        c_lab = _rgb01_to_lab(c_rgb[None, :])[0]
        d = np.linalg.norm(pal_lab - c_lab[None, :], axis=1)
        j = int(np.argmin(d))
        assignments[s.label] = rules[names[j]]
    return assignments

# ------------------------- Enhancer -------------------------

def _load_textures_or_defaults(
    textures: Optional[Mapping[str, Path]],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Image.Image], Dict[str, np.ndarray]]:
    """Return (name->texture image, name->palette rgb01)."""
    tex_images: Dict[str, Image.Image] = {}
    palette_rgb: Dict[str, np.ndarray] = {}

    # Try provided textures first
    if textures:
        for name, p in textures.items():
            try:
                if isinstance(p, Path) and p.exists():
                    img = Image.open(p).convert("RGB")
                    tex_images[name] = img
                    palette_rgb[name] = _image_to_array01(img).reshape(-1, 3).mean(axis=0)
            except Exception:
                # Ignore unreadable textures; synth fallback will kick in
                pass

    # Fill any missing with defaults
    for name, hx in _DEFAULT_PALETTE.items():
        if name not in tex_images:
            rgb01 = _hex_to_rgb01(hx)
            tex_images[name] = _deterministic_noise_texture((512, 512), rgb01, rng)
            palette_rgb[name] = rgb01

    return tex_images, palette_rgb

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
    Enhance an aerial by clustering colors, assigning materials, and compositing textures.

    Returns:
        Path to saved enhanced image.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    rng = np.random.default_rng(seed)

    # Load and prepare analysis image
    src = Image.open(input_path).convert("RGB")
    src_analysis = _downsample_image(src, analysis_max_dim)
    small_arr = _image_to_array01(src_analysis)
    samples = small_arr.reshape(-1, 3)

    # Cluster
    centroids = _kmeans(samples, k=max(1, k), rng=rng)
    labels_small = _assign_full_image(small_arr, centroids)

    # Build rules and textures
    rules = build_material_rules(textures or DEFAULT_TEXTURES)
    tex_images, palette_rgb = _load_textures_or_defaults(textures or DEFAULT_TEXTURES, rng)

    # Assign clusters -> materials using stats on the ORIGINAL image resized to analysis size
    stats = _cluster_stats(small_arr, labels_small)
    assignments = assign_materials(stats, rules)

    # Prepare output canvas dimensions
    src_w, src_h = src.size
    scale = target_width / float(src_w)
    out_w = int(round(target_width))
    out_h = max(1, int(round(src_h * scale)))

    # Upscale base and label map
    base = src.resize((out_w, out_h), _BICUBIC)
    label_img = Image.fromarray(labels_small.astype(np.uint8), mode="L")
    labels_up = label_img.resize((out_w, out_h), _NEAREST)
    labels_up_arr = np.array(labels_up, dtype=np.uint8)

    # Composite
    out = base.copy()
    feather = 1.2  # px feather for seams
    unique_clusters = sorted(int(c) for c in np.unique(labels_small))

    for idx in unique_clusters:
        rule = assignments.get(idx)
        if rule is None:
            continue
        name = rule.name
        target_rgb01 = rules[name].target_rgb01
        blend_alpha = float(np.clip(rule.blend, 0.0, 1.0))

        tex = tex_images[name]
        tex_tiled = _tile_image_to_size(tex, (out_w, out_h))
        tex_tinted = _recolor_texture_to_target(tex_tiled, target_rgb01, strength=0.6)

        # Optional extra tint (legend swatch parity); kept subtle
        if rule.tint and rule.tint_strength > 0.0:
            tint_rgb01 = np.array(rule.tint, dtype=np.float32) / 255.0
            tex_arr = _image_to_array01(tex_tinted)
            tex_arr = np.clip((1.0 - rule.tint_strength) * tex_arr + rule.tint_strength * tint_rgb01, 0.0, 1.0)
            tex_tinted = _array01_to_image(tex_arr)

        mask_bin = (labels_up_arr == idx).astype(np.uint8) * 255
        mask = Image.fromarray(mask_bin, mode="L").filter(ImageFilter.GaussianBlur(radius=feather))
        blended = Image.blend(out, tex_tinted, alpha=blend_alpha)
        out.paste(blended, mask=mask)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return output_path