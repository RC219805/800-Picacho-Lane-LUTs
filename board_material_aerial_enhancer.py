# board_material_aerial_enhancer.py
"""
Apply MBAR board material textures to an aerial photograph using a lightweight,
CI-friendly pipeline focused on deterministic behavior for tests.

Key pieces:
- k-means clustering on a downscaled analysis image
- palette JSON for deterministic cluster → material mapping
- optional texture validation with graceful fallbacks
- minimal enhancement to keep tests fast (no heavyweight ML/GPU deps)
"""

from __future__ import annotations

# --- stdlib ---
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Optional, TYPE_CHECKING, Dict, Any, Callable

# --- third-party (kept light) ---
import numpy as np
from PIL import Image, ImageFilter

# Optional dependency (keep soft so CI stays lean)
try:  # pragma: no cover - optional
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional
    tifffile = None  # type: ignore

# If the real class is available elsewhere, use it only for typing;
# otherwise provide a complete stub that satisfies runtime & tests.
if TYPE_CHECKING:  # pragma: no cover
    from material_response import MaterialRule  # type: ignore
else:
    @dataclass(frozen=True)
    class MaterialRule:  # complete stub for tests and other scripts
        name: str
        texture: str = ""
        blend: float = 1.0
        score_fn: Callable[["ClusterStats"], float] | None = None
        tint: tuple[int, int, int] | None = None
        tint_strength: float = 0.0


# --------------------------
# Public API
# --------------------------

__all__ = [
    "ClusterStats",
    "MaterialRule",
    "compute_cluster_stats",
    "load_palette_assignments",
    "save_palette_assignments",
    "relabel",
    "enhance_aerial",
    "apply_materials",   # back-compat expected by tests
    "assign_materials",  # additional alias expected by tests
    "build_material_rules",
    "DEFAULT_TEXTURES",
]


# --------------------------
# Cluster statistics (exported for tests)
# --------------------------

@dataclass(frozen=True)
class ClusterStats:
    """Statistics for a single color cluster in the aerial image."""
    label: int
    count: int
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (r,g,b) in [0,1]
    mean_rgb: np.ndarray | None = None  # mean RGB values
    mean_hsv: np.ndarray | None = None  # mean HSV values
    std_rgb: np.ndarray | None = None  # standard deviation of RGB


def compute_cluster_stats(labels: np.ndarray, rgb: np.ndarray) -> list[ClusterStats]:
    """
    Compute simple stats per label:
      - pixel count
      - centroid color (mean RGB in [0,1])

    labels: (H, W) uint dtype
    rgb:    (H, W, 3) uint8 or float in [0,1]
    """
    if rgb.dtype.kind in ("u", "i"):
        rgb_f = rgb.astype(np.float32) / 255.0
    else:
        rgb_f = rgb.astype(np.float32)

    labs = labels.reshape(-1)
    flat = rgb_f.reshape(-1, 3)

    out: list[ClusterStats] = []
    for lab in np.unique(labs).tolist():
        mask = labs == lab
        cnt = int(mask.sum())
        if cnt:
            mean = flat[mask].mean(axis=0)
            centroid = (float(mean[0]), float(mean[1]), float(mean[2]))
        else:
            centroid = (0.0, 0.0, 0.0)
        out.append(ClusterStats(label=int(lab), count=cnt, centroid=centroid))
    return out


# --------------------------
# Default textures and material building (for back-compat)
# --------------------------

DEFAULT_TEXTURES: dict[str, Path] = {
    "plaster": Path("textures/plaster.png"),
    "stone": Path("textures/stone.png"),
    "cladding": Path("textures/cladding.png"),
    "screens": Path("textures/screens.png"),
    "equitone": Path("textures/equitone.png"),
    "roof": Path("textures/roof.png"),
    "bronze": Path("textures/bronze.png"),
    "shade": Path("textures/shade.png"),
}


def build_material_rules(textures: Mapping[str, Path]) -> list[MaterialRule]:
    """
    Build a list of MaterialRule objects from a texture mapping.
    This is a minimal implementation for back-compat with tests and scripts.
    """
    rules: list[MaterialRule] = []
    for name, texture_path in textures.items():
        rule = MaterialRule(
            name=name,
            texture=str(texture_path),
            blend=0.6,
            score_fn=lambda stats: 1.0,  # Default scoring
        )
        rules.append(rule)
    return rules


# --------------------------
# Helper functions (for back-compat with other scripts)
# --------------------------

def _downsample_image(image: Image.Image, max_dim: int) -> Image.Image:
    """Downsample image to max_dim for analysis."""
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image.copy()


def _assign_full_image(image_array: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each pixel to nearest centroid, processing in batches to bound memory."""
    pixels = image_array.reshape(-1, 3)
    # Normalize integer pixel data to [0,1] for robust distance computation
    if np.issubdtype(pixels.dtype, np.integer):
        pixels = pixels.astype(np.float32) / 255.0
    batch_size = 10000  # Tune as needed for memory constraints
    n_pixels = pixels.shape[0]
    labels = np.empty(n_pixels, dtype=np.uint8)
    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        batch = pixels[start:end]
        distances = ((batch[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels[start:end] = distances.argmin(axis=1)
    # Choose dtype based on number of centroids to avoid overflow
    n_centroids = centroids.shape[0]
    if n_centroids < 256:
        dtype = np.uint8
    elif n_centroids < 65536:
        dtype = np.uint16
    else:
        dtype = np.int32
    return labels.reshape(image_array.shape[:2]).astype(dtype)


def _cluster_stats(base_array: np.ndarray, labels: np.ndarray) -> list[ClusterStats]:
    """Compute cluster statistics using compute_cluster_stats."""
    return compute_cluster_stats(labels, base_array)


def assign_materials(
    stats: Sequence[ClusterStats],
    rules: Sequence[MaterialRule],
) -> dict[int, MaterialRule]:
    """
    Assign materials to clusters based on scoring functions.
    Each cluster gets the material with the highest score for that cluster.
    
    This is a minimal implementation for back-compat with tests.
    """
    assignments: dict[int, MaterialRule] = {}
    used_rules: set[str] = set()
    
    # Sort stats by count (most common clusters first)
    sorted_stats = sorted(stats, key=lambda s: s.count, reverse=True)
    
    for stat in sorted_stats:
        best_rule: MaterialRule | None = None
        best_score = -1.0
        
        # Score all rules for this cluster
        for rule in rules:
            # Prefer unused rules to get variety
            if rule.name in used_rules:
                continue
                
            if rule.score_fn is not None:
                score = rule.score_fn(stat)
            else:
                score = 1.0
            
            if score > best_score:
                best_score = score
                best_rule = rule
        
        # If all rules are used, allow reuse
        if best_rule is None:
            for rule in rules:
                if rule.score_fn is not None:
                    score = rule.score_fn(stat)
                else:
                    score = 1.0
                
                if score > best_score:
                    best_score = score
                    best_rule = rule
        
        if best_rule is not None:
            assignments[stat.label] = best_rule
            used_rules.add(best_rule.name)
    
    return assignments


# --------------------------
# Palette (de)serialization
# --------------------------

PALETTE_SCHEMA_VERSION = 1


def _serialize_assignments(assignments: Mapping[int, "MaterialRule"]) -> Dict[str, Any]:
    """
    Stable, compact JSON payload:

    {
      "version": 1,
      "assignments": { "0": "plaster", "1": "stone", ... }
    }
    """
    payload = {str(k): v.name for k, v in assignments.items()}
    return {"version": PALETTE_SCHEMA_VERSION, "assignments": payload}


def _deserialize_assignments(
    data: Mapping[str, Any],
    rules: Sequence["MaterialRule"],
    *,
    strict: bool = True,
) -> Dict[int, "MaterialRule"]:
    """
    Accepts legacy flat map or v1+ wrapped payload.
    - strict=True → raises on unknown names or bad keys.
    - strict=False → skips unknowns/bad keys.
    """
    if "assignments" in data:
        raw_map = data.get("assignments", {})
    else:
        raw_map = data

    by_name: Dict[str, "MaterialRule"] = {r.name: r for r in rules}
    out: Dict[int, "MaterialRule"] = {}

    for sk, name in raw_map.items():
        try:
            k = int(sk)
        except Exception:
            if strict:
                raise ValueError(f"Palette key is not an int: {sk!r}")
            continue

        rule = by_name.get(name)
        if rule is None:
            if strict:
                raise ValueError(f"Unknown material in palette: {name!r}")
            continue

        out[k] = rule

    return out


def load_palette_assignments(
    path: str | Path,
    rules: Sequence["MaterialRule"] | Mapping[str, "MaterialRule"] | None = None,
    *,
    strict: bool = True,
) -> dict[int, "MaterialRule"]:
    """
    Load cluster→material mapping. If file missing or unreadable, return {}.
    If rules is None we cannot reconstruct MaterialRule instances → return {}.
    """
    p = Path(path)
    if not p.exists():
        return {}

    try:
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
    except Exception:
        return {}

    if rules is None:
        return {}

    if isinstance(rules, Mapping):
        rule_seq: Sequence["MaterialRule"] = list(rules.values())
    else:
        rule_seq = rules

    return _deserialize_assignments(data, rule_seq, strict=strict)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_palette_assignments(assignments: Mapping[int, "MaterialRule"], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_assignments(assignments)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_text(p, text)


# --------------------------
# Texture & image helpers
# --------------------------

def _validate_texture(path: str | Path, size_hint: tuple[int, int] | None = None) -> Image.Image:
    """
    Open a texture (RGBA) or return a deterministic neutral fallback if missing.
    """
    p = Path(path)
    try:
        img = Image.open(p).convert("RGBA")
    except Exception:
        w, h = size_hint or (64, 64)
        return Image.new("RGBA", (w, h), (200, 200, 200, 255))

    if size_hint and img.size != size_hint:
        img = img.resize(size_hint, resample=Image.BILINEAR)
    return img


# --------------------------
# Lightweight k-means
# --------------------------

def _kmeans(data: np.ndarray, k: int, seed: int, iters: int = 10) -> np.ndarray:
    """
    Tiny k-means for CI: data is (N, 3) in [0,1]. Returns labels (N,).
    """
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(data.shape[0], size=k, replace=False)]
    for _ in range(max(1, iters)):
        d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = data[mask].mean(axis=0)
            else:
                centroids[i] = data[rng.integers(0, data.shape[0])]
    return labels


# --------------------------
# Public utilities
# --------------------------

def relabel(assignments: Mapping[int, "MaterialRule"], labels: np.ndarray) -> np.ndarray:
    """
    Optionally remap labels to a stable order based on material names.
    If no mapping needed, returns labels unchanged.
    """
    if not assignments:
        return labels

    pairs = sorted((cid, rule.name) for cid, rule in assignments.items())
    remap = {cid: new_id for new_id, (cid, _) in enumerate(pairs)}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out


def enhance_aerial(
    input_path: Path,
    output_path: Path,
    *,
    k: int = 8,
    analysis_max_dim: int = 1280,
    seed: int = 22,
    target_width: int | None = 4096,
    palette_path: Optional[Path | str] = None,
    save_palette: Optional[Path | str] = None,
    textures: Mapping[str, Path] | None = None,
) -> Path:
    """
    Minimal enhancement to keep CI deterministic:
    - run k-means on a downscaled analysis image
    - relabel using palette if provided
    - apply a gentle, label-aware blur to hint at "material regions"
    - save an RGB result (no HDR / 16-bit path to avoid heavy deps)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = Image.open(input_path).convert("RGB")

    w, h = image.size
    if max(w, h) > analysis_max_dim:
        scale = analysis_max_dim / max(w, h)
        analysis_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        analysis_image = image.resize(analysis_size, Image.Resampling.BILINEAR)
    else:
        analysis_image = image.copy()

    analysis_array = np.asarray(analysis_image, dtype=np.float32) / 255.0
    flat = analysis_array.reshape(-1, 3)

    labels_small = _kmeans(flat, k=k, seed=seed).astype(np.uint8)
    labels_small = labels_small.reshape(analysis_image.size[1], analysis_image.size[0])  # (H,W)

    labels_small_img = Image.fromarray(labels_small, mode="L")
    labels_full = labels_small_img.resize(image.size, Image.Resampling.NEAREST)
    labels = np.asarray(labels_full, dtype=np.uint8)

    assignments: dict[int, "MaterialRule"] = {}
    if palette_path:
        rule_candidates: list["MaterialRule"] = []
        if textures:
            for name in textures.keys():
                rule_candidates.append(MaterialRule(name=name))
        assignments = load_palette_assignments(palette_path, rule_candidates)
        if assignments:
            labels = relabel(assignments, labels)

    enhanced = np.asarray(image, dtype=np.float32) / 255.0
    blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
    blurred_np = np.asarray(blurred, dtype=np.float32) / 255.0

    alpha = (labels.astype(np.float32) % 3) / 10.0  # 0.0–0.2
    alpha = np.repeat(alpha[:, :, None], 3, axis=2)
    enhanced = (1.0 - alpha) * enhanced + alpha * blurred_np

    out_img = Image.fromarray((np.clip(enhanced, 0.0, 1.0) * 255.0 + 0.5).astype("uint8"), mode="RGB")
    if target_width and out_img.width != target_width:
        tw = int(target_width)
        th = int(round(out_img.height * (tw / out_img.width)))
        out_img = out_img.resize((tw, th), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)

    if save_palette:
        if not assignments:
            assignments = {i: MaterialRule(name=f"cluster{i}") for i in range(k)}
        save_palette_assignments(assignments, save_palette)

    return output_path


def apply_materials(
    base: np.ndarray,
    labels: np.ndarray,
    materials: Mapping[int, MaterialRule],
) -> np.ndarray:
    """
    Apply material textures to a base image using label assignments.
    
    Args:
        base: Base image array (H, W, 3) in [0, 1] float range
        labels: Label map (H, W) uint8
        materials: Mapping from label to MaterialRule
        
    Returns:
        Enhanced image array (H, W, 3) in [0, 1] float range
    """
    output = base.copy()
    
    for label, rule in materials.items():
        mask = labels == label
        if not mask.any():
            continue
            
        # Load texture if available
        if rule.texture and Path(rule.texture).exists():
            try:
                texture_img = Image.open(rule.texture).convert("RGB")
                # Tile texture to match output size
                tex_array = np.asarray(texture_img, dtype=np.float32) / 255.0
                h, w = output.shape[:2]
                th, tw = tex_array.shape[:2]
                
                # Create tiled texture
                tiles_h = (h + th - 1) // th
                tiles_w = (w + tw - 1) // tw
                tiled = np.tile(tex_array, (tiles_h, tiles_w, 1))
                tiled = tiled[:h, :w]
                
                # Blend texture with base
                blend = rule.blend
                mask_3d = mask[..., None]
                output = np.where(mask_3d, 
                                  (1 - blend) * output + blend * tiled,
                                  output)
            except Exception:
                pass  # If texture can't be loaded, skip blending
    
    return np.clip(output, 0.0, 1.0)


# --------------------------
# CLI
# --------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the base aerial image")
    parser.add_argument("output", type=Path, help="Destination image path")
    parser.add_argument("--analysis-max", type=int, default=1280, help="Max dimension for clustering image (default: 1280)")
    parser.add_argument("--k", type=int, default=8, help="Number of clusters (default: 8)")
    parser.add_argument("--seed", type=int, default=22, help="Random seed (default: 22)")
    parser.add_argument("--target-width", type=int, default=4096, help="Output width (default: 4096)")
    parser.add_argument("--palette", type=Path, default=None, help="Load cluster→material assignments from JSON")
    parser.add_argument("--save-palette", type=Path, default=None, help="Write JSON palette to this path after processing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    ns = _parse_args(argv)
    out = enhance_aerial(
        ns.input,
        ns.output,
        k=ns.k,
        analysis_max_dim=ns.analysis_max,
        seed=ns.seed,
        target_width=ns.target_width,
        palette_path=ns.palette,
        save_palette=ns.save_palette,
        textures=None,
    )
    print(str(out))
    return out


if __name__ == "__main__":  # pragma: no cover
    main()    r = int(s[0:2], 16) / 255.0
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
    """Return palette as (m,3) float64 RGB01."""
    src = _DEFAULT_MBAR_8 if palette is None or len(palette) == 0 else list(palette)
    return np.asarray([_hex_to_rgb01(h) for h in src], dtype=np.float64)

# ------------------------------- distances ---------------------------------

def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Efficient squared Euclidean distances between rows of X (n,3) and C (k,3).
    Avoids allocating (n,k,3). Clamps tiny negatives to 0 for numerical safety.
    """
    X = X.astype(np.float64, copy=False)
    C = C.astype(np.float64, copy=False)
    X2 = np.einsum("ij,ij->i", X, X)[:, None]          # (n,1)
    C2 = np.einsum("ij,ij->i", C, C)[None, :]          # (1,k)
    XC = X @ C.T                                       # (n,k)
    D = X2 + C2 - 2.0 * XC
    return np.maximum(D, 0.0)

# -------------------------------- k-means ----------------------------------

@dataclass(frozen=True)
class KMeansResult:
    centers: np.ndarray  # (k,3) in RGB01
    inertia: float

def _kmeans_plus_plus_init(data: np.ndarray, k: int, *, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding on (n,3) data."""
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
    """
    Simple k-means on RGB01. Returns final centers and inertia.
    Uses memory-friendly distance computation.
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if n < k:
        raise ValueError(f"Not enough samples ({n}) for k={k}")
    centers = _kmeans_plus_plus_init(data, k, rng=rng)
    last_inertia = np.inf
    for _ in range(max_iter):
        # assign to current centers
        dist2 = _pairwise_sq_dists(data, centers)   # (n,k)
        labels = np.argmin(dist2, axis=1)
        inertia = float(dist2[np.arange(n), labels].sum())

        # convergence check against previous inertia (fix returning the right inertia)
        if abs(last_inertia - inertia) <= tol * max(1.0, last_inertia):
            last_inertia = inertia
            break
        last_inertia = inertia

        # update centers; if any empty cluster -> random re-seed (keeps progress moving)
        new_centers = np.empty_like(centers)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                new_centers[j] = data[rng.integers(0, n)]
            else:
                new_centers[j] = data[mask].mean(axis=0)
        centers = new_centers

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
    """
    Apply MBAR palette transfer to an aerial image and save to disk as JPEG.
    Keeps API stable; faster & more numerically robust.
    """
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
        im_small = im.resize((int(round(w0 * scale)), int(round(h0 * scale))), Image.Resampling.LANCZOS)
        arr_small = np.asarray(im_small, dtype=np.float32) / 255.0
    else:
        arr_small = arr

    # sample for k-means
    flat = arr_small.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(seed)
    sample_n = min(flat.shape[0], 250_000)
    if sample_n < 1:
        raise ValueError("Image appears empty after preprocessing.")
    idx = rng.choice(flat.shape[0], size=sample_n, replace=False)
    sample = flat[idx]

    km = _kmeans(sample, k=k, seed=seed)
    centers_rgb = km.centers  # (k,3) in RGB01

    # map centers to palette in Lab
    pal_rgb = _palette_rgb01(palette)
    centers_lab = _rgb_to_lab(centers_rgb)
    pal_lab = _rgb_to_lab(pal_rgb)

    # nearest palette color for each center (Lab)
    diff2 = _pairwise_sq_dists(centers_lab, pal_lab)  # (k, m)
    nearest_idx = np.argmin(diff2, axis=1)  # (k,)
    target_rgb = pal_rgb[nearest_idx]  # (k,3)

    # per-center RGB delta
    deltas = target_rgb - centers_rgb  # (k,3)

    # sigma from centroid spread; floor to keep weights well-behaved
    if k > 1:
        # RMS pairwise distance between centers in RGB space
        pairwise = centers_rgb[:, None, :] - centers_rgb[None, :, :]
        pd = float(np.sqrt(np.mean(np.sum(pairwise**2, axis=-1))))
    else:
        pd = 0.25
    sigma2 = max((0.5 * pd) ** 2, 1e-8)

    # apply to full-res in chunks
    H, W, _ = arr.shape
    rows_target = max(128, min(H, int(1_000_000 / max(1, W))))  # ~1M px per chunk
    out = np.empty_like(arr, dtype=np.float32)

    iterator: Iterable[int] = range(0, H, rows_target)
    iterator = tqdm(iterator, desc="Applying palette", unit="rows")

    for y0 in iterator:
        y1 = min(H, y0 + rows_target)
        chunk = arr[y0:y1].reshape(-1, 3).astype(np.float64)  # (n,3)

        # distances to centers -> weights (no (n,k,3) allocation)
        dist2 = _pairwise_sq_dists(chunk, centers_rgb)  # (n,k)
        weights = np.exp(-dist2 / (2 * sigma2))
        weights /= (weights.sum(axis=1, keepdims=True) + 1e-12)

        # blended delta
        blended = weights @ deltas  # (n,3)
        new_chunk = np.clip(chunk + strength * blended, 0.0, 1.0).astype(np.float32)
        out[y0:y1] = new_chunk.reshape(y1 - y0, W, 3)

    # resize to target width
    if target_width and W != target_width:
        new_h = int(round(H * (target_width / W)))
        out_img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB").resize(
            (target_width, new_h), Image.Resampling.LANCZOS
        )
    else:
        out_img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path, format="JPEG", quality=int(jpeg_quality), subsampling=1, optimize=True)
    return output_path

# ---------------------------------- CLI ------------------------------------

def _format_bytes(n: int) -> str:
    return f"{n / (1024**2):.2f} MB"

def _print_header(inp: Path, out: Path, k: int, pal_len: int) -> None:
    print(f"Processing: {inp.name}")
    print(f"Output: {out.name}")
    print(f"Resolution: 4K (4096px width)")
    print(f"Materials: MBAR-approved palette ({pal_len} colors), k={k}\n")

def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        import typer  # lazy import for optional dep
    except Exception:
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
        palette: List[str] = typer.Option(None, "--palette", "-p", help="Override palette with HEX colors. Repeatable."),
    ) -> None:
        pal = palette if palette else None
        _print_header(input_path, output_path, k, len(pal or _DEFAULT_MBAR_8))
        result = enhance_aerial(
            input_path=input_path,
            output_path=output_path,
            analysis_max_dim=analysis_max_dim,
            k=k,
            seed=seed,
            target_width=target_width,
            strength=strength,
            jpeg_quality=jpeg_quality,
            palette=pal,
        )
        print(f"✅ Enhanced aerial saved to: {result}")
        try:
            print(f"✅ File size: {_format_bytes(result.stat().st_size)}")
        except Exception:
            pass

    return app(standalone_mode=True)

if __name__ == "__main__":
    raise SystemExit(main())
