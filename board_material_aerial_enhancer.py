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
from typing import Mapping, Sequence, Optional, TYPE_CHECKING, Dict, Any

# --- third-party (kept light) ---
import numpy as np
from PIL import Image, ImageFilter

# Optional dependency (keep soft so CI stays lean)
try:  # pragma: no cover - optional
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional
    tifffile = None  # type: ignore

# If the real class is available elsewhere, use it only for typing;
# otherwise provide a tiny stub that satisfies runtime & tests.
if TYPE_CHECKING:  # pragma: no cover
    from material_response import MaterialRule  # type: ignore
else:
    from typing import Callable

    @dataclass(frozen=True)
    class MaterialRule:  # minimal stub with extended attributes
        name: str
        texture: str = ""
        blend: float = 0.8
        score_fn: Callable[["ClusterStats"], float] = lambda _: 0.0


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
    "apply_materials",
    "assign_materials",
    "build_material_rules",
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
    mean_rgb: Optional[np.ndarray] = None
    mean_hsv: Optional[np.ndarray] = None
    std_rgb: Optional[np.ndarray] = None


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to HSV [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    delta = maxc - minc
    s = np.where(maxc != 0, delta / maxc, 0.0)

    rc = np.where(delta != 0, (maxc - r) / delta, 0.0)
    gc = np.where(delta != 0, (maxc - g) / delta, 0.0)
    bc = np.where(delta != 0, (maxc - b) / delta, 0.0)

    h = np.where(r == maxc, bc - gc,
                 np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)

    return np.stack([h, s, v], axis=-1)


def compute_cluster_stats(labels: np.ndarray, rgb: np.ndarray) -> list[ClusterStats]:
    """
    Compute simple stats per label:
      - pixel count
      - centroid color (mean RGB in [0,1])
      - mean RGB, HSV, and standard deviation

    labels: (H, W) uint dtype
    rgb:    (H, W, 3) uint8 or float in [0,1]
    """
    if rgb.dtype.kind in ("u", "i"):
        rgb_f = rgb.astype(np.float32) / 255.0
    else:
        rgb_f = rgb.astype(np.float32)

    labs = labels.reshape(-1)
    flat = rgb_f.reshape(-1, 3)

    # Convert to HSV
    hsv_f = _rgb_to_hsv(rgb_f)
    flat_hsv = hsv_f.reshape(-1, 3)

    out: list[ClusterStats] = []
    for lab in np.unique(labs).tolist():
        mask = labs == lab
        cnt = int(mask.sum())
        if cnt:
            mean_rgb = flat[mask].mean(axis=0)
            centroid = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
            mean_hsv = flat_hsv[mask].mean(axis=0)
            std_rgb = flat[mask].std(axis=0)
        else:
            centroid = (0.0, 0.0, 0.0)
            mean_rgb = np.zeros(3, dtype=flat.dtype)
            mean_hsv = np.zeros(3, dtype=flat_hsv.dtype)
            std_rgb = np.zeros(3, dtype=flat.dtype)
        out.append(ClusterStats(
            label=int(lab),
            count=cnt,
            centroid=centroid,
            mean_rgb=mean_rgb,
            mean_hsv=mean_hsv,
            std_rgb=std_rgb
        ))
    return out


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
    analysis_max: int = 1280,
    analysis_max_dim: Optional[int] = None,
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
    # Support both analysis_max and analysis_max_dim for backward compatibility
    if analysis_max_dim is not None:
        analysis_max = analysis_max_dim

    input_path = Path(input_path)
    output_path = Path(output_path)
    image = Image.open(input_path).convert("RGB")

    w, h = image.size
    if max(w, h) > analysis_max:
        scale = analysis_max / max(w, h)
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


def build_material_rules(textures: Mapping[str, Path]) -> list[MaterialRule]:
    """
    Build material rules from texture paths with default scoring functions.

    Each material is assigned a scoring function based on typical properties:
    - plaster: prefers light, neutral colors (high value, low saturation)
    - stone: prefers medium-light colors with some texture variation
    - cladding: prefers medium tones
    - screens: prefers medium-dark colors
    - equitone: prefers darker colors with low saturation
    - roof: prefers medium to dark grays
    - bronze: prefers darker, warmer tones
    - shade: prefers very light colors (shadows/openings)
    """
    def plaster_score(stats: ClusterStats) -> float:
        """Score for plaster - prefers light, neutral colors."""
        if stats.mean_hsv is None:
            return 0.0
        s, v = stats.mean_hsv[1], stats.mean_hsv[2]
        # High value (brightness), low saturation
        return float(v * (1.0 - s * 0.5))

    def stone_score(stats: ClusterStats) -> float:
        """Score for stone - prefers medium-light colors with texture."""
        if stats.mean_hsv is None or stats.std_rgb is None:
            return 0.0
        v = stats.mean_hsv[2]
        texture = float(stats.std_rgb.mean())
        # Medium to light value, some texture variation
        return float((1.0 - abs(v - 0.65)) * (1.0 + texture * 2.0))

    def cladding_score(stats: ClusterStats) -> float:
        """Score for cladding - prefers medium tones."""
        if stats.mean_hsv is None:
            return 0.0
        v = stats.mean_hsv[2]
        # Medium value range
        return float(1.0 - abs(v - 0.55))

    def screens_score(stats: ClusterStats) -> float:
        """Score for screens - prefers medium-dark colors."""
        if stats.mean_hsv is None:
            return 0.0
        v = stats.mean_hsv[2]
        # Medium-dark value
        return float(1.0 - abs(v - 0.45))

    def equitone_score(stats: ClusterStats) -> float:
        """Score for equitone - prefers darker colors with low saturation."""
        if stats.mean_hsv is None:
            return 0.0
        s, v = stats.mean_hsv[1], stats.mean_hsv[2]
        # Dark value, low saturation
        return float((1.0 - v) * (1.0 - s * 0.5))

    def roof_score(stats: ClusterStats) -> float:
        """Score for roof - prefers medium to dark grays."""
        if stats.mean_hsv is None:
            return 0.0
        s, v = stats.mean_hsv[1], stats.mean_hsv[2]
        # Medium-dark value, very low saturation
        return float((1.0 - abs(v - 0.5)) * (1.0 - s))


    def bronze_score(stats: ClusterStats) -> float:
        """Score for bronze - prefers darker, warmer tones."""
        BRONZE_HUE_MIN = 0.0  # Lower bound for orange-brown hue
        BRONZE_HUE_MAX = 0.15  # Upper bound for orange-brown hue characteristic of bronze
        WARM_TONE_MULTIPLIER = 1.0  # Multiplier for warm tones
        NEUTRAL_TONE_MULTIPLIER = 0.5  # Multiplier for non-warm tones
        if stats.mean_hsv is None:
            return 0.0
        h, v = stats.mean_hsv[0], stats.mean_hsv[2]
        # Dark value, warm hue (orange-brown range characteristic of bronze)
        warm = WARM_TONE_MULTIPLIER if BRONZE_HUE_MIN <= h <= BRONZE_HUE_MAX else NEUTRAL_TONE_MULTIPLIER
        return float((1.0 - v) * warm)

    def shade_score(stats: ClusterStats) -> float:
        """Score for shade - prefers very light colors."""
        if stats.mean_hsv is None:
            return 0.0
        v = stats.mean_hsv[2]
        # Very high value (brightness)
        return float(v ** 2)

    # Map material names to scoring functions
    score_functions = {
        "plaster": plaster_score,
        "stone": stone_score,
        "cladding": cladding_score,
        "screens": screens_score,
        "equitone": equitone_score,
        "roof": roof_score,
        "bronze": bronze_score,
        "shade": shade_score,
    }

    rules: list[MaterialRule] = []
    for name, texture_path in textures.items():
        score_fn = score_functions.get(name, lambda _: 0.5)
        rules.append(MaterialRule(
            name=name,
            texture=str(texture_path),
            blend=0.8,
            score_fn=score_fn
        ))

    return rules


def assign_materials(
    stats: Sequence[ClusterStats],
    rules: Sequence[MaterialRule]
) -> dict[int, MaterialRule]:
    """
    Assign materials to clusters based on scoring functions.

    For each cluster, evaluate all material rules and assign the one with
    the highest score. This allows materials to be matched to clusters based
    on their visual properties (color, saturation, texture, etc.).
    """
    assignments: dict[int, MaterialRule] = {}

    for cluster in stats:
        best_rule: Optional[MaterialRule] = None
        best_score = -1.0

        for rule in rules:
            score = rule.score_fn(cluster)
            if score > best_score:
                best_score = score
                best_rule = rule

        if best_rule is not None:
            assignments[cluster.label] = best_rule

    return assignments


def apply_materials(
    base: np.ndarray,
    labels: np.ndarray,
    assignments: dict[int, MaterialRule],
) -> np.ndarray:
    """
    Apply materials to an image array using texture blending.

    Args:
        base: Base image array (H, W, 3) in float [0,1]
        labels: Label array (H, W) with cluster assignments
        assignments: Mapping from cluster labels to MaterialRule instances

    Returns:
        Enhanced image array with textures blended
    """
    result = base.copy()

    # Apply each material's texture
    for label, rule in assignments.items():
        mask = labels == label
        if not mask.any():
            continue

        # Find bounding box of mask
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            raise ValueError(f"Empty mask coordinates detected for label {label}")
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        box_height = y_max - y_min + 1
        box_width = x_max - x_min + 1
        # Load and resize texture only to the bounding box of the mask
        try:
            texture_img = Image.open(rule.texture).convert("RGB")
            # Resize texture to bounding box size
            texture_resized = texture_img.resize((box_width, box_height), Image.Resampling.BILINEAR)
            texture_array = np.asarray(texture_resized, dtype=np.float32) / 255.0
        except (OSError, ValueError):
            # Fallback to neutral color if texture can't be loaded
            texture_array = np.ones((box_height, box_width, 3), dtype=np.float32) * 0.5

        # Blend texture with base image only in bounding box region
        mask_box = mask[y_min:y_max+1, x_min:x_max+1]
        base_box = base[y_min:y_max+1, x_min:x_max+1]
        result_box = result[y_min:y_max+1, x_min:x_max+1]
        # mask_box_3d was unused and removed
        # Only blend masked pixels to avoid unnecessary computation
        blended_pixels = base_box[mask_box] * (1.0 - rule.blend) + texture_array[mask_box] * rule.blend
        result_box[mask_box] = blended_pixels
        result[y_min:y_max+1, x_min:x_max+1] = result_box

    return result


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
        analysis_max=ns.analysis_max,
        seed=ns.seed,
        target_width=ns.target_width,
        palette_path=ns.palette,
        save_palette=ns.save_palette,
        textures=None,
    )
    print(str(out))
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
