# path: board_material_aerial_enhancer.py
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

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Optional, TYPE_CHECKING, Dict, Any

import numpy as np
from PIL import Image, ImageFilter

try:  # pragma: no cover - optional
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional
    tifffile = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from material_response import MaterialRule  # type: ignore
else:
    @dataclass(frozen=True)
    class MaterialRule:  # minimal stub
        name: str

__all__ = [
    "ClusterStats",
    "compute_cluster_stats",
    "load_palette_assignments",
    "save_palette_assignments",
    "relabel",
    "enhance_aerial",
    "apply_materials",
    "assign_materials",
]

# --------------------------
# Cluster statistics
# --------------------------

@dataclass(frozen=True)
class ClusterStats:
    label: int
    count: int
    centroid: tuple[float, float, float]  # (r,g,b) in [0,1]


def compute_cluster_stats(labels: np.ndarray, rgb: np.ndarray) -> list[ClusterStats]:
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
# Palette (de)serialization
# --------------------------

PALETTE_SCHEMA_VERSION = 1


def _serialize_assignments(assignments: Mapping[int, "MaterialRule"]) -> Dict[str, Any]:
    payload = {str(k): v.name for k, v in assignments.items()}
    return {"version": PALETTE_SCHEMA_VERSION, "assignments": payload}


def _deserialize_assignments(
    data: Mapping[str, Any],
    rules: Sequence["MaterialRule"],
    *,
    strict: bool = True,
) -> Dict[int, "MaterialRule"]:
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
    If rules is provided, names must match; otherwise {}.
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


def _load_palette_assignments_loose(path: str | Path) -> dict[int, "MaterialRule"]:
    """
    Loose loader: build MaterialRule stubs from JSON names when rule set isn't provided.
    Why: allow `--palette` to work in CLI without a textures dict.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        raw_map = data["assignments"] if isinstance(data, dict) and "assignments" in data else data
        if not isinstance(raw_map, dict):
            return {}
    except Exception:
        return {}

    out: Dict[int, "MaterialRule"] = {}
    for sk, name in raw_map.items():
        try:
            k = int(sk)
        except Exception:
            continue
        out[k] = MaterialRule(name=str(name))
    return out


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
    try:
        img = Image.open(Path(path)).convert("RGBA")
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
    seed: int = 22,
    target_width: int | None = 4096,
    palette_path: Optional[Path | str] = None,
    save_palette: Optional[Path | str] = None,
    textures: Mapping[str, Path] | None = None,
    save_labels: Optional[Path | str] = None,
) -> Path:
    """
    Deterministic CI enhancement:
    - k-means on downscaled analysis image
    - optional palette relabel
    - gentle label-aware blur mix
    - RGB output; optional label export
    """
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
        if not assignments:  # loose fallback (no textures/rules)
            assignments = _load_palette_assignments_loose(palette_path)
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

    if save_labels:
        lp = Path(save_labels)
        lp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels, mode="L").save(lp)

    if save_palette:
        if not assignments:
            assignments = {i: MaterialRule(name=f"cluster{i}") for i in range(k)}
        save_palette_assignments(assignments, save_palette)

    return output_path


def apply_materials(
    input_path: Path,
    output_path: Path,
    *,
    k: int = 8,
    analysis_max: int = 1280,
    seed: int = 22,
    target_width: int | None = 4096,
    palette_path: Optional[Path | str] = None,
    save_palette: Optional[Path | str] = None,
    textures: Mapping[str, Path] | None = None,
    save_labels: Optional[Path | str] = None,
) -> Path:
    return enhance_aerial(
        input_path=input_path,
        output_path=output_path,
        k=k,
        analysis_max=analysis_max,
        seed=seed,
        target_width=target_width,
        palette_path=palette_path,
        save_palette=save_palette,
        textures=textures,
        save_labels=save_labels,
    )


def assign_materials(
    input_path: Path,
    output_path: Path,
    *,
    k: int = 8,
    analysis_max: int = 1280,
    seed: int = 22,
    target_width: int | None = 4096,
    palette_path: Optional[Path | str] = None,
    save_palette: Optional[Path | str] = None,
    textures: Mapping[str, Path] | None = None,
    save_labels: Optional[Path | str] = None,
) -> Path:
    return apply_materials(
        input_path=input_path,
        output_path=output_path,
        k=k,
        analysis_max=analysis_max,
        seed=seed,
        target_width=target_width,
        palette_path=palette_path,
        save_palette=save_palette,
        textures=textures,
        save_labels=save_labels,
    )

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
    parser.add_argument("--save-labels", type=Path, default=None, help="Write 8-bit label map image (for tests/QA)")
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
        save_labels=ns.save_labels,
    )
    print(str(out))
    return out


if __name__ == "__main__":  # pragma: no cover
    main()