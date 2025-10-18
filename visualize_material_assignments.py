# file: visualize_material_assignments.py
"""Generate a visualization showing MBAR material assignments for an aerial image.

Usage:
  python visualize_material_assignments.py \
      --input /path/to/aerial.tiff \
      --output /path/to/map.jpg \
      --k 8 --seed 22 --max-dim 1280 \
      [--threshold 35.0] \
      [--palette /path/to/assignments.json] \
      [--save-palette /path/to/save.json]

The palette JSON maps cluster index (string) -> material name, e.g.:
  { "0": "sand", "1": "oak", ... }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ------------------------------- color utils -------------------------------

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


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
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883  # D65 white
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    eps = (6 / 29) ** 3
    k = (29 / 3) ** 2 / 3

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(t), k * t + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    return _xyz_to_lab(_rgb_to_xyz(rgb01))


def _hex_to_rgb01(code: str) -> Tuple[float, float, float]:
    s = code.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


# ------------------------------ MBAR palette -------------------------------

DEFAULT_TEXTURES: Mapping[str, str] = {
    "warm_white": "#F5F2EC",
    "sand": "#D8C7A0",
    "stone": "#B5A58B",
    "oak": "#9A7957",
    "slate": "#6A6D6F",
    "charcoal": "#3A3C3E",
    "sky_haze": "#CFD6DF",
    "deep_teal": "#244A5A",
}


@dataclass(frozen=True)
class MaterialRule:
    name: str
    lab: Tuple[float, float, float]
    threshold: float = 35.0  # why: ΔE76 cutoff; above = leave unassigned


def build_material_rules(textures: Mapping[str, str], *, threshold: float = 35.0) -> Dict[str, MaterialRule]:
    rules: Dict[str, MaterialRule] = {}
    pal_rgb = np.asarray([_hex_to_rgb01(h) for h in textures.values()], dtype=np.float64)
    pal_lab = _rgb_to_lab(pal_rgb)
    for (name, _), lab in zip(textures.items(), pal_lab):
        rules[name] = MaterialRule(
            name=name,
            lab=(float(lab[0]), float(lab[1]), float(lab[2])),
            threshold=threshold,
        )
    return rules


# ------------------------------- k-means core ------------------------------

def _downsample_image(image: Image.Image, max_dim: int) -> Image.Image:
    im = ImageOps.exif_transpose(image).convert("RGB")
    w, h = im.size
    if max_dim <= 0 or max(w, h) <= max_dim:
        return im
    scale = max_dim / float(max(w, h))
    return im.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)


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


def _kmeans(sample: np.ndarray, k: int, *, seed: int, max_iter: int = 25, tol: float = 1e-4) -> np.ndarray:
    if sample.ndim != 2 or sample.shape[1] != 3:
        raise ValueError("sample must be (N,3) RGB01")
    if sample.shape[0] < k:
        raise ValueError(f"Not enough samples ({sample.shape[0]}) for k={k}")
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus_init(sample, k, rng=rng)
    last_inertia = np.inf
    for _ in range(max_iter):
        d = sample[:, None, :] - centers[None, :, :]
        dist2 = np.einsum("nik,nik->ni", d, d)
        labels = np.argmin(dist2, axis=1)
        new_centers = np.empty_like(centers)
        for j in range(k):
            mask = labels == j
            new_centers[j] = sample[rng.integers(0, sample.shape[0])] if not np.any(mask) else sample[mask].mean(axis=0)
        centers = new_centers
        inertia = float(dist2[np.arange(sample.shape[0]), labels].sum())
        if abs(last_inertia - inertia) <= tol * max(1.0, last_inertia):
            break
        last_inertia = inertia
    return centers


def _assign_full_image(arr01: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return label map (H,W) by nearest centroid for arr01 in RGB01."""
    h, w, _ = arr01.shape
    flat = arr01.reshape(-1, 3).astype(np.float64)
    d = flat[:, None, :] - centers[None, :, :]
    dist2 = np.einsum("nik,nik->ni", d, d)
    labels = np.argmin(dist2, axis=1).astype(np.uint8)
    return labels.reshape(h, w)


# ----------------------------- cluster analysis ----------------------------

@dataclass
class ClusterStat:
    index: int
    mean_rgb: Tuple[float, float, float]
    mean_lab: Tuple[float, float, float]
    fraction: float
    count: int


def _cluster_stats(base_arr01: np.ndarray, labels: np.ndarray) -> Dict[int, ClusterStat]:
    H, W, _ = base_arr01.shape
    total = H * W
    stats: Dict[int, ClusterStat] = {}
    k = int(labels.max()) + 1
    flat = base_arr01.reshape(-1, 3).astype(np.float64)
    lab_flat = _rgb_to_lab(flat).reshape(H, W, 3).reshape(-1, 3)
    label_flat = labels.reshape(-1)
    for idx in range(k):
        mask = label_flat == idx
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mean_rgb = flat[mask].mean(axis=0)
        mean_lab = lab_flat[mask].mean(axis=0)
        stats[idx] = ClusterStat(
            index=idx,
            mean_rgb=(float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])),
            mean_lab=(float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])),
            fraction=cnt / total,
            count=cnt,
        )
    return stats


# ---------------------------- material assignment --------------------------

def _delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab1 - lab2, axis=-1)


def assign_materials(
    stats: Mapping[int, ClusterStat],
    rules: Mapping[str, MaterialRule],
) -> Dict[int, MaterialRule]:
    """Return mapping cluster index -> MaterialRule for confident matches."""
    assigned: Dict[int, MaterialRule] = {}
    rule_names = list(rules.keys())
    rule_labs = np.asarray([rules[n].lab for n in rule_names], dtype=np.float64)

    for idx, stat in stats.items():
        lab = np.asarray(stat.mean_lab, dtype=np.float64)[None, :]  # (1,3)
        d = _delta_e76(lab, rule_labs)[0]
        j = int(np.argmin(d))
        rule = rules[rule_names[j]]
        if d[j] <= rule.threshold:
            assigned[idx] = rule
        # else: leave unassigned
    return assigned


def save_palette_assignments(assignments: Mapping[int, MaterialRule], path: Path) -> None:
    out = {str(k): v.name for k, v in assignments.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def load_palette_assignments(path: Path, rules: Mapping[str, MaterialRule]) -> Dict[int, MaterialRule]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[int, MaterialRule] = {}
    for k, name in raw.items():
        if name not in rules:
            continue
        try:
            out[int(k)] = rules[name]
        except ValueError:
            continue
    return out


# ----------------------------- viz helpers ---------------------------------

def _fixed_colors(k: int) -> np.ndarray:
    """Return k distinct RGB colors (uint8)."""
    base = np.array(
        [
            (255, 100, 100),  # red
            (100, 255, 100),  # green
            (100, 100, 255),  # blue
            (255, 255, 100),  # yellow
            (255, 100, 255),  # magenta
            (100, 255, 255),  # cyan
            (255, 200, 100),  # orange
            (200, 100, 255),  # purple
        ],
        dtype=np.uint8,
    )
    if k <= len(base):
        return base[:k]
    # extend with a simple hue wheel if k > 8
    extra = []
    for i in range(k - len(base)):
        hue = i / max(1, (k - len(base)))
        r = int(127 + 127 * np.sin(2 * np.pi * hue))
        g = int(127 + 127 * np.sin(2 * np.pi * (hue + 1 / 3)))
        b = int(127 + 127 * np.sin(2 * np.pi * (hue + 2 / 3)))
        extra.append((np.uint8(r), np.uint8(g), np.uint8(b)))
    return np.vstack([base, np.asarray(extra, dtype=np.uint8)])


# ----------------------------------- CLI -----------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize MBAR material assignments (k-means + Lab mapping).")
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to input aerial image (e.g., TIFF/JPEG).")
    p.add_argument("--output", "-o", type=Path, required=True, help="Path to output visualization (JPEG/PNG).")
    p.add_argument("--k", type=int, default=8, help="Number of color clusters.")
    p.add_argument("--seed", type=int, default=22, help="Random seed for k-means++.")
    p.add_argument("--max-dim", type=int, default=1280, help="Analysis max dimension for clustering.")
    p.add_argument("--threshold", type=float, default=35.0, help="ΔE76 threshold for material assignment.")
    p.add_argument("--palette", type=Path, default=None, help="Optional JSON mapping of cluster->material name.")
    p.add_argument("--save-palette", type=Path, default=None, help="Save computed cluster->material mapping.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load image
    with Image.open(args.input) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w0, h0 = im.size
        base_arr01 = np.asarray(im, dtype=np.float32) / 255.0

    # Downsample for analysis
    im_small = _downsample_image(im, args.max_dim)
    arr_small = np.asarray(im_small, dtype=np.float32) / 255.0

    # Sample pixels for k-means
    flat = arr_small.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(args.seed)
    sample_n = min(flat.shape[0], 200_000)
    idx = rng.choice(flat.shape[0], size=sample_n, replace=False) if sample_n < flat.shape[0] else np.arange(flat.shape[0])
    sample = flat[idx]

    # Clustering
    centers = _kmeans(sample, k=int(args.k), seed=int(args.seed))

    # Label downsampled → upsample labels to full-res
    labels_small = _assign_full_image(arr_small, centers)
    labels_small_img = Image.fromarray(labels_small, mode="L")
    labels_full_img = labels_small_img.resize((w0, h0), Image.Resampling.NEAREST)
    labels = np.asarray(labels_full_img, dtype=np.uint8)

    # Stats + rules
    stats = _cluster_stats(base_arr01, labels)
    rules = build_material_rules(DEFAULT_TEXTURES, threshold=float(args.threshold))

    # Load or assign materials
    if args.palette and args.palette.exists():
        assignments = load_palette_assignments(args.palette, rules)
        print(f"Loaded palette mapping from: {args.palette}")
    else:
        assignments = assign_materials(stats, rules)

    if args.save_palette:
        save_palette_assignments(assignments, args.save_palette)
        print(f"Saved palette mapping to: {args.save_palette}")

    # Visualization
    k = int(args.k)
    colors = _fixed_colors(k)
    viz = colors[labels]  # (H,W,3) uint8
    viz_img = Image.fromarray(viz, mode="RGB")

    # Legend sizing
    font = ImageFont.load_default()
    line_h = 24
    pad = 16
    box = 18

    assigned_keys = sorted(assignments.keys())
    unassigned_keys = [i for i in range(k) if i not in assignments]
    n_lines = 1 + len(assigned_keys) + (1 + len(unassigned_keys) if unassigned_keys else 0)
    legend_h = pad + n_lines * (line_h + 6) + pad

    out_w, out_h = viz_img.width, viz_img.height + legend_h
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    canvas.paste(viz_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = viz_img.height + pad
    x = 20

    draw.text((x, y), "MBAR MATERIAL ASSIGNMENTS", fill=(0, 0, 0), font=font)
    y += line_h + 8

    # Assigned entries
    for idx in assigned_keys:
        rule = assignments[idx]
        # color box
        draw.rectangle([x, y, x + box, y + box], fill=tuple(int(v) for v in colors[idx]), outline=(0, 0, 0))
        # coverage
        pct = stats.get(idx).fraction * 100.0 if idx in stats else 0.0
        label_text = f"{rule.name.upper()} – Cluster {idx} ({pct:.1f}%)"
        draw.text((x + box + 10, y - 2), label_text, fill=(0, 0, 0), font=font)
        y += line_h + 6

    # Unassigned
    if unassigned_keys:
        draw.text((x, y), "UNASSIGNED CLUSTERS", fill=(80, 80, 80), font=font)
        y += line_h + 6
        for idx in unassigned_keys:
            draw.rectangle([x, y, x + box, y + box], fill=tuple(int(v) for v in colors[idx]), outline=(0, 0, 0))
            pct = stats.get(idx).fraction * 100.0 if idx in stats else 0.0
            label_text = f"Cluster {idx} ({pct:.1f}%) – no material match"
            draw.text((x + box + 10, y - 2), label_text, fill=(90, 90, 90), font=font)
            y += line_h + 6

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95, optimize=True, subsampling=1)
    print(f"✅ Material assignment map saved to: {args.output}")

    # Console summary
    if assigned_keys:
        print("\nMaterial Assignments:")
        for idx in assigned_keys:
            pct = stats.get(idx).fraction * 100.0 if idx in stats else 0.0
            rule = assignments[idx]
            print(f"  • {rule.name.upper()}: Cluster {idx} ({pct:.1f}% of image)")
    if unassigned_keys:
        print(f"\nUnassigned Clusters: {len(unassigned_keys)}")
        for idx in unassigned_keys:
            pct = stats.get(idx).fraction * 100.0 if idx in stats else 0.0
            print(f"  • Cluster {idx}: {pct:.1f}% of image (below threshold)")


if __name__ == "__main__":
    main()