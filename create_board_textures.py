# path: tools/generate_mbar_textures.py
"""Generate placeholder textures for MBAR board materials.

Procedurally generates texture plates using MBAR palette colors. Designed for
deterministic CI runs and quick visual QA (optional grid preview).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
from PIL import Image
from PIL.ImageFont import FreeTypeFont, ImageFont

# MBAR-approved material colors (RGB 0-255)
MATERIAL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "plaster_marmorino_westwood_beige": (240, 230, 215),
    "stone_bokara_coastal": (195, 175, 155),
    "cladding_sculptform_warm": (200, 170, 140),
    "screens_grey_gum": (145, 140, 135),
    "equitone_lt85": (95, 95, 100),
    "bison_weathered_ipe": (165, 160, 165),
    "dark_bronze_anodized": (75, 60, 50),
    "louvretec_powder_white": (245, 245, 245),
}


def _seed_for(name: str, base_seed: int) -> int:
    """Stable per-material seed; avoids identical noise across plates."""
    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8, person=b"MBAR-TEX")
    return base_seed ^ int.from_bytes(h.digest(), "little")


def create_texture(
    base_color: Tuple[int, int, int],
    size: int = 512,
    *,
    seed: int | None = None,
    noise_sigma: float = 8.0,
    gradient_amp: float = 5.0,
    tileable: bool = False,
) -> Image.Image:
    """Create a subtle procedural texture with low-amplitude variation.

    Why: adds believable micro-variation while keeping output deterministic and fast.
    """
    # Base plane
    base = np.ones((size, size, 3), dtype=np.float32)
    base *= np.array(base_color, dtype=np.float32).reshape(1, 1, 3)

    # Noise (tileable uses wrapped cosines to avoid seams)
    if tileable:
        y = np.arange(size, dtype=np.float32)
        x = np.arange(size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        # Two orthogonal cos components for seamless periodic noise
        periodic = (np.cos(2 * np.pi * xx / size) + np.cos(2 * np.pi * yy / size)) / 2.0
        noise = (periodic[:, :, None]) * noise_sigma
        base += noise
    else:
        rng = np.random.default_rng(seed)
        base += rng.normal(0.0, noise_sigma, (size, size, 3)).astype(np.float32)

    # Gentle 2D gradient for depth
    yg = np.linspace(-gradient_amp, gradient_amp, size, dtype=np.float32)[:, None]
    xg = np.linspace(-gradient_amp, gradient_amp, size, dtype=np.float32)[None, :]
    base += (yg + xg)[:, :, None]

    # Clamp → uint8
    out = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _save_image(img: Image.Image, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt.lower() == "jpg" or fmt.lower() == "jpeg":
        img.save(path.with_suffix(".jpg"), quality=92, subsampling=0, optimize=True)
    elif fmt.lower() == "tiff":
        img.save(path.with_suffix(".tiff"), compression="tiff_lzw")
    else:
        img.save(path.with_suffix(".png"), optimize=True)


def _make_grid(
    tiles: Mapping[str, Image.Image],
    *,
    cols: int = 4,
    pad: int = 12,
    bg: Tuple[int, int, int] = (32, 32, 32),
    label: bool = True,
) -> Image.Image:
    """Simple label grid for QA."""
    if not tiles:
        return Image.new("RGB", (256, 256), bg)
    names = list(tiles.keys())
    w, h = next(iter(tiles.values())).size
    rows = int(np.ceil(len(names) / cols))
    grid_w = cols * w + (cols + 1) * pad
    grid_h = rows * h + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), bg)
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(grid)
        font: FreeTypeFont | ImageFont | None
        try:
            from PIL import ImageFont as IF
            font = IF.truetype("DejaVuSans.ttf", size=max(12, w // 20))
        except Exception:
            from PIL import ImageFont as IF
            font = IF.load_default()
    except Exception:
        draw = None
        font = None

    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        grid.paste(tiles[name], (x, y))
        if draw and font and label:
            text = name
            draw.rectangle((x, y + h - 22, x + w, y + h), fill=(0, 0, 0, 128))
            draw.text((x + 6, y + h - 20), text, fill=(255, 255, 255), font=font)
    return grid


def generate_all_textures(
    out_dir: Path,
    *,
    size: int = 512,
    fmt: str = "png",
    seed: int = 42,
    noise_sigma: float = 8.0,
    gradient_amp: float = 5.0,
    tileable: bool = False,
    make_grid: bool = True,
    grid_cols: int = 4,
) -> Mapping[str, Path]:
    """Generate textures for all MBAR materials and optionally a preview grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    previews: Dict[str, Image.Image] = {}

    for name, color in MATERIAL_COLORS.items():
        mat_seed = _seed_for(name, seed)
        img = create_texture(
            color,
            size=size,
            seed=mat_seed,
            noise_sigma=noise_sigma,
            gradient_amp=gradient_amp,
            tileable=tileable,
        )
        dst = out_dir / f"{name}.{fmt.lower() if fmt.lower() != 'jpeg' else 'jpg'}"
        _save_image(img, dst, fmt)
        written[name] = dst
        previews[name] = img

    if make_grid:
        grid = _make_grid(previews, cols=grid_cols)
        _save_image(grid, out_dir / "preview_grid.png", "png")

    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate deterministic MBAR board material textures."
    )
    p.add_argument("--out", type=Path, default=Path("textures/board_materials"),
                   help="Output directory (default: textures/board_materials)")
    p.add_argument("--size", type=int, default=512, help="Texture size in pixels (square)")
    p.add_argument("--format", choices=("png", "jpg", "jpeg", "tiff"), default="png",
                   help="Image format for individual textures")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    p.add_argument("--tileable", action="store_true", help="Use seamless periodic noise")
    p.add_argument("--noise-sigma", type=float, default=8.0, help="Noise std dev in RGB units")
    p.add_argument("--gradient-amp", type=float, default=5.0, help="Gradient amplitude in RGB units")
    p.add_argument("--no-grid", action="store_true", help="Do not write preview grid")
    p.add_argument("--grid-cols", type=int, default=4, help="Preview grid columns")
    return p.parse_args()


def main(argv: Iterable[str] | None = None) -> None:
    ns = _parse_args()  # argparse already reads sys.argv; argv is for future hookability
    outputs = generate_all_textures(
        ns.out,
        size=ns.size,
        fmt=ns.format,
        seed=ns.seed,
        noise_sigma=ns.noise_sigma,
        gradient_amp=ns.gradient_amp,
        tileable=ns.tileable,
        make_grid=not ns.no_grid,
        grid_cols=ns.grid_cols,
    )
    for name, path in outputs.items():
        print(f"Created: {path}")
    if not ns.no_grid:
        print(f"Created: {ns.out / 'preview_grid.png'}")


if __name__ == "__main__":
    main()


__all__ = [
    "MATERIAL_COLORS",
    "create_texture",
    "generate_all_textures",
    "main",
]