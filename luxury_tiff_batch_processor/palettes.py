# path: luxury_tiff_batch_processor/palettes.py
"""Palette-to-texture resolver for lux-batch.

Resolves a simple palette reference (name, directory path, or JSON mapping file)
into a mapping of material name -> texture file path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _scan_dir_for_textures(folder: Path) -> Dict[str, Path]:
    """Return mapping of '<basename>' -> absolute path for image files in *folder*."""
    textures: Dict[str, Path] = {}
    if not folder.exists() or not folder.is_dir():
        return textures
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            name = p.stem  # e.g., 'plaster_marmorino_westwood_beige'
            textures[name] = p.resolve()
    return textures


def _load_json_mapping(file_path: Path) -> Dict[str, Path]:
    """Load a JSON mapping of name -> path; ignore entries that don't exist."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, Mapping):
        return {}
    out: Dict[str, Path] = {}
    for k, v in data.items():
        try:
            key = str(k).strip()
            path = Path(str(v)).expanduser().resolve()
        except Exception:
            continue
        if key and path.exists() and path.is_file():
            out[key] = path
    return out


def _candidate_dirs_for_known_palette(base_search: Optional[Path]) -> Iterable[Path]:
    """Yield likely texture folders for known palette tokens."""
    # 1) Environment variable
    env_dir = os.getenv("MBAR_TEXTURES_DIR")
    if env_dir:
        yield Path(env_dir).expanduser()

    # 2) CWD textures/board_materials
    cwd = Path.cwd()
    yield cwd / "textures" / "board_materials"

    # 3) Base search (repo/package root) variants
    if base_search:
        yield base_search / "textures" / "board_materials"
        yield base_search / "tools" / "textures" / "board_materials"

    # 4) Walk a few parents looking for 'textures/board_materials'
    probe = cwd
    for _ in range(4):
        probe = probe.parent
        yield probe / "textures" / "board_materials"
        yield probe / "tools" / "textures" / "board_materials"


def resolve_texture_map(
    palette: str | os.PathLike[str],
    *,
    base_search: Optional[Path] = None,
) -> Dict[str, Path]:
    """Resolve *palette* into a mapping of material name -> texture path.

    Accepts:
      - Well-known tokens: 'mbar', 'board_materials' (searches common folders)
      - A directory path: scans for image files
      - A JSON file: loads name -> path mapping

    Returns {} on failure (graceful).
    """
    ref = str(palette).strip()

    # Directory / file?
    p = Path(ref).expanduser()
    if p.exists():
        if p.is_dir():
            return _scan_dir_for_textures(p)
        if p.is_file() and p.suffix.lower() == ".json":
            return _load_json_mapping(p)
        # If it's a single image, map by stem
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return {p.stem: p.resolve()}
        return {}

    # Known tokens
    token = ref.lower()
    if token in {"mbar", "board_materials", "mbar-board"}:
        for folder in _candidate_dirs_for_known_palette(base_search):
            textures = _scan_dir_for_textures(folder)
            if textures:
                return textures

    return {}