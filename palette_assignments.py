"""Utilities for saving and loading aerial material palette assignments."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, MutableMapping

import json

if TYPE_CHECKING:  # pragma: no cover
    from board_material_aerial_enhancer import MaterialRule


_PALETTE_VERSION = "1.0"


def _rule_lookup(rules: Iterable["MaterialRule"]) -> Mapping[str, "MaterialRule"]:
    lookup: MutableMapping[str, "MaterialRule"] = {}
    for rule in rules:
        lookup[rule.name] = rule
    return lookup


def load_palette_assignments(
    path: str | Path,
    rules: Iterable["MaterialRule"] | Mapping[str, "MaterialRule"] | None = None,
) -> dict[int, "MaterialRule"]:
    """Load palette assignments from JSON and map them to material rules."""
    palette_path = Path(path)
    if not palette_path.exists():
        raise FileNotFoundError(palette_path)

    data = json.loads(palette_path.read_text())
    assignments = data.get("assignments", {})
    if not isinstance(assignments, dict):
        raise ValueError("Palette file missing 'assignments' mapping")

    if rules is None:
        raise ValueError("Material rules are required to load palette assignments")

    if isinstance(rules, Mapping):
        lookup = {name: rule for name, rule in rules.items()}
    else:
        lookup = _rule_lookup(rules)

    resolved: dict[int, "MaterialRule"] = {}
    for label_str, material_name in assignments.items():
        try:
            label = int(label_str)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid cluster label '{label_str}' in palette file") from exc

        if material_name not in lookup:
            raise ValueError(f"Unknown material '{material_name}' in palette file")
        resolved[label] = lookup[material_name]

    return resolved


def save_palette_assignments(
    assignments: Mapping[int, "MaterialRule"],
    path: str | Path,
    *,
    version: str = _PALETTE_VERSION,
) -> Path:
    """Persist palette assignments to JSON for reuse."""
    palette_path = Path(path)
    palette_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": version,
        "assignments": {str(label): rule.name for label, rule in sorted(assignments.items())},
    }
    palette_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return palette_path
