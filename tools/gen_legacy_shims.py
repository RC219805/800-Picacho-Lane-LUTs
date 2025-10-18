# file: tools/gen_legacy_shims.py
"""
Auto-generate back-compat shims so legacy imports keep working when code lives under src/.

- Scans tests/ for imports.
- For single-segment modules that fail to import, tries src.<mod>.
- If src.<mod> exists, writes ./<mod>.py that re-exports public names.

Use --fail-on-create in CI to force committing shims instead of generating them ad hoc.
"""
from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path
from typing import Iterable, Set, List

HEADER = "# Auto-generated shim by tools/gen_legacy_shims.py; do not edit."

SKIP_NAMES = {
    "src", "tests", "__future__", "typing", "dataclasses",
    "unittest", "pytest", "numpy", "PIL", "cv2", "scipy",
    "typer", "tqdm", "json", "pathlib", "argparse", "logging",
    "os", "sys", "math", "re", "io", "importlib",
}


def _iter_py(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (q for q in p.rglob("*.py") if q.is_file())
        elif p.is_file() and p.suffix == ".py":
            yield p


def _collect_test_imports(test_root: Path) -> Set[str]:
    mods: Set[str] = set()
    for path in _iter_py([test_root]):
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    name = (n.name or "").split(".")[0]
                    if name and name not in SKIP_NAMES:
                        mods.add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                base = (node.module or "").split(".")[0]
                if base and base not in SKIP_NAMES:
                    mods.add(base)
    return mods


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def _shim_code(mod: str) -> str:
    return f'''"""Auto-generated legacy shim for `{mod}`. Re-exports from `src.{mod}`."""
from __future__ import annotations
{HEADER}

# pylint: disable=unused-wildcard-import,wildcard-import,import-error
# ruff: noqa: F401,F403

import importlib as _importlib
_mod = _importlib.import_module("src.{mod}")
globals().update({{k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")}})
__all__ = ["EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionStatus"] if "{mod}" == "evolutionary_checkpoint" else [k for k in globals() if not k.startswith("_")]
'''


def _write_shim(mod: str, root: Path) -> Path:
    target = root / f"{mod}.py"
    target.write_text(_shim_code(mod), encoding="utf-8")
    return target


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root (default: .)")
    ap.add_argument(
        "--fail-on-create",
        action="store_true",
        help="Exit non-zero if any new shim would be created (do not write).",
    )
    args = ap.parse_args(argv)

    repo = Path(args.root).resolve()
    tests_dir = repo / "tests"
    if not tests_dir.exists():
        print("No tests/ directory found; nothing to do.")
        return 0

    # ensure src is a package
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "__init__.py").touch()

    to_create: List[str] = []
    updated: List[Path] = []

    for name in sorted(_collect_test_imports(tests_dir)):
        if _importable(name):
            continue
        if _importable(f"src.{name}"):
            target = repo / f"{name}.py"
            if not target.exists():
                to_create.append(name)
            else:
                desired = _shim_code(name)
                existing = target.read_text(encoding="utf-8")
                if existing != desired and not args.fail_on_create:
                    target.write_text(desired, encoding="utf-8")
                    updated.append(target)

    if to_create:
        print("Shims missing and required:")
        for mod in to_create:
            print(f" - {mod}.py (re-exports from src.{mod})")
        if args.fail_on_create:
            print("❌ New shims would be created. Commit shims or add modules under src/.")
            return 1
        for mod in to_create:
            _write_shim(mod, repo)
        print(f"Created {len(to_create)} shims.")

    if updated:
        print("Updated existing shims:")
        for p in updated:
            print(f" - {p.relative_to(repo)}")

    if not to_create and not updated:
        print("No shims needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
