# file: tools/gen_legacy_shims.py
"""
Auto-generate back-compat shims so legacy imports keep working when code lives under src/.

- Scans tests/ for imports.
- For single-segment modules that fail to import, tries src.<mod>.
- If src.<mod> exists, writes ./<mod>.py that re-exports public names.

Idempotent and safe on CI worktrees.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import Iterable, Set

HEADER = "# Auto-generated shim by tools/gen_legacy_shims.py; do not edit."

SKIP_NAMES = {
    "src",
    "tests",
    "__future__",
    "typing",
    "dataclasses",
    "unittest",
    "pytest",
    "numpy",
    "PIL",
    "cv2",
    "scipy",
    "typer",
    "tqdm",
    "json",
    "pathlib",
    "argparse",
    "logging",
    "os",
    "sys",
    "math",
    "re",
    "io",
    "importlib",
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
                    continue  # relative imports aren't our concern
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

def _write_shim(mod: str, root: Path) -> Path:
    target = root / f"{mod}.py"
    # Idempotent: if already ours, skip
    if target.exists():
        try:
            if HEADER in target.read_text(encoding="utf-8"):
                return target
        except Exception:
            pass
    code = f'''from __future__ import annotations
{HEADER}

# pylint: disable=unused-wildcard-import,wildcard-import,import-error
# ruff: noqa: F401,F403

import importlib as _importlib as _importlib
_mod = _importlib.import_module("src.{mod}")
globals().update({{k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")}})
__all__ = [k for k in globals() if not k.startswith("_")]
'''
    target.write_text(code, encoding="utf-8")
    return target

def main(argv: list[str] | None = None) -> int:
    repo = Path(".").resolve()
    tests_dir = repo / "tests"
    if not tests_dir.exists():
        print("No tests/ directory found; nothing to do.")
        return 0

    # Ensure src is a package for importlib (helps some toolchains)
    src_pkg = repo / "src" / "__init__.py"
    if not src_pkg.exists():
        src_pkg.parent.mkdir(parents=True, exist_ok=True)
        src_pkg.write_text("", encoding="utf-8")

    mods = _collect_test_imports(tests_dir)
    if not mods:
        print("No candidate legacy imports found.")
        return 0

    created: list[Path] = []
    for name in sorted(mods):
        if _importable(name):
            continue
        if _importable(f"src.{name}"):
            p = _write_shim(name, repo)
            created.append(p)

    if created:
        print("Created shims:")
        for p in created:
            print(f" - {p.relative_to(repo)}")
    else:
        print("No shims needed.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
