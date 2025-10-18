# file: tests/test_evolutionary_shim_guard.py
from __future__ import annotations

import importlib

def test_evolutionary_checkpoint_shim_is_strict():
    # Import shim and canonical module
    sh = importlib.import_module("evolutionary_checkpoint")
    new = importlib.import_module("src.evolutionary")

    # Identity: shim must re-export the exact types/enum
    assert sh.EvolutionStatus is new.EvolutionStatus
    assert sh.EvolutionOutcome is new.EvolutionOutcome
    assert sh.EvolutionaryCheckpoint is new.EvolutionaryCheckpoint

    # Public surface: only these three symbols allowed
    public = {n for n in dir(sh) if not n.startswith("_")}
    expected = {"EvolutionStatus", "EvolutionOutcome", "EvolutionaryCheckpoint"}
    assert public == expected, f"Unexpected public names: {sorted(public - expected)}"

    # __all__ must be exactly these names (guard against accidental edits)
    assert getattr(sh, "__all__", None) == sorted(list(expected)) or getattr(sh, "__all__", None) == list(expected)

# -----------------------------------------------------------------------------


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

def _write_shim(mod: str, root: Path) -> Path:
    target = root / f"{mod}.py"
    code = f'''"""Auto-generated legacy shim for `{mod}`. Re-exports from `src.{mod}`."""
from __future__ import annotations
{HEADER}

# pylint: disable=unused-wildcard-import,wildcard-import,import-error
# ruff: noqa: F401,F403

import importlib as _importlib
_mod = _importlib.import_module("src.{mod}")
globals().update({{k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")}})
__all__ = [k for k in globals() if not k.startswith("_")]
'''
    target.write_text(code, encoding="utf-8")
    return target

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root (default: .)")
    ap.add_argument("--fail-on-create", action="store_true", help="Exit non-zero if any new shim is created.")
    args = ap.parse_args(argv)

    repo = Path(args.root).resolve()
    tests_dir = repo / "tests"
    if not tests_dir.exists():
        print("No tests/ directory found; nothing to do.")
        return 0

    # ensure src is a package
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "__init__.py").touch()

    created: List[Path] = []
    for name in sorted(_collect_test_imports(tests_dir)):
        if _importable(name):
            continue
        if _importable(f"src.{name}"):
            created.append(_write_shim(name, repo))

    if created:
        print("Created shims:")
        for p in created:
            print(f" - {p.relative_to(repo)}")
        if args.fail_on_create:
            print("❌ New shims were needed. Commit shims to the repo or add explicit modules under src/.")
            return 2
    else:
        print("No shims needed.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# -----------------------------------------------------------------------------


# patch: .github/workflows/build.yml  (replace shim step)
# - name: Generate legacy shims from tests
#   run: python tools/gen_legacy_shims.py
# + name: Generate legacy shims from tests (fail if new shims are needed)
# + run: python tools/gen_legacy_shims.py --fail-on-create


# patch: .github/workflows/pylint.yml  (replace shim step)
# - name: Generate legacy shims from tests
#   run: python tools/gen_legacy_shims.py
# + name: Generate legacy shims from tests (fail if new shims are needed)
# + run: python tools/gen_legacy_shims.py --fail-on-create
