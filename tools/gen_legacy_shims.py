# file: tools/gen_legacy_shims.py
"""
Scan tests for legacy top-level imports and re-export from src.<mod> via shims.

Usage:
  Report only:   python tools/gen_legacy_shims.py --report
  Write shims:   python tools/gen_legacy_shims.py --write
  JSON report:   python tools/gen_legacy_shims.py --json
  Limit files:   python tools/gen_legacy_shims.py --modules tests/test_*.py,tests/smoke/**.py
                  (you can repeat --modules; comma-separated globs are supported)

In CI, keep using: --fail-on-create to enforce that shims are already committed.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

HEADER = "# Auto-generated shim by tools/gen_legacy_shims.py; do not edit."

SKIP_NAMES = {
    "src","tests","__future__","typing","dataclasses","unittest","pytest","numpy","PIL","cv2",
    "scipy","typer","tqdm","json","pathlib","argparse","logging","os","sys","math","re","io","importlib",
}

# ------------------------------- file discovery -------------------------------

def _iter_py(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (q for q in p.rglob("*.py") if q.is_file())
        elif p.is_file() and p.suffix == ".py":
            yield p

def _expand_module_globs(root: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()
    for raw in patterns:
        pat = raw.strip()
        if not pat:
            continue
        # Path.glob supports ** recursive patterns
        for m in root.glob(pat):
            if m.is_file() and m.suffix == ".py" and m not in seen:
                files.append(m)
                seen.add(m)
    return files

# ------------------------------- import scanning ------------------------------

def _collect_imports_from_files(files: Iterable[Path]) -> Set[str]:
    mods: Set[str] = set()
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    base = (n.name or "").split(".")[0]
                    if base and base not in SKIP_NAMES:
                        mods.add(base)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                base = (node.module or "").split(".")[0]
                if base and base not in SKIP_NAMES:
                    mods.add(base)
    return mods

def _collect_test_imports(test_root: Path) -> Set[str]:
    return _collect_imports_from_files(_iter_py([test_root]))

# --------------------------------- utilities ---------------------------------

def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

def _strict_evo_shim() -> str:
    return '''"""Compatibility shim for legacy imports. Re-exports src.evolutionary."""
from __future__ import annotations
{header}

from src.evolutionary import (
    EvolutionStatus,
    EvolutionOutcome,
    EvolutionaryCheckpoint,
)

__all__ = ["EvolutionaryCheckpoint", "EvolutionaryOutcome", "EvolutionStatus"]
'''.format(header=HEADER)

def _generic_shim(mod: str) -> str:
    return f'''"""Auto-generated legacy shim for `{mod}`. Re-exports from `src.{mod}`."""
from __future__ import annotations
{HEADER}

# pylint: disable=unused-wildcard-import,wildcard-import,import-error
# ruff: noqa: F401,F403

import importlib as _importlib
_mod = _importlib.import_module("src.{mod}")
globals().update({{k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")}})
__all__ = [k for k in globals() if not k.startswith("_")]
'''

def _shim_code(mod: str) -> str:
    return _strict_evo_shim() if mod == "evolutionary_checkpoint" else _generic_shim(mod)

def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

# ----------------------------------- scan ------------------------------------

def scan(repo: Path, limit_files: Sequence[Path] | None = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (ok, missing_shims, updatable_shims) as module names.
    If limit_files is provided, only those files are scanned; otherwise scans tests/.
    """
    if limit_files is not None:
        files = list(limit_files)
    else:
        tests_dir = repo / "tests"
        if not tests_dir.exists():
            return [], [], []
        files = list(_iter_py([tests_dir]))

    mods = sorted(_collect_imports_from_files(files))
    ok, missing, updatable = [], [], []
    for name in mods:
        if _importable(name):
            ok.append(name)
            continue
        if _importable(f"src.{name}"):
            target = repo / f"{name}.py"
            desired = _shim_code(name)
            if not target.exists():
                missing.append(name)
            else:
                existing = target.read_text(encoding="utf-8")
                if existing != desired:
                    updatable.append(name)
        else:
            # Neither top-level nor src.<name> importable: ignore (not shim-able).
            pass
    return ok, missing, updatable

# ----------------------------------- CLI -------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root (default: .)")
    ap.add_argument("--report", action="store_true", help="Only report; do not write.")
    ap.add_argument("--write", action="store_true", help="Write/refresh shims as needed.")
    ap.add_argument("--json", action="store_true", help="Print JSON result to stdout.")
    ap.add_argument("--fail-on-create", action="store_true",
                    help="Exit non-zero if any new shim would be created (do not write).")
    ap.add_argument(
        "--modules",
        action="append",
        default=[],
        help="Limit scan to given glob(s), e.g. 'tests/test_*.py'. "
             "May be repeated or comma-separated.",
    )
    args = ap.parse_args()

    repo = Path(args.root).resolve()
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "__init__.py").touch()

    # Build limited file list if --modules provided
    limit_files: List[Path] | None = None
    if args.modules:
        patterns: List[str] = []
        for entry in args.modules:
            patterns.extend([p.strip() for p in entry.split(",") if p.strip()])
        matches = _expand_module_globs(repo, patterns)
        if not matches:
            msg = f"No files matched --modules patterns: {patterns}"
            if args.json:
                print(json.dumps({"ok": [], "missing": [], "updatable": [], "note": msg}, indent=2))
            else:
                print(msg)
            return 0
        limit_files = matches
        print(f"Scanning subset ({len(matches)} files) from --modules filter.")

    ok, missing, updatable = scan(repo, limit_files=limit_files)

    if args.json:
        print(json.dumps({"ok": ok, "missing": missing, "updatable": updatable}, indent=2))
    else:
        print(f"OK modules: {ok}")
        if missing:
            print(f"Missing shims   : {missing}")
        if updatable:
            print(f"Updatable shims : {updatable}")

    if args.fail_on_create and missing:
        print("❌ New shims would be created. Commit shims or add src.<mod> modules.")
        return 1

    if args.write:
        for name in missing + updatable:
            p = repo / f"{name}.py"
            _write(p, _shim_code(name))
            print(f"✓ wrote {p.relative_to(repo)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
